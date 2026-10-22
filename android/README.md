# Moshi 在 Snapdragon 手机上的部署

目标设备：HONOR BKQ-AN20，SM8850（HTP v81），Android 16，QAIRT 2.45.0.260326。
本文是当前唯一有效的部署说明。早期 Codex 实验记录和一次性脚本已删除，可在 git tag `moshi-pre-cleanup` 中找回。

## 现状

| 项目 | 当前值 |
| --- | --- |
| App | `com.overmindy.mimicodecdemo` 4.0.0-lpbq4（versionCode 17）；内置演示 36 帧 5.75 s，RTF 2.00，加载 3.4 s |
| 模型包 | `/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-v4-lpbq4`，15 个 context + CPU embedding 表，约 5.0 GB（上一版 `lm-v3-dep8` 仍在手机上） |
| 38 帧输入（3.04 s） | 暖运行 5.7–6.0 s（同期 W8 value/down 版 6.5–7.2 s），首段 PCM 约 0.31 s，输出 2.96 s |
| 每帧耗时 | 约 125–140 ms：Temporal 85–100、DepFormer 15–20、Mimi 13–14、head 3、frontend（CPU）0.03 |
| Temporal 质量（8 段 held-out，1558 帧，4 段串联 QuantSim，对 BF16） | 全 LPBQ W4：文本 top-1 97.95%，音频 cb0 94.4%，8 路全对 85.1%；之前的 W8 value/down：98.4% / 95.4% / 90.1% |
| DepFormer W8 质量（同一批 1558 帧，QuantSim，对 BF16） | 串联 8 路全对 92.4%（旧 FP16 权重的 legacy 图 85.3%，FP32 上限 94.0%） |

质量指标是 teacher-forced 的：每帧给相同的 BF16 历史，比较该帧的文本 argmax 和 DepFormer 贪心音频 token。
它不是听感或 ASR 评测；free-running 贪心解码即使全 W8 也会在十几帧内出现 token 分叉。

## 模型与数据流

```text
24 kHz PCM, 1920 样本/帧 (80 ms)
  → Mimi encoder ──8 codes──┐
                             ▼
          17 路延迟环 (moshi_lm_scheduler.h)
                             ▼
  CPU embedding 求和 → Temporal part 0..3（各 8 层）→ head ──text──┐
                                               ▼          │
                              DepFormer codebook 0..7（顺序执行）
                                               ▼
                         延迟对齐 → Mimi decoder → PCM
```

Temporal 与 DepFormer 都有真实的顺序依赖，不能并行。第一帧只用于预热，所以 N 个输入帧输出 N−1 帧。

| 文件 | 作用 |
| --- | --- |
| `mimi_qnn_runner.cpp` | 原生运行器：加载 context、burst 性能投票、共享内存 KV、Temporal v2、Mimi、DepFormer |
| `moshi_lm_scheduler.h` | Moshi `LMGen` 的 17 路延迟环 |
| `demo/` | Android App（Java UI、JNI、`build.sh`） |
| `moshi_reply_latency_probe.cpp` | 完整链路 CLI：首次/暖运行、首 PCM、逐帧时间线 |
| `moshi_graph_bench.cpp` | 单个 context 计时（≥1 MiB 输入走共享内存） |
| `moshi_temporal_chain_probe.cpp` | 4 个 Temporal part 跑捕获的 layer-0 流，输出最终 hidden 供服务器打分 |
| `test_moshi_scheduler*.cpp` | 延迟环单元测试；`_trace` 读 `scripts/moshi_export_scheduler_trace.py` 生成的上游 trace |

### Temporal part 的图接口

由 `scripts/moshi_temporal_part.py` 导出，布局与 `templates/llm` 中的官方 LLM 一致：

- 投影是 1×1 Conv；q/k/v 按头拆分（SHA）。Moshi 的交错 RoPE 通过置换 q/k 行变成半分布局。
- 输入：`hidden` [1,1,4096]、`attention_mask` [1,1,1,1000]、`position_cos/sin` [1,1,1,64]，
  每层 `past_key_{l}_in` [32,1,128,999]、`past_value_{l}_in` [32,1,999,128]。
- 输出：`hidden_out`，以及每层只有新的一帧 `past_key_{l}_out` [32,1,128,1]、`past_value_{l}_out` [32,1,1,128]。
- I/O 是定点的：hidden/mask/cos/sin 为 uint16，KV 为 uint8（输入输出 encoding 绑定）。
  运行器按 context 里的 scale/offset 做量化和反量化（`real = scale × (q + offset)`）。
- KV ring 由宿主持有，放在注册过的 rpcmem 里，HTP 直接读取，每帧只写入一个槽。
  空槽必须填 0.0 的量化值（零点），不能填字节 0。
- mask：当前帧和最近 999 帧有效（0），其余为 −50；RoPE 使用绝对位置。

## 环境

**Mac（构建 App 和原生程序、ADB）**

```sh
export ANDROID_SDK_ROOT=/Users/e1ainay/Library/Android/sdk          # build-tools 35.0.0、android-34 在这里
export ANDROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/30.0.15729638
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
export QNN_INCLUDE=/Users/e1ainay/d_personal/GenieX/third-party/geniex-qairt/qnn-api/include
export QNN_ANDROID_LIB_DIR=/Users/e1ainay/d_personal/GenieX/third-party/geniex-qairt/third-party/android
export KEYSTORE_PATH=/Users/e1ainay/d_personal/ai-hub-models/android/demo/build/demo.keystore   # 原签名，必须沿用
export SAMPLE_WAV=/path/to/24k_mono_pcm16.wav                          # 打进 APK 的内置演示音频
```

**Linux 服务器（导出、量化、编译）**：`ssh -p 40023 user@117.74.67.2`

| 用途 | 路径 |
| --- | --- |
| BF16 权重 | `/data2/liuguohong/moshiko-pytorch-bf16`（`--model-dir`） |
| QAIRT SDK | `/data2/yejialei/sdk/qairt/2.45.0.260326` |
| 基础 Python（torch、onnx、仓库 editable 安装） | `/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python` |
| AIMET 环境 | `/data2/yejialei/envs/aimet228`（`--system-site-packages`，aimet-onnx 2.28.0 + onnxruntime-gpu 1.22）；运行前先 `source env.sh` |
| 转换器专用 NumPy 1.26 | `/data2/yejialei/tmp/moshi-qdq-0-1-cpu-check-v1/dlc_fusion_diagnostics/python-deps-numpy126` |
| Temporal v2 工作目录 | `/data2/yejialei/moshi-temporal-v2-pilot`（capture、encodings、build、streams） |

`/data2`、`/data3` 和 `/` 都接近满盘，并且与他人共用。AIMET 会把超过 2 GB 的模型写到 `TMPDIR`，
所以一定要把 `TMPDIR` 指到 `/data2` 上的工作目录，结束后清理。8 层 part 构建时磁盘峰值约 15 GB。
这台服务器的 aliyun 镜像下载很慢，大的 wheel 可以先在 Mac 上下载，再分块并行上传。

## Temporal v2：从 BF16 到 context

脚本都在 `scripts/` 下。在服务器上，先 `cd` 到存放脚本的目录再运行，脚本之间互相 import。

1. **采集边界激活**（GPU，约 10 分钟）：用 LibriSpeech 跑流式 Mimi 和贪心 LMGen，
   记录第 0/8/16/24 层的输入和第 31 层的输出。校准用 24 个说话人，held-out 用另外 8 个，两组不重叠。
   每段末尾补 3 s 静音，以覆盖 Moshi 回答的部分。

   ```sh
   NO_CUDA_GRAPH=1 python moshi_temporal_capture.py --model-dir $BF16 \
     --train-roots .../train-clean-100 .../train-other-500 \
     --test-roots .../test-clean .../test-other --output-dir $W/capture
   ```

2. **校准并做 token 级门限**（AIMET + CUDA）：每个 part 生成一个 QuantSim。部署用 `--weights lpbq4`：
   全部权重 LPBQ int4（block 64）；激活 int16；KV 为 8-bit，输入输出绑定。
   `lpbq4_w8vd`（value 和 MLP down 为 int8）质量更好，但每帧多读约 1 GB 权重，慢约 25 ms，见上面的现状表。
   校准样本取 8 段校准音频的前 120 帧，共 64 个 AR1 步，**必须包含第 0–3 帧**（第 0 帧是全初始 token 的 attention sink）。
   部署用的 encodings 就是这一步生成的，没有使用 SeqMSE。

   ```sh
   source /data2/yejialei/envs/aimet228/env.sh
   TMPDIR=$W/tmp python moshi_temporal_segment_eval.py --model-dir $BF16 --capture-dir $W/capture \
     --work-dir $W/lpbq4 --parts 0,1,2,3 --mode each --held-out-clips 8   # 逐段：保存 temporal_part_{p}.encodings
   TMPDIR=$W/tmp python moshi_temporal_segment_eval.py ... --mode together   # 4 段串联；复用 encodings，不重新校准
   ```

   BF16 模型（15 GB）和 7 GB 的 FP32 QuantSim 放不进一块只剩 ~27 GB 的共享 GPU 时，
   用 `CUDA_VISIBLE_DEVICES=a,b MOSHI_ORT_DEVICE=1` 把 QuantSim 放到第二块卡。8 段 held-out 的门限在共享 GPU 上
   每段 `each` 约 70 分钟、`together` 约 3.5–4 小时；筛选变体先用 `moshi_temporal_token_eval.py`。

3. **构建 context**（每段约 8 分钟）：导出 ONNX，拷入 encodings，然后转换、量化、编译。

   ```sh
   python moshi_temporal_part.py --model-dir $BF16 --output-dir $W/build/part$P \
     --first-layer $((8*P)) --layers 8 --parity-steps 4 --calibration-steps 0
   cp $W/lpbq4/temporal_part_$P.encodings $W/build/part$P/
   bash moshi_quantized_build.sh $W/build/part$P temporal_part_$P      # 默认 O3 + vtcm 8 MB
   ```

   `moshi_quantized_build.sh` 把 DLC 的图命名为 `NAME`，这样 `backend.json` 里的 O3 和 VTCM 设置才会生效
   （默认选项下 8 层 part 要 92–101 ms，O3 下只要 28 ms）。转换器经过 `moshi_guard_qairt_converter.py`，
   用来阻止 QAIRT 2.45 错误地融合 RMSNorm。LPBQ 只能用 Conv：LPBQ 的 FullyConnected 会让 context 生成器段错误。

4. **手机数值对照**：`moshi_temporal_device_eval.py --export` 导出 held-out 的 layer-0 流。
   手机上用 `moshi_temporal_chain_probe LIB in.f32 out.f32 part0..3.bin` 跑出结果，
   再把 `out.f32` 传回服务器，不带 `--export` 运行 `moshi_temporal_device_eval.py`，得到与 QuantSim 同口径的 token 指标。

快速筛选工具：`moshi_temporal_token_eval.py` 在 PyTorch 里伪量化权重，每个变体约 1.5 分钟。
`moshi_temporal_sweep.py` 在 2 层子图上一次跑多个 QuantSim 变体，并拆分权重误差和激活误差。

## DepFormer W8

旧流程把 DepFormer 权重留成了 FP16（每个 codebook 约 85M 参数、171 MB）。现在用 AIMET 重新量化：
int8 per-channel 权重、int16 激活，每个 codebook 约 89 MB（codebook 0 含文本 embedding，150 MB）。
手机上每个 codebook 从 4–5 ms 降到约 1.7–2.9 ms。

```sh
source /data2/yejialei/envs/aimet228/env.sh
TMPDIR=$W/tmp python moshi_depformer_quantize.py --model-dir $BF16 --capture-dir $W/capture \
  --work-dir $W/depformer_w8          # 约 25 分钟；--no-legacy 跳过 CPU 上很慢的旧图对照
bash moshi_depformer_build.sh $W/depformer_w8   # 8 个 context 并行构建，约 5 分钟，输出到 bundle/
```

- 输入取自 capture 的第 31 层输出，经 BF16 head 得到 DepFormer 的 `temporal` 和文本 token。
  校准用 8 段校准音频、每段均匀取 32 帧；门限用全部 8 段 held-out（1558 帧），参照是 BF16 贪心 token。
- 门限结果（8 路全对 / 各 codebook）：串联 92.4% / 95.1–99.6%，teacher-forced 93.6% / 98.5–99.6%；
  旧 legacy 图分别为 85.3% 和 87.9%，FP32 上限 94.0%。
- **attention 不能用 −inf mask。** 每个 codebook 的步号是常量，`DepFormerStepBlock` 直接切掉后面的槽位，
  数学上等价。用 −inf mask 时 16-bit 激活的范围被撑爆，只有不需要 mask 的 codebook 0 和 7 正常，其余掉到 55–80%。
- 转换器没有保留 FP32 I/O，图的 I/O 是 uint16 定点。运行器按每个张量自己的 scale/offset 写入和读出，
  FP32 的旧图也能用同一段代码。
- W4 不可用：`--weights pc4` 串联 8 路全对 49.6%，`lpbq4` 62.6%（W8 为 92.4%）。
  另外 DepFormer 的 Linear 导出为 MatMul/FC，LPBQ FC 会让 context 生成器直接退出，要用 W4 得先改成 1×1 Conv。

## CPU frontend

frontend 只是 1 行文本 embedding 加 16 行音频 embedding。`moshi_export_frontend_table.py` 把 17 张 BF16 表
原样导出成 `frontend_embeddings.bin`（531 MB，`MOSHIEMB` 头 + 行数 + BF16 数据），运行器 mmap 后用 FP32 求和，
每帧约 0.03 ms（HTP 上的旧图要 9 ms）。它比旧的 16-bit 图更接近 BF16，所以贪心解码的 token 会有差异：
38 帧验证输入说出的是同一句 "Hello, what's up?"，只是提前一帧开始。

## 其他图（legacy，沿用 Codex 产物）

head 和 Mimi 的量化沿用旧流程的 DLC。`moshi_legacy_o3.sh OUT_DIR` 把这些 DLC
用 O3 + vtcm8 重新编译（经 `moshi_compile_context.sh`，图名从 DLC 读取，旧 DLC 的图都叫 `model`），
token 与默认选项编译的版本完全一致。实测（2026-10-22）：

- 只有 Mimi 变快：encoder 8.9 → 6.1 ms，decoder 6.7 → 5.4 ms（单图连跑）。head 已接近带宽上限，O3 没有收益。
  Mimi 在 vtcm8 下仍有约 8 MB spill；vtcm12 可消除 spill，但手机上加载失败（QNN 5005）。
- 每个图单独重复执行时比在链路里快（DepFormer 2.55 vs 4–5 ms），因为前一个图把缓存里的权重换掉了。
  把 8 个 DepFormer 合进一个 context 或拼成一个图（输出完全一致）都只快 0–2 ms，在噪声范围内，因此没有采用。

| 图 | 来源链路 | 远端产物 |
| --- | --- | --- |
| head | `moshi_export_lm_onnx.py` → `moshi_prepare_calibration_audio.py` → `moshi_quantize_lm_graph_set.py`（AI Hub 量化）→ `moshi_rebuild_guarded_dlc_set.py` | `/data3/yejialei/moshi-lm-guarded-dlc-set-v2/dlc`；QDQ 源 ONNX 在 `/data2/yejialei/tmp/moshi-lm-guarded-dlc-set-v1/graphs`，DepFormer 门限的 legacy 对照也用它 |
| Mimi encoder（W8）、decoder（LPBQ W4） | `moshi_export_mimi_streaming_onnx.py` → `moshi_quantize_mimi_offline.py` → `moshi_validate_mimi_dlc.py` | `/data2/yejialei/moshi-mimi-lpbq-20261018-v2/selected/{encoder,decoder}` |

## 模型包与 App

模型包目录结构（App 只读，用 ADB 放入）：

```text
lm-v4-lpbq4/
  CONTEXT_ONLY
  SHA256SUMS                 # sha256sum 格式，相对路径，16 行
  frontend_embeddings.bin  mimi_encoder.bin  mimi_decoder.bin
  dlc/head.bin  dlc/depformer_codebook_{0..7}.bin  dlc/temporal_part_{0..3}.bin
```

```sh
M=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-v4-lpbq4
adb shell mkdir -p $M/dlc
adb push ... $M/                                   # 放入上面的文件
adb shell chmod 755 $M $M/dlc                      # 目录必须可进入，否则 App 读取失败
adb shell "chmod 644 $M/*.bin $M/dlc/*.bin"
adb shell "cd $M && sha256sum frontend_embeddings.bin mimi_encoder.bin mimi_decoder.bin dlc/*.bin > SHA256SUMS"
```

Mac 上的副本：全 LPBQ Temporal context 在 `/Users/e1ainay/d_personal/moshi-models/temporal-lpbq4/`
（之前的 W8 value/down 版在 `.../temporal-v2/`），
W8 DepFormer 在 `.../depformer_w8/`，embedding 表在 `.../frontend_table/`，O3 编译的 head 和 Mimi 在 `.../legacy-o3/`。

构建并安装 App：

```sh
bash android/demo/build.sh                         # 需要上面的 Mac 环境变量；输出到 android/demo/build/
adb install -r android/demo/build/mimi-codec-demo.apk
```

只能用原 keystore 覆盖安装，不要卸载，否则会丢失 App 数据。
App 启动时按 `SHA256SUMS` 校验全部 16 个文件；文件大小和修改时间不变时，后续启动跳过重新哈希。

## CLI 测量

```sh
APK=$(adb shell pm path com.overmindy.mimicodecdemo | sed 's/package://'); LIB=$(dirname $APK)/lib/arm64
adb shell "LD_LIBRARY_PATH=$LIB ADSP_LIBRARY_PATH=$LIB ./moshi_reply_latency_probe $LIB \
  $M/mimi_encoder.bin $M/mimi_decoder.bin $M input.wav out 0"
adb shell "LD_LIBRARY_PATH=$LIB ADSP_LIBRARY_PATH=$LIB ./moshi_graph_bench $LIB $M/dlc/temporal_part_0.bin 30"
```

`MOSHI_QNN_PROFILE=1`（基础）或 `2`（逐算子）会额外输出 `*.qnn.csv`。`MOSHI_HTP_PERFORMANCE=0` 可关闭 burst 投票做对照。
试听对照可以用 `scripts/moshi_bf16_reply.py` 在服务器上生成 BF16 原模型对同一输入的回复。

## 已知限制与下一步

- 实时率约 2，还没到实时。每帧约 125–140 ms，其中 Temporal 占 85–100 ms，受权重读取带宽限制：
  - 只把带宽档从 MAX 降到 TURBO，一段从约 30 ms 变成 39–45 ms；核心电压档、HMX 投票都不影响速度。
  - 去掉 KV Split 的 batched attention 反而更慢；KV 窗口从 999 缩到 249 也不变快（attention 与权重读取重叠）。
  - 对照（2026-10-22，同一手机、同一计时工具，2.50 运行库）：高通官方 Qwen3-4B 的 W4 context
    （QAIRT 2.50，O3，vtcm 8）有效带宽 32–33 GB/s，我们的 Temporal 31–32 GB/s，效率基本一致。
    官方公布的 Galaxy S26 数字（约 28 tok/s）在这台手机上达不到；换 QAIRT 2.50 运行库也不变快。
  - 继续提速只能减少每帧读取的字节数，代价是质量。
- 全 LPBQ 的 Temporal 在 part 0 误差最大（单独量化时 cb0 96.9%，part 1 为 98.6%）。
  若要找回质量，可只把 part 0 恢复为 W8 value/down，代价约每帧 6 ms。
- 没有做 ASR 和长音频评测；只做了 3 段 held-out 的人工试听。超过 1000 帧窗口后的环绕没有在真实模型上验证。
- context 绑定这台手机和这个 QNN runtime；换芯片或换 runtime 都要重新编译。
