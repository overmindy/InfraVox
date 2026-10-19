# Moshi 在 Snapdragon 手机上的部署

目标设备：HONOR BKQ-AN20，SM8850（HTP v81），Android 16，QAIRT 2.45.0.260326。
本文是当前唯一有效的部署说明。早期 Codex 实验记录和一次性脚本已删除，可在 git tag `moshi-pre-cleanup` 中找回。

## 现状

| 项目 | 当前值 |
| --- | --- |
| App | `com.overmindy.mimicodecdemo` 2.8.0-temporal-v2（versionCode 13） |
| 模型包 | `/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-v2-w4w8vd`，16 个 context，约 6.7 GB |
| 38 帧输入（3.04 s） | 暖运行 8.5–9.0 s，输出 2.96 s，RTF 约 3 |
| 每帧耗时 | 约 200–210 ms：Temporal 135、DepFormer 29–36、Mimi 19、frontend 11、head 3.5 |
| Temporal 质量（8 段 held-out，1558 帧，手机实测） | 文本 top-1 98.7%，音频 cb0 95.3%，8 路全对 90.6% |

质量指标是 teacher-forced 的：每帧给相同的 BF16 历史，比较该帧的文本 argmax 和 DepFormer 贪心音频 token。
它不是听感或 ASR 评测；free-running 贪心解码即使全 W8 也会在十几帧内出现 token 分叉。

## 模型与数据流

```text
24 kHz PCM, 1920 样本/帧 (80 ms)
  → Mimi encoder ──8 codes──┐
                             ▼
          17 路延迟环 (moshi_lm_scheduler.h)
                             ▼
  frontend → Temporal part 0..3（各 8 层）→ head ──text──┐
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

2. **校准并做 token 级门限**（AIMET + CUDA）：每个 part 生成一个 QuantSim。权重 `lpbq4_w8vd` 即 LPBQ int4（block 64），
   其中 value 和 MLP down 保持 int8 per-channel；激活 int16；KV 为 8-bit，输入输出绑定。
   校准样本取 8 段校准音频的前 120 帧，共 64 个 AR1 步，**必须包含第 0–3 帧**（第 0 帧是全初始 token 的 attention sink）。
   部署用的 encodings 就是这一步生成的，没有使用 SeqMSE。

   ```sh
   source /data2/yejialei/envs/aimet228/env.sh
   TMPDIR=$W/tmp python moshi_temporal_segment_eval.py --model-dir $BF16 --capture-dir $W/capture \
     --work-dir $W/w4w8vd --parts 0,1,2,3 --mode each      # 逐段：保存 temporal_part_{p}.encodings
   TMPDIR=$W/tmp python moshi_temporal_segment_eval.py ... --mode together   # 4 段串联；复用 encodings，不重新校准
   ```

3. **构建 context**（每段约 8 分钟）：导出 ONNX，拷入 encodings，然后转换、量化、编译。

   ```sh
   python moshi_temporal_part.py --model-dir $BF16 --output-dir $W/build/part$P \
     --first-layer $((8*P)) --layers 8 --parity-steps 4 --calibration-steps 0
   cp $W/w4w8vd/temporal_part_$P.encodings $W/build/part$P/
   bash moshi_temporal_build.sh $W/build/part$P temporal_part_$P      # 默认 O3 + vtcm 8 MB
   ```

   `moshi_temporal_build.sh` 把 DLC 的图命名为 `NAME`，这样 `backend.json` 里的 O3 和 VTCM 设置才会生效
   （默认选项下 8 层 part 要 92–101 ms，O3 下只要 28 ms）。转换器经过 `moshi_guard_qairt_converter.py`，
   用来阻止 QAIRT 2.45 错误地融合 RMSNorm。LPBQ 只能用 Conv：LPBQ 的 FullyConnected 会让 context 生成器段错误。

4. **手机数值对照**：`moshi_temporal_device_eval.py --export` 导出 held-out 的 layer-0 流。
   手机上用 `moshi_temporal_chain_probe LIB in.f32 out.f32 part0..3.bin` 跑出结果，
   再把 `out.f32` 传回服务器，不带 `--export` 运行 `moshi_temporal_device_eval.py`，得到与 QuantSim 同口径的 token 指标。

快速筛选工具：`moshi_temporal_token_eval.py` 在 PyTorch 里伪量化权重，每个变体约 1.5 分钟。
`moshi_temporal_sweep.py` 在 2 层子图上一次跑多个 QuantSim 变体，并拆分权重误差和激活误差。

## 其他图（legacy，沿用 Codex 产物）

frontend、head、8 个 DepFormer 和 Mimi 仍沿用旧流程的产物，原样复制到新包。它们的 context 不是用 O3 编译的。

| 图 | 来源链路 | 远端产物 |
| --- | --- | --- |
| frontend、head、DepFormer | `moshi_export_lm_onnx.py` → `moshi_prepare_calibration_audio.py` → `moshi_quantize_lm_graph_set.py`（AI Hub 量化）→ `moshi_rebuild_guarded_dlc_set.py` | `/data3/yejialei/moshi-lm-guarded-dlc-set-v2/dlc`；context 当初由旧运行器在手机上编译 |
| Mimi encoder（W8）、decoder（LPBQ W4） | `moshi_export_mimi_streaming_onnx.py` → `moshi_quantize_mimi_offline.py` → `moshi_validate_mimi_dlc.py` | `/data2/yejialei/moshi-mimi-lpbq-20261018-v2/selected/{encoder,decoder}` |

现在的运行器只加载 `.bin` context。要重建这些图，用 `qnn-context-binary-generator` 从上面的 DLC 离线编译，
参照 `moshi_temporal_build.sh` 的编译段，图名要与 DLC 一致。

## 模型包与 App

模型包目录结构（App 只读，用 ADB 放入）：

```text
lm-v2-w4w8vd/
  CONTEXT_ONLY
  SHA256SUMS                 # sha256sum 格式，相对路径，16 行
  mimi_encoder.bin  mimi_decoder.bin
  dlc/frontend.bin  dlc/head.bin  dlc/depformer_codebook_{0..7}.bin  dlc/temporal_part_{0..3}.bin
```

```sh
M=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-v2-w4w8vd
adb shell mkdir -p $M/dlc
adb push ... $M/                                   # 放入上面的文件
adb shell chmod 755 $M $M/dlc                      # 目录必须可进入，否则 App 读取失败
adb shell "chmod 644 $M/*.bin $M/dlc/*.bin"
adb shell "cd $M && sha256sum mimi_encoder.bin mimi_decoder.bin dlc/*.bin > SHA256SUMS"
```

当前包的清单另存在 Mac：`/Users/e1ainay/d_personal/moshi-models/temporal-v2/BUNDLE_SHA256SUMS`，4 个 Temporal context 也在同一目录。

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

- 实时率约 3，还没到实时。下一步：用 O3 + vtcm8 重新编译 legacy 图；评估把 8 个 DepFormer 合并成一个图；
  按 Temporal 的方式重新导出和量化 frontend、head、DepFormer。
- part 3（第 24–31 层）是最主要的误差来源。单独量化 part 3 时 cb0 为 96.3%，其他 part 都在 99% 以上。
  可以考虑 part 3 全部用 W8，代价是每步约多 20 ms。
- 没有做听感、ASR 和长音频评测。超过 1000 帧窗口后的环绕没有在真实模型上验证。
- context 绑定这台手机和这个 QNN runtime；换芯片或换 runtime 都要重新编译。
