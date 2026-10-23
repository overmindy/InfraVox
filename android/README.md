# Moshi 在 Snapdragon 手机上的部署

目标设备：HONOR BKQ-AN20，SM8850（HTP v81），Android 16，QAIRT 2.45.0.260326。
从 Kyutai 原始 BF16 权重到手机上运行，只需要两条命令：服务器上 `scripts/moshi_build_all.sh`，Mac 上 `scripts/moshi_deploy.sh`。
旧的实验代码可在 git tag `moshi-pre-cleanup` 中找回。

## 现状（模型包 `lm-v4-lpbq4`）

| 项目 | 当前值 |
| --- | --- |
| 模型包 | 15 个 context + CPU embedding 表 + 文本词表，约 4.7 GB；App 从 Hugging Face（或镜像）下载到 `/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/<variant id>` |
| 38 帧输入（3.04 s） | 暖运行 5.7–6.0 s（RTF 约 2），首段 PCM 约 0.31 s |
| 每帧耗时 | 约 125–140 ms：Temporal 85–100、DepFormer 15–20、Mimi 13–14、head 3、frontend（CPU）0.03 |
| Temporal 质量（8 段 held-out，1558 帧，对 BF16，teacher-forced） | 全 LPBQ W4：文本 top-1 97.95%，音频 cb0 94.4%，8 路全对 85.1% |
| DepFormer W8 质量（同上） | 串联 8 路全对 92.4%（FP32 上限 94.0%） |

`lm-v4-lpbq4` 的 head 来自已删除的 AI Hub 量化链路。现在的一键流程用本地 AIMET 以 W8A16 重建 head，
默认输出新模型包 `lm-v5`；Temporal、DepFormer 和 Mimi 的量化设置与 `lm-v4-lpbq4` 完全相同。

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

第一帧只用于预热，所以 N 个输入帧输出 N−1 帧。

| 文件 | 作用 |
| --- | --- |
| `scripts/moshi_build_all.sh` | 服务器一键构建：capture → embedding 表 → Mimi → 4 个 Temporal part（并行）→ head + DepFormer → 模型包 |
| `scripts/moshi_capture.py` | BF16 Moshi 跑 8 段 LibriSpeech，记录第 0/8/16/24 层输入和第 31 层输出，作为校准数据 |
| `scripts/moshi_temporal.py` | Temporal part 的 LLM 布局导出 + AIMET 校准（LPBQ W4 / A16 / 8-bit 绑定 KV） |
| `scripts/moshi_head_depformer.py` | head 和 8 个 DepFormer codebook 的导出 + AIMET 校准（W8 / A16） |
| `scripts/moshi_mimi.py` | Mimi 流式 encoder/decoder 导出，并直接写出 QAIRT encodings（encoder W8，decoder LPBQ W4） |
| `scripts/moshi_frontend_table.py` | 17 张 BF16 embedding 表导出为 `frontend_embeddings.bin` |
| `scripts/moshi_qairt_build.sh` | 单个图：转换（经 `moshi_guard_qairt_converter.py`）→ 量化 → O3 + VTCM 8 MB 编译成 context |
| `scripts/moshi_deploy.sh` | 开发用：从服务器经 ADB 推送模型包到手机、构建并安装 App |
| `scripts/hf_publish.py` | 把一个模型变体上传到 HF 模型仓库的 main 分支，并更新 `catalog.json` |
| `android/mimi_qnn_runner.cpp` | 原生运行器：加载 context、burst 性能投票、共享内存 KV、Temporal、Mimi、DepFormer；按帧 `begin/step/end` |
| `android/moshi_lm_scheduler.h` | Moshi `LMGen` 的 17 路延迟环 |
| `android/models/catalog.json` | 所有模型和变体（文件、大小、sha256）的唯一来源，App 内置一份并可从仓库刷新 |
| `android/demo/` | Android App（Java UI、JNI、`build.sh`） |

## App（Duplex 语音）

面向多个全双工语音模型的端侧 App，三个页面：对话、模型、设置。

- **对话**：麦克风（或按实时速度送入的演示音频 / WAV）每 80 ms 一帧进入队列，推理线程逐帧 `step`，
  每产生一帧输出就立即播放并逐字显示文本。模型慢于实时（RTF>1）时队列变长、延迟累积，界面实时显示延迟、
  每帧耗时、RTF 和积压帧。第一次点击停止收音并处理剩余帧，再点一次立即结束。录音、文本、tokens 和性能 CSV 可导出。
  文本 token 默认按 Kyutai 的设置采样（温度 0.7、top-k 25），音频 token 仍是图内 argmax：量化后的 LM 在贪心解码下
  说完开场白就一直选“沉默”，不再回应用户（BF16 贪心则会回应），文本采样后恢复正常。
- **模型**：读取仓库根目录的 `catalog.json`，按设备 SoC 标出可用变体；下载支持断点续传、逐文件 SHA-256 校验，
  `SHA256SUMS` 最后写入作为完成标记。
- **设置**：下载源（hf-mirror.com / huggingface.co / 自定义）、仓库名、私有仓库令牌、HTP 性能档、回声消除、编解码回环调试。

Java 代码按层拆分（`android/demo/src/com/overmindy/duplex/`）：`engine/`（`DuplexEngine` 接口与 Moshi 实现）、
`model/`（catalog、本地存储、下载器）、`audio/`（流式会话、WAV）、`ui/`。
**接入新模型**：实现 `DuplexEngine`，在 `Engines` 里按 catalog 的 `engine` 字段注册，再在 `catalog.json` 加一个 model/variant。
应用包名保持 `com.overmindy.mimicodecdemo`，以便覆盖安装后保留数据；旧版 ADB 放入的 `models/lm-v4-lpbq4` 会自动迁移。

## 发布到 Hugging Face

仓库布局：根目录 `catalog.json`、`README.md`，每个变体在 `<engine>/<variant id>/` 下。App 只读 `main` 分支。
在能访问 huggingface.co 的机器上（服务器只能访问 hf-mirror，镜像不能上传）先 `hf auth login`（写权限令牌），然后：

```sh
python scripts/hf_publish.py --repo <用户名>/duplex-voice-models \
  --variant moshiko-lm-v4-lpbq4-sm8850 --bundle-dir <本地模型包目录>
```

脚本会校验并更新 `android/models/catalog.json`（记得提交），可中断后重跑续传。

## 服务器：从 BF16 到模型包

在带 GPU 的 Linux 服务器上以 editable 方式安装本仓库，然后：

```sh
nohup bash scripts/moshi_build_all.sh > /data2/yejialei/moshi/build.log 2>&1 &
```

所有路径都在 `scripts/moshi_env.sh` 里，可以用环境变量覆盖（例如 `MOSHI_BUNDLE`、`MOSHI_GPUS`）。
输出在 `/data2/yejialei/moshi/bundle/$MOSHI_BUNDLE/`。已完成的图会跳过，失败后重跑即可续上；
成功后删除整个工作目录（`KEEP_WORK=1` 保留）。`/data2/yejialei/moshi/` 下还有：

| 路径 | 内容 |
| --- | --- |
| `inputs/mimi_calibration.wav` | Mimi A16 范围用的音频（前 4 帧） |
| `inputs/validation_input.wav` | 38 帧验证输入，也是 App 内置的演示音频 |
| `python-deps-numpy126/` | QAIRT 2.45 转换器需要的 NumPy 1.26 |
| `bundle/lm-v4-lpbq4/` | 当前手机上的模型包 |

`/data2`、`/data3` 和 `/` 都接近满盘且与他人共用；构建峰值约 70 GB，AIMET 的大文件写在工作目录的 `tmp/` 下。

## Mac：部署到手机

需要 Android SDK（build-tools 35.0.0、android-34）、NDK 30、Android Studio 自带的 JDK、QNN 头文件和
Android 运行库（默认路径见 `scripts/moshi_deploy.sh` 开头），以及 `android/demo/build/demo.keystore`。

```sh
bash scripts/moshi_deploy.sh
```

脚本把服务器上的 `MOSHI_BUNDLE` 推到手机的 `models/$MOSHI_VARIANT`（catalog 里的变体 id）；正常使用时 App 直接从 HF 下载，不需要这一步。
只能用原 keystore 覆盖安装，不要卸载，否则会丢失 App 数据。
App 第一次加载时按 catalog 的 sha256 校验全部文件；之后文件大小和修改时间不变就跳过重新哈希。

## 关键约束

- **Temporal part 接口**（与 `templates/llm` 的官方 LLM 一致）：投影为按头拆分的 1×1 Conv，Moshi 的交错 RoPE
  通过置换 q/k 行变为半分布局。输入 `hidden` [1,1,4096]、`attention_mask` [1,1,1,1000]、`position_cos/sin` [1,1,1,64]、
  每层 `past_key_{l}_in` [32,1,128,999]、`past_value_{l}_in` [32,1,999,128]；输出 `hidden_out` 和每层新的一帧 K/V。
  I/O 为定点，运行器按 context 里的 scale/offset 转换（`real = scale × (q + offset)`）。
- KV ring 由宿主持有，放在注册过的 rpcmem 里；空槽必须填 0.0 的量化值（零点），不能填字节 0。
- 校准样本必须包含第 0–3 帧：第 0 帧是全初始 token 的 attention sink。
- LPBQ 只能用 Conv：LPBQ 的 FullyConnected 会让 context 生成器段错误（Temporal 和 Mimi decoder 因此用 1×1 Conv）。
- DepFormer attention 不能用 −inf mask：每个 codebook 的步号是常量，直接切掉后面的槽位；
  用 mask 时 16-bit 激活范围被撑爆。DepFormer 用 W4 不可用（8 路全对 49.6–62.6%）。
- 图名必须等于 context 名，`backend.json` 里的 O3 + VTCM 才生效（8 层 W4 part：O3 28 ms，默认 92–101 ms）。
- QAIRT 2.45 会错误融合 `gamma * rsqrt(rms(x))` 形式的 RMSNorm，所以转换器经过 `moshi_guard_qairt_converter.py`。
- context 绑定这台手机和这个 QNN runtime；换芯片或换 runtime 都要重新编译。

## 已知限制

- 实时率约 2。Temporal 受权重读取带宽限制（有效带宽 31–32 GB/s，与高通官方 Qwen3-4B W4 context 相当），
  继续提速只能减少每帧读取的字节数。
- 全 LPBQ 的 Temporal 在 part 0 误差最大；若要找回质量，可只把 part 0 的 value/MLP down 恢复为 W8，代价约每帧 6 ms。
- 没有做 ASR 和长音频评测；超过 1000 帧窗口后的环绕没有在真实模型上验证。
