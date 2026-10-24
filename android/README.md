# InfraVox 开发文档

目标设备：HONOR BKQ-AN20，SM8850（HTP v81），Android 16，QAIRT 2.45.0.260326。
每个模型从官方 BF16 权重到手机上运行都只需要两条命令：服务器上 `scripts/<模型>_build_all.sh`，Mac 上 `scripts/<模型>_deploy.sh`。
下面先是推荐使用的 MiniCPM-o 4.5，然后是实验性的 Moshi，最后是两者共用的 App 和发布流程。
Moshi 旧的实验代码可在 git tag `moshi-pre-cleanup` 中找回。

# MiniCPM-o 4.5 全双工语音（推荐）

OpenBMB [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) 的全双工语音部分（不含视觉），
同一台手机、同一个 QAIRT。服务器上 `scripts/minicpmo_build_all.sh` 从 BF16 权重一路做到模型包，
Mac 上 `scripts/minicpmo_deploy.sh` 推到手机并安装 App；路径在 `scripts/minicpmo_env.sh`。

## 模型与数据流

```text
16 kHz PCM，1 s 一个单元（首个单元 1030 ms）
  → 流式 log-mel（CPU，30 s 窗口，前后各 2 帧 CNN 上下文）
  → audio_encoder：Whisper-medium 24 层 + 投影 + 5× 池化，KV 窗口 1450 帧 → 10 个 embedding
  → LLM（Qwen3-8B，4 段 × 9 层）：<unit> + 10 个音频 embedding 一次 AR-16 预填充
       ↓ 每单元最多 20 次 AR-1 解码：<|listen|> 就听；否则输出说话 token，直到 chunk_eos / turn_eos
  → llm_head：logits（CPU 采样：温度 0.7 / top-k 100 / top-p 0.8 / 重复惩罚 1.05）+ TTS 语义向量
  → tts：20 层 Llama，条件 = emb_text(token) + 语义向量，每单元 25 个 s3 码（CPU 采样，温度 0.8）
  → token2wav（28 码窗口，步进 25）：
       flow conformer（CPU / ONNX Runtime）→ DiT × 10 步 Euler（CFG 0.7）→ HiFT f0
       → NSF 源 + STFT（CPU）→ HiFT 解码 → iSTFT（CPU）→ 24 kHz PCM
```

| 文件 | 作用 |
| --- | --- |
| `scripts/minicpmo_build_all.sh` | 一键构建：采集 → token2wav 资源 → LLM 分段边界 → 两张 GPU 并行 AIMET 校准 → QAIRT 编译 → 模型包 |
| `scripts/minicpmo_capture.py` | 用模型自己的 TTS 合成 20 个中英文提问，跑官方双工流程，记录每个图的输入（校准数据 + 对齐基准） |
| `scripts/minicpmo_llm.py` | LLM 的高通 LLM 布局（AR-1 / AR-16 共享权重和 KV 环）+ head；LPBQ W4 / A16 / 8-bit KV |
| `scripts/minicpmo_audio.py` | 流式 Whisper 编码器图（W8） |
| `scripts/minicpmo_tts.py` | TTS 的 AR-1 / AR-16 HTP 图（W8）与 Llama 参考；一键构建已改用 CPU TTS，不再编译它 |
| `scripts/minicpmo_tts_cpu.py` | 手机 CPU 上的 TTS：int8 权重（`tts_cpu.bin`）、同一套算术的参考实现、与浮点 TTS 的对比、主机测试向量 |
| `scripts/minicpmo_flow.py` | DiT 单步图、时间步调制预计算、NSF 源 / STFT / iSTFT、`Token2wavHost`（缓存规则与官方一致） |
| `scripts/minicpmo_token2wav_prepare.py` | 官方 token2wav 侧（minicpmo 环境）：固定音色的提示缓存、conformer ONNX、HiFT 图、golden 数据 |
| `scripts/minicpmo_token2wav.py` | DiT / HiFT 对齐与校准 |
| `scripts/minicpmo_hift_check.py` | NSF 源、STFT、iSTFT 与官方 HiFT 对齐，并导出主机测试向量 |
| `scripts/minicpmo_sim.py` | 手机宿主逻辑的 Python 版（浮点图），C++ 运行器照它移植 |
| `scripts/minicpmo_bundle.py` | `assets.bin`：embedding 表、系统提示 embedding、BPE 字节表、token2wav 固定状态 |
| `scripts/minicpmo_qairt_build.sh` | 转换 + 量化 + 编译；多个图进同一个 context 时开启权重共享 |
| `android/minicpmo_runner.cpp` | 原生运行器；`android/qnn_runtime.h` 是与 Moshi 共用的 QNN 基础设施 |
| `android/minicpmo_host_test.cpp` | 在电脑上校验 CPU 部分（流式 mel、NSF 源、STFT/iSTFT）与官方一致 |

App 里 MiniCPM-o 引擎输入 16 kHz、每帧 80 ms（1280 样本，与 Moshi 的帧时间轴一致），运行器攒满 1 秒算一个单元，
输出 24 kHz（听的时候返回 1 秒静音）。

## 实时性能（SM8850，6 个提问的 4 分钟通话）

LLM 在主线程，TTS + token2wav 在工作线程晚一个单元跑（语音输出因此多 1 秒延迟），CPU 上的 flow conformer、
NSF / STFT 与下一单元的 LLM 重叠；HTP 执行用一把锁串行（各 context 共用一块 spill-fill 缓冲）。
TTS 在 CPU 上跑（`scripts/minicpmo_tts_cpu.py`：int8 权重按行缩放、输入按 64 一组量化成 int8 用 sdot，
与浮点 TTS 的 logits 中位相对误差 0.7%、top-1 一致 96%），不再占 HTP。

| 项目 | 耗时 |
|---|---|
| 说话单元，主线程（mel + Whisper + LLM 解码） | 多为 500–900 ms / 1 s，一单元解 6–8 个 token 时到 1.0–1.1 s |
| 说话单元，工作线程（CPU TTS 25 个码 + token2wav） | 约 230 + 390 ms |
| 说话时 HTP 合计 | 约 810 ms / 1 s（TTS 在 HTP 上时约 920 ms） |
| LLM AR-1 每 token（4 段 + head） | 约 115 ms（CPU TTS 同时占内存带宽），每单元约 4–5 个 token |
| 加载 / 系统提示预填充 | 4.4 s / 1.3 s |

播放端为 MiniCPM-o 保持 1 s 预留（模型安静时用静音补，说话中不插静音），4 分钟测试 0 次 underrun。
head 用 W4 能省约 50 ms/s 的 HTP，但 top-1 与浮点一致率从 0.978 掉到 0.898，所以保留 W8。

## 长对话

LLM 和 TTS 的 KV 环各 2032 个位置。满了以后保留系统提示（TTS 保留本轮第一个 chunk），其余位置循环覆盖最旧的
单元；键保留绝对 RoPE 位置（Qwen3 训练到 40960，约 40 分钟）。不这样做时约 2 分钟后系统提示被覆盖，模型不再说话。

## 关键约束

- **AR-1 与 AR-16 共享权重**：每段 LLM 和 TTS 的两个图放进同一个 context（`weight_sharing_enabled`），
  同一份 KV 环同时绑定到两个图，所以两图必须用同一份编码（AR-1 直接复用 AR-16 的 encodings）。
- 每单元的预填充只要一次 AR-16；不需要 logits 的 token（`</unit>`、`<|chunk_eos|>` 等）先挂起，合进下一次预填充。
- 系统提示（文本 + 参考音色的 Whisper embedding）在服务器上按官方代码算好放进 `assets.bin`，`begin()` 时预填充。
- 音色固定为 `assets/system_ref_audio.wav` 的前 6.76 s：token2wav 的提示缓存（DiT 每步每层的 K/V、conformer 缓存）
  在服务器上算好。DiT 缓存规则照官方：最新帧在前，超过 提示 + 100 帧时保留最新的 338 帧和最早的 100 帧。
- 量化工具链的坑：
  - 手写的 LayerNorm 会被 QAIRT 当成 RMSNorm 并和后面的 modulate 乘法一起融合、回退成浮点；DiT 改用 `layer_norm` 算子。
  - weight_norm 参数化会导出成运行时计算的卷积权重，HTP 不接受；导出前折叠进权重。
  - AIMET 的 Conv + 激活融合组只给融合后的张量编码，QAIRT 不做这个融合；给 `qairt-quantizer` 一份校准输入列表补齐。
- flow conformer 的相对位置编码随缓存长度变化，放在 CPU 上用 ONNX Runtime（动态形状）跑。

# Moshi · Moshiko（实验性）

慢于实时（凉机 RTF 1.25，发热后 1.65），长对话中容易重复，详见本节的已知限制。

## 现状（模型包 `lm-v4-lpbq4`）

| 项目 | 当前值 |
| --- | --- |
| 模型包 | 15 个 context + CPU embedding 表 + 文本词表，约 4.7 GB；App 从 Hugging Face（或镜像）下载到 `/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/<variant id>` |
| 38 帧输入（3.04 s） | 暖运行 5.7–6.0 s（RTF 约 2），首段 PCM 约 0.31 s |
| 每帧耗时 | 凉机约 100 ms（RTF 1.25）：Temporal 65、DepFormer 13、Mimi 13、head 2.4；连续跑几分钟发热后约 130 ms（RTF 1.65，Temporal 85） |
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
Mimi 不放到工作线程：所有图共用 HTP 执行锁，HTP 每帧几乎没有空闲；凉机对比异步 100.4 ms、同步 99.3 ms，没有收益。

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
| `android/moshi_cli.cpp`、`scripts/moshi_phone_cli.sh` | 手机上的命令行会话：把一段 24 kHz WAV 尽快逐帧跑完，写出回复、每步耗时、token 和各图耗时 |
| `android/models/catalog.json` | 所有模型和变体（文件、大小、sha256）的唯一来源，App 内置一份并可从仓库刷新 |
| `android/demo/` | Android App（Java UI、JNI、`build.sh`） |

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
  满 999 帧后固定第 0–3 帧，其余 995 个槽循环覆盖最旧的帧；键保留绝对 RoPE 位置。
- 校准样本必须包含第 0–3 帧：第 0 帧是全初始 token 的 attention sink。
- LPBQ 只能用 Conv：LPBQ 的 FullyConnected 会让 context 生成器段错误（Temporal 和 Mimi decoder 因此用 1×1 Conv）。
- DepFormer attention 不能用 −inf mask：每个 codebook 的步号是常量，直接切掉后面的槽位；
  用 mask 时 16-bit 激活范围被撑爆。DepFormer 用 W4 不可用（8 路全对 49.6–62.6%）。
- 图名必须等于 context 名，`backend.json` 里的 O3 + VTCM 才生效（8 层 W4 part：O3 28 ms，默认 92–101 ms）。
- QAIRT 2.45 会错误融合 `gamma * rsqrt(rms(x))` 形式的 RMSNorm，所以转换器经过 `moshi_guard_qairt_converter.py`。
- context 绑定这台手机和这个 QNN runtime；换芯片或换 runtime 都要重新编译。

## 已知限制

- 实时率凉机 1.25、发热后 1.65。Temporal 受权重读取带宽限制（有效带宽 31–32 GB/s，与高通官方 Qwen3-4B W4 context 相当），
  继续提速只能减少每帧读取的字节数。
- 全 LPBQ 的 Temporal 在 part 0 误差最大；若要找回质量，可只把 part 0 的 value/MLP down 恢复为 W8，代价约每帧 6 ms。
- 没有做 ASR 评测。长对话（`scripts/moshi_phone_cli.sh`，10 个英文提问间隔 22 s）：前两个问题答得好，
  第三个问题（约 52 s，环绕之前）起陷入重复（"will help you develop your skills…"），之后沉默；
  未固定 sink 的旧环在约 150 s 后输出成串的 "j"。固定 sink 之后的版本只跑到 130 s，没有覆盖到这一段。
  重复出现在环绕之前，所以不是 KV 环的问题，更可能来自文本解码（只采样说/不说，说话时取最高分的词）或量化。

# App 与发布

## App：InfraVox

<img src="demo/branding/logo.svg" width="96" alt="InfraVox logo">

InfraVox 是承载多个全双工语音模型的端侧底座（Infra）加语音（Vox）。图标是两组交错的声波，代表两个声音同时进行。
母版在 `android/demo/branding/logo.svg`，
启动器图标是同一几何的自适应图标（`android/demo/res/`）。

面向多个全双工语音模型的端侧 App，主页是聊天界面，左侧抽屉里是历史对话、模型管理和设置。

- **对话**：一通语音通话就是一段对话（模型不跨通话保留上下文）。麦克风（或按实时速度送入的演示音频 / WAV）
  每 80 ms 一帧进入队列，推理线程逐帧 `step`，每产生一帧输出就立即播放。通话按麦克风能量和模型文本停顿
  自动切成聊天气泡（`history/TurnTracker`），每个气泡可回放对应片段；顶部状态条实时显示延迟、每帧耗时、RTF 和积压帧。
  停止收音后会继续送入静音直到模型说完（最多 30 秒），再点一次立即结束。对话保存在 `files/conversations/<id>/`，可导出。
- **Moshi 文本解码**：音频 token 是图内 argmax。量化后的 LM 在纯贪心文本解码下说完开场白就一直选“沉默”（BF16 贪心会回应），
  而完整的 top-k 采样又会选到 `<` 之类的低概率垃圾词。所以只对“说话还是沉默”（pad 与非 pad）按温度 0.7、top-k 25 采样，
  说话时取最高分的非 pad 词。设置里可关回纯贪心。
- **模型管理**：读取仓库根目录的 `catalog.json`，按设备 SoC 标出可用变体；标了 `"recommended": true` 的模型（MiniCPM-o）
  显示“推荐”，用户还没选过模型时默认选它；catalog 的 `revision` 更大的一份优先，内置 catalog 可以盖过仓库里较旧的一份；下载支持断点续传、逐文件 SHA-256 校验，
  `SHA256SUMS` 最后写入作为完成标记。模型仓库是公开的，下载不需要令牌。
- **设置**：下载源（hf-mirror.com / huggingface.co / 自定义）、HTP 性能档、回声消除、文本采样；“高级”里可改模型仓库和编解码回环调试。

Java 代码按层拆分（`android/demo/src/com/overmindy/duplex/`）：`engine/`（`DuplexEngine` 接口与 MiniCPM-o、Moshi 实现）、
`model/`（catalog、本地存储、下载器）、`audio/`（流式会话、WAV）、`history/`（对话存储、气泡切分）、`ui/`。
**接入新模型**：实现 `DuplexEngine`，在 `Engines` 里按 catalog 的 `engine` 字段注册，再在 `catalog.json` 加一个 model/variant。
应用包名保持 `com.overmindy.mimicodecdemo`，以便覆盖安装后保留数据；旧版 ADB 放入的 `models/lm-v4-lpbq4` 会自动迁移。

## 发布到 Hugging Face

仓库布局：根目录 `catalog.json`、`README.md`，每个变体在 `<engine>/<variant id>/` 下。App 只读 `main` 分支。
在能访问 huggingface.co 的机器上（服务器只能访问 hf-mirror，镜像不能上传）先 `hf auth login`（写权限令牌），然后：

```sh
python scripts/hf_publish.py --repo <用户名>/duplex-voice-models \
  --variant minicpmo45-v1-sm8850 --bundle-dir <本地模型包目录>
```

脚本会校验并更新 `android/models/catalog.json`（记得提交），可中断后重跑续传。
