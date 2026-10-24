<p align="center"><img src="android/demo/branding/logo.svg" width="120" alt="InfraVox"></p>

# InfraVox

在手机上本地运行的**全双工语音对话** App：模型边听边说，你可以随时打断它，全程不联网。
名字取自端侧底座（Infra）+ 语音（Vox），一个 App 承载多个全双工语音模型，全部跑在骁龙的 Hexagon NPU 上。

**推荐使用 MiniCPM-o 4.5**：中英文都能聊，在 SM8850 上跑得过实时，几分钟的连续对话也能一直正常回答。
App 首次打开默认选中它。Moshi 作为实验性模型保留，目前慢于实时。

- **下载 App**：[GitHub Releases](https://github.com/overmindy/InfraVox/releases/latest)
- **模型权重**：[ELAINAy/duplex-voice-models](https://huggingface.co/ELAINAy/duplex-voice-models)（App 内直接下载，默认走 hf-mirror.com）

<!-- 演示视频：录好后放在这里 -->

## 功能

- **全双工通话**：麦克风音频持续送入模型，模型一边听一边生成语音并立即播放，可以随时插话打断。
- **多个模型**：MiniCPM-o 4.5（中英文，推荐）和 Moshi（英文，实验性），在模型页切换。
- **聊天式界面**：每通电话按说话轮次切成气泡，可回放每一段；历史对话保存在本地，可导出音频和文本。
- **模型管理**：从 Hugging Face 或镜像下载，断点续传、逐文件 SHA-256 校验，自动标出与本机芯片兼容的版本。
- **实时状态**：通话中显示延迟、每帧耗时、实时率和积压帧。

## 支持的模型与性能

测试设备：HONOR BKQ-AN20（Snapdragon SM8850，HTP v81），QAIRT 2.45。

| | MiniCPM-o 4.5（推荐） | Moshi · Moshiko（实验性） |
| --- | --- | --- |
| 来源 | [OpenBMB](https://huggingface.co/openbmb/MiniCPM-o-4_5)，Qwen3-8B 语音部分 | [Kyutai](https://github.com/kyutai-labs/moshi)，7B |
| 语言 | 中文、英文 | 英文 |
| 量化 | LLM LPBQ W4A16，Whisper / DiT / HiFT W8，TTS 在 CPU 上 int8 | Temporal LPBQ W4A16，DepFormer / Mimi W8 |
| 模型包 | 6.4 GB | 4.7 GB |
| 速度 | 每 1 s 语音约 0.5–0.9 s，**跑得过实时** | 每 80 ms 帧凉机约 100 ms、发热后约 130 ms（RTF 1.25–1.65），慢于实时 |
| 首段语音 | 文字即时出现，语音比文字晚约 2 s | 约 0.3 s |
| 质量 | 4 分钟、6 个问题的连续对话全部完整回答，播放 0 次卡顿 | 文本 token 与 BF16 一致率 97.95%；长对话中第三个问题起容易重复 |

MiniCPM-o 的 LLM 解码、Whisper 和声码器在 NPU 上，TTS 在 CPU 上与 NPU 并行；长对话里会丢弃最早的内容、保留系统提示，
可以一直聊下去。Moshi 每 80 ms 都要完整跑一遍 7B 主干，受 NPU 权重读取带宽限制，长对话中延迟会逐渐累积。

## 使用

1. 安装 Release 里的 APK（Android 10+，arm64）。模型是绑定芯片的 QNN context，目前只支持 **Snapdragon SM8850**。
2. 打开侧边栏的“模型管理”，下载 MiniCPM-o 4.5（标着“推荐”，首次打开已设为当前）。想试 Moshi 再下载它并点“设为当前”。
3. 回到主页点“通话”直接说话；外放时建议开启回声消除或戴耳机。也可以用内置演示音频或自己的 WAV 文件测试。

## 仓库

```sh
# 服务器（GPU）：从官方 BF16 权重量化、编译出手机模型包
bash scripts/minicpmo_build_all.sh
bash scripts/moshi_build_all.sh

# Mac：把模型包推到连着的手机，构建并安装 App（开发用）
bash scripts/minicpmo_deploy.sh
bash scripts/moshi_deploy.sh

# 发布一个模型变体到 Hugging Face，并更新 catalog
python scripts/hf_publish.py --repo ELAINAy/duplex-voice-models --variant <变体 id> --bundle-dir <模型包目录>
```

| 路径 | 内容 |
| --- | --- |
| `android/demo/` | App（Java UI + JNI，`build.sh` 直接出 APK） |
| `android/minicpmo_runner.cpp`、`android/mimi_qnn_runner.cpp` | 两个模型的 C++ 运行时；`android/qnn_runtime.h` 是共用的 QNN 层 |
| `android/models/catalog.json` | 所有模型和文件的清单，App 内置并可从仓库刷新 |
| `scripts/minicpmo_*`、`scripts/moshi_*` | 量化、校准、编译、部署脚本 |

环境要求、每个图的输入输出约定、量化细节和踩过的坑见 [android/README.md](android/README.md)。

## 致谢

基于 [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models)（原 README 见 [QAIHM_README.md](QAIHM_README.md)），
模型来自 [OpenBMB MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)（Apache-2.0）与 [Kyutai Moshi](https://github.com/kyutai-labs/moshi)（CC-BY-4.0）。
