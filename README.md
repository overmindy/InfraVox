# Moshi on Snapdragon

把 Kyutai 的 [Moshi](https://github.com/kyutai-labs/moshi)（7B 全双工语音大模型）从 BF16 权重量化、编译，
跑在骁龙 8 Elite Gen 5 手机的 Hexagon NPU 上，并配套一个 Android 演示 App。

量化方案、C++ 运行时、性能分析和构建部署流水线均由个人独立完成。

| 指标（HONOR BKQ-AN20 / SM8850，38 帧 ≈ 3 s 语音） | 起点 | 现在 |
| --- | --- | --- |
| 端到端单帧延迟 | ~983 ms | **125–160 ms** |
| 实时率 RTF（计算时间 / 音频时长） | 12.6 | **≈ 2.0** |
| 首段音频输出 | 1.33 s | **0.31–0.38 s** |
| 文本 token 与 BF16 一致率（teacher-forced，1558 帧 held-out） | — | **97.95%** |

## 做了什么

**低比特量化**
- Temporal Transformer（32 层）按高通官方 LLM 布局导出为 4 段，每段 8 层：
  投影改为按头拆分的 1×1 Conv，交错 RoPE 通过置换 q/k 行变成半分布局，KV 由宿主持有。
- 用 AIMET-ONNX 做 **LPBQ W4A16**（int4 分块权重、int16 激活、8-bit 绑定 KV）。
  DepFormer 用 W8A16，Mimi 编码器 W8、解码器 LPBQ W4。
- 用 LibriSpeech 搭建校准集和留出集（说话人不重叠），**以 token 一致率而不是张量误差作为验收门槛**。
- 排查出的问题：
  - 校准漏掉第 0 帧（attention sink）时，隐状态误差约 270%。
  - DepFormer 的 −inf mask 会撑爆 16-bit 激活范围，改为直接切掉后面的槽位。
  - QAIRT 转换器会错误融合 RMSNorm，加了一层保护。

**推理运行时（C++ / QNN HTP）**
- 15 个 context 常驻，投票启用 HTP burst 性能模式。
- 64 个 KV 环形缓冲放在注册过的 rpcmem 共享内存里，每步只写新一帧的 K/V；空槽填量化零点，不填字节 0。
- 重置 KV 时用 memset 代替逐元素拷贝，首段音频输出 1.33 s → 0.38 s。
- 17 张 embedding 表求和挪到 CPU，NPU 上这一步从 9 ms 降到 0.03 ms。
- O3 + 8 MB VTCM 编译：8 层 W4 段从 92–101 ms 降到 28 ms（前提是图名等于 context 名，否则这些编译选项会被静默忽略）。

**性能分析**
- 每帧约 125–140 ms：Temporal 85–100，DepFormer 15–20，Mimi 13–14，head 3。
- Temporal 受权重读取带宽限制：有效带宽 31–33 GB/s，与高通官方 Qwen3-4B W4 context 在同一台手机上的实测相当。
  要继续提速，只能减少每帧读取的权重字节数。

## 优化过程

| 阶段 | 38 帧暖运行 | 主要改动 |
| --- | --- | --- |
| 基线 | 37.36 s | 16 个 2 层分片，FP32 I/O |
| 常驻 + 共享内存 | 21.42 s | context 常驻，HTP burst，KV 绑定 rpcmem |
| 官方 LLM 布局 | 9.04 s | 4 段 LPBQ W4 Temporal，单步 Temporal 489 → 131 ms |
| KV 重置 + O3 | 7.0–7.3 s | memset 重置 KV，其余图用 O3 重编译 |
| DepFormer W8 | 6.12 s（App，36 帧） | DepFormer 36 → 15–20 ms/帧，加载 7.0 → 3.4 s |
| Temporal 全 W4 | 5.7–6.0 s | 权重 1.10 → 0.86 GB/段 |

每一步的测量和取舍都记录在对应 commit 的说明里。

## 快速开始

```sh
# 服务器（GPU）：从 Kyutai BF16 权重生成手机模型包
bash scripts/moshi_build_all.sh

# Mac：把模型包推到手机，构建并安装 App
bash scripts/moshi_deploy.sh
```

环境要求、目录结构、图的 I/O 约定和已知坑，见 [android/README.md](android/README.md)。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `scripts/moshi_build_all.sh` | 一键构建：校准数据采集 → 各子图导出与量化 → 编译 → 模型包 |
| `scripts/moshi_temporal.py` | Temporal 段的 LLM 布局导出与 AIMET 校准 |
| `scripts/moshi_head_depformer.py` | head 和 8 个 DepFormer codebook 的导出与校准 |
| `scripts/moshi_mimi.py` | Mimi 流式编解码器导出 |
| `scripts/moshi_qairt_build.sh` | 单个图的转换、量化、编译 |
| `android/mimi_qnn_runner.cpp` | 原生运行时 |
| `android/moshi_lm_scheduler.h` | Moshi 17 路延迟调度 |
| `android/demo/` | Android App（Java UI + JNI） |
| `src/qai_hub_models/models/templates/moshi/` | 显式 KV / 流式状态的 Moshi 与 Mimi 模型定义 |

## 已知限制

- RTF ≈ 2，还没有达到实时。
- context 和这台手机、这个 QNN runtime 绑定，换芯片或 runtime 需要重新编译。
- 没有做 ASR 和长音频评测；超过 1000 帧窗口后的 KV 环绕没有在真实模型上验证。

## 致谢

基于 [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models) fork（原 README 见 [QAIHM_README.md](QAIHM_README.md)），
模型来自 [Kyutai Moshi](https://github.com/kyutai-labs/moshi)。
