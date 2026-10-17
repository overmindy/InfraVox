# Mimi 选择性量化与 LM 窗口容量

日期：2026-10-18。实验目录：`/data2/yejialei/moshi-mimi-quant-20261018-v1`。
Mac 目录：`/Users/e1ainay/d_personal/moshi-models/mimi-w8a16-20261018-v1`。
默认 App、Mimi、LM DLC 和默认 context 不作替换。

## cache1000 的时长

运行器使用 24,000 Hz、每帧 1,920 个采样点，因此每秒 12.5 个 LM 帧。
cache1000 的 1,000 个槽约对应最近 80 秒。60 秒需要 750 个槽。
每帧的 17 条音频/文本流共同推进一个 Temporal 时间步，不应再乘以 17。
这是 Temporal 注意力窗口容量，不是已经通过长音频验证的会话时长。
超过窗口后的环绕与质量仍需要单独验证，不能用 38 帧检查代替。

当前宿主 KV 是 FP32：32 层、每层 K/V 各一个张量、32 个头、头维 128。
不含额外图 I/O、DSP 分配与权重时，大小为：

```text
1000 帧：32 × 2 × 32 × 1000 × 128 × 4 = 1,048,576,000 bytes = 1000 MiB
 750 帧：32 × 2 × 32 ×  750 × 128 × 4 =   786,432,000 bytes =  750 MiB
差值：262,144,000 bytes = 250 MiB
```

运行器按图输入形状读取容量，已有 patch 工具接受 `--capacity 750`。
实际部署仍需生成匹配的 16 个 Temporal 图、编译 context，并检查第 749–751 帧附近的环绕。
不能把旧 1000 槽 context 改名后用于 750 槽图，也不能只改 App 参数。
本次保留 cache1000，先完成 Mimi 最小量化实验。
Mimi 自己的 250 步局部 Transformer 状态独立于 LM 的 1000 槽窗口，本次不改。

## 已有模型与量化方法

复用已经通过的显式状态 ONNX：
`/data2/yejialei/tmp/moshi-mimi-explicit-state-onnx-v5`。
没有重新下载 checkpoint 或重做 FP32 导出。

部署基线来自 `/data2/yejialei/tmp/moshi-mimi-android-bundle-v1`。
真实 DLC 审计表明其静态浮点参数使用 Float32，不能据文件名推测它已经是 W8。
encoder DLC 为 186,568,724 字节，decoder DLC 为 178,361,588 字节。
当前手机对应 SHA256 与这两个文件一致。
HTP 内部浮点数学和编译后的存储不等同于 DLC 中的 Float32 标签。

第一次将部分卷积与线性层一同量化，decoder 未通过留出帧门限，encoder 的 bias-free
下采样卷积被 QAIRT per-channel 转换拒绝。该结果保留在 `run/`，不部署。

随后在 `fc_only/` 中只量化两端各 32 个 Transformer 线性投影：

- 权重：逐输出通道对称 W8，范围搜索尚未启用。
- 激活：对所选投影的输入、输出使用 A16。以现有音频前 4 帧拟合 min/max，并加固定 5% 范围余量。
- 保留：全部卷积及转置卷积、码本、码本搜索和归一化的高精度路径。
- ABI：audio、KV 与卷积状态保持 Float32，codes 和 position 在 QNN 边界使用 Int32。
- CPU 检查：8 帧各自递推，前 4 帧参与校准，后 4 帧不参与范围拟合。
- CPU 模拟执行作者指定的权重与激活网格，不是 QNN CPU 后端执行 DLC。
- DLC 审计：核对实际 W8/A16 类型、公开状态类型，并将实际 DLC 反量化权重与指定网格比较。
- HTP：独立编译 SM8850/v81 context，随后做短手机对照。

这是初步 PTQ 可行性实验，不是完整质量验收。8 帧来自同一音频，不能代表完整使用分布。
不要求编码 token 逐个相同，但必须报告匹配率与音频差异。

## 可复现入口

`scripts/moshi_quantize_mimi_offline.py` 生成策略并执行 CPU 检查与转换。
`scripts/moshi_validate_mimi_dlc.py` 审计实际 DLC 并离线编译 HTP context。
`scripts/moshi_run_mimi_phone_probe.py` 在独立缓存中各跑一次 4 帧 codec 对照。
手机检查复用基线 DLC 作为现有运行器的标签和缓存键，但候选图必须从候选 context 恢复。
若发生 DLC compose/finalize 回退，检查失败。
候选实际 DLC、哈希与编译收据保留在远端，不覆盖默认缓存。

各阶段命令与结果分别记录在 `RESULT.json`、`PRECISION_AUDIT.json`、`conversion.json` 和手机 `commands.json`。
短手机耗时不是统计稳定的延迟结论，也不是完整 LM 回复延迟。

## 首轮通过结果

两端各量化 25,165,824 个线性权重。两片都通过实际 DLC 位宽、权重网格与公开 ABI 审计，
并成功编译 SM8850/v81 context。8 帧 CPU 模拟中 encoder 的编码匹配率为 63/64；
decoder 在固定参考编码下的逐帧 PCM 相对 RMS 差异最大为 2.646%。

| 文件 | 原默认字节数 | 候选字节数 |
| --- | ---: | ---: |
| encoder DLC | 186,568,724 | 106,161,412 |
| decoder DLC | 178,361,588 | 95,874,580 |
| encoder HTP context | 152,854,528 | 128,548,864 |
| decoder HTP context | 89,935,872 | 65,470,464 |

DLC 合计减少约 44.6%。HTP context 合计减少 48,771,072 字节，即 46.51 MiB、20.09%。
没有对候选进行独立 DMA-BUF/RAM 测量，不能把 context 文件降幅报告为实际内存降幅。

4 帧手机 codec 对照位于 `phone/PHONE_RESULT.json`：

- 两组均恢复两个已有 context，没有 DLC compose/finalize 回退。
- encoder code 匹配为 29/32，即 90.625%。
- PCM 相对 RMS 差异为 0.6485%，相关系数 0.999979，无削波。
- encoder 平均图执行：36.249 → 37.698 ms。
- decoder 平均图执行：22.413 → 23.960 ms。
- codec 流水线：245.119 → 254.769 ms。单次短对照未显示加速。
- 没有独立听感、长音频质量或完整 LM 链路验收。

手机旧 `moshi_qnn_benchmark` 未使用指定的 Mimi cache 目录，因此被拒绝。
另一次校验使用了字节相同但 mtime 不同的 `models/mimi` 副本，缓存键未命中，也被拒绝。
两次日志分别保存在 `phone_rejected_old_runner/`、`phone_rejected_mtime/`，不计入通过结果。
最终程序使用当前 `android/mimi_qnn_runner.cpp` 重新构建，并使用实际部署包中的 Mimi DLC 路径。
`MOSHI_CODEC_CONTEXT_CACHE_DIR` 指向独立候选目录，初始化恢复证据从完整 native 日志读取，
因为 run() 会在开始处理帧前清空 CSV 时序列表。

本次约新增 0.2 GiB 的选定 context 与小型记录，未将候选 DLC 下载到 Mac 或手机。
两端候选 DLC 保存在远端，手机默认包和默认缓存未被替换。
