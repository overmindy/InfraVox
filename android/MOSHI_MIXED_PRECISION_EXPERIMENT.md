# Moshi Temporal 按敏感性选择精度

本实验只处理 `temporal_layers_2_3`。用户暂不连接手机。
候选保存在独立目录，不替换默认模型包。

## 方法

保留原 cache1000、delta KV 输出、Scatter 更新和 guarded RMSNorm 来源。
不修改 Mimi，不重新下载权重，不重新导出已通过的 FP32 权重。

投影采用分级精度。低敏感投影可以使用 W4。敏感投影提高到 W8。
如果 W8 仍超过局部误差上限，则保留该投影的浮点精度。
RMSNorm 内部、attention mask 等非投影计算保留浮点。
量化投影的输入和输出采用 A16，并直接读取 DLC 检查实际类型。

使用已有 clean/overlap 两组共 16 份校准输入。
从不变的源图采集 8 个投影的输入和输出，保存小型 NPZ。
诊断图只增加输出元数据，外部权重使用原文件的硬链接。

每个投影分别计算现有 W4 min-max、W4 MSE 和 W8 的线性输出误差。
W4 可按输出通道选择两个已有编码中校准误差较小的一个。
QKV 分成 Q、K、V 三组检查，FFN 上投影按两个分支检查。
使用所有校准样本中的最差分组误差，防止平均数掩盖 V 或门控分支。

这些是校准敏感性统计，不是独立质量集，也不是完整链路试验。

## 已测敏感性

下表是隔离投影时的最大相对 RMS 误差。

| 层内投影 | W4 误差 | W8 误差 | 保守混合候选 |
| --- | ---: | ---: | --- |
| layer 2 QKV | 27.81% | 4.68% | W8 |
| layer 2 attention 输出 | 12.09% | 0.94% | W8 |
| layer 2 FFN 上投影 | 12.10% | 0.99% | W8 |
| layer 2 FFN 下投影 | 10.35% | 0.82% | W4 |
| layer 3 QKV | 27.59% | 3.04% | W8 |
| layer 3 attention 输出 | 13.42% | 0.88% | W8 |
| layer 3 FFN 上投影 | 11.51% | 1.03% | W8 |
| layer 3 FFN 下投影 | 13.77% | 0.93% | W8 |

局部 10% 上限给出全 W8 对照。该对照先通过两个样本的分片检查。
然后只将 W4 误差最低的 layer 2 FFN 下投影降为 W4。
该步骤限制 W4 投影数量为 1，并设置局部 15% 误差上限。
分片整体的 10% 筛选上限保持不变。局部上限不代表质量通过。

## 对照结果

全 W8 对照的 ONNX CPU 模拟检查通过：

| 输出 | 第 0 帧相对 RMS | 第 1 帧相对 RMS |
| --- | ---: | ---: |
| hidden | 0.481% | 2.806% |
| layer 2 K | 0.262% | 0.399% |
| layer 2 V | 1.927% | 4.794% |
| layer 3 K | 0.291% | 0.374% |
| layer 3 V | 0.493% | 3.210% |

两帧输出均有限，新写入 K/V 不为零。
该检查读取实际 DLC 的权重和编码，在源图上模拟。
它不是实际 DLC 执行，也不是手机 HTP 执行。

混合候选也通过相同的两帧筛选，整体上限仍为 10%：

| 输出 | 第 0 帧相对 RMS | 第 1 帧相对 RMS |
| --- | ---: | ---: |
| hidden | 0.496% | 2.817% |
| layer 2 K | 0.262% | 0.399% |
| layer 2 V | 1.927% | 4.794% |
| layer 3 K | 0.394% | 0.386% |
| layer 3 V | 0.601% | 3.196% |

两个方案的样本相同，均为 teacher-forced 的源图校准样本。
没有独立保留集、递归漂移、音频或长窗口质量证明。

实际 DLC 确认：`onnx::MatMul_535` 为 W4，其余 7 个投影为 W8。
8 个投影的输入和输出均为 `uFxp_16`。
W4 覆盖 46,137,344 个参数，约占这片投影权重的 11.2%。
本轮只先降低这一处。后续扩大 W4 覆盖需保留当前通过的组合。

源 DLC 为 1,644,432,748 字节。混合候选为 413,739,716 字节，减少约 74.84%。
DLC 仍以一字节容器保存 W4 编码值，其大小不是最终 HTP 内存。
新的 HTP context 必须从本候选生成，不能复用原模型的 context。

SM8850 / HTP v81 离线编译已通过，返回码为 0，用时 17.23 秒。
新 context 为 391,090,176 字节，约 372.97 MiB。
SHA256：`4605c9df9c974c949d1384d81c2eeed2196e0de05e426b5abb2702d4aeebed6a`。
这项结果只证明离线编译接受本图。没有在手机上执行该 context。
不能据此推断实际峰值内存、常驻数量、装载次数或延迟改善。

## 文件及复现入口

- 源图：`/data3/yejialei/moshi-cache1000-scatter-delta-lm-v2/graphs/temporal_layers_2_3/candidate/model.onnx`。
- 投影校准：`/data3/yejialei/moshi-mixed-20261016-calibration`。
- 敏感性报告：`/data3/yejialei/moshi-mixed-20261016-sensitivity`。
- 全 W8 对照：`/data3/yejialei/moshi-mixed-sensitive-20261016-v1`。
- 单投影 W4 策略：`/data3/yejialei/moshi-mixed-20261016-one-w4-policy`。
- 混合候选：`/data3/yejialei/moshi-mixed-one-w4-20261016-v1`。
- 本地交付：`/Users/e1ainay/d_personal/moshi-models/lm-mixed-one-w4-20261016-v1`。

脚本入口：

1. `moshi_calibrate_fc_boundaries.py --save-projections`：保存 16 份投影校准样本。
2. `moshi_select_sensitive_precision.py`：生成逐投影敏感性报告和保守对照策略。
3. `moshi_make_one_w4_policy.py`：只降低最不敏感的一个投影。
4. `moshi_author_w4_encodings.py`：复用明确编码，由 guarded QAIRT 转换并检查实际位宽。
5. `moshi_simulate_offline_dlc.py`：完成两帧分片筛选。

每个脚本使用 `--help` 查看必需路径。输出目录必须是新的独立目录。
源图哈希、样本哈希、编码哈希、执行命令及源码哈希保存在各目录的 JSON 中。
混合候选的结果以其 `RESULT.json` 为准。
离线编译命令和日志位于 `htp_offline/compile.json`、`htp_offline/compile.log`。
本地 DLC 命名为 `candidate/model.mixed-w4-w8-a16.dlc`，明确区分统一 W4。
已生成的 context 位于 `htp_offline/temporal_layers_2_3_mixed.bin`。

本轮另外清理 8,448,573,898 字节可重新生成的 CPU 浮点模拟文件，
以及停止的 SDK 搜索所使用的临时输入和转换副本。
候选 DLC、context、原始校准 NPZ、投影样本、编码、报告和脚本均保留。
详细删除清单位于 `cleanup.json`。没有删除当前默认模型或已完成的本地模型下载。

## 先前方案的证据边界

统一 W4 min-max 和统一 W4 MSE 没有通过分片数值筛选，不扩大部署。
旧激活范围不能直接复用于新的转换策略。源图 hidden 范围最低约为 -219.27，
旧部分编码只有约正负 20，可能裁剪有效输出。
此外，保留 norm 浮点时，转换器可能把投影边界也保留为浮点。
本流程明确指定每个投影的边界，并检查实际 DLC，防止误报 A16。

SDK 内置 MSE 在持续单核计算 645.58 秒且未写出候选后被停止。
独立 NumPy W4 范围搜索用时 66.92 秒。后续复用这些编码，没有重复搜索权重。

手机单片执行、4 帧和完整 38 帧、音频、内存、实际常驻数量和 context 装载次数待检查。
离线编译成功或文件变小均不能代替这些手机结果。
