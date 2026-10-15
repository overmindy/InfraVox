# Moshi 激活感知与 LPBQ 离线量化

本轮处理 `temporal_layers_2_3` 和 `temporal_layers_30_31`。用户暂不连接手机。
所有候选使用独立目录，不替换默认模型包。

## 当前已通过的候选

`temporal_layers_2_3` 使用 2 个逐通道 W4 和 6 个 W8 投影，FC 输入输出均为 A16。
W4 位于 Layer 2 FFN 下投影和 Layer 3 FFN 上投影。
敏感的 QKV、attention 输出及其余投影保留 W8，RMSNorm 内部保留 FP32。
W4 覆盖 138,412,032 个投影权重参数，占本片的 33.67%。
上一轮为 46,137,344 个参数，占 11.22%。本片 W4 参数数量提高到 3 倍。

| 同 SDK、同 HTP 配置的离线 context | 字节数 | MiB |
| --- | ---: | ---: |
| 交接清单中的默认 2_3 分片 | 824,279,040 | 786.09 |
| 上一轮 1 个 W4、7 个 W8 候选 | 391,090,176 | 372.97 |
| 本轮 2 个 W4、6 个 W8 候选 | 346,349,568 | 330.30 |

本轮 context 相对默认分片减少约 58.0%，相对上一轮候选减少约 11.4%。
这是 context 文件大小比较，不是手机峰值内存或延迟比较。
默认分片的 DLC 哈希与本地部署清单相同，两个编译都使用 QAIRT 2.45、SM8850、HTP v81。

| 数值检查范围 | hidden 最大相对 RMS | KV 最大相对 RMS |
| --- | ---: | ---: |
| 16 份同输入校准样本 | 8.86% | 5.30% |
| 两个片段各连续 3 帧，分别递归 K/V | 8.70% | 5.21% |

所有输出有限，新写入 K/V 非零，host 保留槽未变化。每个输出均低于 10% 上限。
该结果使用实际 DLC 权重与映射后的激活编码，在 ONNX CPU 上模拟。
样本中包含第 37 帧，但这不是完整 38 帧递归，也不是手机执行。
HTP v81 离线编译返回 0。

候选目录：`/data3/yejialei/moshi-regularized-priority-candidate-20261016-v1`。
其 `RESULT.json`、`candidate/calibration_validation.json` 和 `htp_offline/compile.json` 为结果依据。

`temporal_layers_30_31` 使用 1 个 W4、7 个 W8 投影，FC 输入输出均为 A16。
W4 位于 Layer 31 FFN 上投影，占本片投影参数的 22.45%。
其余投影由分片误差筛选保留 W8，没有沿用 2_3 的位宽分配。
16 份样本的最大 hidden/KV 相对 RMS 为 9.35% / 1.29%。
两个片段各 3 帧递归的最大 hidden/KV 相对 RMS 为 7.41% / 1.29%。
所有输出有限，新 K/V 非零，host 保留槽不变，HTP 离线编译返回 0。

| 分片 | W4 参数比例 | 默认 context MiB | 候选 context MiB | 文件减少 |
| --- | ---: | ---: | ---: | ---: |
| 2_3 | 33.67% | 786.09 | 330.30 | 57.98% |
| 30_31 | 22.45% | 786.10 | 352.94 | 55.10% |
| 两片合计 | 28.06% | 1572.20 | 683.25 | 56.54% |

两片合计节省 932,130,816 字节 context 文件，约 889 MiB。
比例只针对这两个分片，不代表完整 16 片 Temporal 或整个 LM。
DLC 使用字节容器保存低位宽值，DLC 大小不能直接体现 W4 的打包收益。
这里使用相同 SDK 与配置生成的 context 比较文件大小。
手机内存、延迟、实际常驻数量和装载次数仍未测量。
两片实际 DLC 的输入输出名称、序列化顺序、形状和 dtype 均与各自源 DLC 一致。
两个 `candidate/state_precision_guard.json` 记录 ABI、FP32 RMSNorm 和 Float16 Scatter 检查。

30_31 候选目录：`/data3/yejialei/moshi-temporal-30-31-offline-20261016-v1/candidate_workflow`。
合并证据目录：`/data3/yejialei/moshi-quantization-advanced-20261016-v1`。
本地交付目录：`/Users/e1ainay/d_personal/moshi-models/lm-mixed-advanced-20261016-v1`。
两个 DLC 和两个 context 的下载哈希写入 `DOWNLOAD_VERIFIED.json`。

本轮清理已完成或已排除方案的可再生 CPU 模拟文件。
清单与回收字节数记录在合并证据目录的 `cleanup.json`。
原始 FP32 权重、校准输入、投影快照、候选 DLC、context、编码、报告和脚本均保留。

## 研究依据

- [AWQ 论文](https://arxiv.org/abs/2306.00978)和[作者实现](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)：
  根据输入激活调整通道尺度，再量化权重。
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)：
  使用输入协方差补偿权重量化误差。
- [高通 SeqMSE](https://qualcomm.github.io/aimet-pages/releases/2.35.1/ptq_techniques/seq_mse.html)：
  用层输出重构误差选择量化范围。
- [高通 LPBQ](https://qualcomm.github.io/aimet-pages/releases/2.3.0/quantsim/blockwise.html)：
  将分块低位宽编码放到共同的逐通道高位宽网格中。
  LPBQ 支持低位宽存储，并可使用现有逐通道计算内核。

本机远端环境没有 AIMET。实验使用已有 PyTorch、NumPy、ONNX Runtime 和 QAIRT。
这是依据上述方法实现的实验脚本，不是 AIMET 或论文作者的完整工作流。
没有使用论文中的速度数字推算手机收益。

## 来源与保留项

原始 ONNX：
`/data3/yejialei/moshi-cache1000-scatter-delta-lm-v2/graphs/temporal_layers_2_3/candidate/model.onnx`。

ONNX SHA256：`56a3ab1dde6e40d291c7b9f4f2f545b51b0fe7d966a6cdb885c2b167249b9aba`。
外部权重 SHA256：`92ab32961ccc41b49442ca012026680b9e1be5940ba2256fe171e5d74abc892d`。

30_31 使用同一 cache1000 scatter-delta 目录内对应分片的原始 ONNX。
源 DLC SHA256：`5a11cda77f338e32d53cf93c88870aaab32731de9a5c2bc740ddcd53a523a6f3`。
它与本地部署清单完全一致。开始前确认两片源 DLC 的 8 个投影权重均为 FP32。

复用原有 clean-000、overlap-000 两个片段。
每个片段使用帧 0、1、2、4、8、16、24、37，共 16 份输入。
投影快照来自 `/data3/yejialei/moshi-mixed-20261016-calibration`。
30_31 复用同一份原始校准清单，在不变的源图上采集对应投影快照。
新快照位于 `/data3/yejialei/moshi-temporal-30-31-offline-20261016-v1/calibration`。

保留 cache1000、绝对 position、delta K/V 输出、Scatter 更新和 RMSNorm 转换保护。
Mimi encoder/decoder、当前默认模型和运行调度均不修改。
不重新下载模型，不重新导出 PyTorch FP32 模型。
离线处理只生成原图的通道缩放副本和量化权重副本。

通道缩放使用 `x / s` 与 `s * W`。
浮点缩放图先与不变的原始图比较，再检查量化图。
最终输出误差始终相对于不变的原始图计算。
转换器输入中已补偿的权重不作为质量比较基准。

ONNX 元数据的哈希不足以识别全部权重。
不同候选可能使用相同的张量形状、外部文件位置和 ONNX 哈希。
必须同时保存并检查 `model.data` 的 SHA256。

## 数值筛选方法

逐投影搜索使用 12 份快照拟合，另外 4 份检查局部误差。
随后，16 份样本全部参与精度选择。
因此，这 4 份最终也不能称作独立质量集。

精度选择比较整个分片的 hidden 和四个新写入 K/V。
它还检查每个片段的连续帧 0、1、2。
递归时，参考图和候选图分别使用各自产生的 K/V。
上游 hidden 仍来自捕获数据，所以这不是完整 LM 递归。

最终 DLC 数值检查使用实际 DLC 权重和激活编码，在原始 ONNX 上模拟。
检查包括 16 份同输入样本和两个片段各 3 帧的分片递归。
最终每个输出的相对 RMS 上限保持 10%。
NaN、Inf、新写入 KV 全零、ABI 变化或 host 保留槽变化均判失败。

这些检查不是原生 DLC 执行。
模拟器记录未映射的编码，并将 FC 的转换器别名映射回原始投影边界。
QNN 内部融合、非 FC 的 FP16 运算和其他转换器变换未全部重现。
HTP 离线编译成功也不等于手机执行通过。

## 已排除的方案

| 方案 | W4 参数比例 | 结果 |
| --- | ---: | --- |
| 直接把 6 个非 QKV 投影降为 W4 | 75.51% | 第 1 帧 hidden 误差 28.81%，停止编译 |
| AWQ 类缩放与逐层输出范围选择 | 41.84% | 16 样本最大误差 11.55%，短递归最大误差 11.58% |
| 加入弱正则协方差补偿 | 100% | 16 样本最大误差 27.01%，短递归最大误差 20.98% |

弱正则使用归一化输入协方差与 0.01 阻尼。
少量拟合样本上的误差很低，但其他样本变差。
没有将这个结果解释成 W4 质量通过。

强正则与完整校准筛选给出逐通道 W4 回退候选。
它保留 2 个 W4、6 个 W8 投影，W4 参数比例为 26.53%。
其 16 样本和短递归检查通过，HTP v81 离线编译也通过。
context 为 361,046,016 字节。它仍不是设备验证结果。

## LPBQ 支持检查

先构造一个 128 输入、256 输出的微型 FullyConnected，分块大小为 32。
使用 QAIRT 2.45 的 1.0.0 编码格式。
实际 DLC 权重编码为 `BLOCKWISE_EXPANSION`，FC 输入和输出均为 `uFxp_16`。
读取实际 DLC 后，32,768 个权重编码全部与预期一致。
SM8850、HTP v81 离线编译返回 0。

LPBQ 使用 W4 压缩值、逐块整数扩展系数和逐通道浮点尺度。
本实验的扩展网格为 8 位，压缩权重为 4 位。
因此，不能把它描述成所有运算都采用 4 位。
每个输入通道块使用 32 个权重。
未通过分片数值门槛的投影保留 W8。

支持检查目录：`/data3/yejialei/moshi-lpbq-support-20261016-v1`。

## 真实分片及 HTP 兼容处理

LPBQ 的完整校准搜索保留 7 个 W4、1 个 W8 投影。
W4 参数比例为 95.92%。Layer 2 attention 输出投影保留 W8。
该候选通过实际 DLC 权重与编码的数值检查，但完整 HTP 离线编译发生段错误，返回 -11。
它不作为可用的 HTP 候选交付。

调试日志显示，崩溃发生在 11264 输入的 FFN 下投影附近。
前面 4096 输入的 LPBQ 投影已经完成组图。
这将问题范围缩小到下投影的 LPBQ 处理，但不足以证明具体的 SDK 缺陷原因。

兼容候选将两个 FFN 下投影改成等价的逐通道 W8 表示。
计算 `W8_code = W4_code * block_integer_scale`，逐通道浮点尺度保持不变。
W4 码值在 -8 到 7 之间，整数扩展系数在 1 到 16 之间，乘积可由有符号 8 位表示。
没有重新拟合权重，也没有对该权重再做一次有损量化。

实际读取新旧两个 DLC 后，两处共 92,274,688 个整数码值全部匹配，逐通道尺度也完全一致。
该兼容诊断候选实际存储为 5 个 LPBQ W4、3 个 W8 投影，W4 参数比例为 73.47%。
其中一处 W8 来自敏感性选择，两处 W8 来自 HTP 兼容处理。
8 个 FC 的输入和输出均为 A16。

RMSNorm 的 4 个平方计算及其关键中间操作在实际 DLC 中仍为 FP32。
4 个 Scatter 的 data、update 和 output 均为 FP16，内部缓存形状保持 `[1,32,1000,128]`。
这项检查没有把 host 槽保留证明当成 HTP 内部状态证明。

SDK 的 1.0.0 编码 JSON 将部分实际 FP32 张量标成 `INT32`、`scale=0`。
模拟器以实际 DLC IR 的类型为准，并在 `encoding_metadata_conflicts` 中记录冲突。
它还记录无效或未映射的编码，不把这些条目注入浮点计算图。
该修复只影响检查器，不改变候选权重。

兼容候选目录：`/data3/yejialei/moshi-lpbq-htp-compatible-20261016-v1`。

该诊断候选通过数值检查，但 HTP 图最终化仍失败，返回 15。
日志包含 `GenPad` 缺少属性，以及 LPBQ FC 无法创建的错误。
因此，73.47% 候选也不作为可运行的 HTP 结果交付。

进一步使用独立的 `[1,4096] × [4096,12288]` FC 复现。
分块大小 32 和 128 都无法完成 HTP 图最终化。
这两个检查不包含 Moshi 的注意力、RMSNorm 或 KV。
微型矩阵可编译不足以证明真实尺寸支持。
本轮停止扩大 LPBQ，保留日志供后续 SDK 或表示方式修复使用。

另按[高通经验偏置校正说明](https://qualcomm.github.io/aimet-pages/releases/1.32.0/user_guide/post_training_quant_techniques.html)
检查输出均值偏移。校正使用 12 份样本拟合，再用全部样本和短递归选择。
它没有提高 W4 覆盖率，因此没有为这个方案再生成 DLC。

## 复现入口

1. `moshi_optimize_activation_aware.py`：搜索输入通道缩放，并检查等价浮点图。
2. `moshi_refine_weight_rounding.py`：复现弱正则补偿的诊断结果。
3. `moshi_select_regularized_policy.py`：复现逐通道回退候选。
4. `moshi_probe_lpbq_support.py`：复现微型 LPBQ/A16 支持检查。
5. `moshi_build_lpbq_candidate.py`：按校准敏感性选择 LPBQ W4 与 W8。
6. `moshi_simulate_offline_dlc.py`：读取真实 DLC 权重与编码，检查最初两帧。
7. `moshi_validate_temporal_calibration.py`：检查其余校准输入和三帧分片递归。
8. `moshi_compile_temporal_candidate.py`：仅在数值门槛通过后生成 HTP context。
9. `moshi_reencode_lpbq_compat.py`：将两个下投影转换成等价 W8，并重新检查候选。
10. `moshi_resume_offline_checks.py`：从缺少的检查继续，不重复量化或已通过的检查。

每个脚本的 `--help` 提供路径参数。输出目录必须是新的独立目录。
模型、校准数据和大型中间文件不进入 Git。
各候选目录中的 JSON 保存脚本哈希、模型哈希、位宽、编码和执行命令。

## 待完成的设备检查

手机接入后，先做同输入 HTP 单片比较，再做 4 帧和完整 38 帧链路检查。
比较 hidden、KV、文本、音频、耗时、峰值内存、实际常驻数量和 context 装载次数。
未取得这些结果前，不报告实际手机提速或内存下降，也不替换默认模型包。
