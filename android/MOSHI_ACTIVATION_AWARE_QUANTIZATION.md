# Moshi 激活感知与 LPBQ 离线量化

本轮处理 `temporal_layers_2_3`。用户暂不连接手机。
所有候选使用独立目录，不替换默认模型包。

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

复用原有 clean-000、overlap-000 两个片段。
每个片段使用帧 0、1、2、4、8、16、24、37，共 16 份输入。
投影快照来自 `/data3/yejialei/moshi-mixed-20261016-calibration`。

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

## 复现入口

1. `moshi_optimize_activation_aware.py`：搜索输入通道缩放，并检查等价浮点图。
2. `moshi_refine_weight_rounding.py`：复现弱正则补偿的诊断结果。
3. `moshi_select_regularized_policy.py`：复现逐通道回退候选。
4. `moshi_probe_lpbq_support.py`：复现微型 LPBQ/A16 支持检查。
5. `moshi_build_lpbq_candidate.py`：按校准敏感性选择 LPBQ W4 与 W8。
6. `moshi_simulate_offline_dlc.py`：读取真实 DLC 权重与编码，检查最初两帧。
7. `moshi_validate_temporal_calibration.py`：检查其余校准输入和三帧分片递归。
8. `moshi_compile_temporal_candidate.py`：仅在数值门槛通过后生成 HTP context。

每个脚本的 `--help` 提供路径参数。输出目录必须是新的独立目录。
模型、校准数据和大型中间文件不进入 Git。
各候选目录中的 JSON 保存脚本哈希、模型哈希、位宽、编码和执行命令。

## 待完成的设备检查

手机接入后，先做同输入 HTP 单片比较，再做 4 帧和完整 38 帧链路检查。
比较 hidden、KV、文本、音频、耗时、峰值内存、实际常驻数量和 context 装载次数。
未取得这些结果前，不报告实际手机提速或内存下降，也不替换默认模型包。
