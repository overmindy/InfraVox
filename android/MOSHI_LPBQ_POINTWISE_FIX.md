# Moshi LPBQ 算子表示修复与正确性审计

日期：2026-10-16。目标：修复真实 Temporal 分片的 LPBQ 离线编译失败。
手机未连接。默认模型包未替换。只处理 `temporal_layers_2_3`。

## 已完成结果

QAIRT 2.45.0.260326、SM8850、HTP v81 的完整分片 context 编译返回 0。
候选保留 7 个 LPBQ W4、1 个逐通道 W8 投影。全部投影输入输出为整数 A16。
W4 覆盖 394,264,576 / 411,041,792 个投影权重参数，即 95.9184%。
Layer 2 attention 输出投影因校准误差保留 W8。其余投影保留已有 LPBQ 选择。

| 同 SDK、同设备配置 | Context 字节 | MiB |
| --- | ---: | ---: |
| 当前默认 2_3 基线 | 824,279,040 | 786.09375 |
| 前一轮 2 个逐通道 W4、6 个 W8 | 346,349,568 | 330.3046875 |
| 本轮 7 个 LPBQ W4、1 个 W8 | 230,772,736 | 220.08203125 |

相对默认 context 文件减少约 72.00%，相对前一轮候选减少约 33.37%。
这不是手机内存或延迟的测量。LPBQ 包含 W4 到 W8 网格的解码开销。

远端候选：`/data3/yejialei/moshi-lpbq-pointwise-20261016-v2`。
本地交付：`/Users/e1ainay/d_personal/moshi-models/lm-lpbq-pointwise-20261016-v2`。

## 失败范围与修复

原先的微型 128×256 FullyConnected 可编译。
独立真实尺寸 4096×12288 FullyConnected 在 HTP 图最终化失败，日志包含 GenPad 错误。
完整分片还在 11264 输入的下投影附近发生段错误。
将部分权重展开为等价 W8 也没有解决剩余 FC 的图最终化失败。

本轮先使用同样 4096×12288 尺寸、block size 32 和 LPBQ/A16，
改写为 1×1 Conv。该独立算子通过 HTP 离线编译。
随后对真实分片的八个静态投影做同样改写：

```text
原始：X[1,1,I] @ W[I,O] -> Y[1,1,O]
改写：Reshape X -> [1,I,1,1]
      Conv1x1，权重 W.T[O,I,1,1]
      Reshape 输出 -> [1,1,O]
```

单 token 下，这个变换保持浮点线性映射。新增 reshape 不改变元素顺序。
两个采集输入的完整分片浮点输出最大相对 RMS 差异为 1.2646e-6。
差异来自 CPU MatMul 与 Conv 的浮点累加顺序。允许上限为 1e-5。
没有重新导出 PyTorch，没有重新拟合量化权重，没有修改 cache 或注意力计算。

这证明当前检查范围内，等价 Conv 表示避开了 QAIRT 2.45 的大型 FC 路径失败。
没有据此断言所有 FC、所有设备或所有 SDK 版本都存在同一缺陷。

## 量化正确性审计

依据：

- [AIMET LPBQ 文档](https://qualcomm.github.io/aimet-pages/releases/2.20.0/techniques/lpbq.html)。
- [AIMET LPBQ 尺度源码](https://github.com/qualcomm/aimet/blob/develop/TrainingExtensions/onnx/src/python/aimet_onnx/lpbq_utils.py)。
- [AIMET 量化配方](https://github.com/qualcomm/aimet/blob/develop/Docs/tutorials/quantization_recipe.rst)。

本轮保存了官方尺度源码及 SHA256，从中提取三个 NumPy 函数独立运行。
没有安装或宣称运行完整 AIMET 包。

对每个输出通道，已有实现计算：

```python
block_step = max(-block.min() / 8, block.max() / 7)
channel_step = max(block_steps_for_channel) / 16
integer_step = clip(round(block_step / channel_step), 1, 16)
effective_step = channel_step * integer_step
W_hat = clip(round(W / effective_step), -8, 7) * effective_step
```

官方 `grouped_dynamic_quantize(..., grouping=[1,-1], bitwidth=4)`
对七个 LPBQ 权重产生的 12,320,768 个逐块整数系数全部与已有实现一致。
全部逐通道浮点尺度也精确一致。
这里核对的是 LPBQ 尺度压缩公式。初始 block range 仍是本实验的 min/max 选择。

另直接读取改写前后的两个真实 DLC，完成以下核对：

1. 八个投影共 411,041,792 个权重码值逐个相同。
2. 每个输出通道的位宽、浮点尺度和零点完全相同。
3. 七个 LPBQ 投影的 block size、逐块整数系数和扩展位宽完全相同。
4. QNN Conv 的 HWIO 布局被明确转换回 OI，与原 FC 权重比较。
5. LPBQ 编码仍为 `BLOCKWISE_EXPANSION`，压缩位宽为 4，扩展网格为 8。
6. 转换器为 Conv 增加的 bias 全为零。
7. 实际 DLC 的公开输入输出名称、顺序、shape 和 dtype 与源 DLC 完全相同。
8. 四个 RMSNorm 的平方及保留链路均为 FP32。
9. 四个 Scatter 的 data/update/output 均为 Float16，cache 保持 `[1,32,1000,128]`。

检查器最初按 `ElementWisePower` 查找平方，但 QAIRT IR 使用 `Eltwise_Binary`。
已按保留的 `/norm*/Pow` 节点名修正检查器，再核对全部归一化链路类型。
原始诊断保存在 `lpbq_audit_ir_type_diagnostic.json`，候选权重未因该修正改变。

## 数值检查

读取新 DLC 的权重码值和实际激活编码，在 ONNX Runtime CPU 中模拟。
Conv 的实际编码轴为 3，FC 的实际编码轴为 0，检查器明确处理该差异。
参考仍为原始 cache1000、RMSNorm 已修复的 FP32 ONNX。

| 检查范围 | hidden 最大相对 RMS | KV 最大相对 RMS |
| --- | ---: | ---: |
| 两段音频的 16 个采集输入 | 9.3028% | 9.4285% |
| 两段各 3 帧、候选使用自己的 KV | 8.8798% | 9.2167% |

全部输出有限，新 K/V 非零，host 保留槽未改变。每个输出低于既定 10% 上限。
这不是直接执行 DLC，也不是完整 LM 递归或独立音频质量检查。
已有权重拟合使用 12 个采集样本，策略选择使用 16 个样本。
这些样本已参与策略选择，不能再作为独立测试集。

## 当前做法的适用边界

训练后校准、逐通道/分块量化、敏感矩阵保留 W8，以及转换后的真实编码核对，
均属于合理的 PTQ 工作流程。LPBQ 的尺度公式已与官方函数逐项核对。
本轮只是改变等价算子表示，未增加拟合或搜索。

AWQ 风格缩放、自写 GPTQ 风格舍入补偿和贪心 W4/W8 分配仍是实验实现。
强正则系数 32.0 和 10% 输出误差上限是本实验设置，不是官方默认值或通用质量标准。
尚未证明这套参数优于官方 LPBQ+SeqMSE，或已找到最优位宽分配。
因此，候选可用于下一步设备验证，不能据此宣布产品质量通过。

## 执行入口与复现

代码：

- `scripts/moshi_probe_lpbq_support.py --operator conv`：真实尺寸独立支持检查。
- `scripts/moshi_rewrite_lpbq_pointwise.py`：复用已通过数值筛选的 LPBQ 权重，改写、转换、数值检查、编译。
- `scripts/moshi_audit_lpbq_pointwise.py`：官方尺度公式、真实码值、布局、公开 ABI 和状态精度审计。
- `scripts/moshi_simulate_offline_dlc.py`：支持 FC/OI 与 Conv/HWIO 的 DLC 参数模拟。

远端 Linux 使用：

```bash
cd /home/user/yejialei/qairt_moshi
PY=/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python
SDK=/data2/yejialei/sdk/qairt/2.45.0.260326
NUMPY=/data2/yejialei/tmp/moshi-qdq-0-1-cpu-check-v1/dlc_fusion_diagnostics/python-deps-numpy126
RUN=/data3/yejialei/moshi-lpbq-pointwise-20261016-v2

# 复现时将 RUN 改为新目录。现有目录有防覆盖检查。
"$PY" scripts/moshi_rewrite_lpbq_pointwise.py \
  --parent-directory /data3/yejialei/moshi-lpbq-candidate-20261016-v1 \
  --output-dir "$RUN" --sdk-root "$SDK" --numpy-dir "$NUMPY"

"$PY" scripts/moshi_audit_lpbq_pointwise.py \
  --directory "$RUN" \
  --official-source /data3/yejialei/moshi-lpbq-pointwise-20261016-v2/references/aimet_lpbq_utils.py \
  --awq-source /data3/yejialei/moshi-awq-20261016-v1/source/model.onnx \
  --sdk-root "$SDK" --numpy-dir "$NUMPY"
```

实际脚本快照在候选目录的 `runtime_scripts/`。JSON 记录对应脚本及模型哈希。
原始 FP32 导出、父候选和校准数据均保留。Mimi encoder/decoder 未修改。

## 尚未测量

手机 HTP 执行、4 帧及完整 38 帧链路、独立音频/文本质量、手机耗时、峰值内存、
实际常驻数量和 context 装载次数均待设备检查。不得用离线编译代替这些结果。
