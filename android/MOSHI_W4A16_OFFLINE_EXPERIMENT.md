# Moshi Temporal W4A16 离线实验：2026-10-15

本轮先处理一个 Temporal 分片。用户暂不连接手机。候选不替换默认模型包。

## 输入和环境

- 分片：`temporal_layers_2_3`。
- 远端输出根目录：`/data3/yejialei/moshi-w4a16-offline-20261015-v1`。
- 候选子目录：`temporal_layers_2_3_int32`。
- 本地交付目录：`/Users/e1ainay/d_personal/moshi-models/lm-w4a16-offline-20261015-v1`。
- Python：`/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python`。
- SDK：`/data2/yejialei/sdk/qairt/2.45.0.260326`。
- SDK 使用隔离的 NumPy 1.26.4。普通 ONNX 环境原有 NumPy 2.2.6 不变。

直接复用已经转换的浮点 cache1000 DLC。其 SHA256 为：
`456c05a979b4d283c4796906c1bbbb947324358df8096487401db82c8b002082`。

源 ONNX：
`/data3/yejialei/moshi-cache1000-scatter-delta-lm-v2/graphs/temporal_layers_2_3/candidate/model.onnx`。

原始校准目录：`/data2/yejialei/tmp/moshi-lm-calibration-mp-v1`。
使用已有 clean/overlap 两组共 16 个样本。没有重新下载权重或重新导出 FP32。

保留 1000-slot cache、delta KV 输出、原有 Scatter 更新和 guarded RMSNorm 来源。
DLC 的 `position` 输入为 int32。源 ONNX 的 `position` 仍为 int64。
原始 NPZ 在生成 DLC 输入文件时转换为 int32。公共浮点输入和输出保留原 dtype。
Mimi、手机运行器、常驻策略和默认模型包没有修改。

## 实际精度和大小

16 个选定 Temporal 源图共有 128 个静态投影。其中 34 个为 FP32，94 个已有 INT8 QDQ。
`2_3` 和 `30_31` 的 8 个投影均为 FP32。清单来自实际图的初始化器及 QDQ 节点。
完整结果见输出根目录的 `source_precision_inventory.json`。

`2_3` 使用 QAIRT 离线量化器。配置如下：

- 权重 4 位，MatMul/FullyConnected 按行量化。
- 权重采用 symmetric min-max，激活采用 16 位 min-max。
- 保留公共 I/O dtype。偏置位宽选项为 32。
- 保存完整 encodings、输入样本哈希、源码哈希和执行命令。

实际 DLC 检查确认 8 个静态投影均为 4 位编码。其数据类型为 `sFxp_8`，
量化编码为 `BW_AXIS_SCALE_OFFSET`，编码位宽为 4。投影激活为 `uFxp_16`。
DLC 用一字节容器保存一个 4 位编码值。不能把 DLC 大小视为最终 HTP 常驻内存。

| 项目 | 原 DLC | W4A16 候选 DLC |
| --- | ---: | ---: |
| 文件大小，字节 | 1,644,432,748 | 413,691,292 |
| 静态投影权重 | 8 个 FP32 | 8 个 W4 编码 |
| 文件大小减少 | — | 74.843% |

量化器运行约 51.83 秒。候选 SHA256：
`056705c156782ad24f635d416b707bd1858a3c25158c5f6722551bac3e5cb039`。

## 必要检查及证据边界

`precision_audit.json` 直接读取候选 DLC，不根据文件名判断位宽。

直接调用 QNN CPU 后端时，图在 `ElementWiseBinary` 算子校验阶段失败。
错误为 `QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE`。没有取得候选 DLC 的 CPU 输出。
该错误不构成 HTP 不支持 W4 的证据。日志保存为 `cpu_run.log` 和 `cpu_check.json`。

ONNX CPU 模拟器从候选 DLC 读取实际 W4 整数和 scale。
它逐项检查这些整数与原权重量化后的整数一致。
它保留源图中的计算和 KV ABI，添加能够映射回源图的 A16 编码。
无法映射的 QNN 内部变换单独记录。该检查不等于实际 DLC 执行。

数值检查只使用 clean-000 的第 0、1 帧。
每个输出检查 NaN/Inf、零输出及相对 RMS 误差。
10% 是首轮筛选上限，不是音频质量标准。
这两个样本也不构成独立质量集或完整递归链路检查。

CPU 模拟检查失败。第 0、1 帧 hidden 相对 RMS 误差为 83.48%、91.83%。
第 0 帧 layer_2 V 误差为 26.95%。第 1 帧两层 V 误差为 35.88%、33.39%。
所有输出均有限，但数值误差超过本轮 10% 上限。候选不进入 HTP 或整链测试。
完整结果见候选目录的 `cpu_simulation.json`。

`moshi_diagnose_w4_weights.py` 复用 CPU 模拟模型中的实际 W4 权重，
移除激活量化，再检查相同的两个样本。它只定位误差来源，不产生部署候选。

| CPU 检查 | 第 0 帧 hidden 相对 RMS | 第 1 帧 hidden 相对 RMS |
| --- | ---: | ---: |
| 实际 W4 权重及映射的 A16 编码 | 83.48% | 91.83% |
| 相同 W4 权重，源图浮点激活 | 4.70% | 42.93% |

仅保留 W4 权重时，layer_2 V 的两帧误差仍为 26.92%、35.86%。
因此，不能仅靠恢复激活精度解决本候选的全部误差。
当前按行 min-max W4 方案停止扩大。没有启动 `30_31` 或其余分片量化。
下一轮先改进首片的权重量化方法及敏感激活编码，再检查相同的最小数值关卡。

输出根目录的 `RESULT.json` 标记为 `rejected_by_offline_numerical_screen`。
候选 DLC 和实际编码已经下载到独立本地目录，并重新检查 SHA256。
手机 HTP 单片、4 帧、38 帧、音频质量、实际常驻数量、内存和 context 装载次数均待检查。
原模型手机耗时仅作后续对照，不能用于推算本候选速度。

## 复现

在远端仓库运行以下命令。`NEW_OUTPUT_DIRECTORY` 必须是新的独立目录。

```sh
/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python \
  scripts/moshi_quantize_temporal_offline.py \
  --source-onnx /data3/yejialei/moshi-cache1000-scatter-delta-lm-v2/graphs/temporal_layers_2_3/candidate/model.onnx \
  --source-dlc /data3/yejialei/moshi-cache1000-scatter-delta-lm-v2/graphs/temporal_layers_2_3/model.dlc \
  --expected-dlc-sha256 456c05a979b4d283c4796906c1bbbb947324358df8096487401db82c8b002082 \
  --calibration-dir /data2/yejialei/tmp/moshi-lm-calibration-mp-v1 \
  --graph temporal_layers_2_3 \
  --sdk-root /data2/yejialei/sdk/qairt/2.45.0.260326 \
  --numpy-dir /data2/yejialei/tmp/moshi-qdq-0-1-cpu-check-v1/dlc_fusion_diagnostics/python-deps-numpy126 \
  --output-dir NEW_OUTPUT_DIRECTORY
```

`moshi_check_offline_dlc.py` 接受 `--directory`、`--sdk-root` 和 `--numpy-dir`。
它检查实际 DLC。加 `--audit-only` 可以只读取编码。

`moshi_simulate_offline_dlc.py` 接受相同的三个路径参数。
启动 SDK Python 工具前，将 `LD_LIBRARY_PATH` 指向 SDK 的 `lib/x86_64-linux-clang`。
CPU 模拟文件可根据源码和实际 encodings 重新生成。保留报告即可避免重复运行。

## 空间清理

用户授权删除适量无用产物。首轮删除约 4.04 GB 逻辑数据：

- 旧 v1 目录中 3 个已有恢复来源的损坏 ZIP 下载。
- 旧 Scatter 原型中 HTP 数值检查失败的 DLC。保留其源图及失败报告。
- 本次停止的 int64 输入准备目录中的可重新生成 raw 输入。

清理清单保存在远端输出根目录的 `cleanup.json`。
没有删除原始校准 NPZ、当前模型、已验证候选或转换来源。

完成数值检查后，继续删除可重新生成的 CPU 浮点模拟权重文件和 raw 输入副本。
保留本次样本 000、001 的 DLC 原生输入、所有报告、候选 DLC 和实际 encodings。
两轮共删除 6,606,047,897 字节逻辑数据，约 6.61 GB。
`input_list.txt` 其余行的 raw 输入可从保留的原始 NPZ 重新生成。
