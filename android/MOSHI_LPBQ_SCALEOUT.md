# Moshi Temporal LPBQ 扩展实验

实验日期：2026-10-16 至 2026-10-17。目标设备：BKQ-AN20、SM8850、HTP v81。
QAIRT：2.45.0.260326。保留 cache1000、增量 KV 输出、现有 RMSNorm 修复和 Mimi encoder/decoder。
不重做 PyTorch/FP32 导出，不替换手机默认模型或 App。

本地结果目录：`/Users/e1ainay/d_personal/moshi-models/lm-lpbq-scaleout-20261016-v1`。
远端结果目录：`/data2/yejialei/moshi-lpbq-scaleout-20261016-v1`。
远端仓库：`/home/user/yejialei/qairt_moshi`。
Python：`/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python`。
SDK：`/data2/yejialei/sdk/qairt/2.45.0.260326`。

## 已完成的速度方案

只替换原来高精度权重保留最多的 `temporal_layers_2_3` 和 `temporal_layers_30_31`。
两片各保留 7 个 LPBQ W4、1 个 W8 投影，投影激活使用 A16。
该方案的记录为 `TWO_MANIFEST.json`、`TWO_RESULT.json`、`phone_bundle_two/`。

| 完整 38 帧指标 | 本次基线 | 两片 LPBQ |
| --- | ---: | ---: |
| 首次完整处理 s | 45.056248 | 37.488367 |
| 暖运行完整处理 s | 31.327961 | 26.886624 |
| 暖运行首个 PCM s | 1.415760 | 1.435648 |
| 暖运行首个超过 -40 dBFS 的块 s | 3.817030 | 3.547620 |
| 暖运行实际常驻 Temporal 数量 | 16 | 16 |
| 暖运行 Temporal context 装载次数 | 0 | 0 |
| 暖运行输出实时率 RTF | 10.5838 | 9.0833 |

暖运行总耗时减少 14.18%，首次完整处理减少 16.80%。首个 PCM 没有提前。
这是单个 native session 中的首次/重复运行对照，不是多会话统计分布，也不包含 Java UI 和扬声器延迟。
仍未达到实时运行。

两个替换分片的 context 合计从 1,648,418,816 字节降至 462,209,024 字节。
全部 16 个 Temporal context 合计从 8,342,355,968 字节降至 7,156,146,176 字节，减少约 14.22%。
不得把替换分片的 71.96% 降幅写成整个 LM 的降幅。

4 帧阶段两组都实际常驻 16 片。在首轮完整输出之后，读取同一 native 进程的 `dmabuf_dump PID`：
基线拥有 11,991,476 KiB，候选拥有 10,808,820 KiB，减少 1,182,656 KiB，约 1.129 GiB。
这是进程拥有的 DMA-BUF PSS，包含该测试进程的模型/设备缓冲，不是整机总 RAM 或 App 峰值。
38 帧的进程 VmHWM 反而增加了 182 MiB；文件映射、可回收页面、CPU 堆和 DSP 分配不能混成一个指标。
原始 RSS/HWM、MemAvailable 和 DMA 明细均保留。

38 个输入帧产生 37 个输出帧、71040 个 PCM 样本、2.96 秒音频，无削波。
基线和候选各自首次/暖运行的 WAV 字节一致。
37/37 个文本 token 相同；287/296 个音频 code 相同。
PCM 相关系数 0.997817，候选相对基线的 PCM RMS 差异为 6.614%。
这些数值不是独立听感验收，也不是语音识别结果。

## 扩大范围时的实际发现

16 个 Temporal 分片均已完成离线候选构建。中间的 11 片组合已通过手机单片门限并组成完整链路。
11 片方案记录为 `ELEVEN_MANIFEST.json`、`ELEVEN_RESULT.json`、`phone_bundle_eleven/`。
其替换分片 context 合计减少约 50.10%，但本次 38 帧暖运行只减少 3.62%。
两组 38 帧实际常驻数量均为 16，暖运行 context 装载次数均为零。
37/37 个文本 token、272/296 个音频 code 相同，PCM 相关系数为 0.998523。

多数已经是 W8 的分片，在短单片检查中改成 LPBQ 后并未更快。
因此保留两类选择：优先速度的少量替换，以及优先权重/context 内存的扩大替换。
所有计时按各自实验的基线比较，不把不同会话的基线混用。

初版 0–1、22–23、24–25、26–27、28–29 候选虽然通过 CPU 模拟和离线编译，仍被手机门限拦住。
它们的失败记录保留，不能用于默认包。
24–25 的第 0 帧第二层 K/V 明显失真，随后污染其自己的递推缓存。
参考图的归一化输入峰值约为 290，平方值超过 FP16 有限范围。

只把 DLC 中的张量标为 FP32 不能保证 HTP 内部按 FP32 计算。
[官方 QNN Execution Provider 文档](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
明确说明 QAIRT 2.35 以后相关 HTP 浮点运算使用 FP16 数学。
因此本轮将“DLC 中保留 FP32 边界”和“设备实际数学精度”分开记录。

稳定的 RMS 分母采用等价公式：

```text
s = max(max(abs(x)), floor)
rms = sqrt(mean((x/s)^2) + epsilon/s/s) * s
```

恢复分母单位后，原有 gamma、分子和后续量化边界保持原单位。
该改写不改变权重、公开输入输出、缓存容量或 KV 更新方式。
24–25 采用 floor=1 后，通过手机门限：候选对 DLC CPU 模拟的最大误差从约 206% 降至 0.447%。
22–23、26–27、28–29 应用同一修复后也通过了手机门限。
这四片对 DLC CPU 模拟的最大误差为 0.257%–0.494%，结果位于 `stable_rms/`。

0–1 另有 3 个平方节点使用 `node_Mul_*` 名称，但输出名称带 `/norm`。
原来只按节点名称选择 FP32 的规则会漏掉它们，现已增加按输出张量名称匹配和实际 DLC 审计。
只补标记、以及恢复原基线三个浮点 FC 的备选，都没有解决该片的 HTP 偏差。
进一步用较小缩放下限改写归一化后，该片的 HTP 偏差仍未解决。
这些备选属于失败诊断，不是最终量化结果。
最终低内存组合保留原默认的 0–1：3 个 Float16 投影、5 个 W8 投影。
不将失败候选计为通过，也不扩大误差阈值。

## 十五片低内存组合

`LOW_MEMORY_MANIFEST.json` 选择已通过手机门限的 15 片，0–1 使用默认版本。
全部 16 个 Temporal context 从 8,342,355,968 字节降至 4,420,370,432 字节，减少 47.01%。
公开 ABI、cache1000 和每片原有的 Scatter/Where 选择均保留。

实际 DLC 逐矩阵审计保存在 `BASELINE_ACTUAL_PRECISION.json`、`PRECISION_COMPARISON.csv` 和 `PRECISION_COMPARISON.json`。
以下只统计 Temporal 的 128 个大投影矩阵，共 6,576,668,672 个参数；不包含 Mimi、DepFormer、head 或 frontend。

| 权重类型 | 原默认矩阵数量 | 低内存方案矩阵数量 |
| --- | ---: | ---: |
| FP32 | 16 | 0 |
| FP16 | 18 | 3 |
| W8 | 94 | 36 |
| LPBQ W4 | 0 | 89 |

大矩阵中的浮点保留从 34 个降至 3 个。
W4 覆盖全部 Temporal 投影参数的 77.10%，浮点保留占参数约 2.17%。
被替换的 15 片有 31 个 W8 和 89 个 W4，来自实际编码审计；原 0–1 的 5 个 W8 计入全模型的 36 个 W8。
这里的 FP32/FP16 是 DLC 张量类型，HTP 内部浮点数学精度的限制见前文。

`LOW_MEMORY_RESULT.json` 和 `phone_bundle_low_memory/` 记录最终 15 片组合的手机结果。
4 帧和 38 帧均通过基本检查，没有 DLC 回退；默认包清单 SHA256 保持一致。

| 完整 38 帧指标 | 该组基线 | 15 片低内存组合 |
| --- | ---: | ---: |
| 首次完整处理 s | 46.825867 | 39.004808 |
| 暖运行完整处理 s | 31.259536 | 32.736988 |
| 首次运行首个 PCM s | 16.701175 | 9.165043 |
| 暖运行首个 PCM s | 1.686177 | 1.698032 |
| 暖运行首个超过 -40 dBFS 的块 s | 4.286420 | 4.176740 |
| 暖运行实际常驻 Temporal 数量 | 16 | 16 |
| 暖运行 Temporal context 装载次数 | 0 | 0 |
| 暖运行输出实时率 RTF | 10.5607 | 11.0598 |

该组合首次完整处理减少约 16.70%，首次运行的首个 PCM 提前约 45.12%；暖运行则慢了 4.73%。
本轮默认建议按目标选择：速度使用两片方案，降低权重/context 内存使用 15 片方案。
不把 15 片方案称为速度提升，也不将低内存 4 帧阶段的 42.42% 时间下降解释为算子提升。
该阶段基线实际常驻 15 片、候选 16 片，含装载次数的差异。

低内存组 4 帧的进程拥有 DMA-BUF PSS 为：基线 11,089,728 KiB（15 片）、候选 8,066,012 KiB（16 片）。
候选多保留一片，实际 DMA-BUF 仍少约 2.884 GiB。
另与同机、同 runtime、同 4 帧阶段的已保留 16 片基线快照 11,991,476 KiB 比较，
16 片候选少约 3.744 GiB、32.74%；这项基线快照来自 `phone_bundle_two/f4/baseline/`，并非低内存组同一次基线。
两种比较都保留原始数量、阶段和来源，不以进程 RSS 推导上述节约。

38 帧仍输出 37 帧、71040 个样本、2.96 秒音频，无削波，各组自身首次/暖运行 WAV 字节相同。
37/37 个文本 token、278/296 个音频 code 相同，PCM 相关系数 0.998573，PCM RMS 差异 5.348%。
音频 code 与 PCM 不同不自动构成质量失败；当前也没有独立听感或长音频质量验收。

温度记录也需一起读取。两片速度对照的电池温度约从基线 31–32°C 变化到候选 32–34°C；
15 片对照约从基线 37–39°C 变化到候选 39–40°C。
每种方案仅有本组首次/暖运行对照，不能据此给出跨会话统计置信度或把两组绝对耗时直接混算。
“首次”是本 native session 的第一次运行，文件页缓存没有做冷启动控制。

## 执行流程与方法

1. 从已部署的 `temporal_candidate_inventory.json` 读取每片实际采用的 Scatter 或 Where 版本。
2. 对带 Q/DQ 的图，移除成对的量化/反量化节点，并从已通过的 FP32 导出恢复 8 个投影矩阵。
   部分 Q/DQ 图中的浮点权重已被调整，不能直接再次量化。
   每个恢复矩阵均核对原始权重存储 SHA256；不对反量化后的 W8 权重再做 W4。
3. 复用原来两段音频的 16 个捕获样本，读取 8 个投影的输入/输出。
4. 做 AWQ 风格的输入通道缩放和按投影输出误差的范围搜索。
   12 个样本用于权重拟合，4 个样本用于该阶段的保留检查。
5. 对权重采用 block size 32 的 LPBQ：compressed bitwidth 4，展开为 8 位网格，块整数比例范围 1–16。
   使用强正则的协方差补偿舍入，并逐输出通道比较最近舍入和补偿舍入。
6. 从 W8 控制开始，按矩阵大小尝试 W4。每次接受必须通过 16 个捕获样本和两段各 3 帧的本片 KV 递推。
   选择门限 9.5%，实际 DLC CPU 数值门限 10%。不通过的矩阵保留 W8。
   扩大批次采用首个通过完整门限的舍入方案，不重复搜寻已经足够的备选。
7. 将 singleton-token 静态 MatMul 等价改写为 1×1 Conv，避免真实大尺寸 LPBQ FC 的 HTP GenPad/finalize 失败。
8. 交给带 RMSNorm 融合保护的 QAIRT 转换器，读取实际 DLC IR，核对位宽、A16 边界、归一化和状态更新。
9. 用实际 DLC 的权重编码重建 CPU 模拟，完成捕获样本和短 KV 递推检查，再离线编译 SM8850/v81 context。
10. 手机先做 3 帧单片检查，再做组合后的 4 帧和完整 38 帧真实音频链路。

这是借鉴 AWQ/GPTQ 思路的离线 PTQ 实现，不是完整官方 AIMET AWQ、GPTQ 或 QAT 流程。
校准数据参与了方案选择，全部数值检查也不是独立质量集。
CPU 检查使用实际 DLC 参数与编码的 ONNX 模拟，不是 QNN CPU 后端直接执行该 DLC。
2–3 层原有的 2.9576% CPU/HTP 对齐告警继续保留，未放宽 2% 诊断门限。

LPBQ 的 DLC 可以保存展开后的 8 位数据和块缩放描述；已经是 W8 的部分 DLC 文件可能略增大。
本轮内存比较使用实际 HTP 编译 context 大小和手机 DMA-BUF，不按 DLC 文件名推测 4 位存储或速度。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `scripts/moshi_scaleout_lpbq.py` | 分片调度、来源校验、校准复用、CPU/HTP 编译门限和中间产物清理 |
| `scripts/moshi_restore_float_source.py` | 从原始 FP32 导出恢复矩阵，保留当前图的缓存和公开 ABI |
| `scripts/moshi_optimize_activation_aware.py` | AWQ 风格缩放、输出误差范围搜索、准备 A16 探针 |
| `scripts/moshi_build_lpbq_candidate.py` | LPBQ 编码、补偿舍入、逐矩阵 W4/W8 选择及实际 DLC 审计 |
| `scripts/moshi_projection_layout.py` | MatMul 到等价 pointwise Conv 的布局变换 |
| `scripts/moshi_stabilize_rmsnorm.py` | 保持分母单位的归一化缩放 |
| `scripts/moshi_retain_sensitive_projections.py` | 独立精度/稳定性修复候选；复用已有权重选择 |
| `scripts/moshi_repair_rms_batch.py` | 对已知异常分片并行应用同一修复 |
| `scripts/moshi_collect_lpbq_scaleout.py` | 只收集完成候选，串行执行手机单片检查 |
| `scripts/moshi_run_lpbq_phone_shards.py` | 实际 context 哈希、3 帧 own-KV 数值门限、禁止 DLC 回退 |
| `scripts/moshi_run_lpbq_phone_bundle.py` | 独立缓存中的组合音频、常驻数量、装载次数和内存检查 |
| `scripts/moshi_summarize_lpbq_scaleout.py` | 汇总逐片与整链证据，保留失败和对齐告警 |
| `scripts/moshi_analyze_lpbq_phone_chain.py` | 检查已输出的 WAV、逐帧时序和常驻记录；不依赖本地未提交的旧分析入口 |

远端批量入口使用 `moshi_scaleout_lpbq.py`，设置 `--output-root` 为独立目录。
`--inventory` 指向 `/data3/yejialei/moshi-cache1000-selected-temporal-v1/temporal_candidate_inventory.json`，
`--export-root` 指向 `/data2/yejialei/tmp/moshi-lm-onnx-v2`，
`--calibration-manifest` 指向 `/data2/yejialei/tmp/moshi-lm-calibration-mp-v1/manifest.json`。
`--layers` 使用每片的起始层号；`--workers` 和 `--first-gpu` 控制实际并行任务。

## 手机 context 实验的边界

扩大阶段直接传输已编译的 candidate context，以减少重复传输大 DLC 的时间。
独立实验目录的 manifest 明确记录候选 DLC 哈希、编译 context 哈希和每片单片门限。
现有 DLC 路径仅用于本 native runner 的文件存在检查、图标签和缓存键；实际执行图来自候选 context。
候选缓存单独映射到该 context，不修改默认缓存。
每次运行检查 context 哈希、`restore_context`、公开输出和数值结果。
出现 `open_dlc`、`compose`、`finalize` 或任何 DLC 回退就判定实验无效。
这不是把默认 DLC 或默认 context 当作候选执行，也不是完成了新 App 模型包的安装验证。
完整候选 DLC、量化输入、编码和编译收据仍保存在远端。

CPU 模拟和 AWQ 中间权重在完成对应检查后可重建清理。
所有实际清理动作都记录文件名、大小、SHA256 和原因；不删除原始导出、校准输入或当前默认模型。
