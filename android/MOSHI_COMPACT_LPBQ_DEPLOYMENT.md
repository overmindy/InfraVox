# 单份编译模型包与 Mimi W4 实验

日期：2026-10-18。设备：BKQ-AN20，SM8850/v81，ADB `ATWV026602002286`。

## 最终选择

本轮选择已经通过组件数值检查和手机链路检查的最小组合，不宣称全局最优：

- Temporal：15 片 LPBQ W4/W8，0–1 保留已部署的混合浮点/W8 图。
- Mimi encoder：32 个线性投影保留 W8/A16。W4 encoder 在手机短检查中失败，不部署。
- Mimi decoder：32 个线性投影使用 LPBQ W4/A16，block size 32。
- Mimi 卷积、转置卷积、码本、码本搜索和归一化保留浮点路径。
- cache1000、增量 KV、RMSNorm 稳定性修复及各片 Scatter/Where 选择保留。

手机包：`/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-cache1000-v1`。
Mac 记录与选定量化 context：`/Users/e1ainay/d_personal/moshi-models/moshi-compact-lpbq-20261018-v1`。
Mimi 远端最终 DLC、编译输入和 context：`/data2/yejialei/moshi-mimi-lpbq-20261018-v2/selected`。
LM 远端来源见此前 `SELECTED_REMOTE_ARTIFACTS.json`，交付目录保存其副本。

## 手机存储格式

新 App 版本为 `2.6.0-compact-lpbq`，沿用原签名密钥升级，不卸载 App。
设备包只保存 26 个 LM 编译 context 和 2 个 Mimi 编译 context，共 **6,700,310,528 bytes，6.240 GiB**。
不再同时保存一套 DLC 和另一套编译权重，也不以占位 DLC 冒充真实模型。

```text
lm-cache1000-v1/
  CONTEXT_ONLY
  dlc_inventory.json
  SHA256SUMS
  mimi_encoder.bin
  mimi_decoder.bin
  dlc/
    frontend.bin
    temporal_layers_0_1.bin ... temporal_layers_30_31.bin
    head.bin
    depformer_codebook_0.bin ... depformer_codebook_7.bin
```

`dlc/` 是沿用的目录名，其中的 `.bin` 是真实的、包含权重的 QNN 编译模型。
清单格式为 `moshi-context-only-app-inventory-v1`，包含 28 个文件的 SHA256、大小、来源 DLC 哈希、
当前设备指纹和已检查的 runtime API。
App 在加载前校验 28 个文件；运行器直接调用 QNN context 反序列化。
反序列化失败则停止，不尝试把 `.bin` 当作 DLC 编译，也不回退执行另一份模型。

此包绑定当前手机与匹配的 QNN runtime。换芯片、换不兼容 runtime 或更新系统后，可能需要重新编译。
远端保留原始导出、校准数据及最终 DLC，因此不依赖旧手机 DLC 副本恢复来源。

## W4 的实际门限

复用既有显式状态 FP32 导出和上一轮 A16 校准，不重做导出。
逐 32 个输入通道分块，采用 compressed bitwidth 4、展开网格 8 位的 LPBQ。
线性投影等价改写为 pointwise Conv。Mimi 每个音频帧有两个内部 Transformer 时间步，
因此改写为 batch=2；用 CPU 两帧检查改写前后数值等价，公开 ABI 不变。

encoder 全线性 W4 的 CPU KV 误差为 10.895%，超过 10% 门限。
只保留首层 W8 的备选仍未通过。MLP W4/attention W8 虽通过 CPU 和 HTP 编译，
手机 codec 4 帧的 PCM 差异为 55.10%，也被拒绝。没有放宽这些门限。

最终 W8 encoder＋W4 decoder 在手机 codec 4 帧中：

- code 匹配率 29/32，即 90.625%；
- PCM 相对 RMS 差异 2.597%，相关系数 0.999664，无削波；
- 两端都从选定 context 恢复，没有 DLC 编译回退。

Mimi context 文件从原默认 242,790,400 bytes 降至 182,239,232 bytes，减少 24.94%。
相较上一轮全 W8 Mimi 的 194,019,328 bytes，再减少 11,780,096 bytes。
这不是实际 RAM 降幅；本轮不新增 RAM 对照实验。

## 完整链路与相同 token 检查

精简包 4 帧检查通过。随后只进行一次完整 38 帧首次/暖运行检查：

- 38 个输入帧产生 37 个输出帧，71,040 个 PCM 样本，即 2.96 秒；
- 16 个 Temporal 实际常驻，暖运行 context 重装载为零；
- 首次与暖运行 WAV 字节相同，28 个模型均从 `.bin` 恢复；
- 本次首次完整处理 44.147 s，暖运行 35.804 s，仍未实时；
- 与此前同输入的低内存 LM＋浮点 Mimi 记录相比，文本 37/37 一致，音频 token 288/296 一致；
- 直接 PCM 差异为 36.273%，相关系数 0.935704，原直接 PCM 检查为失败，记录不改写。

直接比较不同生成 token 的波形，不能隔离 decoder 数值误差。
因此将本次同一组 37 帧生成 token 输入原浮点 decoder，额外做一次短 decoder 回放，不再运行 LM。
W4 decoder 与这个同 token 浮点参考的 PCM 差异为 **7.151%**、相关系数 **0.997466**，
通过原有 10% decoder 数值门限。该检查与生成 token 匹配率一起作为组件部署依据。

`chain_f38/RESULT.json` 保留直接 PCM 未通过的结果。
`COMPONENT_VERIFICATION.json` 分别记录 token 变化和同 token decoder 误差。
这不是独立听感、独立语义或长音频质量验收，不能称为与浮点输出完全一致。

## 交付与清理记录

安装后，实际 App 界面通过 28 文件校验，并完成 2 帧 Encoder → LM → Decoder 推理。
输出 1 帧、0.08 秒音频，界面显示本次推理用时 8.89 秒。
另用链接已安装 App 原生库的短探针检查首次与暖运行，结果一致。
首次界面检查发现 ADB 新建的 `dlc/` 目录不可供 App 读取。
将目录权限改为 755 后，界面校验和推理通过。暂存脚本已包含该修复。

完整精简包校验通过后，App 删除已被该包替代的两个私有 Mimi 导入副本。
删除条件要求当前选用的 encoder 和 decoder 均来自这个已校验的精简包。
该操作不删除输入音频、输出音频或 tokens。
最终安装后，App 日志确认删除两个私有 DLC，共 364,930,312 bytes，约 348 MiB。
最终 APK 的原生库与完成界面推理检查的版本逐字节相同。
最后一次 APK 更新只增加私有旧副本清理，安装后再次通过完整模型校验。

本轮 Mac 清理实际释放 12.827 GiB。手机外部目录清理实际释放 27.269 GiB。
手机删除旧默认 DLC 包、旧编译缓存、旧 Mimi 包和实验模型副本。
清理后再次校验保留的 28 个模型 SHA256，全部通过。
Mac 保留 17 个选定量化 context，共 4,046,622,720 bytes，约 3.769 GiB。
另外 11 个未修改模型的 context 在手机保留，来源 DLC 在远端保留。
Mac 只保留当前签名 APK。原签名密钥保留。

- `COMPACT_INVENTORY.json`：手机最终包清单。
- `PROMOTION.json`：安装版本、签名 APK 哈希、切换位置。
- `APP_VERIFICATION.json`：实际 App 加载与短推理检查。
- `MAC_CLEANUP.json`、`PHONE_CLEANUP.json`、`REMOTE_CLEANUP.json`：具体删除项和释放空间。
- `SELECTED_LOCAL_FILES.json`：Mac 只保留选定版本的量化 context；未修改模块的 context 只在手机保留。
- `FAIL_CLOSED_CHECK.json`：无效 context 被拒绝，未触发 DLC 编译。

清理保留原始远端 FP32 导出、校准数据、签名密钥、本地配置和小型检查记录。
旧下载校验文件只是历史记录；出现 `LOCAL_ARTIFACTS_PRUNED.json` 后，不应据旧清单重下全部旧模型。
