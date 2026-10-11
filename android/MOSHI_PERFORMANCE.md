# 手机 LM 运行架构与性能排查

先看 `mimi_qnn_runner.cpp` 的 `PreparedMimiCodec::run`、`runMoshiFrames`、
`PagedMoshiLm::step` 和 `Graph`。Java 的 `MainActivity.runCodec` 通过
`mimi_codec_jni.cpp` 调用它们。不要把“子图执行有顺序”与“每帧重编译子图”混为一谈：
Temporal 层和 DepFormer codebook 存在真实数据依赖，仍须依次执行。

## 2.2 的已验证基线

手机 ATWV026602002286，已验证 26 个 DLC，App 两输入帧完整链路通过。
565.72 秒生成一帧 80 ms 音频；绝非实时。这个总时间包含每帧的
contextCreate、DLC compose、graphFinalize、缓冲分配、execute、释放和宿主处理。
看到应用占用一个 CPU 核，不能据此断言 NPU 没有参与：调用的是 HTP backend，
还需要 QNN profiling 才能解释设备端算子耗时。

## 2.3 改什么，为什么

1. `Graph::saveBinary/loadBinary` 使用 SDK 提供的 contextGetBinary 和
   contextCreateFromBinary。第一次保存已 finalize 的 context；下一次直接恢复，
   不再 compose/finalize。metadata 由 System context 保持有效，mmap 保留至 contextFree。
   文件先写唯一临时文件，再 fsync/rename，失败不替换正常 DLC，不把坏缓存当模型。
   缓存不改模型精度，也不能跨手机/SDK直接当通用部署模型。
2. `PagedMoshiLm::resident` 将 frontend、head、八个 DepFormer 常驻同一会话。
   再次开始一段音频仍清空流式状态，但不销毁这些图。
   DepFormer 的首 codebook 必须重新清零深度 cache 输入；否则常驻会引入上一帧状态。
3. Temporal 使用内存预算内常驻，其余按需恢复/释放。每个图完整 cache I/O 约 375 MiB；
   直接常驻这些独立分配的缓冲就约 6 GiB，还没算 12 GB 模型和 SDK 工作内存。
   不能凭“手机 24 GB RAM”认为全部常驻安全。下一轮若 restore 明显主导，
   `GraphIoPool` 已共用这组 scratch：执行完成且宿主取走结果后，
   `Graph::suspendIo` 归还缓冲，但保留入选图的 context；下次 `resumeIo` 重新绑定。
   `/proc/meminfo` 的 MemAvailable 至少 8 GiB 时才扩大常驻，source-weight 预算
   不超过可用内存四分之一、最多 3 GiB；按 DLC 大小优先选择小图。
   新常驻图加载前还检查至少 4 GiB 可用。它不是硬 RSS 上限，也不是 OOM 保证。
   实测该手机入选 7 个 Temporal 图，另 9 个分页；CLI 用
   `MOSHI_TEMPORAL_RESIDENT_MIB=0` 可关闭扩展常驻作对照。
4. `runMoshiFrames` 每输入帧回调完成进度和该帧 PCM。JNI 在原 worker 线程回调
   Java `onNativeFrame`，UI 更新发回主线程；可选 AudioTrack MODE_STREAM 立即播放。
   播放缓冲有界，使用 backpressure，不无限积压。当前输入仍为录音后/导入 WAV，
   不是麦克风采集与推理同时运行；一帧计算若大于 80 ms，播放仍有间隔。
5. `.performance.csv` 区分 open_dlc、compose、finalize、save_context、restore_context、
   allocate_io、restore_kv、pack_kv、execute、resident_reuse 和 wall_total。
   wall_total 与其他行重叠，不能把它们加在一起。失败也尽量留下 CSV。
   App 导出 WAV/tokens 时一起导出 CSV。显示 RTF = 用时 / 输出音频时长，小于 1 才实时。
6. `Runtime::beginPerformanceVote/endPerformanceVote` 请求 HTP performance/DCVS
   core/bus target TURBO，保留 DCVS scaling，结束/异常都销毁 powerConfigId。
   App 可关闭性能开关作对照；CLI 以 `MOSHI_HTP_PERFORMANCE=1` 开启。
   `performance_vote_accepted` 只证明 SDK 接受请求，不代表持续满频或绕过温控。

App 缓存位于 `getExternalFilesDir("qnn-context-v1")`，即该手机
`/sdcard/Android/data/com.overmindy.mimicodecdemo/files/qnn-context-v1/`；
外部存储不可用时回退私有目录。CLI 默认输出 WAV 同目录的 `OUTPUT.wav.contexts-v1/`，
可用 `MOSHI_CONTEXT_CACHE_DIR` 指定已经验证的本机缓存，避免重复构建。
该测试手机已有 26 个缓存，不必再建：安装并打开 App 2.3（让 App 自己创建目录）后，
执行 `bash scripts/moshi_android_reuse_context_cache.sh /Users/e1ainay/Android/platform-tools/adb`。
脚本复制该手机 CLI 已验证的缓存，逐个比较 SHA256，不覆盖已有文件，不删除源文件，
只给新复制的 ADB 文件增加 App 读取所需的 644 权限。不能用来给另一部手机分发缓存。
不要求给只读模型目录写权限。缓存文件名含 DLC size/mtime、QNN core/backend API
版本；模型必须先经过 App SHA256 校验。替换模型后重新加载会话。
SDK 或设备不兼容时恢复失败会回退 DLC 重建。首次构建缓存可能更慢且需额外磁盘，
不要删除已经验证的 DLC 来腾空间；缓存本身可在没有执行任务时清理。

## 可复用的测量顺序

先运行确定性 frontend 微基准：`moshi_qnn_benchmark --benchmark-frontend LIB_DIR
FRONTEND.dlc CACHE.bin`。它对照首次 DLC 执行、同图重复执行、保存后恢复及重复执行，
要求四次 hidden 逐字节一致。CLI 需要正确的 LD_LIBRARY_PATH 和 ADSP_LIBRARY_PATH。
已在手机通过：compose 0.672 s、finalize 1.106 s、缓存恢复 0.317 s；
恢复后的 execute 23.5 / 21.7 ms。只证明此图与此输入，不代表全部 LM。

完整链路 CLI 的最后可加 `--repeat`：同一个 session 连续处理同一输入两次，
分别生成 OUTPUT.wav 和 OUTPUT.wav.repeat.wav，以及各自 tokens/CSV。
重复测试必须比较全部输出，不能只看程序退出码或 finite=true。

```sh
python3 scripts/moshi_summarize_phone_performance.py FIRST.performance.csv REPEAT.performance.csv
```

按冷构建、磁盘缓存恢复、同会话常驻三种情况分别看结果。
若 finalize 主导，先缓存；若恢复/内存搬运主导，考虑共享缓冲与常驻；
若 execute 主导，再做设备 profiling、HTP performance vote、模型/算子改写。
CPU/NPU 流水线必须遵守帧与 cache 所有权，不能用多个线程同时改同一 graph 输入。
当前没有宣称 NPU 利用率、全链路 2.3 精度或实时性已通过。

## 本轮完整原生测试（同一手机、同一 WAV、每次两输入帧）

| 模式 | 第一段两帧 | 同 session 重跑两帧 |
| --- | ---: | ---: |
| 小图常驻＋Temporal context 缓存，首次无缓存 | 372.6 s（包括构建） | 43.7 s |
| 再加共享缓冲＋7 个 Temporal 常驻，已有缓存 | 38.50 s | 29.68 s |
| 再加 HTP 性能请求，已有缓存 | 24.74 s | 17.61 s |

三种实现的输出 WAV、文本 tokens、八路音频 tokens 的 SHA256 完全相同。
这是短输入的实现一致性，不是所有中间张量与 FP32 的完整精度证明。
暖运行输出一帧 80 ms，最后一行输出 RTF 约 220，仍非实时。
最后一次暖运行有 55 次 execute（52 LM +2 Encoder +1 Decoder），
18 次分页恢复、34 次 resident_reuse；执行约 4.93 s，context 恢复约 11.41 s。
因此后续优先研究剩余 context 的常驻/RSS 和更小的 cache ABI，不是盲目增加线程。
App 2.3 已构建签名；USB 安装本轮被手机拒绝，尚未验证 Java 播放回调与 App 计时。
