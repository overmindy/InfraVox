# Temporal KV cache optimization checkpoint

The deployed app still uses the old 3000-slot full-cache ABI. Do not point it
at these experimental models until its tensor-shape and delta-output handling
have been updated and checked on device.

## Call chain and changes

`TemporalBlock.forward_split` in
`src/qai_hub_models/models/templates/moshi/explicit_cache.py` creates the new
K/V vectors, computes `position % capacity`, and uses `Where` over full caches.
Attention consumes the updated cache. Absolute position is also used for RoPE.
Android `PagedMoshiLm::step` restores prefixes before graph execution and packs
complete updated outputs afterward; capacities are currently hard-coded 3000.

`scripts/moshi_patch_temporal_cache.py` patches an existing guarded QDQ ONNX,
retaining all weight files/encodings. Its default experiment changes capacity
to 1000 and replaces four cache-update Where nodes with ScatterElements(axis=2).
Position-to-slot modulo and valid-history masks use the new capacity; absolute
position itself is unchanged. ONNX Scatter is functional, not verified in-place
QNN execution. Old exported intermediates remain in use; no full LM re-export.

`--cache-output delta` leaves the complete cache inside attention but returns
only the updated slot after its existing output QDQ. Public cache outputs then
have shape [1,32,1,128], instead of [1,32,1000,128]. This is an ABI change.
The full-cache input still exists; avoiding its RPC/DDR transfer requires separate
QNN registered/shared-buffer work and cannot be claimed from this graph rewrite.

`MoshiKvRing` in `android/moshi_kv_ring.h` provides the host state for that ABI:
allocate once, append per-head K/V delta into `absolutePosition % capacity`, and
reset without reallocating. Its tests cover repeated wrap, preserved slots,
stable allocation, incorrect position rejection, and reset. It is not wired
into the production runner yet. Do not alias QNN RAW input/output buffers without
backend guarantees and a device correctness test.

## Current verified and pending results

- Existing real 6_7 guarded model, clean-000 frame 0 and 1: 1000-slot scatter
  CPU ORT hidden and retained cache outputs exactly equal the 3000-slot source.
- Host ring C++ unit test passes. These are state-storage tests, not model tests.
- Delta-output CPU ORT comparison also passed: hidden and newly written K/V
  slots match the original guarded QDQ model exactly at clean-000 frames 0/1.
  These are independent captured-input probes, not a recurrent chain test.
- Full-cache scatter DLC conversion was started on the remote host. A started
  job is not a successful conversion or HTP test.
- 1000 slots correspond to about 80 seconds at 12.5 frames/s. After wrap, this
  model intentionally has a shorter history than the original 3000-slot model.
  Early-frame parity does not prove long-context quality or wrap correctness.

Remote experiments (new artifacts on data3):

    /data3/yejialei/moshi-cache-1000-scatter-6-7-v1
    /data3/yejialei/moshi-cache-1000-scatter-delta-6-7-v1

Source weights are hard-linked to the existing model.data on the same filesystem;
no re-download or weight requantization. Patch output directories must be new.

## Next checks, in order

1. Read patch_report.json, conversion.json and bounded log tails on remote.
2. Verify the converted DLC on HTP with identical captured inputs.
3. Compare 3000/Where, 1000/Where and 1000/Scatter with the same performance vote
   and profiling disabled, to separate capacity benefit from update benefit.
4. Validate delta outputs; feed them into preallocated host rings on subsequent
   frames. Compare states and logits, not just finite outputs.
5. Test positions 999, 1000, 1001 against a 1000-window reference, keeping RoPE
   position absolute. Use a small operator/state test first; then real-model test.
6. Make runner capacity manifest-driven, integrate delta ABI and registered
   buffers, then test the complete LM/Mimi chain before replacing app models.

## Official references and HTP checkpoint (2026-10-12)

The pinned Moshi `external_repos/moshi/moshi/moshi/modules/transformer.py`
defines `RingKVCache.complete`: preallocate cache, modulo slot indices, and
`scatter_` / `index_copy_` the new K/V. Its absolute-position bookkeeping is
independent of physical slot order. `scripts/moshi_test_ring_cache_reference.py`
tests the actual upstream classes against our slot/mask formulas. Positions
0, 1, 998, 999, 1000, 1001 and reset pass at capacity 1000. This is a state unit
test, not a 1000-frame full-model accuracy test.

The Qualcomm implementation `templates/llm/sha_dynamic_kvcache.py` already
returns new K/V values only, leaving accumulation to the caller. This is a useful
local reference for the delta-output ABI, not an interchangeable Moshi cache.

The initial UNALIGNED Scatter DLC successfully composed/executed on the phone
and retained four ScatterElements ops in DLC info. Adjacent no-profile warm
medians were 131.336 ms (3000/Where) and 42.9319 ms (1000/Scatter). However its
HTP accuracy FAILED: hidden relative RMS 0.02089; written cache relative RMS
0.454, 6.152, 0.829, 14.509 vs guarded QDQ CPU. These timings are diagnostic,
not validated deployment speedups. It must not replace app models.

The cache input, new K/V update and output use DIFFERENT quantization encodings.
For layer_6_value an example newly written value is about 0.0021. Quantizing
with update scale 0.00035295 / zero point 31285 yields integer 31291. Copying
that integer and decoding with output scale 0.000055024710 / zero point 32768
yields -0.08127149667, matching the observed HTP -0.0812714994. This strongly
indicates unaligned encoded-payload copying, not harmless FP16 roundoff.

The patch now defaults to `--scatter-encoding aligned`: explicit Q/DQ converts
both data and updates into the EXISTING output encoding before Scatter. No
weight encoding is changed. `--scatter-encoding source` retains the failing
diagnostic construction for reproducibility. The aligned candidate passed
captured-input CPU checks at frame 0 exactly. At frame 1 hidden max error is
1.7844e-5 and four cache elements differ by one existing output quantization
step (max 6.0588e-5). Validation explicitly allows 1.1 output LSB for aligned
cache outputs, rather than a large relative tolerance. Converted-encoding and
phone checks are still needed before deployment claims.

The 1000/Where control also composed/executed on device: warm median 38.7228 ms
(six same-input repeats, exclude first). Frame-0 hidden and all four written
cache slots are byte-identical to the deployed 3000/Where DLC. Full-cache
preservation across recurrent device frames is not checked by these dumps.
This separates the clear capacity benefit from the unproven Scatter benefit:
the unaligned Scatter is not faster than this control and is numerically wrong.

SDK 2.45 references on remote:

    docs/QAIRT-Docs/QNN/OpDef/HtpOpDefSupplement.html#scatterelements
    docs/QAIRT-Docs/QNN/general/htp/htp_shared_buffer_tutorial.html

The shared-buffer tutorial uses `memRegister`, MEMHANDLE tensors and explicit
buffer lifetime management; it does not make a RAW-buffer graph in-place by
changing its ONNX operator alone.

Aligned full-cache and aligned delta DLC conversion receipts are now successful.
DLC info confirms layer_6_value Scatter data/update/output have the same scale
0.000055024710 and offset -32768. The aligned full-cache model was downloaded,
hash-verified (SHA256 7ffd1c521b67940a9daeaf1e1a544e4625004998228b3f5fff4bdb4174b6a6e0)
and pushed to the phone. Its redirected benchmark command returned exit 0,
but adb disconnected before output/timing retrieval; device accuracy is NOT
confirmed. Retrieve `/data/local/tmp/moshi_cache1000_scatter_aligned_6_7.log`
and `moshi_cache1000_scatter_aligned_timing.csv*` after reconnection.

The aligned delta candidate also passed independent captured-input CPU probes:
frame 0 all outputs exact; frame 1 all four newly written slots exact, hidden
max error 1.7844e-5. This still does not establish recurrent device state parity.
Remote directory: `/data3/yejialei/moshi-cache-1000-scatter-aligned-delta-6-7-v1`.

## Reconnected phone: results retrieved and recurrent probe passed

Aligned full-cache frame-0 HTP hidden/value slots are byte-identical to the
deployed original; key relative RMS is 0.000830 / 0.000909 (max one output LSB).
The preceding large unaligned errors are fixed at this checkpoint.

`MOSHI_BENCH_RECURRENT=1` changes `--benchmark-temporal` into a captured-hidden
replay probe. Its raw input contains N consecutive 4096-float hidden frames.
The Graph/input caches stay allocated. Each execute advances absolute position;
only the newly written per-head slot is copied into persistent input buffers.
Full-cache or delta-cache outputs are supported. No RAW input/output aliasing
is used; the full input ABI and its transfer still exist.

`scripts/moshi_pack_temporal_hidden.py` packs the real frame-0/1 calibration
hidden values remotely (32 KiB). On-device original, 1000/Where, aligned full
and aligned delta variants all completed this two-frame probe. For BOTH frames:
Where hidden/written slots are byte-identical to original; aligned full and
aligned delta are byte-identical to each other. Aligned-vs-original frame-1
hidden relative RMS is 0.0008097; written-cache relative RMS is <=0.0009364.
This is an isolated graph-state test, NOT complete LM/token/audio parity.

An adjacent six-repeat no-detailed-profile control (same frame-0 input and
maximum requests, exclude first) gave warm medians:

| 1000-slot variant | Execute median |
| --- | ---: |
| Where, complete cache output | 37.7561 ms |
| Aligned Scatter, complete cache output | 40.8009 ms |
| Aligned Scatter, delta cache output | 32.7374 ms |

Delta is about 19.8% less execute time than aligned full output, and 13.3% less
than Where in this short control. Do not extrapolate to sustained App speed.
Next deployment checks: manifest-driven capacities and delta feedback in
PagedMoshiLm, remaining Temporal graph ABI variants (including numbered 0_1
outputs), whole-chain parity, then long-context/wrap and registered-buffer work.

## 2026-10-13: batch ABI repair and strict numerical fallback

`PagedMoshiLm` now reads cache capacity and output shape from graph metadata.
A preallocated `MoshiKvRing` stores each delta slot; the complete input cache
still crosses the RAW tensor ABI. This is not zero-copy or proven in-place HTP
memory updating. A two-frame full Mimi/LM/Mimi run with only Temporal 6_7
replaced completed; first/repeat WAV hashes match the deployed baseline.
Warm whole-chain times were 21.349 s mixed versus 21.153 s control: no overall
speedup demonstrated by this single replacement.

The first 16-graph remote batch converted six graphs, without device validation.
The 0_1 source hoists 3000-slot range/reshape constants into external integer
initializers. `moshi_patch_temporal_cache.py::patch` now changes these metadata
tensors while preserving weight and encoding initializers. Its new two-frame
CPU check passes; hidden outputs are exact at both frames. Cache layer-0 key
maximum differences are about 0.0002203, within the existing output-LSB bound.

`semantic_bytes` ignores only the presence of an empty default operator domain,
not arbitrary model differences. This fixes false rejection of the verified
6_7 prototype. `--reuse-build-root` reuses matching data3 source snapshots and
structurally identical converted candidates. Reports now retain failed numerical
checks instead of terminating before writing their metrics.

| Captured input graph | Frame-1 aligned Scatter hidden relative RMS | Capacity1000 Where outputs vs baseline, frames 0/1 |
| --- | ---: | --- |
| 4_5 | 0.0040933904 | All outputs elementwise exact |
| 8_9 | 0.0002559016 | All outputs elementwise exact |

The Scatter errors therefore are not caused solely by shrinking capacity in
these two tests. Additional Q/DQ boundaries remain the suspect, not a proven
universal backend fault. Strict checks were not loosened. Optional
`moshi_build_cache1000_set.py --fallback-where` tries a separate Where candidate
ONLY after Scatter exits with a numerical checkpoint failure (exit 2); shape,
conversion, or runtime failures do not silently fall back. Both candidates and
reports remain available. A converted fallback is still not a device pass.

Remote roots:
- `/data3/yejialei/moshi-cache1000-scatter-delta-lm-v2`
- `/data3/yejialei/moshi-cache1000-where-fallback-lm-v1`
- `/data3/yejialei/moshi-performance-inspect-v1/cache1000_where_diag_4_5`
- `/data3/yejialei/moshi-performance-inspect-v1/cache1000_where_diag_8_9`

Next gates: collect all strict reports; device same-input/delta recurrence;
short mixed/full-chain token and audio checks; same-device timing; long-context
and wrap checks before default App model replacement.

## 2026-10-13: complete candidate collection and real 4_5 device checkpoint

Both builds finished. Eight Scatter and eight strictly checked Where fallback
graphs cover all 16 Temporal shards. `moshi_collect_cache1000_candidates.py`
checks successful state/receipt and strict two-frame reports, hashes each DLC,
and creates data3 symlinks without duplicating weights. It deliberately emits
`temporal_candidate_inventory.json` with `device_verified=false` and
`all_graphs_verified=false`, NOT a misleading App-ready inventory.
Collection: `/data3/yejialei/moshi-cache1000-selected-temporal-v1`.
Local resumable download: `/Users/e1ainay/d_personal/moshi-models/lm-cache1000-candidate-v1`.

The 4_5 Where/delta DLC remote/local SHA256 agrees:
`50906c6b422b0bc7fef7d7f7ff01d219c7433adb52e559847f584093cd2737f9`.
Real captured hidden frames 0/1 were replayed on the connected SM8850 phone.
Each run feeds its OWN new slot into its next input cache; baseline and candidate
start from zero state. `moshi_compare_recurrent_dumps.py` compares hidden and
the four newly written K/V slices, not megabytes of unwritten zero slots.

- Frame 0: all five outputs byte-identical to baseline DLC.
- Frame 1: hidden relative RMS 0.0067021823, max abs 0.0011291504;
  layer4 K/V exact; layer5 key/value relative RMS 0.0002838911/0.0007203285.
- All finite; passes the explicit 0.02 experimental device checkpoint, but
  NOT byte-identical and NOT a long-context/whole-LM quality pass.
- Independent same-frame resident six-repeat control, maximum HTP requests,
  no detailed profiler, exclude first: baseline median 182.486 ms versus
  candidate 39.0936 ms (4.668x execute-only speedup).

A separate full Mimi/LM/Mimi two-frame run replacing ONLY 4_5 completed first
and repeat. Both generated WAV hashes equal the baseline:
`ea84d9bc97ac46c32d9726a2175ed288c2f333951eee2eb98cb27c51b9b72da4`.
Wall totals: 75.599 s first / 33.3944 s repeat. This is NOT whole-App speedup
evidence; other 25 graphs are unchanged and context/lifecycle costs remain.
Artifacts: local `device_4_5/report.json`, `chain_f2.log`, `chain_f2.wav`, and
the original/candidate `*_repeat6.csv` files under the candidate download root.

Next: finish/verify download hashes; device checks for remaining shards; full
candidate chain/token comparison; same-device whole-chain timing; only then
publish a validated App inventory and replace its default model selection.

## 2026-10-13: whole-chain QNN profile and bounded residency experiment

Same phone/input and the 4_5-only candidate bundle; two input frames, warm repeat.
The 3072/4096 runs enable QNN BASIC profiling. The 3584 run uses a clean indexed
build with only the independent residency-budget change and no QNN profiler.
These are single adjacent trials, not statistically established improvements.

| Source-weight budget MiB | Warm wall s | Restore s | Restore calls | Sampled minimum MemAvailable MiB |
| ---: | ---: | ---: | ---: | ---: |
| 3072 | 27.9973 | 20.3315 | 18 | 3029.48 |
| 3584 | 28.8274 | 21.1649 | 16 | 2505.92 |
| 4096 | 26.3460 | 18.8302 | 14 | 1581.25 |

All three repeat WAVs match the baseline SHA256 above. The 4096 trial saves
about 5.9% wall time versus 3072 but has less memory headroom; 3584 does not
improve wall time. Keep the 3072 default. Actual allocations are NOT bounded
by source file sizes, and planned residency is not guaranteed by nominal RAM.

`MOSHI_TEMPORAL_BUDGET_MIB` explicitly requests an experiment budget, capped
at 6144 MiB and initial available memory minus 6 GiB; existing per-admission
checks still apply. It is independent of unverified maximum/shared-context
experiments, which are NOT included in the committed change or enabled here.

QNN BASIC profiles show 55 graphExecute calls per warm run. For 3072/4096,
summed accelerator event time is 3.6094/3.5481 s; native execute phase totals
are 5.9036/5.7052 s. Event 8001 reports eight HVX threads per call. This confirms
the reported QNN execution path, not HMX occupancy, NPU-wide utilization, or
a device clock frequency. RPC/accelerator/parent events overlap and must not
be added as independent elapsed-time components.

The clean profiling build also ran a three-repeat DETAILED probe on 4_5.
Excluding first, top reported node-cycle medians are gating_1/MatMul 18.605M,
gating/MatMul 18.599M, input projections 10.186M/10.130M; cache Where and
FP32-to-uint16 conversions also appear among the top nodes. Cycle values rank
nodes within this capture; they are not converted to milliseconds or summed
as a fabricated utilization percentage.

Reproducible source entrypoints:
- `Runtime`: opt-in profileCreate/collectProfile/profileFree and QNN CSV export.
- `PagedMoshiLm`: independent bounded source-weight residency experiment.
- `scripts/moshi_summarize_runtime_profile.py`: phases, restore counts, QNN units,
  and sampled memory; reports overlap explicitly.
- `scripts/moshi_profile_cache1000_phone.sh`: wait for verified complete downloads,
  preserve existing phone models, push/check SHA256, native full candidate run,
  warm profile capture. Does not publish an App-ready inventory or change defaults.

Local evidence: `lm-cache1000-candidate-v1/profile_budget/budget*.json` and
`cache1000_4_5_detailed.csv.qnn.csv`. Full candidate driver status is recorded
in `profile_full_driver.log`, with eventual artifacts in `profile_full_v1/`.

## Full replacement and lifecycle optimization: verified device results

500-cache construction was deferred at the user's request to focus on the
dominant overhead. All 16 1000-cache Temporal candidates are now downloaded,
SHA256 checked, and installed in a separate native-test bundle on this phone:
`/data/local/tmp/moshi-cache1000-full-profile-v1`. App defaults/APK are unchanged.

The first mixed cold-build/resident run failed at 14_15 with graphExecute 6001
(QNN_GRAPH_ERROR_INVALID_HANDLE). The exact saved 14_15 context subsequently
passed an isolated two-frame replay. A nonresident run prepared the remaining
contexts and passed the complete chain. Subsequent cached resident runs pass.
This isolates the cold/multiple-context lifecycle boundary but does NOT establish
a universal SDK root cause. Preserve generated caches; don't reconvert DLCs.

| Complete new Temporal set, warm two-frame run | Wall s | Restore s | Restore calls |
| --- | ---: | ---: | ---: |
| No Temporal residency, init acceleration | 24.9774 | 20.1454 | 32 |
| 3 GiB DLC proxy budget | 19.1252 | 15.1597 | 18 |
| 6 GiB DLC proxy budget | 11.0792 | 6.8788 | 6 |
| Codec cached, 6 GiB DLC proxy | 8.3896 | 5.4108 | 6 |
| Codec cached, shared workspace, 6 GiB DLC proxy | 4.8200 | 2.7433 | 6 |
| Codec cached, shared workspace, 8 GiB binary proxy | 1.7180 | 0 | 0 |

These are observed short trials, not statistically isolated effects of each
flag. The final warm run really retains all 16 Temporal contexts, not merely
a planned admission list. Total execute time is 1.4307 s for two input frames;
sampled minimum MemAvailable is 2.730 GiB. First/repeat WAV, audio codes, and
text tokens match the existing baseline (text `0,3`, audio codes recorded above).

A longer independent adjacent four-frame control also passes byte comparisons:
- Original 3000-cache bundle, 3 GiB budget, no context group: warm 36.5433 s,
  restore 25.2419 s (36 calls), execute 9.2417 s.
- Full 1000-cache set + shared workspace + binary proxy: warm 3.42445 s,
  restore 0 (0 calls), execute 2.8637 s; about 10.67x wall speedup in this trial.
- Text and audio-code files compare byte-identical. Optimized first/repeat WAV
  SHA256 both match baseline repeat:
  `2701e5b3b339e135c9a4a55cb776b76f82732329c24099d4a65c486a739b64e1`.
- Sampled minimum MemAvailable 3.000 GiB, peak process RSS 6.655 GiB.

Why the selected changes work:
1. `MOSHI_CONTEXT_GROUP=1`: inspect all 26 cached HTP graph blobs and register
   contexts to one sequential group using the maximum declared spill/fill size,
   50,790,400 bytes. Frontend group leader remains alive until all members free.
   This uses register-multi-contexts, NOT unverified asynchronous batch loading.
2. `MOSHI_RESIDENT_COST_USE_BINARY=1`: replan using actual device-context file
   sizes, still only a RAM proxy. 2_3/30_31 DLCs are ~1.64 GB each, but their
   contexts ~824 MB each. Using raw DLC size had overestimated their cost.
   The experiment supports up to 8 GiB with initial headroom and per-admission
   checks. Resource allocation failures retire completed Temporal contexts,
   preserving host KV and refusing expensive verified-DLC rebuilds.
3. `MOSHI_CODEC_CONTEXT_CACHE_DIR`: cache encoder/decoder contexts too. Their
   constructor cost occurs BEFORE pipeline wall_total. Priming measured
   native_session_prepare_ms=8813.32; a new process with cached codecs measured
   556.925 ms. Explicit preparation timing prevents hiding startup latency.

Rejected/default-off experiments: persistent binary did not clearly improve
wall time; direct RAW ring binding removed explicit copy but did not demonstrate
whole-chain speedup in its trial (different admission count). Keep the pooled
copy path for this verified fast configuration. Ring binding is an opt-in
experiment, not NPU zero-copy; its cache storage outlives bound contexts.

Reproduce with the new clean native binary and
`MOSHI_FAST_PROFILE=1 bash scripts/moshi_profile_cache1000_phone.sh CANDIDATE_ROOT BINARY LIB_DIR`.
On a fresh device/cache set, first use `MOSHI_PREPARE_ONLY=1` with the same script;
this uses nonresident preparation and preserves all models. Fast mode refuses
missing binary-cost metadata rather than guessing readiness.

Evidence directories under local `lm-cache1000-candidate-v1`:
`full_budget0`, `full_budget3`, `full_budget6`, `codec_normal`, `codec_persistent`,
`group6`, `group_binary8`, `f4_baseline`, `f4_optimized`.
This is a real native phone-chain proof, NOT an updated installed-App/UI proof,
long-context wrap proof, sustained thermal benchmark, or real-time result.
Four inputs emit three 80 ms chunks; 3.424 s for 240 ms output is still not real time.
