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
