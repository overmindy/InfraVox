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
- Full-cache scatter DLC conversion and delta-output CPU comparison were started
  on the remote host. A started job is not a successful conversion or HTP test.
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
