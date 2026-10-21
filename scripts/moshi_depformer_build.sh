#!/bin/bash
# Build the 8 W8 DepFormer contexts from moshi_depformer_quantize.py output, in parallel.
# Usage: moshi_depformer_build.sh WORK_DIR   -> WORK_DIR/bundle/depformer_codebook_{0..7}.bin
set -euo pipefail
W=$1
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
CACHE=""
for layer in 0 1 2 3 4 5; do for kind in key value; do CACHE="$CACHE layer_${layer}_$kind"; done; done
OUTPUTS=""
for name in $CACHE; do OUTPUTS="$OUTPUTS output_$name"; done
export FLOAT_IO="temporal $CACHE $OUTPUTS"
export INT32_INPUTS=previous_token
mkdir -p "$W/bundle"
for k in 0 1 2 3 4 5 6 7; do
  bash "$SCRIPTS/moshi_quantized_build.sh" "$W/depformer_codebook_$k" "depformer_codebook_$k" > "$W/build_$k.log" 2>&1 &
done
wait
for k in 0 1 2 3 4 5 6 7; do
  grep -E "spill_bytes|graphName" "$W/build_$k.log" | tr '\n' ' '; echo
  cp "$W/depformer_codebook_$k/htp_depformer_codebook_${k}_o3-vtcm8/depformer_codebook_$k.bin" "$W/bundle/"
done
sha256sum "$W"/bundle/*.bin
