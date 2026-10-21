#!/bin/bash
# Recompile the legacy frontend, head, DepFormer and Mimi DLCs with O3 + VTCM 8 MB.
# Usage: moshi_legacy_o3.sh OUT_DIR [COMPILE]
set -euo pipefail
OUT=$1; COMPILE=${2:-o3-vtcm8}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
LM=${MOSHI_LEGACY_DLC_DIR:-/data3/yejialei/moshi-lm-guarded-dlc-set-v2/dlc}
MIMI=${MOSHI_MIMI_DIR:-/data2/yejialei/moshi-mimi-lpbq-20261018-v2/selected}

compile() { "$SCRIPTS/moshi_compile_context.sh" "$1" "$2" "$OUT/work/$2" "$COMPILE" > "$OUT/work/$2.log" 2>&1; }
mkdir -p "$OUT/work" "$OUT/bundle/dlc"
for name in frontend head depformer_codebook_{0..7}; do
  compile "$LM/$name.dlc" "$name" &
done
compile "$MIMI/encoder/model.dlc" mimi_encoder &
compile "$MIMI/decoder/model.dlc" mimi_decoder &
wait
for name in frontend head depformer_codebook_{0..7}; do cp "$OUT/work/$name/$name.bin" "$OUT/bundle/dlc/"; done
cp "$OUT/work/mimi_encoder/mimi_encoder.bin" "$OUT/work/mimi_decoder/mimi_decoder.bin" "$OUT/bundle/"
grep -h -E "spill|graphName" "$OUT"/work/*.log
