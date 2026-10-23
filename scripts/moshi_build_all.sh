#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# One-click server build: Kyutai BF16 checkpoint -> phone bundle $MOSHI_ROOT/bundle/$MOSHI_BUNDLE.
# Paths come from moshi_env.sh. Finished graphs are skipped, so a failed run resumes where it stopped;
# KEEP_WORK=1 keeps the work dir (ONNX, DLCs, encodings) afterwards. Peak disk ~70 GB on /data2.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
source "$S/moshi_env.sh"
W=$MOSHI_WORK
B=$MOSHI_ROOT/bundle/$MOSHI_BUNDLE
read -ra GPUS <<< "$MOSHI_GPUS"
mkdir -p "$W/tmp" "$W/logs" "$B/dlc"
export TMPDIR=$W/tmp  # AIMET writes multi-GB model copies to TMPDIR.

# shellcheck disable=SC1090
aimet() { (source "$AIMET_ENV"; "$AIMET_PY" "$@"); }
build() { "$S/moshi_qairt_build.sh" "$W/$1" "$1" > "$W/logs/$1.build.log" 2>&1; }
await() { local pid; for pid in "$@"; do wait "$pid"; done; }

echo "== capture: BF16 Moshi on LibriSpeech, Temporal boundary activations"
[ -f "$W/capture/manifest.json" ] || NO_CUDA_GRAPH=1 CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$S/moshi_capture.py" \
  --model-dir "$MOSHI_BF16" --librispeech-dir "$LIBRISPEECH_TRAIN_CLEAN" --output-dir "$W/capture"

echo "== frontend embedding table, Mimi"
[ -f "$B/frontend_embeddings.bin" ] || "$PY" "$S/moshi_frontend_table.py" --model-dir "$MOSHI_BF16" \
  --output "$B/frontend_embeddings.bin"
[ -f "$W/mimi_decoder/mimi_decoder.encodings" ] || CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$S/moshi_mimi.py" \
  --model-dir "$MOSHI_BF16" --audio-wav "$MOSHI_MIMI_WAV" --work-dir "$W"
pids=()
for name in mimi_encoder mimi_decoder; do
  [ -f "$W/$name/$name.bin" ] || { QUANTIZE=0 INT32_INPUTS=position \
    FLOAT_IO="audio kv_cache conv_state kv_cache_out conv_state_out" build "$name" & pids+=($!); }
done
await ${pids[@]+"${pids[@]}"}

echo "== 4 Temporal parts in parallel (one GPU each)"
pids=()
for p in 0 1 2 3; do
  name=temporal_part_$p
  [ -f "$W/$name/$name.bin" ] || {
    { [ -f "$W/$name/$name.encodings" ] || CUDA_VISIBLE_DEVICES=${GPUS[$p % ${#GPUS[@]}]} aimet "$S/moshi_temporal.py" \
        --model-dir "$MOSHI_BF16" --capture-dir "$W/capture" --work-dir "$W" --part "$p"; } > "$W/logs/$name.log" 2>&1 \
      && build "$name"
  } & pids+=($!)
done
await "${pids[@]}"

echo "== head + 8 DepFormer codebooks"
[ -f "$W/depformer_codebook_7/depformer_codebook_7.encodings" ] || CUDA_VISIBLE_DEVICES=${GPUS[0]} aimet \
  "$S/moshi_head_depformer.py" --model-dir "$MOSHI_BF16" --capture-dir "$W/capture" --work-dir "$W"
pids=()
[ -f "$W/head/head.bin" ] || { build head & pids+=($!); }
for k in 0 1 2 3 4 5 6 7; do
  [ -f "$W/depformer_codebook_$k/depformer_codebook_$k.bin" ] || { INT32_INPUTS=previous_token build "depformer_codebook_$k" & pids+=($!); }
done
await ${pids[@]+"${pids[@]}"}

echo "== bundle $B"
ln -f "$W/mimi_encoder/mimi_encoder.bin" "$W/mimi_decoder/mimi_decoder.bin" "$B/"
for name in head depformer_codebook_{0..7} temporal_part_{0..3}; do ln -f "$W/$name/$name.bin" "$B/dlc/"; done
"$PY" "$S/moshi_text_vocab.py" --model-dir "$MOSHI_BF16" --output "$B/text_vocab.txt"
(cd "$B" && sha256sum frontend_embeddings.bin mimi_encoder.bin mimi_decoder.bin text_vocab.txt dlc/*.bin > SHA256SUMS)
cat "$B/SHA256SUMS"
[ "${KEEP_WORK:-0}" = 1 ] || rm -rf "$W"
