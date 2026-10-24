#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# One-click server build: openbmb/MiniCPM-o-4_5 BF16 -> SM8850 contexts in $MCPO_WORK/contexts.
# Paths come from minicpmo_env.sh. Every finished step is skipped, so a failed run resumes where it stopped.
# Calibration runs as two queues, one per GPU in $MCPO_GPUS; QAIRT compilation runs on the CPU in parallel.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
source "$S/minicpmo_env.sh"
W=$MCPO_WORK
read -ra GPUS <<< "$MCPO_GPUS"
mkdir -p "$W/tmp" "$W/logs" "$W/contexts"
export TMPDIR=$W/tmp  # AIMET writes multi-GB model copies to TMPDIR.
COMMON=(--model-dir "$MCPO_HF" --capture-dir "$MCPO_CAPTURE" --work-dir "$W")

# shellcheck disable=SC1090
aimet() { local gpu=$1; shift; (source "$AIMET_ENV"; CUDA_VISIBLE_DEVICES=$gpu "$AIMET_PY" "$@"); }
step() { local done=$1 log=$2; shift 2; [ -f "$done" ] || "$@" > "$W/logs/$log.log" 2>&1; }
await() { local pid; for pid in "$@"; do wait "$pid"; done; }

echo "== capture: BF16 duplex sessions on synthesized questions"
[ -f "$MCPO_CAPTURE/manifest.json" ] || CUDA_VISIBLE_DEVICES=${GPUS[0]} "$MCPO_PY" "$S/minicpmo_capture.py" \
  --model-dir "$MCPO_HF" --output-dir "$MCPO_CAPTURE" --ref-wav "$MCPO_REF_WAV" > "$W/logs/capture.log" 2>&1

echo "== official token2wav assets, LLM part boundaries"
step "$W/token2wav/golden.npz" token2wav_prepare env CUDA_VISIBLE_DEVICES=${GPUS[0]} "$MCPO_PY" "$S/minicpmo_token2wav_prepare.py" \
  "${COMMON[@]}" --ref-wav "$MCPO_REF_WAV" &
p1=$!
step "$W/boundaries/done" boundaries aimet "${GPUS[1]}" "$S/minicpmo_llm.py" boundaries "${COMMON[@]}" && touch "$W/boundaries/done" &
p2=$!
await $p1 $p2

echo "== calibration (AIMET), two GPU queues"
{
  step "$W/token2wav/hift_decode/hift_decode.encodings" token2wav aimet "${GPUS[0]}" "$S/minicpmo_token2wav.py" --work-dir "$W"
  step "$W/audio_encoder/audio_encoder.encodings" audio_encoder aimet "${GPUS[0]}" "$S/minicpmo_audio.py" "${COMMON[@]}"
  step "$W/tts_cpu/tts_cpu.bin" tts_cpu aimet "${GPUS[0]}" "$S/minicpmo_tts_cpu.py" --model-dir "$MCPO_HF" \
    --capture-dir "$MCPO_CAPTURE" --output "$W/tts_cpu/tts_cpu.bin" --host-test-dir "$W/hosttest"
  for p in 1 3; do
    step "$W/llm_part_${p}_ar16/llm_part_${p}_ar16.encodings" "llm_part_$p" aimet "${GPUS[0]}" "$S/minicpmo_llm.py" part --part $p "${COMMON[@]}"
  done
  step "$W/llm_head/llm_head.encodings" llm_head aimet "${GPUS[0]}" "$S/minicpmo_llm.py" head "${COMMON[@]}"
} &
q1=$!
{
  for p in 0 2; do
    step "$W/llm_part_${p}_ar16/llm_part_${p}_ar16.encodings" "llm_part_$p" aimet "${GPUS[1]}" "$S/minicpmo_llm.py" part --part $p "${COMMON[@]}"
  done
} &
q2=$!
await $q1 $q2

echo "== QAIRT: convert, quantize, compile contexts"
ln -sfn "$W/token2wav/dit" "$W/dit"
ln -sfn "$W/token2wav/hift_f0" "$W/hift_f0"
ln -sfn "$W/token2wav/hift_decode" "$W/hift_decode"
build() { local out=$1; shift; [ -f "$W/contexts/$out.bin" ] || "$S/minicpmo_qairt_build.sh" "$W/contexts/$out.bin" "$@" > "$W/logs/$out.build.log" 2>&1; }
pids=()
for p in 0 1 2 3; do build "llm_part_$p" "llm_part_${p}_ar1" "llm_part_${p}_ar16" & pids+=($!); done
build llm_head llm_head & pids+=($!)
build audio_encoder audio_encoder & pids+=($!)
build dit dit & pids+=($!)
build hift_f0 hift_f0 & pids+=($!)
build hift_decode hift_decode & pids+=($!)
await "${pids[@]}"

B=$MCPO_ROOT/bundle/$MCPO_BUNDLE
echo "== bundle $B"
mkdir -p "$B"
for name in llm_part_0 llm_part_1 llm_part_2 llm_part_3 llm_head audio_encoder dit hift_f0 hift_decode; do
  ln -f "$W/contexts/$name.bin" "$B/"
done
rm -f "$B/tts.bin"  # bundles before the TTS moved to the CPU
ln -f "$W/token2wav/flow_encoder.onnx" "$W/tts_cpu/tts_cpu.bin" "$B/"
[ -f "$B/assets.bin" ] || aimet "${GPUS[0]}" "$S/minicpmo_bundle.py" "${COMMON[@]}" --output "$B/assets.bin" > "$W/logs/bundle.log" 2>&1
(cd "$B" && sha256sum assets.bin flow_encoder.onnx ./*.bin | sed 's#  \./#  #' | sort -k2 -u > SHA256SUMS)
cat "$B/SHA256SUMS"
