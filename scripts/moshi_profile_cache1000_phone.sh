#!/bin/bash
# Candidate-only native test. Never switches the App's default model bundle.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
candidate_root="${1:?Pass local cache1000 candidate directory}"
native_binary="${2:?Pass the clean profiling binary}"
adb_bin="${ADB:-/Users/e1ainay/Android/platform-tools/adb}"
library_dir="${3:?Pass the currently installed App native library directory}"
phone_root=/data/local/tmp/moshi-cache1000-full-profile-v1
baseline_root=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-guarded-v2
result_root="$candidate_root/profile_full_v1"
mkdir -p "$result_root"
# Bound the wait and keep incomplete downloads, rather than restarting them.
for attempt in $(seq 1 240); do
  set +e
  python3 "$repo_root/scripts/moshi_collect_cache1000_candidates.py" --verify-dir "$candidate_root" > "$result_root/download_checkpoint.json"
  checkpoint=$?
  set -e
  if [ "$checkpoint" = 0 ]; then break; fi
  if [ "$checkpoint" != 3 ]; then echo "Download hash/inventory failure; stop."; exit "$checkpoint"; fi
  if [ "$attempt" = 240 ]; then echo "Download wait expired; all artifacts retained."; exit 3; fi
  echo "Waiting for complete verified downloads ($attempt/240)."
  sleep 15
done
"$adb_bin" get-state
"$adb_bin" shell "mkdir -p $phone_root/dlc"
for layer in $(seq 0 2 30); do
  graph="temporal_layers_${layer}_$((layer + 1))"
  model="$candidate_root/dlc/$graph.dlc"
  expected="$(shasum -a 256 "$model" | awk '{print $1}')"
  existing="$("$adb_bin" shell "if test -f $phone_root/dlc/$graph.dlc; then sha256sum $phone_root/dlc/$graph.dlc; fi" | awk '{print $1}')"
  if [ -n "$existing" ] && [ "$existing" != "$expected" ]; then
    echo "Existing phone model differs: $graph. Stop without overwriting."; exit 2
  fi
  if [ -z "$existing" ]; then "$adb_bin" push "$model" "$phone_root/dlc/$graph.dlc"; fi
  actual="$("$adb_bin" shell "sha256sum $phone_root/dlc/$graph.dlc" | awk '{print $1}')"
  [ "$actual" = "$expected" ] || { echo "Phone SHA256 failed: $graph"; exit 2; }
done
for graph in frontend head depformer_codebook_{0..7}; do
  "$adb_bin" shell "test -f $baseline_root/dlc/$graph.dlc; if ! test -e $phone_root/dlc/$graph.dlc; then ln -s $baseline_root/dlc/$graph.dlc $phone_root/dlc/$graph.dlc; fi"
done
"$adb_bin" push "$native_binary" "$phone_root/runner"
echo "Starting real two-frame full candidate chain. No maximum/all-resident mode."
"$adb_bin" shell "env MOSHI_TEMPORAL_BUDGET_MIB=3072 MOSHI_QNN_PROFILE=1 MOSHI_HTP_PERFORMANCE=1 MOSHI_CONTEXT_CACHE_DIR=/data/local/tmp/moshi_context_chain_f2_v1.wav.contexts-v1 LD_LIBRARY_PATH=$library_dir ADSP_LIBRARY_PATH=$library_dir $phone_root/runner $library_dir /data/local/tmp/moshi-mimi-codec-v1/mimi_encoder.dlc /data/local/tmp/moshi-mimi-codec-v1/mimi_decoder.dlc /data/local/tmp/moshi-mimi-codec-v1/input.wav $phone_root/full_f2.wav 2 $phone_root --repeat > $phone_root/run.log 2>&1"
for file in run.log full_f2.wav full_f2.wav.repeat.wav full_f2.wav.repeat.wav.performance.csv full_f2.wav.repeat.wav.performance.csv.qnn.csv full_f2.wav.repeat.wav.codes.txt; do
  "$adb_bin" pull "$phone_root/$file" "$result_root/"
done
python3 "$repo_root/scripts/moshi_summarize_runtime_profile.py" \
  --timing "$result_root/full_f2.wav.repeat.wav.performance.csv" \
  --qnn "$result_root/full_f2.wav.repeat.wav.performance.csv.qnn.csv" \
  --report "$result_root/report.json"
"$adb_bin" shell "sha256sum $phone_root/full_f2.wav $phone_root/full_f2.wav.repeat.wav /data/local/tmp/moshi_context_chain_f2_v1.wav"
echo "Native candidate profiling complete; inspect accuracy before App deployment."
