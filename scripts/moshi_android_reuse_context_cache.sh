#!/usr/bin/env bash
# Reuse only this phone's already benchmarked contexts; never replace DLCs.
set -euo pipefail
ADB_TOOL=${1:-adb}
SOURCE=/data/local/tmp/moshi_context_chain_f2_v1.wav.contexts-v1
TARGET=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/qnn-context-v1
"$ADB_TOOL" get-state
if ! "$ADB_TOOL" shell "test -d '$SOURCE' && test -d '$TARGET'"; then
  echo 'Install and launch App 2.3 first so the app creates its writable qnn-context-v1 directory.' >&2
  exit 2
fi
"$ADB_TOOL" shell "
  set -e
  for source in '$SOURCE'/*.bin; do
    test -f \"\$source\"
    target='$TARGET'/\"\${source##*/}\"
    if ! test -e \"\$target\"; then
      cp -n \"\$source\" \"\$target\"
      chmod 644 \"\$target\"
    fi
    source_hash=\$(sha256sum \"\$source\")
    target_hash=\$(sha256sum \"\$target\")
    test \"\${source_hash%% *}\" = \"\${target_hash%% *}\"
    echo \"CONTEXT VERIFIED \${source##*/}\"
  done
"
echo 'Context cache copied and byte-verified. Existing files were not overwritten; source cache retained.'
