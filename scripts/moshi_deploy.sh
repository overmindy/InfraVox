#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# One-click Mac deploy for development: stream a server bundle straight to the phone over ADB
# (the app itself downloads from Hugging Face instead), then build and install the APK with the original keystore.
# The phone copy goes to files/models/$MOSHI_VARIANT, the catalog variant id the app looks for.
# Files already on the phone with the right checksum are skipped; SHA256SUMS goes last, so the app never sees a partial copy.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
source "$R/scripts/moshi_env.sh"  # server-side paths: MOSHI_ROOT, MOSHI_BF16, PY
: "${MOSHI_SERVER:=user@117.74.67.2}"
: "${MOSHI_SERVER_PORT:=40023}"
: "${MOSHI_BUNDLE:=lm-v4-lpbq4}"
: "${MOSHI_VARIANT:=moshiko-lm-v4-lpbq4-sm8850}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/30.0.15729638}"
: "${QNN_INCLUDE:=$HOME/d_personal/GenieX/third-party/geniex-qairt/qnn-api/include}"
: "${QNN_ANDROID_LIB_DIR:=$HOME/d_personal/GenieX/third-party/geniex-qairt/third-party/android}"
: "${KEYSTORE_PATH:=$R/android/demo/build/demo.keystore}"  # The app only updates with its original key.
export ANDROID_SDK_ROOT ANDROID_NDK_HOME QNN_INCLUDE QNN_ANDROID_LIB_DIR KEYSTORE_PATH
[ -f "$KEYSTORE_PATH" ] || { echo "Missing $KEYSTORE_PATH; never regenerate it (reinstalling loses app data)" >&2; exit 2; }

M=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/$MOSHI_VARIANT
REMOTE=$MOSHI_ROOT/bundle/$MOSHI_BUNDLE
remote() { ssh -p "$MOSHI_SERVER_PORT" "$MOSHI_SERVER" "$@"; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== bundle $MOSHI_BUNDLE -> phone $M"
remote cat "$REMOTE/SHA256SUMS" > "$TMP/SHA256SUMS"
VOCAB=
if ! grep -q ' text_vocab.txt$' "$TMP/SHA256SUMS"; then  # bundles built before moshi_text_vocab.py
  VOCAB=$TMP/text_vocab.txt
  remote "$PY" - --model-dir "$MOSHI_BF16" --output /dev/stdout < "$R/scripts/moshi_text_vocab.py" > "$VOCAB"
  echo "$(shasum -a 256 "$VOCAB" | cut -d' ' -f1)  text_vocab.txt" >> "$TMP/SHA256SUMS"
fi
adb shell rm -f "$M/SHA256SUMS"
adb shell mkdir -p "$M/dlc"
adb shell "cd $M && sha256sum $(cut -d' ' -f3 "$TMP/SHA256SUMS" | tr '\n' ' ') 2>/dev/null" > "$TMP/phone" || true
while read -r sum file; do
  file=${file#\*}
  grep -q "^$sum  $file\$" "$TMP/phone" && continue
  echo "push $file"
  if [ "$file" = text_vocab.txt ] && [ -n "$VOCAB" ]; then adb push "$VOCAB" "$M/$file" > /dev/null
  else remote cat "$REMOTE/$file" | adb exec-in "cat > $M/$file"; fi
done < "$TMP/SHA256SUMS"
adb push "$TMP/SHA256SUMS" "$M/SHA256SUMS" > /dev/null
adb shell "chmod 755 $M $M/dlc && chmod 644 $M/*.bin $M/*.txt $M/dlc/*.bin $M/SHA256SUMS && cd $M && sha256sum -c SHA256SUMS"

echo "== app"
remote cat "$MOSHI_ROOT/inputs/validation_input.wav" > "$TMP/demo.wav"
SAMPLE_WAV=$TMP/demo.wav OUT_DIR=$TMP/apk bash "$R/android/demo/build.sh"
adb install -r "$TMP/apk/duplex-voice.apk"
