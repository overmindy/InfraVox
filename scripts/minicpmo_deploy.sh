#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# One-click Mac deploy for development: stream the server's MiniCPM-o bundle straight to the phone
# over ADB (the app itself downloads from Hugging Face instead), then build and install the APK.
# The phone copy goes to files/models/$MCPO_VARIANT, the catalog variant id the app looks for.
# Files already on the phone with the right checksum are skipped; SHA256SUMS goes last.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
source "$R/scripts/minicpmo_env.sh"  # server-side paths: MCPO_ROOT, MCPO_BUNDLE
: "${MCPO_SERVER:=user@117.74.67.2}"
: "${MCPO_SERVER_PORT:=40023}"
: "${MCPO_VARIANT:=minicpmo45-v1-sm8850}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/30.0.15729638}"
: "${QNN_INCLUDE:=$HOME/d_personal/GenieX/third-party/geniex-qairt/qnn-api/include}"
: "${QNN_ANDROID_LIB_DIR:=$HOME/d_personal/GenieX/third-party/geniex-qairt/third-party/android}"
: "${KEYSTORE_PATH:=$R/android/demo/build/demo.keystore}"  # The app only updates with its original key.
: "${ORT_DIR:=$R/android/demo/build/ort/aar}"
: "${ORT_VERSION:=1.20.0}"
export ANDROID_SDK_ROOT ANDROID_NDK_HOME QNN_INCLUDE QNN_ANDROID_LIB_DIR KEYSTORE_PATH ORT_DIR
[ -f "$KEYSTORE_PATH" ] || { echo "Missing $KEYSTORE_PATH; never regenerate it (reinstalling loses app data)" >&2; exit 2; }
if [ ! -f "$ORT_DIR/jni/arm64-v8a/libonnxruntime.so" ]; then
  mkdir -p "$ORT_DIR"
  curl -fsSL -o "$ORT_DIR.aar" "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VERSION/onnxruntime-android-$ORT_VERSION.aar"
  unzip -o -q "$ORT_DIR.aar" -d "$ORT_DIR"
fi

M=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/$MCPO_VARIANT
REMOTE=$MCPO_ROOT/bundle/$MCPO_BUNDLE
remote() { ssh -n -p "$MCPO_SERVER_PORT" "$MCPO_SERVER" "$@"; }  # -n: keep ssh off the while-read loop's stdin
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== bundle $MCPO_BUNDLE -> phone $M"
remote cat "$REMOTE/SHA256SUMS" > "$TMP/SHA256SUMS"
adb shell rm -f "$M/SHA256SUMS"
adb shell mkdir -p "$M"
adb shell "cd $M && sha256sum $(cut -d' ' -f3 "$TMP/SHA256SUMS" | tr '\n' ' ') 2>/dev/null" > "$TMP/phone" || true
while read -r sum file; do
  file=${file#\*}
  grep -q "^$sum  $file\$" "$TMP/phone" && continue
  echo "push $file"
  remote cat "$REMOTE/$file" | adb exec-in "cat > $M/$file"
done < "$TMP/SHA256SUMS"
adb push "$TMP/SHA256SUMS" "$M/SHA256SUMS" > /dev/null
for file in $(adb shell ls "$M"); do  # files of an older bundle layout (e.g. tts.bin)
  [ "$file" = SHA256SUMS ] || grep -q "  $file\$" "$TMP/SHA256SUMS" || { echo "remove $file"; adb shell rm -f "$M/$file"; }
done
adb shell "chmod 755 $M 2>/dev/null; chmod 644 $M/* 2>/dev/null; cd $M && sha256sum -c SHA256SUMS"  # chmod fails once the app owns the dir

echo "== app"
if [ -n "${MCPO_DEMO_WAV:-}" ]; then cp "$MCPO_DEMO_WAV" "$TMP/demo.wav"; else remote cat "$MCPO_ROOT/capture/questions/s00.wav" > "$TMP/demo.wav"; fi
SAMPLE_WAV=$TMP/demo.wav OUT_DIR=$TMP/apk bash "$R/android/demo/build.sh"
adb install -r "$TMP/apk/infravox.apk"
