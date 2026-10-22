#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# One-click Mac deploy: stream the app's bundle (MainActivity.BUNDLE) from the server to the phone,
# build the APK with the original keystore and install it. Nothing large is kept on the Mac.
# Files already on the phone with the right checksum are skipped; older bundles on the phone are removed.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
: "${MOSHI_SERVER:=user@117.74.67.2}"
: "${MOSHI_SERVER_PORT:=40023}"
: "${MOSHI_ROOT:=/data2/yejialei/moshi}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/30.0.15729638}"
: "${QNN_INCLUDE:=$HOME/d_personal/GenieX/third-party/geniex-qairt/qnn-api/include}"
: "${QNN_ANDROID_LIB_DIR:=$HOME/d_personal/GenieX/third-party/geniex-qairt/third-party/android}"
: "${KEYSTORE_PATH:=$R/android/demo/build/demo.keystore}"  # The app only updates with its original key.
export ANDROID_SDK_ROOT ANDROID_NDK_HOME QNN_INCLUDE QNN_ANDROID_LIB_DIR KEYSTORE_PATH
[ -f "$KEYSTORE_PATH" ] || { echo "Missing $KEYSTORE_PATH; never regenerate it (reinstalling loses app data)" >&2; exit 2; }

BUNDLE=$(sed -n 's/.*String BUNDLE = "\(.*\)";/\1/p' "$R/android/demo/src/com/overmindy/mimicodecdemo/MainActivity.java")
MODELS=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models
M=$MODELS/$BUNDLE
REMOTE=$MOSHI_ROOT/bundle/$BUNDLE
remote() { ssh -p "$MOSHI_SERVER_PORT" "$MOSHI_SERVER" "$@"; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== bundle $BUNDLE -> phone"
remote cat "$REMOTE/SHA256SUMS" > "$TMP/SHA256SUMS"
adb shell mkdir -p "$M/dlc"
adb shell "cd $M && sha256sum $(cut -d' ' -f3 "$TMP/SHA256SUMS" | tr '\n' ' ') 2>/dev/null" > "$TMP/phone" || true
while read -r sum file; do
  file=${file#\*}
  grep -q "^$sum  $file\$" "$TMP/phone" && continue
  echo "push $file"
  remote cat "$REMOTE/$file" | adb exec-in "cat > $M/$file"
done < "$TMP/SHA256SUMS"
adb push "$TMP/SHA256SUMS" "$M/SHA256SUMS" > /dev/null  # Last, so the app never accepts a partial bundle.
adb shell "chmod 755 $M $M/dlc && chmod 644 $M/*.bin $M/dlc/*.bin $M/SHA256SUMS && cd $M && sha256sum -c SHA256SUMS"
for old in $(adb shell ls "$MODELS"); do [ "$old" = "$BUNDLE" ] || adb shell rm -rf "$MODELS/$old"; done

echo "== app"
remote cat "$MOSHI_ROOT/inputs/validation_input.wav" > "$TMP/demo.wav"
SAMPLE_WAV=$TMP/demo.wav OUT_DIR=$TMP/apk bash "$R/android/demo/build.sh"
adb install -r "$TMP/apk/mimi-codec-demo.apk"
