#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# One-click Mac deploy for development: copy the server's MiniCPM-o bundle to a local cache, push it to
# the phone over ADB (the app itself downloads from Hugging Face instead), then build and install the APK.
# The phone copy goes to files/models/$MCPO_VARIANT, the catalog variant id the app looks for.
# Files already on the phone with the right checksum are skipped; SHA256SUMS goes last.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
source "$R/scripts/minicpmo_env.sh"  # server-side paths: MCPO_ROOT, MCPO_BUNDLE, MCPO_TARGET
: "${MCPO_SERVER:=user@117.74.67.2}"
: "${MCPO_SERVER_PORT:=40023}"
: "${MCPO_VARIANT:=minicpmo45-$MCPO_BUNDLE-$MCPO_TARGET}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/30.0.15729638}"
# The app's QNN runtime must be the QAIRT that compiled the contexts: include/QNN, plus the aarch64-android
# libraries and hexagon-v8x/unsigned skels of $QAIRT_SDK_ROOT copied flat into lib/.
: "${QAIRT_LOCAL:=$R/android/demo/build/qairt/$(basename "$QAIRT_SDK_ROOT")}"
: "${QNN_INCLUDE:=$QAIRT_LOCAL/include}"
: "${QNN_ANDROID_LIB_DIR:=$QAIRT_LOCAL/lib}"
: "${MCPO_CACHE:=$R/android/demo/build/bundles/$MCPO_BUNDLE-$MCPO_TARGET}"  # local copy of the server bundle
: "${MCPO_STREAMS:=8}"  # the server link is limited per connection; sshd resets bursts of more than ~10
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
REMOTE=$MCPO_ROOT/bundle/$MCPO_BUNDLE-$MCPO_TARGET
remote() { ssh -n -p "$MCPO_SERVER_PORT" "$MCPO_SERVER" "$@"; }  # -n: keep ssh off the while-read loop's stdin
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$MCPO_CACHE/.parts"
export MCPO_SERVER MCPO_SERVER_PORT REMOTE MCPO_CACHE

# Download FILE into $MCPO_CACHE as 32 MiB chunks over $MCPO_STREAMS ssh connections; finished chunks survive a rerun.
fetch() {
  local file=$1 sum=$2 size chunks i
  size=$(remote stat -c %s "$REMOTE/$file")
  chunks=$(( (size + 33554431) / 33554432 ))
  echo "fetch $file ($((size >> 20)) MiB)"
  seq 0 $((chunks - 1)) | xargs -P "$MCPO_STREAMS" -n 1 bash -c 'part=$MCPO_CACHE/.parts/$0.$1; [ -e "$part.done" ] && exit 0
    for try in 1 2 3 4 5; do  # a connection can hang without closing: perl alarm caps each chunk at 5 minutes
      perl -e "alarm shift; exec @ARGV" 300 ssh -n -o ServerAliveInterval=15 -p "$MCPO_SERVER_PORT" "$MCPO_SERVER" \
        "dd if=$REMOTE/$0 bs=32M skip=$1 count=1 iflag=fullblock status=none" > "$part" \
        && { touch "$part.done"; exit 0; }
      sleep $((try * 3))
    done
    exit 1' "$file" || true  # exit 1 (not ssh 255) so xargs keeps the other chunks going
  for ((i = 0; i < chunks; i++)); do
    [ -e "$MCPO_CACHE/.parts/$file.$i.done" ] || { echo "$file: chunk $i failed, rerun to resume" >&2; exit 1; }
  done
  : > "$MCPO_CACHE/$file"
  for ((i = 0; i < chunks; i++)); do cat "$MCPO_CACHE/.parts/$file.$i" >> "$MCPO_CACHE/$file"; done
  [ "$(shasum -a 256 "$MCPO_CACHE/$file" | cut -d' ' -f1)" = "$sum" ] || { echo "$file: checksum mismatch, rerun to retry" >&2; rm -f "$MCPO_CACHE/.parts/$file".*; exit 1; }
  rm -f "$MCPO_CACHE/.parts/$file".*
}

echo "== bundle $MCPO_BUNDLE -> phone $M"
remote cat "$REMOTE/SHA256SUMS" > "$TMP/SHA256SUMS"
adb shell rm -f "$M/SHA256SUMS"
adb shell mkdir -p "$M"
adb shell "cd $M && sha256sum $(cut -d' ' -f3 "$TMP/SHA256SUMS" | tr '\n' ' ') 2>/dev/null" > "$TMP/phone" || true
while read -r sum file; do
  file=${file#\*}
  grep -q "^$sum  $file\$" "$TMP/phone" && continue
  [ "$(shasum -a 256 "$MCPO_CACHE/$file" 2>/dev/null | cut -d' ' -f1)" = "$sum" ] || fetch "$file" "$sum"
  echo "push $file"
  adb push "$MCPO_CACHE/$file" "$M/$file" < /dev/null > /dev/null
done < "$TMP/SHA256SUMS"
adb push "$TMP/SHA256SUMS" "$M/SHA256SUMS" > /dev/null
for file in $(adb shell ls "$M"); do  # files of an older bundle layout (e.g. tts.bin)
  [ "$file" = SHA256SUMS ] || grep -q "  $file\$" "$TMP/SHA256SUMS" || { echo "remove $file"; adb shell rm -f "$M/$file"; }
done
# A dir adb created stays owned by shell and the app reaches it as "other": it must be able to write SHA256SUMS.tmp
# there. chmod fails, harmlessly, once the app owns the dir.
adb shell "chmod 777 $M 2>/dev/null; chmod 644 $M/* 2>/dev/null; cd $M && sha256sum -c SHA256SUMS"

echo "== app"
if [ -n "${MCPO_DEMO_WAV:-}" ]; then cp "$MCPO_DEMO_WAV" "$TMP/demo.wav"; else remote cat "$MCPO_ROOT/capture/questions/s00.wav" > "$TMP/demo.wav"; fi
SAMPLE_WAV=$TMP/demo.wav OUT_DIR=$TMP/apk bash "$R/android/demo/build.sh"
adb install -r "$TMP/apk/infravox.apk"
