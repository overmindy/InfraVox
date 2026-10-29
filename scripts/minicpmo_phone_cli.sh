#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build android/minicpmo_cli.cpp, run it on the phone against a pushed MiniCPM-o bundle, and pull the results.
#   bash scripts/minicpmo_phone_cli.sh <16 kHz mono WAV> <local output dir> [image.rgb width height] [seed]
# image.rgb is raw interleaved 8-bit RGB (e.g. the *_rgb.u8 vectors of minicpmo_vision_reference.py hosttest).
# Android 17's NPU scheduler refuses shell processes (deviceCreate 14001, see engine/NpuGate.java), so on such
# phones test through the app instead.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
source "$R/scripts/minicpmo_env.sh"
: "${MCPO_VARIANT:=minicpmo45-$MCPO_BUNDLE-$MCPO_TARGET}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/30.0.15729638}"
: "${QAIRT_LOCAL:=$R/android/demo/build/qairt/$(basename "$QAIRT_SDK_ROOT")}"
: "${ORT_DIR:=$R/android/demo/build/ort/aar}"
INPUT=$1 OUT=$2
shift 2
M=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/$MCPO_VARIANT
D=/data/local/tmp/minicpmo_cli
HOST=$([ "$(uname -s)" = Darwin ] && echo darwin-x86_64 || echo linux-x86_64)
mkdir -p "$OUT"

"$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST/bin/aarch64-linux-android29-clang++" -std=c++17 -O2 -march=armv8.2-a+dotprod \
  -Wall -Wextra -Werror -static-libstdc++ -I "$QAIRT_LOCAL/include" -I "$ORT_DIR/headers" \
  "$R/android/minicpmo_runner.cpp" "$R/android/minicpmo_cli.cpp" -o "$OUT/minicpmo_cli" \
  -L "$ORT_DIR/jni/arm64-v8a" -lonnxruntime -ldl
adb shell rm -rf "$D/run"
adb shell mkdir -p "$D/lib" "$D/run"
for library in "$QAIRT_LOCAL"/lib/*.so "$ORT_DIR/jni/arm64-v8a/libonnxruntime.so"; do
  adb push "$library" "$D/lib/" > /dev/null
done
adb push "$OUT/minicpmo_cli" "$D/" > /dev/null
adb push "$INPUT" "$D/input.wav" > /dev/null
ARGS=()
if [ $# -ge 3 ]; then
  adb push "$1" "$D/image.rgb" > /dev/null
  ARGS=("$D/image.rgb" "$2" "$3")
  shift 3
fi
adb shell "chmod 755 $D/minicpmo_cli && LD_LIBRARY_PATH=$D/lib $D/minicpmo_cli $D/lib $M $D/input.wav $D/run ${ARGS[*]-} $*"
adb pull "$D/run/." "$OUT/" > /dev/null
echo "results in $OUT"
