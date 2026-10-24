#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build android/moshi_cli.cpp, run it on the phone against the installed Moshi bundle, and pull the results.
#   bash scripts/moshi_phone_cli.sh <input 24 kHz mono WAV> <local output dir> [seed]
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
: "${MOSHI_VARIANT:=moshiko-lm-v4-lpbq4-sm8850}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/30.0.15729638}"
: "${QNN_INCLUDE:=$HOME/d_personal/GenieX/third-party/geniex-qairt/qnn-api/include}"
: "${QNN_ANDROID_LIB_DIR:=$HOME/d_personal/GenieX/third-party/geniex-qairt/third-party/android}"
INPUT=$1 OUT=$2 SEED=${3:-1}
M=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/$MOSHI_VARIANT
D=/data/local/tmp/moshi_cli
HOST=$([ "$(uname -s)" = Darwin ] && echo darwin-x86_64 || echo linux-x86_64)
mkdir -p "$OUT"

"$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST/bin/aarch64-linux-android29-clang++" -std=c++17 -O2 -Wall -Wextra -Werror \
  -static-libstdc++ -I "$QNN_INCLUDE" "$R/android/mimi_qnn_runner.cpp" "$R/android/moshi_cli.cpp" -o "$OUT/moshi_cli" -ldl
adb shell rm -rf "$D/run"
adb shell mkdir -p "$D/lib" "$D/run"
for library in libQnnHtp.so libQnnSystem.so libQnnHtpV81.so libQnnHtpV81Stub.so libQnnHtpV81Skel.so libQnnHtpPrepare.so; do
  adb push "$QNN_ANDROID_LIB_DIR/$library" "$D/lib/" > /dev/null
done
adb push "$OUT/moshi_cli" "$D/" > /dev/null
adb push "$INPUT" "$D/input.wav" > /dev/null
adb shell "chmod 755 $D/moshi_cli && LD_LIBRARY_PATH=$D/lib $D/moshi_cli $D/lib $M $D/input.wav $D/run $SEED"
adb pull "$D/run/." "$OUT/" > /dev/null
echo "results in $OUT"
