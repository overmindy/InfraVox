#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SDK=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}
NDK=${ANDROID_NDK_HOME:-}
QNN_HEADERS=${QNN_INCLUDE:-}
QNN_LIBS=${QNN_ANDROID_LIB_DIR:-}
SAMPLE=${SAMPLE_WAV:-}  # Optional 24 kHz mono PCM16 WAV shipped as the "演示音频" input.
OUT=${OUT_DIR:-$SCRIPT_DIR/build}

if [[ -z "$SDK" || -z "$NDK" || -z "$QNN_HEADERS" || -z "$QNN_LIBS" ]]; then
  echo 'Set ANDROID_SDK_ROOT, ANDROID_NDK_HOME, QNN_INCLUDE, QNN_ANDROID_LIB_DIR (SAMPLE_WAV optional).' >&2
  exit 2
fi
if [[ "$(uname -s)" == Darwin ]]; then
  HOST=darwin-x86_64
  if [[ -z "${JAVA_HOME:-}" && -x '/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/javac' ]]; then
    export JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home'
  fi
else
  HOST=linux-x86_64
fi
if [[ -z "${JAVA_HOME:-}" || ! -x "$JAVA_HOME/bin/javac" ]]; then
  echo 'Set JAVA_HOME to a JDK installation.' >&2
  exit 2
fi
BUILD_TOOLS="$SDK/build-tools/35.0.0"
PLATFORM="$SDK/platforms/android-34/android.jar"
CXX="$NDK/toolchains/llvm/prebuilt/$HOST/bin/aarch64-linux-android29-clang++"
rm -rf "$OUT/classes" "$OUT/dex" "$OUT/stage"
mkdir -p "$OUT/classes" "$OUT/dex" "$OUT/stage/lib/arm64-v8a" "$OUT/stage/assets"

"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared \
  -static-libstdc++ -Wl,--no-undefined -I "$QNN_HEADERS" \
  "$REPO_ROOT/android/mimi_qnn_runner.cpp" "$SCRIPT_DIR/moshi_jni.cpp" \
  -o "$OUT/stage/lib/arm64-v8a/libmoshi_qnn.so" -ldl

for library in libQnnHtp.so libQnnSystem.so libQnnHtpV81.so \
               libQnnHtpV81Stub.so libQnnHtpV81Skel.so \
               libQnnHtpPrepare.so libQnnModelDlc.so; do
  cp "$QNN_LIBS/$library" "$OUT/stage/lib/arm64-v8a/$library"
done
cp "$REPO_ROOT/android/models/catalog.json" "$OUT/stage/assets/catalog.json"
if [[ -n "$SAMPLE" ]]; then cp "$SAMPLE" "$OUT/stage/assets/demo.wav"; fi

find "$SCRIPT_DIR/src" -name '*.java' > "$OUT/sources.txt"
"$JAVA_HOME/bin/javac" -source 8 -target 8 -Xlint:-options -encoding UTF-8 \
  -classpath "$PLATFORM" -d "$OUT/classes" @"$OUT/sources.txt"
find "$OUT/classes" -name '*.class' > "$OUT/classes.txt"
"$BUILD_TOOLS/d8" --min-api 29 --lib "$PLATFORM" --output "$OUT/dex" @"$OUT/classes.txt"

"$BUILD_TOOLS/aapt2" link --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
  -I "$PLATFORM" --min-sdk-version 29 --target-sdk-version 34 \
  -o "$OUT/unsigned.apk"
cp "$OUT/dex/classes.dex" "$OUT/stage/classes.dex"
(cd "$OUT/stage" && zip -q -0 -r "$OUT/unsigned.apk" classes.dex lib assets)
"$BUILD_TOOLS/zipalign" -f 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"

KEYSTORE=${KEYSTORE_PATH:-$OUT/demo.keystore}
if [[ ! -f "$KEYSTORE" ]]; then
  "$JAVA_HOME/bin/keytool" -genkeypair -keystore "$KEYSTORE" -alias mimi-demo \
    -storepass android -keypass android -dname 'CN=Mimi Demo' \
    -keyalg RSA -keysize 2048 -validity 3650 -noprompt >/dev/null
fi
"$BUILD_TOOLS/apksigner" sign --ks "$KEYSTORE" --ks-key-alias mimi-demo \
  --ks-pass pass:android --key-pass pass:android \
  --out "$OUT/duplex-voice.apk" "$OUT/aligned.apk"
"$BUILD_TOOLS/apksigner" verify "$OUT/duplex-voice.apk"
echo "Built $OUT/duplex-voice.apk"
