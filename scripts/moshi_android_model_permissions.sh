#!/usr/bin/env bash
# ADB owns pushed files: make the model files readable by the app, not just shell.
set -euo pipefail
ADB_TOOL=${1:-adb}
DEVICE_MODEL_ROOT=/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models
"$ADB_TOOL" get-state
"$ADB_TOOL" shell "test -d '$DEVICE_MODEL_ROOT/mimi' && test -d '$DEVICE_MODEL_ROOT/lm-guarded-v2/dlc'"
"$ADB_TOOL" shell "find '$DEVICE_MODEL_ROOT/mimi' '$DEVICE_MODEL_ROOT/lm-guarded-v2' -type d -user 2000 -exec chmod 755 {} \;"
"$ADB_TOOL" shell "find '$DEVICE_MODEL_ROOT/mimi' '$DEVICE_MODEL_ROOT/lm-guarded-v2' -type f -user 2000 -exec chmod 644 {} \;"
"$ADB_TOOL" shell "cd '$DEVICE_MODEL_ROOT/lm-guarded-v2' && sha256sum -c SHA256SUMS"
