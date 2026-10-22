#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Convert, quantize and compile one Moshi graph into an SM8850 (HTP v81) context DIR/NAME.bin.
# Usage: moshi_qairt_build.sh DIR NAME   (reads DIR/onnx/model.onnx and DIR/NAME.encodings)
#   QUANTIZE=0    the encodings already fix every tensor (Mimi): converter only, no qairt-quantizer.
#   FLOAT_IO      I/O tensors the converter keeps FP32.  INT32_INPUTS  int64 inputs fed as int32.
# The graph is named NAME so the O3 + VTCM 8 MB options apply (W4 Temporal part ~28 ms vs ~95 ms).
set -euo pipefail
DIR=$1; NAME=$2
S=$(cd "$(dirname "$0")" && pwd)
source "$S/moshi_env.sh"
BIN=$QAIRT_SDK_ROOT/bin/x86_64-linux-clang
LIB=$QAIRT_SDK_ROOT/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB
ARGS=()
for name in ${INT32_INPUTS:-}; do ARGS+=(--source_model_input_datatype "$name" int32); done
if [ -n "${FLOAT_IO:-}" ]; then read -ra IO <<< "$FLOAT_IO"; ARGS+=(--preserve_io_datatype "${IO[@]}"); fi

if [ ! -f "$DIR/$NAME.dlc" ]; then
  CONVERTED=$DIR/$NAME.dlc
  if [ "${QUANTIZE:-1}" = 1 ]; then CONVERTED=$DIR/float/$NAME.dlc; fi
  mkdir -p "$(dirname "$CONVERTED")"
  "$PY" "$S/moshi_guard_qairt_converter.py" --sdk-root "$QAIRT_SDK_ROOT" --numpy-dir "$QAIRT_NUMPY_DIR" -- \
    --input_network "$DIR/onnx/model.onnx" --output_path "$CONVERTED" \
    --quantization_overrides "$DIR/$NAME.encodings" --preserve_onnx_output_order --onnx_skip_simplification \
    ${ARGS[@]+"${ARGS[@]}"}
  if [ "${QUANTIZE:-1}" = 1 ]; then
    env PYTHONPATH="$QAIRT_NUMPY_DIR:$QAIRT_SDK_ROOT/lib/python" "$PY" "$BIN/qairt-quantizer" \
      --input_dlc "$CONVERTED" --output_dlc "$DIR/$NAME.dlc" --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
    rm -rf "$DIR/float"
  fi
fi

OUT=$DIR/htp
mkdir -p "$OUT"
echo "{\"graphs\": [{\"graph_names\": [\"$NAME\"], \"O\": 3, \"vtcm_mb\": 8}], \"devices\": [{\"soc_model\": 87, \"dsp_arch\": \"v81\"}]}" \
  > "$OUT/backend.json"
echo "{\"backend_extensions\": {\"shared_library_path\": \"$LIB/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$OUT/backend.json\"}}" \
  > "$OUT/config.json"
"$BIN/qnn-context-binary-generator" --backend "$LIB/libQnnHtp.so" --model "$LIB/libQnnModelDlc.so" \
  --dlc_path "$DIR/$NAME.dlc" --config_file "$OUT/config.json" --binary_file "$NAME" --output_dir "$OUT" --log_level warn
"$BIN/qnn-context-binary-utility" --context_binary "$OUT/$NAME.bin" --json_file "$OUT/$NAME.json" > /dev/null
if ! grep -q "\"graphName\": *\"$NAME\"" "$OUT/$NAME.json"; then
  echo "$NAME: the DLC graph is not named $NAME, so the O3/VTCM options did not apply" >&2
  exit 1
fi
mv "$OUT/$NAME.bin" "$DIR/$NAME.bin"
rm -rf "$OUT"
