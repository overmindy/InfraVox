#!/bin/bash
# Convert + quantize + compile one calibrated Moshi Temporal part for SM8850 (HTP v81).
# Usage: moshi_temporal_build.sh PART_DIR NAME [COMPILE]
#   COMPILE: o3-vtcm8 (default; no spill on SM8850), o3, o3-vtcm<N>, or default (no graph options).
#   On the phone an 8-layer W4 LPBQ part runs ~28 ms with O3 vs ~95 ms with default options.
# Reads PART_DIR/NAME.encodings; reuses PART_DIR/NAME.dlc when it already exists.
# The graph is named NAME so the per-graph compile options actually apply.
set -euo pipefail
PART=$1; NAME=$2; COMPILE=${3:-o3-vtcm8}
SDK=${QAIRT_SDK_ROOT:-/data2/yejialei/sdk/qairt/2.45.0.260326}
NUMPY=${QAIRT_NUMPY_DIR:-/data2/yejialei/tmp/moshi-qdq-0-1-cpu-check-v1/dlc_fusion_diagnostics/python-deps-numpy126}
PY=${QAIRT_PYTHON:-/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BIN=$SDK/bin/x86_64-linux-clang
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang
FLOAT_IO="hidden attention_mask position_cos position_sin hidden_out"

if [ ! -f "$PART/$NAME.dlc" ]; then
  mkdir -p "$PART/float"
  time "$PY" "$SCRIPTS/moshi_guard_qairt_converter.py" --sdk-root "$SDK" --numpy-dir "$NUMPY" \
    --receipt "$PART/$NAME.convert.json" -- \
    --input_network "$PART/onnx/model.onnx" --output_path "$PART/float/$NAME.dlc" \
    --quantization_overrides "$PART/$NAME.encodings" --preserve_onnx_output_order --onnx_skip_simplification
  time env PYTHONPATH="$NUMPY:$SDK/lib/python" "$PY" "$BIN/qairt-quantizer" \
    --input_dlc "$PART/float/$NAME.dlc" --output_dlc "$PART/$NAME.dlc" \
    --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
    --preserve_io_datatype $FLOAT_IO
  rm -f "$PART/float/$NAME.dlc"
fi

"$SCRIPTS/moshi_compile_context.sh" "$PART/$NAME.dlc" "$NAME" "$PART/htp_${NAME}_$COMPILE" "$COMPILE"
