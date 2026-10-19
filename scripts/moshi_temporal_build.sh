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
LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB
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

case "$COMPILE" in
  default) GRAPHS="" ;;
  o3) GRAPHS="\"graphs\": [{\"graph_names\": [\"$NAME\"], \"O\": 3}]," ;;
  o3-vtcm*) GRAPHS="\"graphs\": [{\"graph_names\": [\"$NAME\"], \"O\": 3, \"vtcm_mb\": ${COMPILE#o3-vtcm}}]," ;;
  *) echo "Unknown compile mode $COMPILE" >&2; exit 2 ;;
esac
OUT=$PART/htp_${NAME}_$COMPILE
mkdir -p "$OUT"
echo "{$GRAPHS \"devices\": [{\"soc_model\": 87, \"dsp_arch\": \"v81\"}]}" > "$OUT/backend.json"
cat > "$OUT/config.json" <<EOF
{"backend_extensions": {"shared_library_path": "$LIB/libQnnHtpNetRunExtensions.so",
                        "config_file_path": "$OUT/backend.json"}}
EOF
time "$BIN/qnn-context-binary-generator" --backend "$LIB/libQnnHtp.so" --model "$LIB/libQnnModelDlc.so" \
  --dlc_path "$PART/$NAME.dlc" --config_file "$OUT/config.json" \
  --binary_file "$NAME" --output_dir "$OUT" --log_level warn 2>&1 | grep -E "spill_bytes|fill_bytes|read_total" || true
"$BIN/qnn-context-binary-utility" --context_binary "$OUT/$NAME.bin" --json_file "$OUT/context.json" >/dev/null
grep -o '"graphName"[^,]*\|"optimizationLevel"[^,]*\|"vtcmSize[^,]*' "$OUT/context.json" | tr '\n' ' '; echo
ls -la "$OUT/$NAME.bin"
sha256sum "$OUT/$NAME.bin"
