#!/bin/bash
# Compile a quantized DLC into an SM8850 (HTP v81) context binary OUT_DIR/NAME.bin.
# Usage: moshi_compile_context.sh DLC NAME OUT_DIR [COMPILE]
#   COMPILE: o3-vtcm8 (default; no spill on SM8850), o3, o3-vtcm<N>, or default (no graph options).
# Graph options only apply when graph_names matches the graph inside the DLC, so it is read from the DLC.
set -euo pipefail
DLC=$1; NAME=$2; OUT=$3; COMPILE=${4:-o3-vtcm8}
SDK=${QAIRT_SDK_ROOT:-/data2/yejialei/sdk/qairt/2.45.0.260326}
NUMPY=${QAIRT_NUMPY_DIR:-/data2/yejialei/tmp/moshi-qdq-0-1-cpu-check-v1/dlc_fusion_diagnostics/python-deps-numpy126}
PY=${QAIRT_PYTHON:-/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python}
BIN=$SDK/bin/x86_64-linux-clang
LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB

GRAPH=$(env PYTHONPATH="$NUMPY:$SDK/lib/python" "$PY" "$BIN/qairt-dlc-info" -i "$DLC" 2>/dev/null \
  | sed -n 's/^Info of graph: //p' | head -n 1)
[ -n "$GRAPH" ] || { echo "No graph name found in $DLC" >&2; exit 1; }
case "$COMPILE" in
  default) GRAPHS="" ;;
  o3) GRAPHS="\"graphs\": [{\"graph_names\": [\"$GRAPH\"], \"O\": 3}]," ;;
  o3-vtcm*) GRAPHS="\"graphs\": [{\"graph_names\": [\"$GRAPH\"], \"O\": 3, \"vtcm_mb\": ${COMPILE#o3-vtcm}}]," ;;
  *) echo "Unknown compile mode $COMPILE" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
echo "{$GRAPHS \"devices\": [{\"soc_model\": 87, \"dsp_arch\": \"v81\"}]}" > "$OUT/backend.json"
cat > "$OUT/config.json" <<EOF
{"backend_extensions": {"shared_library_path": "$LIB/libQnnHtpNetRunExtensions.so",
                        "config_file_path": "$OUT/backend.json"}}
EOF
time "$BIN/qnn-context-binary-generator" --backend "$LIB/libQnnHtp.so" --model "$LIB/libQnnModelDlc.so" \
  --dlc_path "$DLC" --config_file "$OUT/config.json" \
  --binary_file "$NAME" --output_dir "$OUT" --log_level warn 2>&1 | grep -E "spill_bytes|fill_bytes|read_total" || true
"$BIN/qnn-context-binary-utility" --context_binary "$OUT/$NAME.bin" --json_file "$OUT/$NAME.context.json" >/dev/null
grep -o '"graphName"[^,]*\|"optimizationLevel"[^,]*\|"vtcmSize[^,]*' "$OUT/$NAME.context.json" | tr '\n' ' '; echo
ls -la "$OUT/$NAME.bin"
sha256sum "$OUT/$NAME.bin"
