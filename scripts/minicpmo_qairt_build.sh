#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Convert + quantize graphs and compile them into one context OUT.bin for $MCPO_SOC_MODEL / $MCPO_DSP_ARCH.
# Usage: minicpmo_qairt_build.sh OUT GRAPH [GRAPH...]
#   Each GRAPH reads $MCPO_WORK/GRAPH/onnx/model.onnx and $MCPO_WORK/GRAPH/GRAPH.encodings.
#   Several graphs (AR-1 + AR-16 of one LLM part) share their weights inside the context.
# Graph names equal the DLC names so the O3 + VTCM 8 MB options apply.
set -euo pipefail
OUT=$1; shift
S=$(cd "$(dirname "$0")" && pwd)
source "$S/minicpmo_env.sh"
BIN=$QAIRT_SDK_ROOT/bin/x86_64-linux-clang
LIB=$QAIRT_SDK_ROOT/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB
DLCS=()
for NAME in "$@"; do
  DIR=$MCPO_WORK/$NAME
  if [ ! -f "$DIR/$NAME.dlc" ]; then
    mkdir -p "$DIR/float"
    "$PY" "$S/moshi_guard_qairt_converter.py" --sdk-root "$QAIRT_SDK_ROOT" --numpy-dir "$QAIRT_NUMPY_DIR" -- \
      --input_network "$DIR/onnx/model.onnx" --output_path "$DIR/float/$NAME.dlc" \
      --quantization_overrides "$DIR/$NAME.encodings" --preserve_onnx_output_order --onnx_skip_simplification
    CALIB=()  # tensors AIMET left unencoded (inside its fused groups) get calibrated here
    [ -f "$DIR/calib/input_list.txt" ] && CALIB=(--input_list "$DIR/calib/input_list.txt")
    env PYTHONPATH="$QAIRT_NUMPY_DIR:$QAIRT_SDK_ROOT/lib/python" "$PY" "$BIN/qairt-quantizer" \
      --input_dlc "$DIR/float/$NAME.dlc" --output_dlc "$DIR/$NAME.dlc" --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
      ${CALIB[@]+"${CALIB[@]}"}
    rm -rf "$DIR/float"
  fi
  DLCS+=("$DIR/$NAME.dlc")
done

HTP=$(dirname "$OUT")/htp_$(basename "$OUT" .bin)
mkdir -p "$HTP"
GRAPHS=$(printf '"%s",' "$@"); GRAPHS=${GRAPHS%,}
SHARE=false; [ $# -gt 1 ] && SHARE=true
echo "{\"graphs\": [{\"graph_names\": [$GRAPHS], \"O\": 3, \"vtcm_mb\": 8}], \"devices\": [{\"soc_model\": $MCPO_SOC_MODEL, \"dsp_arch\": \"$MCPO_DSP_ARCH\"}], \"context\": {\"weight_sharing_enabled\": $SHARE}}" \
  > "$HTP/backend.json"
echo "{\"backend_extensions\": {\"shared_library_path\": \"$LIB/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$HTP/backend.json\"}}" \
  > "$HTP/config.json"
"$BIN/qnn-context-binary-generator" --backend "$LIB/libQnnHtp.so" --model "$LIB/libQnnModelDlc.so" \
  --dlc_path "$(IFS=,; echo "${DLCS[*]}")" --config_file "$HTP/config.json" --binary_file "$(basename "$OUT" .bin)" \
  --output_dir "$HTP" --log_level warn
"$BIN/qnn-context-binary-utility" --context_binary "$HTP/$(basename "$OUT" .bin).bin" --json_file "$HTP/info.json" > /dev/null
for NAME in "$@"; do
  grep -q "\"graphName\": *\"$NAME\"" "$HTP/info.json" || { echo "$OUT: graph $NAME missing (O3/VTCM would not apply)" >&2; exit 1; }
done
mv "$HTP/$(basename "$OUT" .bin).bin" "$OUT"
mv "$HTP/info.json" "${OUT%.bin}.json"
rm -rf "$HTP"
