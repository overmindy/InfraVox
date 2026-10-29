# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Server paths for the MiniCPM-o build (sourced by the other minicpmo_*.sh scripts); override any in the environment.
: "${MCPO_ROOT:=/data2/yejialei/minicpmo}"                      # bundles, captures, work dir
: "${MCPO_HF:=$MCPO_ROOT/hf}"                                   # openbmb/MiniCPM-o-4_5 snapshot
: "${MCPO_REF_WAV:=$MCPO_ROOT/ref_voice.wav}"                   # assistant voice: first 6.76 s of assets/system_ref_audio.wav
: "${MCPO_CAPTURE:=$MCPO_ROOT/capture}"
: "${MCPO_WORK:=$MCPO_ROOT/work}"
: "${MCPO_BUNDLE:=v1}"
: "${MCPO_TARGET:=sm8975}"                                      # contexts are bound to one SoC
case $MCPO_TARGET in
  sm8850) : "${MCPO_SOC_MODEL:=87}" "${MCPO_DSP_ARCH:=v81}" ;;
  sm8975) : "${MCPO_SOC_MODEL:=103}" "${MCPO_DSP_ARCH:=v85}" ;;
  *) : "${MCPO_SOC_MODEL:?set for target $MCPO_TARGET}" "${MCPO_DSP_ARCH:?set for target $MCPO_TARGET}" ;;
esac
: "${MCPO_GPUS:=0 5}"
: "${QAIRT_SDK_ROOT:=/data2/yejialei/sdk/qairt/2.51.40.260930}"  # first QAIRT with HTP v85; v1 SM8850 was 2.45.0.260326
: "${QAIRT_NUMPY_DIR:=/data2/yejialei/moshi/python-deps-numpy126}"  # the QAIRT 2.45 converter needs NumPy 1.26
: "${PY:=/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python}"   # torch, onnx, this repo (editable)
: "${AIMET_PY:=/data2/yejialei/envs/aimet228/bin/python}"             # adds aimet-onnx 2.28, onnxruntime-gpu
: "${AIMET_ENV:=/data2/yejialei/envs/aimet228/env.sh}"
: "${MCPO_PY:=/data2/yejialei/envs/minicpmo/bin/python}"              # transformers 4.51 + minicpmo-utils (official code)
