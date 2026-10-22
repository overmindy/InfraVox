# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Server paths for the Moshi build (sourced by the other moshi_*.sh scripts); override any in the environment.
: "${MOSHI_ROOT:=/data2/yejialei/moshi}"                       # bundles, pinned inputs, work dir
: "${MOSHI_BF16:=/data2/liuguohong/moshiko-pytorch-bf16}"      # kyutai/moshiko-pytorch-bf16
: "${LIBRISPEECH_TRAIN_CLEAN:=/data2/guozhengliang/LibriSpeech_full960_20261013/train-clean-100}"
: "${MOSHI_MIMI_WAV:=$MOSHI_ROOT/inputs/mimi_calibration.wav}" # Mimi A16 ranges come from its first 4 frames
: "${MOSHI_WORK:=$MOSHI_ROOT/work}"
: "${MOSHI_BUNDLE:=lm-v5}"
: "${MOSHI_GPUS:=1 2 3 4}"
: "${QAIRT_SDK_ROOT:=/data2/yejialei/sdk/qairt/2.45.0.260326}"
: "${QAIRT_NUMPY_DIR:=$MOSHI_ROOT/python-deps-numpy126}"     # the QAIRT 2.45 converter needs NumPy 1.26
: "${PY:=/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python}"  # torch, onnx, this repo (editable)
: "${AIMET_PY:=/data2/yejialei/envs/aimet228/bin/python}"            # adds aimet-onnx 2.28, onnxruntime-gpu
: "${AIMET_ENV:=/data2/yejialei/envs/aimet228/env.sh}"
