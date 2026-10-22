# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

import onnx
import torch

from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.lm import (
    LMModel,
)
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.loaders import (
    CheckpointInfo,
)
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import (
    no_compile,
)

FRAME_SAMPLES = 1920
DEFAULT_HF_REPO = "kyutai/moshiko-pytorch-bf16"
MODEL_WEIGHTS_NAME = "model.safetensors"
MIMI_WEIGHTS_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_NAME = "tokenizer_spm_32k_3.model"


def resolve_moshi_checkpoint(
    hf_repo: str = DEFAULT_HF_REPO,
    model_dir: str | Path | None = None,
) -> CheckpointInfo:
    """Resolve local or Hugging Face checkpoint files without loading weights."""
    if model_dir is None:
        return CheckpointInfo.from_hf_repo(hf_repo)

    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Moshi model directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    moshi_weights = model_path / config.get("moshi_name", MODEL_WEIGHTS_NAME)
    mimi_weights = model_path / config.get("mimi_name", MIMI_WEIGHTS_NAME)
    tokenizer = model_path / config.get("tokenizer_name", TOKENIZER_NAME)
    missing = [
        str(p) for p in (moshi_weights, mimi_weights, tokenizer) if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing files in local Moshi directory: {', '.join(missing)}"
        )
    if not config_path.is_file():
        return CheckpointInfo(moshi_weights, mimi_weights, tokenizer)

    mimi_config_name = config.get("mimi_config_name")
    mimi_config_path = model_path / mimi_config_name if mimi_config_name else None
    lora_name = config.get("lora_name")
    lora_path = model_path / lora_name if lora_name else None
    missing_optional = [
        str(p) for p in (mimi_config_path, lora_path) if p and not p.is_file()
    ]
    if missing_optional:
        raise FileNotFoundError(
            "Missing files referenced by config.json: " + ", ".join(missing_optional)
        )
    return CheckpointInfo.from_hf_repo(
        hf_repo,
        moshi_weights=moshi_weights,
        mimi_weights=mimi_weights,
        tokenizer=tokenizer,
        config_path=config_path,
        mimi_config_path=mimi_config_path,
        lora_weights=lora_path,
    )


def load_moshi_lm(
    hf_repo: str = DEFAULT_HF_REPO,
    model_dir: str | Path | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LMModel:
    checkpoint = resolve_moshi_checkpoint(hf_repo=hf_repo, model_dir=model_dir)
    return checkpoint.get_moshi(device=device, dtype=dtype)


def export_onnx(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    # Kyutai wraps RMSNorm and a few eager functions with torch.compile, which the
    # TorchScript-based ONNX exporter cannot trace into.
    path.parent.mkdir(parents=True, exist_ok=True)
    with no_compile():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )
    onnx.checker.check_model(str(path))
