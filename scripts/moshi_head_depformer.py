# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Export and calibrate the Moshi head and the 8 DepFormer codebook graphs with AIMET.

int8 per-channel weights, int16 activations. Calibration frames come from the capture
(BF16 layer-31 output, 8 calibration clips x 32 evenly spaced frames): the head sees that
hidden state, and each codebook sees the FP32 chain's own previous token and depth cache,
which is what the phone runs. Writes WORK/<graph>/onnx/model.onnx and WORK/<graph>/<graph>.encodings.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import torch
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme
from moshi_temporal import cuda_providers, load_clips

from qai_hub_models.models.templates.moshi.explicit_cache import (
    ExplicitDepFormerStep,
    TemporalHead,
)
from qai_hub_models.models.templates.moshi.model import export_onnx, load_moshi_lm

BOOKS = 8
CACHE = [f"layer_{layer}_{kind}" for layer in range(6) for kind in ("key", "value")]
INPUTS = ["previous_token", "temporal", *CACHE]
OUTPUTS = ["audio_token", *(f"output_{name}" for name in CACHE)]


class StepWithoutLogits(torch.nn.Module):
    """The runner only reads the token and caches; dropping logits saves an output copy."""

    def __init__(self, step: ExplicitDepFormerStep) -> None:
        super().__init__()
        self.step = step

    def forward(self, previous_token, temporal, *caches):
        token, _, *updated = self.step(previous_token, temporal, *caches)
        return token, *updated


def calibrate(
    graph: torch.nn.Module,
    name: str,
    samples: list[dict],
    work: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    onnx_path = work / name / "onnx" / "model.onnx"
    example = tuple(torch.from_numpy(samples[0][n]) for n in input_names)
    export_onnx(graph, example, onnx_path, input_names, output_names)
    sim = QuantizationSimModel(
        onnx.load(str(onnx_path)),
        param_type="int8",
        activation_type="int16",
        quant_scheme=QuantScheme.min_max,
        config_file="htp_v81",
        providers=cuda_providers(),
    )
    sim.compute_encodings(samples)
    sim.export(str(work / name), name, export_model=False)
    print(f"{name}: {len(samples)} calibration frames", flush=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    lm = load_moshi_lm(model_dir=args.model_dir, device=device, dtype=torch.bfloat16)
    head = TemporalHead(lm)
    steps = [
        copy.deepcopy(ExplicitDepFormerStep(lm, k)).float().eval() for k in range(BOOKS)
    ]
    head_fp32 = copy.deepcopy(head).float().eval()
    frames = []
    for hidden in load_clips(args.capture_dir, "calibration")[:8]:
        for t in np.linspace(0, len(hidden) - 1, 32).round().astype(int):
            final = hidden[t, 4].reshape(1, 1, -1).astype(np.float32)
            temporal, logits = head(torch.from_numpy(final).to(device, torch.bfloat16))
            frames.append(
                (
                    final,
                    temporal.float().cpu().numpy(),
                    int(logits.reshape(-1).argmax()),
                )
            )
    del lm, head

    samples = [[] for _ in range(BOOKS)]
    for _, temporal, text in frames:
        previous = np.array([text], np.int64)
        caches = [np.zeros((1, 16, 8, 64), np.float32) for _ in CACHE]
        for book in range(BOOKS):
            feed = dict(zip(INPUTS, [previous, temporal, *caches], strict=False))
            samples[book].append(feed)
            token, _, *updated = steps[book](
                *(torch.from_numpy(feed[n]).to(device) for n in INPUTS)
            )
            previous, caches = (
                token.cpu().numpy().astype(np.int64),
                [c.cpu().numpy() for c in updated],
            )

    calibrate(
        head_fp32.cpu(),
        "head",
        [{"hidden": f[0]} for f in frames],
        args.work_dir,
        ["hidden"],
        ["temporal", "text_logits"],
    )
    for book in range(BOOKS):
        calibrate(
            StepWithoutLogits(steps[book].cpu()),
            f"depformer_codebook_{book}",
            samples[book],
            args.work_dir,
            INPUTS,
            OUTPUTS,
        )


if __name__ == "__main__":
    main()
