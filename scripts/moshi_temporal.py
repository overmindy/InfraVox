# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Moshi Temporal parts (8 consecutive layers each) in the Qualcomm LLM graph layout.

Mirrors templates/llm: per-head 1x1-conv projections, host-owned KV inputs
(past_key [H,1,D,P], past_value [H,1,P,D]), an additive attention mask and RoPE
cos/sin supplied by the host, and only the new key/value as outputs. Moshi's
interleaved RoPE becomes half-split by permuting q/k rows inside every head.

Exports one part to WORK/temporal_part_P/onnx and calibrates it with AIMET on captured
speech: int4 LPBQ weights (block 64), int16 activations, 8-bit KV with in/out tied.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import aimet_onnx.common.quantsim as qs
import numpy as np
import onnx
import torch
from aimet_onnx import quantsim
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme, set_lpbq_for_params
from torch import nn
from torch.nn import functional

from qai_hub_models.models.templates.llm._utils import (
    _get_kv_io_map,
    _set_matmul_second_input_to_8b,
    _set_tensors_to_output_8b_sym,
    _tie_quantizers_for_kv_cache,
)
from qai_hub_models.models.templates.moshi.explicit_cache import TemporalBlock
from qai_hub_models.models.templates.moshi.model import load_moshi_lm

HEADS, HEAD_DIM, DIM = 32, 128, 4096
LAYERS_PER_PART, PARTS = 8, 4
PAST = 999  # 999 past slots + the current token = Moshi's 1000-frame window.
MASK_MIN = -50.0  # Same clip as the official w4a16 LLM recipes.
MAX_PERIOD = 10000.0


def rope_tables(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.exp(
        torch.arange(HEAD_DIM // 2, dtype=torch.float32)
        * (-math.log(MAX_PERIOD) * 2 / HEAD_DIM)
    )
    angle = positions.float()[:, None] * freqs[None]
    return angle.cos().reshape(-1, 1, 1, 1, HEAD_DIM // 2), angle.sin().reshape(
        -1, 1, 1, 1, HEAD_DIM // 2
    )


def conv(weight: torch.Tensor) -> nn.Conv2d:
    # LPBQ only works on Conv: an LPBQ FullyConnected crashes the QAIRT 2.45 context generator.
    layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=False)
    layer.weight.data.copy_(weight[:, :, None, None])
    return layer


class RMSNorm(nn.Module):
    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.eps = float(norm.eps)
        self.alpha = nn.Parameter(norm.alpha.detach().float().reshape(1, DIM, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps) * self.alpha
        )


class PartLayer(nn.Module):
    """One Temporal layer on a [1, DIM, 1, 1] residual stream."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        attention = layer.self_attn
        projection, output = attention.in_projs[0], attention.out_projs[0]
        if (
            projection.bias is not None
            or output.bias is not None
            or attention.kv_repeat != 1
        ):
            raise ValueError("Expected bias-free multi-head Temporal attention")
        if layer.gating is None or not isinstance(layer.layer_scale_1, nn.Identity):
            raise ValueError("Expected SiLU gating and no layer scale")
        query, key, value = projection.weight.detach().float().split(DIM)
        # Interleaved (2i, 2i+1) RoPE pairs -> half-split (i, i + D/2) per head.
        order = torch.cat([torch.arange(0, HEAD_DIM, 2), torch.arange(1, HEAD_DIM, 2)])

        def per_head(weight: torch.Tensor, permute: bool) -> list[nn.Conv2d]:
            heads = weight.reshape(HEADS, HEAD_DIM, DIM)
            return [
                conv(heads[h][order] if permute else heads[h]) for h in range(HEADS)
            ]

        self.norm1, self.norm2 = RMSNorm(layer.norm1), RMSNorm(layer.norm2)
        self.query = nn.ModuleList(per_head(query, True))
        self.key = nn.ModuleList(per_head(key, True))
        self.value = nn.ModuleList(per_head(value, False))
        self.output = conv(output.weight.detach().float())
        gate_in = layer.gating.linear_in.weight.detach().float()
        hidden = gate_in.shape[0] // 2
        self.gate, self.up = conv(gate_in[:hidden]), conv(gate_in[hidden:])
        self.down = conv(layer.gating.linear_out.weight.detach().float())
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    @staticmethod
    def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        first, second = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return torch.cat(
            (first * cos - second * sin, first * sin + second * cos), dim=-1
        )

    def forward(self, x, mask, cos, sin, past_key, past_value):
        normed = self.norm1(x)
        past_keys = torch.split(past_key, 1, dim=0)
        past_values = torch.split(past_value, 1, dim=0)
        heads, new_keys, new_values = [], [], []
        for h in range(HEADS):
            q = self.rope(self.query[h](normed).reshape(1, 1, 1, HEAD_DIM), cos, sin)
            k = self.rope(
                self.key[h](normed).reshape(1, 1, 1, HEAD_DIM), cos, sin
            ).transpose(2, 3)
            v = self.value[h](normed).reshape(1, 1, 1, HEAD_DIM)
            new_keys.append(k)
            new_values.append(v)
            scores = (
                torch.matmul(q, torch.cat((past_keys[h], k), dim=3)) * self.scale + mask
            )
            heads.append(
                torch.matmul(
                    torch.softmax(scores, dim=-1), torch.cat((past_values[h], v), dim=2)
                )
            )
        attended = torch.cat(heads, dim=3).reshape(1, DIM, 1, 1)
        x = x + self.output(attended)
        normed = self.norm2(x)
        x = x + self.down(functional.silu(self.gate(normed)) * self.up(normed))
        return x, torch.cat(new_keys, dim=0), torch.cat(new_values, dim=0)


class TemporalPart(nn.Module):
    def __init__(self, lm: nn.Module, part: int) -> None:
        super().__init__()
        self.first = part * LAYERS_PER_PART
        self.layers = nn.ModuleList(
            PartLayer(lm.transformer.layers[i])
            for i in range(self.first, self.first + LAYERS_PER_PART)
        )

    def forward(self, hidden, attention_mask, position_cos, position_sin, *past):
        x = hidden.reshape(1, DIM, 1, 1)
        outputs = []
        for i, layer in enumerate(self.layers):
            x, key, value = layer(
                x,
                attention_mask,
                position_cos,
                position_sin,
                past[2 * i],
                past[2 * i + 1],
            )
            outputs += [key, value]
        return (x.reshape(1, 1, DIM), *outputs)


def part_names(part: int) -> tuple[list[str], list[str]]:
    inputs = ["hidden", "attention_mask", "position_cos", "position_sin"]
    outputs = ["hidden_out"]
    for i in range(part * LAYERS_PER_PART, (part + 1) * LAYERS_PER_PART):
        inputs += [f"past_key_{i}_in", f"past_value_{i}_in"]
        outputs += [f"past_key_{i}_out", f"past_value_{i}_out"]
    return inputs, outputs


class HostRing:
    """Host-side KV window in the graph layout; the slot for position t is t % PAST."""

    def __init__(self, device: torch.device) -> None:
        self.keys = [
            torch.zeros(HEADS, 1, HEAD_DIM, PAST, device=device)
            for _ in range(LAYERS_PER_PART)
        ]
        self.values = [
            torch.zeros(HEADS, 1, PAST, HEAD_DIM, device=device)
            for _ in range(LAYERS_PER_PART)
        ]

    def inputs(self, position: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mask = torch.full((1, 1, 1, PAST + 1), MASK_MIN, device=self.keys[0].device)
        mask[..., PAST] = 0
        if position >= PAST:
            mask[..., :PAST] = 0
        else:
            mask[..., :position] = (
                0  # Slots 0..t-1 hold positions 0..t-1 until the ring wraps.
            )
        return mask, [
            t for pair in zip(self.keys, self.values, strict=False) for t in pair
        ]

    def write(self, position: int, outputs) -> None:
        slot = position % PAST
        for i in range(LAYERS_PER_PART):
            self.keys[i][..., slot] = outputs[2 * i][..., 0]
            self.values[i][:, :, slot] = outputs[2 * i + 1][:, :, 0]


@torch.no_grad()
def iterate(
    step, hidden: np.ndarray, device: torch.device, keep: set[int] = frozenset()
):
    """AR1 over one clip with a host ring; `step` maps graph inputs to outputs (torch or QuantSim).

    Returns the per-frame hidden_out and the full graph inputs at the `keep` steps.
    """
    ring = HostRing(device)
    cos, sin = rope_tables(torch.arange(len(hidden)))
    outputs, samples = [], []
    for t in range(len(hidden)):
        mask, past = ring.inputs(t)
        x = (
            torch.from_numpy(np.ascontiguousarray(hidden[t], np.float32))
            .to(device)
            .reshape(1, 1, -1)
        )
        inputs = (x, mask, cos[t].to(device), sin[t].to(device), *past)
        if t in keep:
            samples.append([v.cpu().numpy() for v in inputs])
        result = step(inputs)
        ring.write(t, result[1:])
        outputs.append(result[0].reshape(-1).cpu().numpy())
    return np.stack(outputs), samples


def cuda_providers() -> list:
    # TF32 matmuls perturb the simulation by ~1e-4 relative; keep full FP32. MOSHI_ORT_DEVICE puts
    # the QuantSim on another GPU than the BF16 model when one GPU cannot hold both.
    options = {"use_tf32": "0", "device_id": os.environ.get("MOSHI_ORT_DEVICE", "0")}
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]


def lpbq_four_bit(sim: QuantizationSimModel, block_size: int) -> None:
    four_bit = set()
    for op in sim.connected_graph.ordered_ops:
        weight = (
            sim.get_op_quantizers(op)[2].get("weight")
            if op.type in {"Conv", "Gemm", "MatMul"}
            else None
        )
        if weight is not None and weight.bitwidth == 4:
            four_bit.add(op.name)
    set_lpbq_for_params(
        sim, bitwidth=4, block_size=block_size, nodes_to_include=four_bit
    )


def create_sim(model: onnx.ModelProto) -> QuantizationSimModel:
    """QuantSim configured like the official w4a16 LLM recipes, before any calibration."""
    AimetLogger.set_level_for_all_areas(logging.WARNING)
    quantsim.op_types_to_tie_qtzrs = ["Concat"]
    quantsim._tie_qtzrs = True
    if "Slice" not in quantsim.op_outputs_to_ignore:
        quantsim.op_outputs_to_ignore += ["Slice", "Constant"]
    qs.encoding_version = "1.0.0"
    sim = QuantizationSimModel(
        model,
        param_type="int4",
        activation_type="int16",
        quant_scheme=QuantScheme.min_max,
        config_file="htp_v81",
        providers=cuda_providers(),
    )
    lpbq_four_bit(sim, 64)
    kv_io_map = _get_kv_io_map(sim)
    sim._tie_quantizers_for_op_types(["Concat"])
    sim._rebuild_session()
    _set_tensors_to_output_8b_sym(sim, list(kv_io_map) + list(kv_io_map.values()))
    _tie_quantizers_for_kv_cache(sim, kv_io_map)
    _set_matmul_second_input_to_8b(sim)
    return sim


def load_clips(capture: Path, split: str) -> list[np.ndarray]:
    """Clips in capture order, which fixes which ones calibrate."""
    manifest = json.loads((capture / "manifest.json").read_text())
    return [
        np.load(capture / f"{c['name']}.npz")["hidden"]
        for c in manifest
        if c["split"] == split
    ]


def export_onnx(part: TemporalPart, index: int, onnx_path: Path) -> None:
    cpu = torch.device("cpu")
    part = part.to(cpu)
    mask, past = HostRing(cpu).inputs(0)
    cos, sin = rope_tables(torch.zeros(1))
    input_names, output_names = part_names(index)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        part,
        (torch.zeros(1, 1, DIM), mask, cos[0], sin[0], *past),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


@torch.no_grad()
def reference_layers(
    lm, first: int, hidden: np.ndarray, device: torch.device
) -> np.ndarray:
    blocks = [
        TemporalBlock(lm.transformer.layers[i]).to(device).float()
        for i in range(first, first + LAYERS_PER_PART)
    ]
    caches = [block.empty_cache().float() for block in blocks]
    outputs = []
    for t in range(len(hidden)):
        x = torch.from_numpy(hidden[t]).to(device).reshape(1, 1, -1)
        for i, block in enumerate(blocks):
            x, caches[i] = block(x, caches[i], torch.tensor([t], device=device))
        outputs.append(x.reshape(-1).cpu().numpy())
    return np.stack(outputs)


def build(args: argparse.Namespace) -> None:
    """Calibration samples are AR1 steps of the float part replayed over 8 calibration clips (first
    120 frames, 64 steps). Steps 0-3 are always kept: frame 0 (all-initial tokens) is an attention
    sink for every later step, and clipping its K/V corrupts the whole clip.
    """
    device = torch.device(args.device)
    name = f"temporal_part_{args.part}"
    out = args.work_dir / name
    lm = load_moshi_lm(model_dir=args.model_dir, device="cpu", dtype=torch.bfloat16)
    part = TemporalPart(lm, args.part).to(device).eval()
    clips = [h[:120, args.part] for h in load_clips(args.capture_dir, "calibration")][
        :8
    ]

    ours, _ = iterate(lambda inputs: part(*inputs), clips[0][:4], device)
    reference = reference_layers(lm, part.first, clips[0][:4], device)
    del lm
    parity = float(
        np.max(
            np.linalg.norm(ours - reference, axis=1) / np.linalg.norm(reference, axis=1)
        )
    )
    print(f"{name}: parity relative error vs TemporalBlock {parity:.2e}", flush=True)
    if parity > 1e-4:
        raise SystemExit("Rewritten part does not match the TemporalBlock reference")

    per_clip = math.ceil(64 / len(clips))
    samples = []
    for hidden in clips:
        keep = {0, 1, 2, 3} | set(
            np.linspace(4, len(hidden) - 1, max(per_clip - 4, 1))
            .round()
            .astype(int)
            .tolist()
        )
        samples += iterate(lambda inputs: part(*inputs), hidden, device, keep)[1]
    names, _ = part_names(args.part)
    onnx_path = out / "onnx" / "model.onnx"
    export_onnx(part, args.part, onnx_path)
    sim = create_sim(onnx.load(str(onnx_path)))
    sim.compute_encodings([dict(zip(names, s, strict=False)) for s in samples])
    sim.export(str(out), name, export_model=False)
    print(
        f"{name}: {len(samples)} calibration steps -> {out / (name + '.encodings')}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=range(PARTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    start = time.monotonic()
    build(args)
    print(
        f"temporal_part_{args.part} done in {time.monotonic() - start:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
