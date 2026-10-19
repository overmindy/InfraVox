"""Export one Moshi Temporal part (consecutive layers) in the Qualcomm LLM graph layout.

Mirrors templates/llm: per-head 1x1-conv projections, host-owned KV inputs
(past_key [H,1,D,P], past_value [H,1,P,D]), an additive attention mask and RoPE
cos/sin supplied by the host, and only the new key/value as outputs. Moshi's
interleaved RoPE becomes half-split by permuting q/k rows inside every head.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from qai_hub_models.models.templates.moshi.explicit_cache import TemporalBlock, TemporalFrontend
from qai_hub_models.models.templates.moshi.model import load_moshi_lm

HEADS, HEAD_DIM, DIM = 32, 128, 4096
PAST = 999  # 999 past slots + the current token = Moshi's 1000-frame window.
MASK_MIN = -50.0  # Same clip as the official w4a16 LLM recipes.
MAX_PERIOD = 10000.0


def rope_tables(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.exp(torch.arange(HEAD_DIM // 2, dtype=torch.float32) * (-math.log(MAX_PERIOD) * 2 / HEAD_DIM))
    angle = positions.float()[:, None] * freqs[None]
    return angle.cos().reshape(-1, 1, 1, 1, HEAD_DIM // 2), angle.sin().reshape(-1, 1, 1, 1, HEAD_DIM // 2)


class Linear1x1(nn.Module):
    """Same math as a 1x1 conv on [1, C, 1, 1], exported as MatMul/Gemm (FullyConnected)."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.linear(x.reshape(1, -1), self.weight).reshape(1, -1, 1, 1)


PROJECTION = "conv"


def conv(weight: torch.Tensor) -> nn.Module:
    if PROJECTION == "linear":
        return Linear1x1(weight)
    layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=False)
    layer.weight.data.copy_(weight[:, :, None, None])
    return layer


class RMSNorm(nn.Module):
    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.eps = float(norm.eps)
        self.alpha = nn.Parameter(norm.alpha.detach().float().reshape(1, DIM, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps) * self.alpha


class PartLayer(nn.Module):
    """One Temporal layer on a [1, DIM, 1, 1] residual stream."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        attention = layer.self_attn
        projection, output = attention.in_projs[0], attention.out_projs[0]
        if projection.bias is not None or output.bias is not None or attention.kv_repeat != 1:
            raise ValueError("Expected bias-free multi-head Temporal attention")
        if layer.gating is None or not isinstance(layer.layer_scale_1, nn.Identity):
            raise ValueError("Expected SiLU gating and no layer scale")
        query, key, value = projection.weight.detach().float().split(DIM)
        # Interleaved (2i, 2i+1) RoPE pairs -> half-split (i, i + D/2) per head.
        order = torch.cat([torch.arange(0, HEAD_DIM, 2), torch.arange(1, HEAD_DIM, 2)])

        def per_head(weight: torch.Tensor, permute: bool) -> list[nn.Conv2d]:
            heads = weight.reshape(HEADS, HEAD_DIM, DIM)
            return [conv(heads[h][order] if permute else heads[h]) for h in range(HEADS)]

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
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    def forward(self, x, mask, cos, sin, past_key, past_value):
        normed = self.norm1(x)
        past_keys = torch.split(past_key, 1, dim=0)
        past_values = torch.split(past_value, 1, dim=0)
        heads, new_keys, new_values = [], [], []
        for h in range(HEADS):
            q = self.rope(self.query[h](normed).reshape(1, 1, 1, HEAD_DIM), cos, sin)
            k = self.rope(self.key[h](normed).reshape(1, 1, 1, HEAD_DIM), cos, sin).transpose(2, 3)
            v = self.value[h](normed).reshape(1, 1, 1, HEAD_DIM)
            new_keys.append(k)
            new_values.append(v)
            scores = torch.matmul(q, torch.cat((past_keys[h], k), dim=3)) * self.scale + mask
            heads.append(torch.matmul(torch.softmax(scores, dim=-1), torch.cat((past_values[h], v), dim=2)))
        attended = torch.cat(heads, dim=3).reshape(1, DIM, 1, 1)
        x = x + self.output(attended)
        normed = self.norm2(x)
        x = x + self.down(functional.silu(self.gate(normed)) * self.up(normed))
        return x, torch.cat(new_keys, dim=0), torch.cat(new_values, dim=0)


class TemporalPart(nn.Module):
    def __init__(self, lm: nn.Module, first: int, count: int) -> None:
        super().__init__()
        self.first = first
        self.layers = nn.ModuleList(PartLayer(lm.transformer.layers[i]) for i in range(first, first + count))

    def forward(self, hidden, attention_mask, position_cos, position_sin, *past):
        x = hidden.reshape(1, DIM, 1, 1)
        outputs = []
        for i, layer in enumerate(self.layers):
            x, key, value = layer(x, attention_mask, position_cos, position_sin, past[2 * i], past[2 * i + 1])
            outputs += [key, value]
        return (x.reshape(1, 1, DIM), *outputs)

    def names(self) -> tuple[list[str], list[str]]:
        layers = range(self.first, self.first + len(self.layers))
        inputs = ["hidden", "attention_mask", "position_cos", "position_sin"]
        outputs = ["hidden_out"]
        for i in layers:
            inputs += [f"past_key_{i}_in", f"past_value_{i}_in"]
            outputs += [f"past_key_{i}_out", f"past_value_{i}_out"]
        return inputs, outputs


class HostRing:
    """Host-side KV window in the graph layout; the slot for position t is t % PAST."""

    def __init__(self, layers: int, device: torch.device) -> None:
        self.keys = [torch.zeros(HEADS, 1, HEAD_DIM, PAST, device=device) for _ in range(layers)]
        self.values = [torch.zeros(HEADS, 1, PAST, HEAD_DIM, device=device) for _ in range(layers)]

    def inputs(self, position: int, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mask = torch.full((1, 1, 1, PAST + 1), MASK_MIN, device=device)
        mask[..., PAST] = 0
        valid = min(position, PAST)
        for back in range(1, valid + 1):
            mask[..., (position - back) % PAST] = 0
        past = [t for pair in zip(self.keys, self.values) for t in pair]
        return mask, past

    def write(self, position: int, outputs: tuple[torch.Tensor, ...]) -> None:
        slot = position % PAST
        for i in range(len(self.keys)):
            self.keys[i][..., slot] = outputs[2 * i][..., 0]
            self.values[i][:, :, slot] = outputs[2 * i + 1][:, :, 0]


def random_hidden(lm: nn.Module, steps: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    frontend = TemporalFrontend(lm).to(device).float()
    sequence = torch.cat([torch.randint(0, 32000, (steps, 1), generator=generator),
                          torch.randint(0, 2048, (steps, 16), generator=generator)], dim=1)
    with torch.no_grad():
        return torch.stack([frontend(sequence[t : t + 1, :, None].to(device)).reshape(1, 1, DIM) for t in range(steps)])


@torch.no_grad()
def run_part(part: TemporalPart, hidden: torch.Tensor, keep_inputs: int = 0):
    device = hidden.device
    ring = HostRing(len(part.layers), device)
    cos, sin = rope_tables(torch.arange(hidden.shape[0]))
    outputs, samples = [], []
    for t in range(hidden.shape[0]):
        mask, past = ring.inputs(t, device)
        inputs = (hidden[t], mask, cos[t].to(device), sin[t].to(device), *past)
        if t >= hidden.shape[0] - keep_inputs:
            samples.append([x.detach().cpu().numpy().copy() for x in inputs])
        result = part(*inputs)
        ring.write(t, result[1:])
        outputs.append(result[0])
    return torch.stack(outputs), samples


@torch.no_grad()
def run_reference(lm: nn.Module, first: int, count: int, hidden: torch.Tensor) -> torch.Tensor:
    blocks = [TemporalBlock(lm.transformer.layers[i]).to(hidden.device).float() for i in range(first, first + count)]
    caches = [block.empty_cache().float() for block in blocks]
    outputs = []
    for t in range(hidden.shape[0]):
        x = hidden[t]
        position = torch.tensor([t], device=hidden.device)
        for i, block in enumerate(blocks):
            x, caches[i] = block(x, caches[i], position)
        outputs.append(x)
    return torch.stack(outputs)


def relative_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).float().pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--parity-steps", type=int, default=12)
    parser.add_argument("--calibration-steps", type=int, default=0, help="Save the last N step inputs as .npz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projection", choices=("conv", "linear"), default="conv")
    args = parser.parse_args()
    global PROJECTION
    PROJECTION = args.projection
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"first_layer": args.first_layer, "layers": args.layers, "past": PAST, "mask_min": MASK_MIN,
              "projection": args.projection}

    start = time.monotonic()
    lm = load_moshi_lm(model_dir=args.model_dir, device="cpu", dtype=torch.bfloat16)
    report["load_s"] = time.monotonic() - start

    part = TemporalPart(lm, args.first_layer, args.layers).to(device).eval()
    steps = max(args.parity_steps, args.calibration_steps)
    hidden = random_hidden(lm, steps, seed=0, device=device)
    if args.first_layer:
        hidden = run_reference(lm, 0, args.first_layer, hidden)
    ours, samples = run_part(part, hidden, keep_inputs=args.calibration_steps)
    reference = run_reference(lm, args.first_layer, args.layers, hidden)
    report["parity_relative_rms"] = [relative_rms(ours[t], reference[t]) for t in range(steps)]
    print("parity relative RMS per step:", report["parity_relative_rms"], flush=True)
    if max(report["parity_relative_rms"]) > 1e-4:
        raise SystemExit("Rewritten part does not match the TemporalBlock reference")
    if samples:
        input_names, _ = part.names()
        for index, sample in enumerate(samples):
            # Unfilled KV slots are zeros, so compression keeps early-step samples small.
            np.savez_compressed(args.output_dir / f"calibration_{index:03d}.npz", **dict(zip(input_names, sample)))

    part = part.cpu()
    mask, past = HostRing(args.layers, torch.device("cpu")).inputs(0, torch.device("cpu"))
    cos, sin = rope_tables(torch.zeros(1))
    example = (torch.zeros(1, 1, DIM), mask, cos[0], sin[0], *past)
    input_names, output_names = part.names()
    onnx_path = args.output_dir / "onnx" / "model.onnx"
    onnx_path.parent.mkdir()
    start = time.monotonic()
    torch.onnx.export(part, example, str(onnx_path), input_names=input_names, output_names=output_names,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    report["onnx_export_s"] = time.monotonic() - start
    report["onnx_bytes"] = sum(p.stat().st_size for p in onnx_path.parent.iterdir())
    (args.output_dir / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
