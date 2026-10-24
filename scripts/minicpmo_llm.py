# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""MiniCPM-o 4.5 LLM (Qwen3-8B) as 4 parts of 9 layers plus a head, in the Qualcomm LLM graph layout.

Every part exists as an AR-1 graph (decode) and an AR-16 graph (prefill) that share weights and
the host KV ring: past_key [8,1,128,P], past_value [8,1,P,128] per layer with P = 2032 slots, an
additive mask [1,1,N,P+N] and RoPE cos/sin [1,1,N,64] from the host, and only the new K/V as
outputs. The head graph is the final RMSNorm, lm_head and the TTS semantic projector.

Steps (each resumable):
  boundaries  BF16 LLM over the captured sessions -> part inputs at layers 0/9/18/27
  part P      parity check, ONNX for AR-16 and AR-1, AIMET calibration of AR-16 (LPBQ W4 / A16 /
              8-bit KV); AR-1 reuses those encodings so both graphs quantize KV identically
  head        ONNX + AIMET (W8 / A16) on part-3 outputs
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
from safetensors import safe_open
from torch import nn
from torch.nn import functional
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from qai_hub_models.models.templates.llm._utils import (
    _get_kv_io_map,
    _set_matmul_second_input_to_8b,
    _set_tensors_to_output_8b_sym,
    _tie_quantizers_for_kv_cache,
)

DIM, HEADS, KV_HEADS, HEAD_DIM, FFN = 4096, 32, 8, 128, 12288
LAYERS, PARTS = 36, 4
LAYERS_PER_PART = LAYERS // PARTS
PAST = 2032  # AR-16 attends 2032 past slots + 16 new = a 2048 window.
AR_N = 16
ROPE_THETA = 1e6
EPS = 1e-6
VOCAB = 151748
TTS_DIM = 768
MASK_MIN = -50.0


class Weights:
    """Lazy reader for the sharded MiniCPM-o checkpoint (keys as in model.safetensors.index.json)."""

    def __init__(self, model_dir: Path) -> None:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.dir, self.index, self.files = model_dir, index, {}

    def __call__(self, key: str) -> torch.Tensor:
        name = self.index[key]
        if name not in self.files:
            self.files[name] = safe_open(str(self.dir / name), framework="pt")
        return self.files[name].get_tensor(key).float()


def rope_tables(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    angle = positions.double()[:, None] * freqs[None]
    return angle.cos().float(), angle.sin().float()  # [T, 64] each


def conv(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Conv2d:
    # LPBQ only works on Conv: an LPBQ FullyConnected crashes the QAIRT 2.45 context generator.
    layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    layer.weight.data.copy_(weight[:, :, None, None])
    if bias is not None:
        layer.bias.data.copy_(bias)
    return layer


class ChannelNorm(nn.Module):
    """RMSNorm over dim 1 of [1, C, 1, N]."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.reshape(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS) * self.weight


class HeadNorm(nn.Module):
    """Qwen3 q/k RMSNorm over the last (head) dim."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * self.weight


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class PartLayer(nn.Module):
    """One Qwen3 decoder layer on a [1, DIM, 1, N] residual stream."""

    def __init__(self, w: Weights, index: int) -> None:
        super().__init__()
        p = f"llm.model.layers.{index}."
        query = w(p + "self_attn.q_proj.weight").reshape(HEADS, HEAD_DIM, DIM)
        key = w(p + "self_attn.k_proj.weight").reshape(KV_HEADS, HEAD_DIM, DIM)
        value = w(p + "self_attn.v_proj.weight").reshape(KV_HEADS, HEAD_DIM, DIM)
        self.norm1 = ChannelNorm(w(p + "input_layernorm.weight"))
        self.norm2 = ChannelNorm(w(p + "post_attention_layernorm.weight"))
        self.query = nn.ModuleList(conv(query[h]) for h in range(HEADS))
        self.key = nn.ModuleList(conv(key[h]) for h in range(KV_HEADS))
        self.value = nn.ModuleList(conv(value[h]) for h in range(KV_HEADS))
        self.q_norm = HeadNorm(w(p + "self_attn.q_norm.weight"))
        self.k_norm = HeadNorm(w(p + "self_attn.k_norm.weight"))
        self.output = conv(w(p + "self_attn.o_proj.weight"))
        self.gate = conv(w(p + "mlp.gate_proj.weight"))
        self.up = conv(w(p + "mlp.up_proj.weight"))
        self.down = conv(w(p + "mlp.down_proj.weight"))
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    @staticmethod
    def heads(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)  # [1, 128, 1, N] -> [1, 1, N, 128]

    def forward(self, x, mask, cos, sin, past_key, past_value):
        n = x.shape[3]
        normed = self.norm1(x)
        past_keys = torch.split(past_key, 1, dim=0)
        past_values = torch.split(past_value, 1, dim=0)
        keys, values, new_keys, new_values = [], [], [], []
        for g in range(KV_HEADS):
            k = rope(self.k_norm(self.heads(self.key[g](normed))), cos, sin).transpose(2, 3)
            v = self.heads(self.value[g](normed))
            new_keys.append(k)
            new_values.append(v)
            keys.append(torch.cat((past_keys[g], k), dim=3))
            values.append(torch.cat((past_values[g], v), dim=2))
        group = HEADS // KV_HEADS
        attended = []
        for h in range(HEADS):
            q = rope(self.q_norm(self.heads(self.query[h](normed))), cos, sin) * self.scale
            scores = torch.matmul(q, keys[h // group]) + mask
            attended.append(torch.matmul(torch.softmax(scores, dim=-1), values[h // group]))
        x = x + self.output(torch.cat(attended, dim=3).permute(0, 3, 1, 2))
        normed = self.norm2(x)
        x = x + self.down(functional.silu(self.gate(normed)) * self.up(normed))
        assert x.shape[3] == n
        return x, torch.cat(new_keys, dim=0), torch.cat(new_values, dim=0)


class LLMPart(nn.Module):
    def __init__(self, w: Weights, part: int) -> None:
        super().__init__()
        self.first = part * LAYERS_PER_PART
        self.layers = nn.ModuleList(PartLayer(w, i) for i in range(self.first, self.first + LAYERS_PER_PART))

    def forward(self, hidden, attention_mask, position_cos, position_sin, *past):
        n = hidden.shape[1]
        x = hidden.permute(0, 2, 1).reshape(1, DIM, 1, n)
        outputs = []
        for i, layer in enumerate(self.layers):
            x, key, value = layer(x, attention_mask, position_cos, position_sin, past[2 * i], past[2 * i + 1])
            outputs += [key, value]
        return (x.reshape(1, DIM, n).permute(0, 2, 1), *outputs)


class LLMHead(nn.Module):
    """Final norm, logits, and the TTS condition (normalized semantic projection of the normed hidden)."""

    def __init__(self, w: Weights) -> None:
        super().__init__()
        self.norm = ChannelNorm(w("llm.model.norm.weight"))
        self.lm_head = conv(w("llm.lm_head.weight"))
        p = "tts.projector_semantic."
        self.proj1 = conv(w(p + "linear1.weight"), w(p + "linear1.bias"))
        self.proj2 = conv(w(p + "linear2.weight"), w(p + "linear2.bias"))

    def forward(self, hidden):
        normed = self.norm(hidden.reshape(1, DIM, 1, 1))
        logits = self.lm_head(normed).reshape(1, VOCAB)
        semantic = self.proj2(functional.relu(self.proj1(normed))).reshape(1, TTS_DIM)
        semantic = semantic * torch.rsqrt(semantic.pow(2).sum(dim=-1, keepdim=True) + 1e-24)
        return logits, semantic


def part_names(part: int) -> tuple[list[str], list[str]]:
    inputs = ["hidden", "attention_mask", "position_cos", "position_sin"]
    outputs = ["hidden_out"]
    for i in range(part * LAYERS_PER_PART, (part + 1) * LAYERS_PER_PART):
        inputs += [f"past_key_{i}_in", f"past_value_{i}_in"]
        outputs += [f"past_key_{i}_out", f"past_value_{i}_out"]
    return inputs, outputs


class HostRing:
    """Host KV window in the graph layout. Position t lives in slot t while it fits, then cycles over
    the slots after `pinned` (no wrap in calibration)."""

    def __init__(self, device: torch.device) -> None:
        self.keys = [torch.zeros(KV_HEADS, 1, HEAD_DIM, PAST, device=device) for _ in range(LAYERS_PER_PART)]
        self.values = [torch.zeros(KV_HEADS, 1, PAST, HEAD_DIM, device=device) for _ in range(LAYERS_PER_PART)]
        self.device = device
        self.pinned = 0  # positions kept once the ring is full (minicpmo_runner.cpp KvModel::pin)

    def slot(self, position: int) -> int:
        return position if position < self.pinned else self.pinned + (position - self.pinned) % (PAST - self.pinned)

    def inputs(self, position: int, count: int, n: int):
        """Mask, cos, sin and past for `count` real tokens at positions position.. padded to n rows."""
        mask = torch.full((1, 1, n, PAST + n), MASK_MIN, device=self.device)
        valid = min(position, PAST)
        mask[..., :valid] = 0
        for i in range(n):
            mask[0, 0, i, PAST : PAST + min(i, count - 1) + 1] = 0
        cos, sin = rope_tables(torch.arange(position, position + n))
        cos, sin = cos.reshape(1, 1, n, -1).to(self.device), sin.reshape(1, 1, n, -1).to(self.device)
        past = [t for pair in zip(self.keys, self.values, strict=False) for t in pair]
        return mask, cos, sin, past

    def write(self, position: int, count: int, outputs) -> None:
        for j in range(count):
            slot = self.slot(position + j)
            for i in range(LAYERS_PER_PART):
                self.keys[i][..., slot] = outputs[2 * i][..., j]
                self.values[i][:, :, slot] = outputs[2 * i + 1][:, :, j]


@torch.no_grad()
def iterate(step, hidden: np.ndarray, device: torch.device, n: int, keep: set[int] = frozenset()):
    """Feeds a [T, DIM] sequence in chunks of n through `step`; returns outputs [T, DIM] and the
    graph inputs of the chunks whose index is in `keep`."""
    if len(hidden) > PAST:
        hidden = hidden[:PAST]
    ring = HostRing(device)
    outputs, samples = [], []
    for chunk, start in enumerate(range(0, len(hidden), n)):
        count = min(n, len(hidden) - start)
        x = torch.zeros(1, n, DIM, device=device)
        x[0, :count] = torch.from_numpy(np.ascontiguousarray(hidden[start : start + count], np.float32)).to(device)
        mask, cos, sin, past = ring.inputs(start, count, n)
        inputs = (x, mask, cos, sin, *past)
        if chunk in keep:
            samples.append([v.cpu().numpy() for v in inputs])
        result = step(inputs)
        ring.write(start, count, result[1:])
        outputs.append(result[0][0, :count].cpu().numpy())
    return np.concatenate(outputs), samples


def cuda_providers() -> list:
    options = {"use_tf32": "0", "device_id": os.environ.get("ORT_DEVICE", "0")}
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]


def create_sim(model: onnx.ModelProto, param_type: str, kv: bool) -> QuantizationSimModel:
    """QuantSim configured like the official w4a16 LLM recipes, before any calibration."""
    AimetLogger.set_level_for_all_areas(logging.WARNING)
    quantsim.op_types_to_tie_qtzrs = ["Concat"]
    quantsim._tie_qtzrs = True
    if "Slice" not in quantsim.op_outputs_to_ignore:
        quantsim.op_outputs_to_ignore += ["Slice", "Constant"]
    qs.encoding_version = "1.0.0"
    sim = QuantizationSimModel(
        model, param_type=param_type, activation_type="int16", quant_scheme=QuantScheme.min_max,
        config_file="htp_v81", providers=cuda_providers(),
    )
    if param_type == "int4":
        four_bit = {
            op.name for op in sim.connected_graph.ordered_ops
            if op.type in {"Conv", "Gemm", "MatMul"} and (q := sim.get_op_quantizers(op)[2].get("weight")) is not None
            and q.bitwidth == 4
        }
        set_lpbq_for_params(sim, bitwidth=4, block_size=64, nodes_to_include=four_bit)
    if kv:
        kv_io_map = _get_kv_io_map(sim)
        sim._tie_quantizers_for_op_types(["Concat"])
        sim._rebuild_session()
        _set_tensors_to_output_8b_sym(sim, list(kv_io_map) + list(kv_io_map.values()))
        _tie_quantizers_for_kv_cache(sim, kv_io_map)
    _set_matmul_second_input_to_8b(sim)
    return sim


def write_input_list(directory: Path, names: list[str], samples: list[list[np.ndarray]], count: int = 8) -> None:
    """Raw float32 calibration inputs for qairt-quantizer: tensors that AIMET leaves without an
    encoding (outputs inside its fused op groups) are calibrated there instead of staying float."""
    lines = []
    for i, sample in enumerate(samples[:: max(1, len(samples) // count)][:count]):
        (directory / "calib" / str(i)).mkdir(parents=True, exist_ok=True)
        paths = []
        for name, value in zip(names, sample, strict=True):
            path = directory / "calib" / str(i) / f"{name}.raw"
            np.ascontiguousarray(value, np.float32).tofile(path)
            paths.append(f"{name}:={path}")
        lines.append(" ".join(paths))
    (directory / "calib" / "input_list.txt").write_text("\n".join(lines) + "\n")


def export_onnx(module: nn.Module, example: tuple, names: tuple[list[str], list[str]], onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module.cpu(), tuple(t.cpu() for t in example), str(onnx_path), input_names=names[0], output_names=names[1],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )


def load_sessions(capture: Path, split: str) -> list[str]:
    return [s["name"] for s in json.loads((capture / "manifest.json").read_text()) if s["split"] == split]


def qwen3_config(model_dir: Path) -> Qwen3Config:
    # The checkpoint's own MiniCPMOConfig needs transformers 4.51; the LLM fields are plain Qwen3.
    c = json.loads((model_dir / "config.json").read_text())
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "rms_norm_eps", "rope_theta", "vocab_size", "max_position_embeddings", "tie_word_embeddings")
    config = Qwen3Config(**{k: c[k] for k in keys})
    config._attn_implementation = "eager"
    return config


@torch.no_grad()
def boundaries(args: argparse.Namespace) -> None:
    """Part-boundary activations from the BF16 LLM over every captured session."""
    out = args.work_dir / "boundaries"
    out.mkdir(parents=True, exist_ok=True)
    config = qwen3_config(args.model_dir)
    config._attn_implementation = "sdpa"
    llm = Qwen3ForCausalLM(config).to(torch.bfloat16)
    w = Weights(args.model_dir)
    llm.load_state_dict({k[len("llm."):]: w(k).to(torch.bfloat16) for k in w.index if k.startswith("llm.")}, strict=True)
    llm = llm.to(args.device).eval()
    for split in ("calibration", "validation"):
        for name in load_sessions(args.capture_dir, split):
            if (out / f"{name}.npz").exists():
                continue
            embeds = torch.from_numpy(np.load(args.capture_dir / f"{name}.npz")["llm_embeds"]).to(args.device, torch.bfloat16)
            hidden = llm.model(inputs_embeds=embeds[None], output_hidden_states=True).hidden_states
            np.savez(out / f"{name}.npz", hidden=np.stack([hidden[p * LAYERS_PER_PART][0].float().cpu().numpy() for p in range(PARTS)], 1))
            print("boundaries", name, embeds.shape[0], flush=True)


@torch.no_grad()
def reference_layers(w: Weights, first: int, hidden: np.ndarray, device: torch.device) -> np.ndarray:
    config = qwen3_config(w.dir)
    x = torch.from_numpy(hidden).to(device)[None]
    positions = torch.arange(len(hidden), device=device)[None]
    rotary = Qwen3RotaryEmbedding(config).to(device)
    embeddings = rotary(x, positions)
    mask = torch.full((len(hidden), len(hidden)), float("-inf"), device=device).triu(1)[None, None]
    for i in range(first, first + LAYERS_PER_PART):
        layer = Qwen3DecoderLayer(config, i).to(device)
        prefix = f"llm.model.layers.{i}."
        layer.load_state_dict({k[len(prefix):]: w(k) for k in w.index if k.startswith(prefix)}, strict=True)
        out = layer(x, attention_mask=mask, position_ids=positions, position_embeddings=embeddings)
        x = out[0] if isinstance(out, tuple) else out
    return x[0].cpu().numpy()


def calibration_samples(part_module: nn.Module, clips: list[np.ndarray], device: torch.device, n: int, total: int):
    """Chunks 0-1 of every clip (system prompt start: attention sinks) plus evenly spaced later ones."""
    per_clip = math.ceil(total / len(clips))
    samples = []
    for hidden in clips:
        chunks = math.ceil(len(hidden) / n)
        keep = {0, 1} | set(np.linspace(2, chunks - 1, max(per_clip - 2, 1)).round().astype(int).tolist())
        samples += iterate(lambda inputs: part_module(*inputs), hidden, device, n, keep)[1]
    return samples


def build_part(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    w = Weights(args.model_dir)
    part = LLMPart(w, args.part).to(device).eval()
    names = part_names(args.part)
    bdir = args.work_dir / "boundaries"
    clips = [np.load(bdir / f"{s}.npz")["hidden"][:, args.part] for s in load_sessions(args.capture_dir, "calibration")]

    probe = clips[0][:40]
    reference = reference_layers(w, part.first, probe, device)
    for n in (1, AR_N):
        ours, _ = iterate(lambda inputs: part(*inputs), probe, device, n)
        parity = float(np.max(np.linalg.norm(ours - reference, axis=1) / np.linalg.norm(reference, axis=1)))
        print(f"llm_part_{args.part} AR-{n}: parity relative error vs Qwen3DecoderLayer {parity:.2e}", flush=True)
        # AR-16 differs by the finite in-block mask (the phone feeds the same one); AR-1 checks the rewrite.
        if parity > (1e-3 if n == 1 else 1e-2):
            raise SystemExit("Rewritten part does not match the Qwen3 reference")

    if args.part == PARTS - 1:  # head calibration inputs: this part's float outputs
        head_inputs = [iterate(lambda inputs: part(*inputs), c, device, AR_N)[0] for c in clips]
        np.savez(args.work_dir / "head_inputs.npz", *head_inputs)

    for n in (AR_N, 1):
        name = f"llm_part_{args.part}_ar{n}"
        ring = HostRing(torch.device("cpu"))
        mask, cos, sin, past = ring.inputs(0, n, n)
        export_onnx(part, (torch.zeros(1, n, DIM), mask, cos, sin, *past), names, args.work_dir / name / "onnx" / "model.onnx")
        part.to(device)

    name = f"llm_part_{args.part}_ar{AR_N}"
    out = args.work_dir / name
    samples = calibration_samples(part, clips, device, AR_N, args.samples)
    del part
    torch.cuda.empty_cache()
    sim = create_sim(onnx.load(str(out / "onnx" / "model.onnx")), "int4", kv=True)
    sim.compute_encodings([dict(zip(names[0], s, strict=False)) for s in samples])
    sim.export(str(out), name, export_model=False)
    # AR-1 shares weights and KV ring with AR-16, so it takes the same encodings verbatim.
    ar1 = args.work_dir / f"llm_part_{args.part}_ar1"
    (ar1 / f"llm_part_{args.part}_ar1.encodings").write_text((out / f"{name}.encodings").read_text())
    print(f"{name}: {len(samples)} calibration chunks -> {out / (name + '.encodings')}", flush=True)


def build_head(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    head = LLMHead(Weights(args.model_dir)).eval()
    out = args.work_dir / "llm_head"
    export_onnx(head, (torch.zeros(1, 1, DIM),), (["hidden"], ["logits", "semantic"]), out / "onnx" / "model.onnx")
    data = np.load(args.work_dir / "head_inputs.npz")
    rows = np.concatenate([data[k] for k in data.files])
    picks = np.linspace(0, len(rows) - 1, args.samples).round().astype(int)
    held_out = rows[np.setdiff1d(np.arange(len(rows)), picks)[:: max(1, len(rows) // 512)]]
    rows = rows[picks]
    sim =create_sim(onnx.load(str(out / "onnx" / "model.onnx")), f"int{args.head_bits}", kv=False)
    sim.compute_encodings([{"hidden": r.reshape(1, 1, DIM).astype(np.float32)} for r in rows])
    sim.export(str(out), "llm_head", export_model=False)
    with torch.no_grad():
        reference = np.stack([head(torch.from_numpy(r.reshape(1, 1, DIM).astype(np.float32)))[0][0].numpy() for r in held_out])
    quantized = np.stack([sim.session.run(None, {"hidden": r.reshape(1, 1, DIM).astype(np.float32)})[0][0] for r in held_out])
    top1 = float(np.mean(reference.argmax(-1) == quantized.argmax(-1)))
    print(f"llm_head W{args.head_bits}: {len(rows)} calibration rows on {device}; top-1 agreement with float "
          f"{top1:.3f} over {len(held_out)} held-out rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["boundaries", "part", "head"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=range(PARTS))
    parser.add_argument("--samples", type=int, default=64)
    # W4 cut top-1 agreement with the float head from 0.978 to 0.898 for ~50 ms of HTP per second of speech.
    parser.add_argument("--head-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    start = time.monotonic()
    {"boundaries": boundaries, "part": build_part, "head": build_head}[args.step](args)
    print(f"{args.step} done in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
