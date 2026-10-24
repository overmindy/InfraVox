# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""MiniCPM-o 4.5 streaming TTS (20-layer Llama, 768 wide) with its audio-code head, as one graph.

Like the LLM parts it comes as AR-1 (one code per step) and AR-16 (the per-unit condition) graphs
that share weights and a host KV ring of 2032 slots per layer (past_key [12,1,64,P],
past_value [12,1,P,64]); RoPE cos/sin [1,1,N,32] (theta 1e4) and the mask come from the host.
Outputs the code logits [1,N,6562] and the new K/V. emb_text / emb_code stay on the CPU.

  build   parity vs transformers LlamaModel on captured TTS inputs, ONNX, AIMET (W8 / A16 / 8-bit KV)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional
from transformers import LlamaConfig, LlamaModel

from minicpmo_llm import ChannelNorm, Weights, conv, create_sim, export_onnx, load_sessions

DIM, HEADS, HEAD_DIM, LAYERS, FFN = 768, 12, 64, 20, 3072
CODES = 6562
PAST, AR_N = 2032, 16
ROPE_THETA = 1e4
MASK_MIN = -50.0


def rope_tables(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    angle = positions.double()[:, None] * freqs[None]
    return angle.cos().float(), angle.sin().float()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class TTSLayer(nn.Module):
    def __init__(self, w: Weights, index: int) -> None:
        super().__init__()
        p = f"tts.model.layers.{index}."
        q = w(p + "self_attn.q_proj.weight").reshape(HEADS, HEAD_DIM, DIM)
        k = w(p + "self_attn.k_proj.weight").reshape(HEADS, HEAD_DIM, DIM)
        v = w(p + "self_attn.v_proj.weight").reshape(HEADS, HEAD_DIM, DIM)
        self.norm1 = ChannelNorm(w(p + "input_layernorm.weight"))
        self.norm2 = ChannelNorm(w(p + "post_attention_layernorm.weight"))
        self.query = nn.ModuleList(conv(q[h] / math.sqrt(HEAD_DIM)) for h in range(HEADS))
        self.key = nn.ModuleList(conv(k[h]) for h in range(HEADS))
        self.value = nn.ModuleList(conv(v[h]) for h in range(HEADS))
        self.output = conv(w(p + "self_attn.o_proj.weight"))
        self.gate = conv(w(p + "mlp.gate_proj.weight"))
        self.up = conv(w(p + "mlp.up_proj.weight"))
        self.down = conv(w(p + "mlp.down_proj.weight"))

    def forward(self, x, mask, cos, sin, past_key, past_value):
        normed = self.norm1(x)
        past_keys, past_values = torch.split(past_key, 1, dim=0), torch.split(past_value, 1, dim=0)
        heads, new_keys, new_values = [], [], []
        for h in range(HEADS):
            # RoPE is linear, so scaling the query weights by 1/sqrt(d) is exact.
            q = rope(self.query[h](normed).permute(0, 2, 3, 1), cos, sin)
            k = rope(self.key[h](normed).permute(0, 2, 3, 1), cos, sin).transpose(2, 3)
            v = self.value[h](normed).permute(0, 2, 3, 1)
            new_keys.append(k)
            new_values.append(v)
            scores = torch.matmul(q, torch.cat((past_keys[h], k), dim=3)) + mask
            heads.append(torch.matmul(torch.softmax(scores, dim=-1), torch.cat((past_values[h], v), dim=2)))
        x = x + self.output(torch.cat(heads, dim=3).permute(0, 3, 1, 2))
        normed = self.norm2(x)
        x = x + self.down(functional.silu(self.gate(normed)) * self.up(normed))
        return x, torch.cat(new_keys, dim=0), torch.cat(new_values, dim=0)


class TTSModel(nn.Module):
    def __init__(self, w: Weights) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TTSLayer(w, i) for i in range(LAYERS))
        self.norm = ChannelNorm(w("tts.model.norm.weight"))
        g, v = w("tts.head_code.0.parametrizations.weight.original0"), w("tts.head_code.0.parametrizations.weight.original1")
        self.head = conv(g * v / v.norm(dim=1, keepdim=True))

    def forward(self, embeds, attention_mask, position_cos, position_sin, *past):
        n = embeds.shape[1]
        x = embeds.permute(0, 2, 1).reshape(1, DIM, 1, n)
        outputs = []
        for i, layer in enumerate(self.layers):
            x, key, value = layer(x, attention_mask, position_cos, position_sin, past[2 * i], past[2 * i + 1])
            outputs += [key, value]
        logits = self.head(self.norm(x)).reshape(1, CODES, n).permute(0, 2, 1)
        return (logits, *outputs)


def names() -> tuple[list[str], list[str]]:
    inputs, outputs = ["embeds", "attention_mask", "position_cos", "position_sin"], ["logits"]
    for i in range(LAYERS):
        inputs += [f"past_key_{i}_in", f"past_value_{i}_in"]
        outputs += [f"past_key_{i}_out", f"past_value_{i}_out"]
    return inputs, outputs


class HostRing:
    def __init__(self, device: torch.device) -> None:
        self.keys = [torch.zeros(HEADS, 1, HEAD_DIM, PAST, device=device) for _ in range(LAYERS)]
        self.values = [torch.zeros(HEADS, 1, PAST, HEAD_DIM, device=device) for _ in range(LAYERS)]
        self.device = device

    def inputs(self, position: int, count: int, n: int):
        mask = torch.full((1, 1, n, PAST + n), MASK_MIN, device=self.device)
        mask[..., : min(position, PAST)] = 0
        for i in range(n):
            mask[0, 0, i, PAST : PAST + min(i, count - 1) + 1] = 0
        cos, sin = rope_tables(torch.arange(position, position + n))
        cos, sin = cos.reshape(1, 1, n, -1).to(self.device), sin.reshape(1, 1, n, -1).to(self.device)
        return mask, cos, sin, [t for pair in zip(self.keys, self.values, strict=False) for t in pair]

    def write(self, position: int, count: int, outputs) -> None:
        for j in range(count):
            slot = (position + j) % PAST
            for i in range(LAYERS):
                self.keys[i][..., slot] = outputs[2 * i][..., j]
                self.values[i][:, :, slot] = outputs[2 * i + 1][:, :, j]


@torch.no_grad()
def iterate(step, embeds: np.ndarray, device: torch.device, n: int, keep: set[int] = frozenset()):
    embeds = embeds[:PAST]
    ring = HostRing(device)
    outputs, samples = [], []
    for chunk, start in enumerate(range(0, len(embeds), n)):
        count = min(n, len(embeds) - start)
        x = torch.zeros(1, n, DIM, device=device)
        x[0, :count] = torch.from_numpy(embeds[start : start + count]).to(device)
        mask, cos, sin, past = ring.inputs(start, count, n)
        inputs = (x, mask, cos, sin, *past)
        if chunk in keep:
            samples.append([v.cpu().numpy() for v in inputs])
        result = step(inputs)
        ring.write(start, count, result[1:])
        outputs.append(result[0][0, :count].cpu().numpy())
    return np.concatenate(outputs), samples


@torch.no_grad()
def reference_logits(w: Weights, embeds: np.ndarray, device: torch.device) -> np.ndarray:
    config = LlamaConfig(hidden_size=DIM, intermediate_size=FFN, num_attention_heads=HEADS, num_hidden_layers=LAYERS,
                         num_key_value_heads=HEADS, max_position_embeddings=4096)
    config._attn_implementation = "eager"
    model = LlamaModel(config)
    model.load_state_dict({k[len("tts.model."):]: w(k) for k in w.index if k.startswith("tts.model.")}, strict=True)
    model = model.to(device).eval()
    hidden = model(inputs_embeds=torch.from_numpy(embeds).to(device)[None]).last_hidden_state
    g, v = w("tts.head_code.0.parametrizations.weight.original0"), w("tts.head_code.0.parametrizations.weight.original1")
    return (hidden[0] @ (g * v / v.norm(dim=1, keepdim=True)).to(device).T).cpu().numpy()


def build(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    w = Weights(args.model_dir)
    tts = TTSModel(w).to(device).eval()
    sessions = load_sessions(args.capture_dir, "calibration")
    clips = [np.load(args.capture_dir / f"{s}.npz")["tts_inputs"][:PAST] for s in sessions]
    clips = [c for c in clips if len(c)]
    probe = clips[0][:200]
    reference = reference_logits(w, probe, device)
    for n in (1, AR_N):
        ours, _ = iterate(lambda inputs: tts(*inputs), probe, device, n)
        parity = float(np.max(np.linalg.norm(ours - reference, axis=1) / np.linalg.norm(reference, axis=1)))
        print(f"tts AR-{n}: parity relative error vs LlamaModel {parity:.2e}", flush=True)
        if parity > 1e-3:
            raise SystemExit("Rewritten TTS does not match the Llama reference")

    for n in (AR_N, 1):
        mask, cos, sin, past = HostRing(torch.device("cpu")).inputs(0, n, n)
        export_onnx(tts, (torch.zeros(1, n, DIM), mask, cos, sin, *past), names(), args.work_dir / f"tts_ar{n}" / "onnx" / "model.onnx")
        tts.to(device)
    per = math.ceil(args.samples / len(clips))
    samples = []
    for c in clips:
        chunks = math.ceil(len(c) / AR_N)
        keep = {0, 1} | set(np.linspace(2, chunks - 1, max(per - 2, 1)).round().astype(int).tolist())
        samples += iterate(lambda inputs: tts(*inputs), c, device, AR_N, keep)[1]
    del tts
    torch.cuda.empty_cache()
    out = args.work_dir / f"tts_ar{AR_N}"
    sim = create_sim(onnx.load(str(out / "onnx" / "model.onnx")), "int8", kv=True)
    sim.compute_encodings([dict(zip(names()[0], s, strict=False)) for s in samples])
    sim.export(str(out), f"tts_ar{AR_N}", export_model=False)
    (args.work_dir / "tts_ar1" / "tts_ar1.encodings").write_text((out / f"tts_ar{AR_N}.encodings").read_text())
    print(f"tts: {len(samples)} calibration chunks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    start = time.monotonic()
    build(args)
    print(f"tts done in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
