# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""MiniCPM-o 4.5 TTS for the phone CPU (android/minicpmo_runner.cpp CpuTts), off the HTP.

Linear weights are int8 with one scale per output row; each step quantizes the input vector to
int8 per group of 64 (sdot-friendly), accumulates int32 per group and applies both scales. RMSNorm,
RoPE, attention and the float KV ring (pinned first chunk, as KvModel::pin) stay in float32.
This file is the reference of that arithmetic: it checks it against transformers' LlamaModel on a
captured session, writes OUT/tts_cpu.bin (minicpmo_bundle container) and host-test vectors.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from minicpmo_bundle import write_container
from minicpmo_llm import Weights, load_sessions
from minicpmo_tts import CODES, DIM, FFN, HEAD_DIM, HEADS, LAYERS, PAST, ROPE_THETA, reference_logits

GROUP = 64
EPS = 1e-6


def quantize_rows(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    scale = weight.abs().amax(dim=1).clamp(min=1e-12) / 127
    q = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.numpy(), scale.float().numpy()


def tensors(w: Weights) -> dict[str, np.ndarray]:
    out = {}
    for i in range(LAYERS):
        p = f"tts.model.layers.{i}."
        q = w(p + "self_attn.q_proj.weight") / math.sqrt(HEAD_DIM)  # RoPE is linear: scaling q first is exact
        qkv = torch.cat([q, w(p + "self_attn.k_proj.weight"), w(p + "self_attn.v_proj.weight")])
        gate_up = torch.cat([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")])
        for name, weight in (("qkv", qkv), ("o", w(p + "self_attn.o_proj.weight")), ("gate_up", gate_up),
                             ("down", w(p + "mlp.down_proj.weight"))):
            out[f"tts_{i}_{name}"], out[f"tts_{i}_{name}_scale"] = quantize_rows(weight.float())
        out[f"tts_{i}_norm1"] = w(p + "input_layernorm.weight").float().numpy()
        out[f"tts_{i}_norm2"] = w(p + "post_attention_layernorm.weight").float().numpy()
    out["tts_norm"] = w("tts.model.norm.weight").float().numpy()
    g, v = w("tts.head_code.0.parametrizations.weight.original0"), w("tts.head_code.0.parametrizations.weight.original1")
    out["tts_head"], out["tts_head_scale"] = quantize_rows((g * v / v.norm(dim=1, keepdim=True)).float())
    return out


class CpuTts:
    """Step-by-step forward with the phone's arithmetic (torch float32 on any device)."""

    def __init__(self, t: dict[str, np.ndarray], device: torch.device) -> None:
        self.d = device
        self.t = {k: torch.from_numpy(v.astype(np.float32)).to(device) for k, v in t.items()}
        freqs = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
        self.freqs = freqs.to(device)
        self.reset()

    def reset(self) -> None:
        self.keys = torch.zeros(LAYERS, HEADS, PAST, HEAD_DIM, device=self.d)
        self.values = torch.zeros(LAYERS, HEADS, PAST, HEAD_DIM, device=self.d)
        self.position, self.pinned = 0, 0

    def slot(self, position: int) -> int:
        return position if position < self.pinned else self.pinned + (position - self.pinned) % (PAST - self.pinned)

    def linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        groups = x.reshape(-1, GROUP)
        sx = groups.abs().amax(dim=1).clamp(min=1e-12) / 127
        qx = torch.round(groups / sx[:, None]).clamp(-127, 127)
        qw = self.t[name].reshape(self.t[name].shape[0], -1, GROUP)
        per_group = torch.einsum("ogi,gi->og", qw, qx)  # exact: |sum| < 2^24
        return (per_group * sx[None]).sum(dim=1) * self.t[name + "_scale"]

    def norm(self, x: torch.Tensor, weight: str) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean() + EPS) * self.t[weight]

    def rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        angle = position * self.freqs
        cos, sin = angle.cos().float(), angle.sin().float()
        first, second = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    @torch.no_grad()
    def step(self, embed: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(embed, dtype=torch.float32, device=self.d)
        slot, filled = self.slot(self.position), min(self.position + 1, PAST)
        for i in range(LAYERS):
            qkv = self.linear(f"tts_{i}_qkv", self.norm(x, f"tts_{i}_norm1")).reshape(3, HEADS, HEAD_DIM)
            q, k = self.rope(qkv[0], self.position), self.rope(qkv[1], self.position)
            self.keys[i, :, slot], self.values[i, :, slot] = k, qkv[2]
            scores = torch.einsum("hd,hpd->hp", q, self.keys[i, :, :filled])
            attended = torch.einsum("hp,hpd->hd", torch.softmax(scores, dim=-1), self.values[i, :, :filled])
            x = x + self.linear(f"tts_{i}_o", attended.reshape(-1))
            gate, up = self.linear(f"tts_{i}_gate_up", self.norm(x, f"tts_{i}_norm2")).reshape(2, FFN)
            x = x + self.linear(f"tts_{i}_down", torch.nn.functional.silu(gate) * up)
        self.position += 1
        return self.linear("tts_head", self.norm(x, "tts_norm")).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-test-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    w = Weights(args.model_dir)
    t = tensors(w)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_container(args.output, t)
    print(f"tts_cpu: {sum(v.nbytes for v in t.values()) / 2**20:.0f} MiB -> {args.output}", flush=True)

    model = CpuTts(t, device)
    worst = []
    for name in load_sessions(args.capture_dir, "validation"):
        embeds = np.load(args.capture_dir / f"{name}.npz")["tts_inputs"][:300]
        if not len(embeds):
            continue
        reference = reference_logits(w, embeds, device)
        model.reset()
        ours = np.stack([model.step(e) for e in embeds])
        relative = np.linalg.norm(ours - reference, axis=1) / np.linalg.norm(reference, axis=1)
        p, q = (torch.softmax(torch.from_numpy(a / 0.8), dim=-1) for a in (reference, ours))
        kl = float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).mean())
        top1 = float(np.mean(ours.argmax(-1) == reference.argmax(-1)))
        worst.append(float(np.median(relative)))
        print(f"tts_cpu {name}: {len(embeds)} steps, relative error median {np.median(relative):.3f} "
              f"max {relative.max():.3f}, top-1 {top1:.3f}, KL(T=0.8) {kl:.4f}", flush=True)
    if not worst or max(worst) > 0.05:
        raise SystemExit("CPU TTS arithmetic is too far from the float TTS")

    if args.host_test_dir:
        embeds = np.load(args.capture_dir / f"{load_sessions(args.capture_dir, 'validation')[0]}.npz")["tts_inputs"][:48]
        model.reset()
        logits = np.stack([model.step(e) for e in embeds])
        args.host_test_dir.mkdir(parents=True, exist_ok=True)
        embeds.astype(np.float32).tofile(args.host_test_dir / "tts_embeds.f32")
        logits.astype(np.float32).tofile(args.host_test_dir / "tts_logits.f32")
        print(f"host test vectors: {len(embeds)} steps -> {args.host_test_dir}", flush=True)
    assert CODES == t["tts_head"].shape[0] and DIM == t["tts_norm"].shape[0]


if __name__ == "__main__":
    main()
