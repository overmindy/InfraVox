# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""MiniCPM-o 4.5 streaming audio encoder (Whisper-medium + projector + 5x pooling) as one graph.

One call encodes a 1 s unit: mel [1,80,104] (unit 0 has no 2-frame prefix, so the host left-pads
it with zeros and zeroes the conv1 frame they produce through frame_mask [1,1,102]), Whisper positions
[1,1024,1,50] (channel-first) from the host, and a host-owned KV window of 1450 past frames per layer (the official
encoder resets its cache once 1500 frames would be exceeded). Outputs the 10 LLM embeddings
[1,10,4096] and the 50 new K/V frames per layer.

  build   parity vs the captured BF16 audio embeddings, ONNX, AIMET (W8 / A16 / 8-bit KV)
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

from minicpmo_llm import Weights, create_sim, export_onnx, load_sessions

D, HEADS, HEAD_DIM, LAYERS, FFN = 1024, 16, 64, 24, 4096
FRAMES, MEL_IN = 50, 104
PAST = 1450
LLM_DIM, POOL = 4096, 5
MASK_MIN = -50.0
UNIT_TOKEN = 151683  # "<unit>"


def linear_conv(w: Weights, prefix: str, bias: bool = True) -> nn.Conv2d:
    weight = w(prefix + ".weight")
    layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias)
    layer.weight.data.copy_(weight[:, :, None, None])
    if bias:
        layer.bias.data.copy_(w(prefix + ".bias"))
    return layer


class ChannelLayerNorm(nn.Module):
    def __init__(self, w: Weights, prefix: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(w(prefix + ".weight"))
        self.bias = nn.Parameter(w(prefix + ".bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A channel-axis LayerNorm gets its weight mis-laid-out by the converter; normalize channel-last instead.
        y = functional.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, 1e-5)
        return y.permute(0, 3, 1, 2)


class EncoderLayer(nn.Module):
    """Whisper encoder layer on [1, D, 1, T] with per-head projections and a host KV window."""

    def __init__(self, w: Weights, index: int) -> None:
        super().__init__()
        p = f"apm.layers.{index}."
        self.norm1 = ChannelLayerNorm(w, p + "self_attn_layer_norm")
        self.norm2 = ChannelLayerNorm(w, p + "final_layer_norm")
        q_w, q_b = w(p + "self_attn.q_proj.weight").reshape(HEADS, HEAD_DIM, D), w(p + "self_attn.q_proj.bias").reshape(HEADS, HEAD_DIM)
        k_w = w(p + "self_attn.k_proj.weight").reshape(HEADS, HEAD_DIM, D)
        v_w, v_b = w(p + "self_attn.v_proj.weight").reshape(HEADS, HEAD_DIM, D), w(p + "self_attn.v_proj.bias").reshape(HEADS, HEAD_DIM)
        scale = HEAD_DIM**-0.5  # WhisperAttention scales the query projection, bias included

        def head(weight, bias):
            layer = nn.Conv2d(D, HEAD_DIM, 1, bias=bias is not None)
            layer.weight.data.copy_(weight[:, :, None, None])
            if bias is not None:
                layer.bias.data.copy_(bias)
            return layer

        self.query = nn.ModuleList(head(q_w[h] * scale, q_b[h] * scale) for h in range(HEADS))
        self.key = nn.ModuleList(head(k_w[h], None) for h in range(HEADS))
        self.value = nn.ModuleList(head(v_w[h], v_b[h]) for h in range(HEADS))
        self.output = linear_conv(w, p + "self_attn.out_proj")
        self.fc1 = linear_conv(w, p + "fc1")
        self.fc2 = linear_conv(w, p + "fc2")

    def forward(self, x, mask, past_key, past_value):
        normed = self.norm1(x)
        past_keys, past_values = torch.split(past_key, 1, dim=0), torch.split(past_value, 1, dim=0)
        heads, new_keys, new_values = [], [], []
        for h in range(HEADS):
            q = self.query[h](normed).permute(0, 2, 3, 1)  # [1,1,T,64]
            k = self.key[h](normed).permute(0, 2, 1, 3)  # [1,1,64,T]
            v = self.value[h](normed).permute(0, 2, 3, 1)
            new_keys.append(k)
            new_values.append(v)
            scores = torch.matmul(q, torch.cat((past_keys[h], k), dim=3)) + mask
            heads.append(torch.matmul(torch.softmax(scores, dim=-1), torch.cat((past_values[h], v), dim=2)))
        x = x + self.output(torch.cat(heads, dim=3).permute(0, 3, 1, 2))
        x = x + self.fc2(functional.gelu(self.fc1(self.norm2(x))))
        return x, torch.cat(new_keys, dim=0), torch.cat(new_values, dim=0)


class AudioEncoder(nn.Module):
    def __init__(self, w: Weights) -> None:
        super().__init__()
        # Unpadded convs yield exactly the kept frames 1..50 (padding only touched the dropped edge frames):
        # an offset-1 slice here fails HTP graph prepare.
        self.conv1 = nn.Conv2d(80, D, (1, 3))
        self.conv2 = nn.Conv2d(D, D, (1, 3), stride=(1, 2))
        for conv, name in ((self.conv1, "apm.conv1"), (self.conv2, "apm.conv2")):
            conv.weight.data.copy_(w(name + ".weight")[:, :, None])
            conv.bias.data.copy_(w(name + ".bias"))
        self.layers = nn.ModuleList(EncoderLayer(w, i) for i in range(LAYERS))
        self.norm = ChannelLayerNorm(w, "apm.layer_norm")
        self.proj1 = linear_conv(w, "audio_projection_layer.linear1")
        self.proj2 = linear_conv(w, "audio_projection_layer.linear2")

    def forward(self, mel, frame_mask, positions, attention_mask, *past):
        x = functional.gelu(self.conv1(mel.reshape(1, 80, 1, MEL_IN))) * frame_mask.reshape(1, 1, 1, MEL_IN - 2)
        x = functional.gelu(self.conv2(x)) + positions
        outputs = []
        for i, layer in enumerate(self.layers):
            x, key, value = layer(x, attention_mask, past[2 * i], past[2 * i + 1])
            outputs += [key, value]
        x = self.proj2(functional.relu(self.proj1(self.norm(x))))  # [1, 4096, 1, 50]
        pooled = x.reshape(1, LLM_DIM, FRAMES // POOL, POOL).mean(dim=3)
        return (pooled.permute(0, 2, 1), *outputs)


def names() -> tuple[list[str], list[str]]:
    inputs, outputs = ["mel", "frame_mask", "positions", "attention_mask"], ["audio_embeds"]
    for i in range(LAYERS):
        inputs += [f"past_key_{i}_in", f"past_value_{i}_in"]
        outputs += [f"past_key_{i}_out", f"past_value_{i}_out"]
    return inputs, outputs


class AudioRing:
    """Host KV window: frames are appended in order and the window is cleared once full (as upstream)."""

    def __init__(self, device: torch.device) -> None:
        self.keys = [torch.zeros(HEADS, 1, HEAD_DIM, PAST, device=device) for _ in range(LAYERS)]
        self.values = [torch.zeros(HEADS, 1, PAST, HEAD_DIM, device=device) for _ in range(LAYERS)]
        self.length, self.device = 0, device

    def inputs(self, first_unit: bool, table: torch.Tensor):
        if self.length + FRAMES >= 1500:
            self.length = 0
        frame_mask = torch.ones(1, 1, MEL_IN - 2, device=self.device)  # conv1 frames 1..102
        if first_unit:
            frame_mask[..., :1] = 0
        mask = torch.full((1, 1, FRAMES, PAST + FRAMES), MASK_MIN, device=self.device)
        mask[..., : self.length] = 0
        mask[..., PAST:] = 0
        positions = table[self.length : self.length + FRAMES].T.reshape(1, D, 1, FRAMES).to(self.device)
        past = [t for pair in zip(self.keys, self.values, strict=False) for t in pair]
        return frame_mask, positions, mask, past

    def write(self, outputs) -> None:
        for i in range(LAYERS):
            self.keys[i][..., self.length : self.length + FRAMES] = outputs[2 * i]
            self.values[i][:, :, self.length : self.length + FRAMES] = outputs[2 * i + 1]
        self.length += FRAMES


@torch.no_grad()
def iterate(step, mel_chunks: np.ndarray, table: torch.Tensor, device: torch.device, keep: set[int] = frozenset()):
    ring = AudioRing(device)
    outputs, samples = [], []
    for c, mel in enumerate(mel_chunks):
        frame_mask, positions, mask, past = ring.inputs(c == 0, table)
        inputs = (torch.from_numpy(mel[None]).to(device), frame_mask, positions, mask, *past)
        if c in keep:
            samples.append([v.cpu().numpy() for v in inputs])
        result = step(inputs)
        ring.write(result[1:])
        outputs.append(result[0][0].cpu().numpy())
    return np.stack(outputs), samples


def unit_rows(embeds: np.ndarray, unit: np.ndarray) -> list[int]:
    """Rows right after every <unit> embedding: where each unit's 10 audio embeddings start."""
    return [i + 1 for i in np.flatnonzero((embeds == unit[None]).all(axis=1))]


def build(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    w = Weights(args.model_dir)
    unit = w("llm.model.embed_tokens.weight")[UNIT_TOKEN].to(torch.bfloat16).float().numpy()
    encoder = AudioEncoder(w).to(device).eval()
    table = w("apm.embed_positions.weight")
    out = args.work_dir / "audio_encoder"

    session = np.load(args.capture_dir / f"{load_sessions(args.capture_dir, 'calibration')[0]}.npz")
    ours, _ = iterate(lambda inputs: encoder(*inputs), session["mel_chunks"][:40], table, device)
    starts = unit_rows(session["llm_embeds"], unit)
    reference = np.stack([session["llm_embeds"][s : s + FRAMES // POOL] for s in starts[: len(ours)]])
    error = np.linalg.norm(ours - reference, axis=-1) / np.linalg.norm(reference, axis=-1)
    print(f"audio_encoder: relative error vs BF16 capture median {np.median(error):.2e} max {error.max():.2e} "
          f"over {error.size} embeddings", flush=True)
    if np.median(error) > 2e-2:
        raise SystemExit("Rewritten audio encoder does not match the BF16 capture")

    ring = AudioRing(torch.device("cpu"))
    frame_mask, positions, mask, past = ring.inputs(False, table)
    export_onnx(encoder, (torch.zeros(1, 80, MEL_IN), frame_mask, positions, mask, *past), names(), out / "onnx" / "model.onnx")
    encoder.to(device)
    samples = []
    sessions = load_sessions(args.capture_dir, "calibration")
    per = math.ceil(args.samples / len(sessions))
    for name in sessions:
        mel = np.load(args.capture_dir / f"{name}.npz")["mel_chunks"]
        keep = {0, 1} | set(np.linspace(2, len(mel) - 1, max(per - 2, 1)).round().astype(int).tolist())
        samples += iterate(lambda inputs: encoder(*inputs), mel, table, device, keep)[1]
    del encoder
    torch.cuda.empty_cache()
    sim = create_sim(onnx.load(str(out / "onnx" / "model.onnx")), "int8", kv=True)
    sim.compute_encodings([dict(zip(names()[0], s, strict=False)) for s in samples])
    sim.export(str(out), "audio_encoder", export_model=False)
    np.save(out / "positions.npy", table.numpy().astype(np.float16))
    print(f"audio_encoder: {len(samples)} calibration units", flush=True)


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
    print(f"audio_encoder done in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
