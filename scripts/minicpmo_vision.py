# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""MiniCPM-o 4.5 vision encoder (SigLIP-400M NaViT, 27 layers) + resampler as one fixed-shape graph.

Duplex feeds each photo as one source image (max_slice_nums=1): resized to ~448x448 area with its
aspect ratio, so a grid of h x w patches with h*w <= 1056 for aspect ratios up to 4:1 (the phone
centre-crops wider ones). The graph always takes N = 1056 patch slots and the host fills, channel-first:
  patches [1,588,1,N]          normalized pixels per 14x14 patch, (channel, row, column) order
  positions [1,1152,1,N]       NaViT position embeddings (bucketized 70x70 table rows)
  attention_mask [1,1,1,N]     0 for the h*w real patches, MASK_MIN for padding
  resampler_pos [1,4096,1,N]   the resampler's 2D sin-cos key positions for the h x w grid
and gets image_embeds [1,64,4096], the 64 LLM rows between <image> and </image>.

  build   parity vs the official float32 embeddings (minicpmo_vision_reference.py embeds), ONNX,
          AIMET W8 / A16 on real photos, held-out accuracy, and the host tables (vision_assets.bin)
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

from minicpmo_bundle import write_container
from minicpmo_llm import Weights, create_sim, export_onnx, write_input_list

D, HEADS, HEAD_DIM, LAYERS = 1152, 16, 72, 27
PATCH, GRID, SLOTS = 14, 70, 1056
LLM_DIM, QUERIES, R_HEADS = 4096, 64, 32
R_HEAD_DIM = LLM_DIM // R_HEADS
MAX_SIDE = 128  # bucket tables for grids up to 128 patches per side
MASK_MIN = -50.0
EPS = 1e-6
NAMES = (["patches", "positions", "attention_mask", "resampler_pos"], ["image_embeds"])


def conv(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Conv2d:
    weight = weight.reshape(weight.shape[0], -1)
    layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    layer.weight.data.copy_(weight[:, :, None, None])
    if bias is not None:
        layer.bias.data.copy_(bias)
    return layer


class ChannelLayerNorm(nn.Module):
    def __init__(self, w: Weights, prefix: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(w(prefix + ".weight"))
        self.bias = nn.Parameter(w(prefix + ".bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-last layer_norm: the converter mis-lays-out a channel-axis LayerNorm (minicpmo_audio.py).
        y = functional.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, EPS)
        return y.permute(0, 3, 1, 2)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # gelu(approximate="tanh") with the cube taken of clamp(x, -10, 10): same values (tanh is already +-1
    # there), but fc1 reaches +-280 from layer 9 on and its cube (+-2e7) wrecks 16-bit activation encodings.
    inner = x.clamp(-10.0, 10.0)
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (inner + 0.044715 * inner * inner * inner)))


def heads(w: Weights, prefix: str, count: int, dim: int, scale: float = 1.0) -> nn.ModuleList:
    weight = w(prefix + ".weight").reshape(count, dim, -1) * scale
    bias = w(prefix + ".bias").reshape(count, dim) * scale
    return nn.ModuleList(conv(weight[h], bias[h]) for h in range(count))


class EncoderLayer(nn.Module):
    """SigLIP encoder layer on [1, D, 1, N] with per-head 1x1 conv projections."""

    def __init__(self, w: Weights, index: int) -> None:
        super().__init__()
        p = f"vpm.encoder.layers.{index}."
        self.norm1 = ChannelLayerNorm(w, p + "layer_norm1")
        self.norm2 = ChannelLayerNorm(w, p + "layer_norm2")
        self.query = heads(w, p + "self_attn.q_proj", HEADS, HEAD_DIM, HEAD_DIM**-0.5)
        self.key = heads(w, p + "self_attn.k_proj", HEADS, HEAD_DIM)
        self.value = heads(w, p + "self_attn.v_proj", HEADS, HEAD_DIM)
        self.output = conv(w(p + "self_attn.out_proj.weight"), w(p + "self_attn.out_proj.bias"))
        self.fc1 = conv(w(p + "mlp.fc1.weight"), w(p + "mlp.fc1.bias"))
        self.fc2 = conv(w(p + "mlp.fc2.weight"), w(p + "mlp.fc2.bias"))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended = []
        for h in range(HEADS):
            q = self.query[h](normed).permute(0, 2, 3, 1)  # [1,1,N,72]
            k = self.key[h](normed).permute(0, 2, 1, 3)  # [1,1,72,N]
            v = self.value[h](normed).permute(0, 2, 3, 1)
            attended.append(torch.matmul(torch.softmax(torch.matmul(q, k) + mask, dim=-1), v))
        x = x + self.output(torch.cat(attended, dim=3).permute(0, 3, 1, 2))
        return x + self.fc2(gelu_tanh(self.fc1(self.norm2(x))))


class VisionEncoder(nn.Module):
    def __init__(self, w: Weights) -> None:
        super().__init__()
        self.patch = conv(w("vpm.embeddings.patch_embedding.weight"), w("vpm.embeddings.patch_embedding.bias"))
        self.layers = nn.ModuleList(EncoderLayer(w, i) for i in range(LAYERS))
        self.post_norm = ChannelLayerNorm(w, "vpm.post_layernorm")
        r = "resampler."
        self.kv_proj = conv(w(r + "kv_proj.weight"), None)
        self.ln_kv = ChannelLayerNorm(w, r + "ln_kv")
        in_w, in_b = w(r + "attn.in_proj_weight"), w(r + "attn.in_proj_bias")
        query = functional.layer_norm(w(r + "query"), (LLM_DIM,), w(r + "ln_q.weight"), w(r + "ln_q.bias"), EPS)
        query = (query @ in_w[:LLM_DIM].T + in_b[:LLM_DIM]) * R_HEAD_DIM**-0.5  # constant: learned queries
        self.register_buffer("queries", query.reshape(1, QUERIES, R_HEADS, R_HEAD_DIM).permute(2, 0, 1, 3).contiguous())
        key_w, value_w = in_w[LLM_DIM : 2 * LLM_DIM].reshape(R_HEADS, R_HEAD_DIM, -1), in_w[2 * LLM_DIM :].reshape(R_HEADS, R_HEAD_DIM, -1)
        key_b, value_b = in_b[LLM_DIM : 2 * LLM_DIM].reshape(R_HEADS, -1), in_b[2 * LLM_DIM :].reshape(R_HEADS, -1)
        self.r_key = nn.ModuleList(conv(key_w[h], key_b[h]) for h in range(R_HEADS))
        self.r_value = nn.ModuleList(conv(value_w[h], value_b[h]) for h in range(R_HEADS))
        self.r_output = conv(w(r + "attn.out_proj.weight"), w(r + "attn.out_proj.bias"))
        self.ln_post = ChannelLayerNorm(w, r + "ln_post")
        self.proj = conv(w(r + "proj").T.contiguous(), None)

    def forward(self, patches, positions, attention_mask, resampler_pos):
        x = self.patch(patches) + positions
        for layer in self.layers:
            x = layer(x, attention_mask)
        kv = self.ln_kv(self.kv_proj(self.post_norm(x)))
        keyed = kv + resampler_pos
        attended = []
        for h in range(R_HEADS):
            k = self.r_key[h](keyed).permute(0, 2, 1, 3)  # [1,1,128,N]
            v = self.r_value[h](kv).permute(0, 2, 3, 1)  # [1,1,N,128]
            scores = torch.matmul(self.queries[h : h + 1], k) + attention_mask
            attended.append(torch.matmul(torch.softmax(scores, dim=-1), v))
        out = self.r_output(torch.cat(attended, dim=3).permute(0, 3, 1, 2))  # [1,4096,1,64]
        out = self.proj(self.ln_post(out))
        return out.reshape(1, LLM_DIM, QUERIES).permute(0, 2, 1)


def bucket_table() -> np.ndarray:
    """Row s-1 holds the NaViT position bucket of each of s patches along one side (upstream bucketize)."""
    boundaries = torch.arange(1 / GRID, 1.0, 1 / GRID)
    table = np.zeros((MAX_SIDE, MAX_SIDE), np.int32)
    for side in range(1, MAX_SIDE + 1):
        step = 1 / torch.tensor(side, dtype=torch.int32)  # float32, as upstream divides by tgt_sizes; a float64 step moves buckets
        table[side - 1, :side] = torch.bucketize(torch.arange(0, 1 - 1e-6, step), boundaries, right=True).numpy()
    return table


def sincos(height: int, width: int) -> np.ndarray:
    """Resampler.pos_embed[:h, :w]: first half encodes the column, second half the row (upstream grid order)."""
    omega = 1.0 / 10000 ** (np.arange(LLM_DIM // 4, dtype=np.float32) / (LLM_DIM / 4.0))
    rows, cols = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    def encode(pos):
        out = pos.reshape(-1)[:, None] * omega[None]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    return np.concatenate([encode(cols), encode(rows)], axis=1)  # [h*w, 4096]


def host_inputs(pixels: np.ndarray, size: tuple[int, int], table: np.ndarray, buckets: np.ndarray) -> list[np.ndarray]:
    """Graph inputs for one processed image: pixel_values [3,14,L*14] and its (h, w) patch grid."""
    h, w = int(size[0]), int(size[1])
    count = h * w
    patches = np.zeros((1, 3 * PATCH * PATCH, 1, SLOTS), np.float32)
    patches[0, :, 0, :count] = pixels.reshape(3, PATCH, count, PATCH).transpose(0, 1, 3, 2).reshape(-1, count)
    ids = (buckets[h - 1, :h, None] * GRID + buckets[w - 1, None, :w]).reshape(-1)
    positions = np.zeros((1, D, 1, SLOTS), np.float32)
    positions[0, :, 0, :count] = table[ids].T
    mask = np.full((1, 1, 1, SLOTS), MASK_MIN, np.float32)
    mask[..., :count] = 0
    resampler = np.zeros((1, LLM_DIM, 1, SLOTS), np.float32)
    resampler[0, :, 0, :count] = sincos(h, w).T
    return [patches, positions, mask, resampler]


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


@torch.no_grad()
def build(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    w = Weights(args.model_dir)
    table = w("vpm.embeddings.position_embedding.weight").numpy()
    buckets = bucket_table()
    out = args.work_dir / "vision"
    files = sorted(args.reference_dir.glob("*.npz"))
    docs = [f for f in files if f.name.startswith("docs_")]
    photos = [f for f in files if not f.name.startswith("docs_")]
    held_out = set(photos[:: args.hold_every]) | set(docs[:: args.hold_every])
    calibration = [f for f in files if f not in held_out]
    print(f"vision: {len(calibration)} calibration / {len(held_out)} held-out images", flush=True)

    def load(path: Path):
        data = np.load(path)
        return host_inputs(data["pixel_values"], tuple(data["tgt_size"]), table, buckets), data["embeds"], data["embeds_bf16"]

    encoder = VisionEncoder(w).to(device).eval()

    def run(inputs: list[np.ndarray]) -> np.ndarray:
        return encoder(*(torch.from_numpy(v).to(device) for v in inputs))[0].cpu().numpy()

    floats = {}
    for path in sorted(held_out):
        inputs, reference, bf16 = load(path)
        floats[path] = run(inputs)
        print(f"  {path.stem}: float vs official fp32 rel {np.median(relative(floats[path], reference)):.2e}, "
              f"official bf16 vs fp32 rel {np.median(relative(bf16, reference)):.2e}", flush=True)
    parity = np.median([np.median(relative(floats[p], load(p)[1])) for p in floats])
    if parity > 5e-3:  # GPU matmul noise is ~1e-3; the official BF16 encoder is ~2e-2 off its float32 self
        raise SystemExit(f"Rewritten vision encoder does not match the official float32 one ({parity:.2e})")

    samples = [load(p)[0] for p in calibration]
    export_onnx(encoder, tuple(torch.from_numpy(v) for v in samples[0]), NAMES, out / "onnx" / "model.onnx")
    del encoder
    torch.cuda.empty_cache()
    sim = create_sim(onnx.load(str(out / "onnx" / "model.onnx")), "int8", kv=False)
    sim.compute_encodings([dict(zip(NAMES[0], s, strict=True)) for s in samples])
    sim.export(str(out), "vision", export_model=False)
    write_input_list(out, NAMES[0], samples)

    rel, cos, rel_bf16 = [], [], []
    for path in sorted(held_out):
        inputs, reference, bf16 = load(path)
        quantized = sim.session.run(None, dict(zip(NAMES[0], inputs, strict=True)))[0][0]
        rel.append(np.median(relative(quantized, floats[path])))
        cos.append(np.median(cosine(quantized, floats[path])))
        rel_bf16.append(np.median(relative(bf16, reference)))
        print(f"  {path.stem}: W8A16 vs float rel {rel[-1]:.3f} cos {cos[-1]:.4f}", flush=True)
    print(f"vision W8A16 held-out: median rel {np.median(rel):.3f} (official bf16 vs fp32 {np.median(rel_bf16):.3f}), "
          f"median cos {np.median(cos):.4f}, min cos {np.min(cos):.4f}", flush=True)

    write_container(out / "vision_assets.bin", {
        "vision_positions": table.astype(np.float16),
        "vision_buckets": buckets,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True, help="minicpmo_vision_reference.py embeds output")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hold-every", type=int, default=8, help="every n-th image is held out")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    start = time.monotonic()
    build(args)
    print(f"vision done in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
