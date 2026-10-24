# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Pure-torch pieces of MiniCPM-o's token2wav (Step-Audio2 / CosyVoice2 flow + HiFT) shared by the
server scripts: the DiT estimator step in phone layout, its precomputed timestep modulation, the
HiFT neural source, the n_fft=16 STFT pair, and Token2wavHost, the streaming cache bookkeeping
that the phone runner mirrors.

DiT step graph I/O (one Euler step, CFG rows [cond, uncond]):
  x [2,80,T], cond [2,240,T] (mu | speaker | zeros), attention_mask [1,1,T,P+T],
  select [1,T+2,2] (one-hot: the last two valid frames become the next causal-conv cache),
  mod_0..mod_15 [1,9,512] and mod_final [1,2,512] (adaLN of this step's timestep),
  per block past_key_i [16,1,64,P], past_value_i [16,1,P,64], cnn_i [2,1024,2]
  -> velocity [2,80,T], new_key_i [16,1,64,T], new_value_i [16,1,T,64], new_cnn_i [2,1024,2].
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional

DIT_BLOCKS, DIT_DIM, DIT_HEADS, DIT_HEAD_DIM = 16, 512, 8, 64
MEL, CHUNK_TOKENS, LOOKAHEAD, UP = 80, 25, 3, 2
T = CHUNK_TOKENS * UP  # mel frames per regular chunk
STEPS, CFG = 10, 0.7
SILENCE_TOKEN = 4218
MASK_MIN = -50.0
MEL_CACHE, SOURCE_CACHE = 8, 8 * 480
HIFT_FRAMES = T + MEL_CACHE
N_FFT, HOP = 16, 4


def timesteps() -> tuple[list[float], list[float]]:
    """Per-step (t, dt) exactly as CausalConditionalCFM.solve_euler_chunk accumulates them in fp32."""
    span = 1 - torch.cos(torch.linspace(0, 1, STEPS + 1, dtype=torch.float32) * 0.5 * torch.pi)
    t, dt = span[0].reshape(1), span[1] - span[0]
    ts, dts = [], []
    for step in range(1, STEPS + 1):
        ts.append(float(t))
        dts.append(float(dt))
        t = t + dt
        if step < STEPS:
            dt = span[step + 1] - t
    return ts, dts


def modulations(state: dict, t: float) -> list[torch.Tensor]:
    """adaLN outputs of every block and the final layer for timestep t: 16 x [1,9,512] + [1,2,512]."""
    p = "decoder.estimator."
    half = 128
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = torch.tensor([t * 1000.0])[:, None] * freqs[None]
    freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    c = functional.linear(freq, state[p + "t_embedder.mlp.0.weight"], state[p + "t_embedder.mlp.0.bias"])
    c = functional.linear(functional.silu(c), state[p + "t_embedder.mlp.2.weight"], state[p + "t_embedder.mlp.2.bias"])
    out = []
    for i in range(DIT_BLOCKS):
        m = functional.linear(functional.silu(c), state[p + f"blocks.{i}.adaLN_modulation.1.weight"], state[p + f"blocks.{i}.adaLN_modulation.1.bias"])
        out.append(m.reshape(1, 9, DIT_DIM))
    m = functional.linear(functional.silu(c), state[p + "final_layer.adaLN_modulation.1.weight"], state[p + "final_layer.adaLN_modulation.1.bias"])
    out.append(m.reshape(1, 2, DIT_DIM))
    return out


def linear(state: dict, prefix: str) -> nn.Linear:
    weight = state[prefix + ".weight"]
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=prefix + ".bias" in state)
    layer.weight.data.copy_(weight)
    if layer.bias is not None:
        layer.bias.data.copy_(state[prefix + ".bias"])
    return layer


def layer_norm(state: dict, prefix: str, dim: int) -> nn.LayerNorm:
    norm = nn.LayerNorm(dim)
    norm.weight.data.copy_(state[prefix + ".weight"])
    norm.bias.data.copy_(state[prefix + ".bias"])
    return norm


def causal_conv(state: dict, prefix: str) -> nn.Conv1d:
    weight = state[prefix + ".weight"]
    conv = nn.Conv1d(weight.shape[1], weight.shape[0], weight.shape[2])
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(state[prefix + ".bias"])
    return conv


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def plain_norm(x: torch.Tensor) -> torch.Tensor:
    # A LayerNormalization op: written out, QAIRT fuses it with modulate into a float RMSNorm.
    return functional.layer_norm(x, (x.shape[-1],), eps=1e-6)


class DiTBlock(nn.Module):
    def __init__(self, state: dict, i: int) -> None:
        super().__init__()
        p = f"decoder.estimator.blocks.{i}."
        self.to_q, self.to_k, self.to_v = (linear(state, p + f"attn.to_{n}") for n in "qkv")
        self.q_norm = layer_norm(state, p + "attn.q_norm", DIT_HEAD_DIM)
        self.k_norm = layer_norm(state, p + "attn.k_norm", DIT_HEAD_DIM)
        self.proj = linear(state, p + "attn.proj")
        self.conv1 = causal_conv(state, p + "conv.block.1")
        self.conv_norm = layer_norm(state, p + "conv.block.3", DIT_DIM)
        self.conv2 = causal_conv(state, p + "conv.block.6")
        self.fc1, self.fc2 = linear(state, p + "mlp.fc1"), linear(state, p + "mlp.fc2")

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, -1, DIT_HEADS, DIT_HEAD_DIM).transpose(1, 2)  # [2,8,T,64]

    def forward(self, x, mod, mask, select, past_key, past_value, cnn):
        n = x.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = torch.split(mod, 1, dim=1)
        h = modulate(plain_norm(x), shift_msa, scale_msa)
        q = self.q_norm(self.heads(self.to_q(h))) * DIT_HEAD_DIM**-0.5
        k = self.k_norm(self.heads(self.to_k(h)))
        v = self.heads(self.to_v(h))
        keys = torch.cat((past_key.reshape(2, DIT_HEADS, DIT_HEAD_DIM, -1), k.transpose(2, 3)), dim=3)
        values = torch.cat((past_value.reshape(2, DIT_HEADS, -1, DIT_HEAD_DIM), v), dim=2)
        attended = torch.matmul(torch.softmax(torch.matmul(q, keys) + mask, dim=-1), values)
        x = x + gate_msa * self.proj(attended.transpose(1, 2).reshape(2, n, DIT_DIM))

        h = modulate(plain_norm(x), shift_conv, scale_conv).transpose(1, 2)  # [2,512,T]
        h = torch.cat((cnn[:, :DIT_DIM], h), dim=2)
        cache1 = torch.matmul(h, select)
        h = self.conv1(h).transpose(1, 2)
        h = functional.mish(self.conv_norm(h)).transpose(1, 2)
        h = torch.cat((cnn[:, DIT_DIM:], h), dim=2)
        cache2 = torch.matmul(h, select)
        x = x + gate_conv * self.conv2(h).transpose(1, 2)

        h = modulate(plain_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.fc2(functional.gelu(self.fc1(h), approximate="tanh"))
        new_key = k.transpose(2, 3).reshape(2 * DIT_HEADS, 1, DIT_HEAD_DIM, n)
        new_value = v.reshape(2 * DIT_HEADS, 1, n, DIT_HEAD_DIM)
        return x, new_key, new_value, torch.cat((cache1, cache2), dim=1)


class DiTStep(nn.Module):
    def __init__(self, state: dict) -> None:
        super().__init__()
        p = "decoder.estimator."
        self.in_proj = linear(state, p + "in_proj")
        self.blocks = nn.ModuleList(DiTBlock(state, i) for i in range(DIT_BLOCKS))
        self.out = linear(state, p + "final_layer.linear")

    def forward(self, x, cond, attention_mask, select, *rest):
        mods, caches = rest[: DIT_BLOCKS + 1], rest[DIT_BLOCKS + 1 :]
        h = self.in_proj(torch.cat((x, cond), dim=1).transpose(1, 2))  # [2,T,512]
        outputs = []
        for i, block in enumerate(self.blocks):
            h, key, value, cnn = block(h, mods[i], attention_mask, select, *caches[3 * i : 3 * i + 3])
            outputs += [key, value, cnn]
        shift, scale = torch.split(mods[-1], 1, dim=1)
        velocity = self.out(modulate(plain_norm(h), shift, scale)).transpose(1, 2)
        return (velocity, *outputs)


def dit_names() -> tuple[list[str], list[str]]:
    inputs = ["x", "cond", "attention_mask", "select"] + [f"mod_{i}" for i in range(DIT_BLOCKS)] + ["mod_final"]
    outputs = ["velocity"]
    for i in range(DIT_BLOCKS):
        inputs += [f"past_key_{i}_in", f"past_value_{i}_in", f"cnn_{i}_in"]
        outputs += [f"past_key_{i}_out", f"past_value_{i}_out", f"cnn_{i}_out"]
    return inputs, outputs


def source(f0: np.ndarray, noise: np.random.Generator, l_linear: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """SourceModuleHnNSF2 on one chunk's f0 [F]: 9 harmonics, unvoiced noise, merged by tanh(linear).

    The random initial phase upstream never survives its 1/480 linear resampling, so it is omitted.
    """
    rate, up, harmonics, amp, noise_std = 24000, 480, 9, 0.1, 0.003
    frames = len(f0)
    fn = f0[:, None].astype(np.float64) * np.arange(1, harmonics + 1)[None]  # [F, 9]
    rad = (fn / rate) % 1  # one value per frame (upstream: per sample after nearest upsampling)
    phase = np.cumsum(rad, axis=0) * 2 * np.pi * up  # cumulative phase at every frame centre
    # linear interpolation x480 with align_corners=False, edges clamped (torch.nn.functional.interpolate)
    pos = (np.arange(frames * up) + 0.5) / up - 0.5
    lo = np.clip(np.floor(pos).astype(np.int64), 0, frames - 1)
    hi = np.clip(lo + 1, 0, frames - 1)
    frac = np.clip(pos - np.floor(pos), 0, 1)
    frac[pos < 0] = 0
    sample_phase = phase[lo] * (1 - frac[:, None]) + phase[hi] * frac[:, None]
    sines = np.sin(sample_phase) * amp
    uv = np.repeat((f0 > 10).astype(np.float64), up)[:, None]
    noise_amp = uv * noise_std + (1 - uv) * amp / 3
    sines = sines * uv + noise_amp * noise.standard_normal(sines.shape)
    weight, bias = l_linear
    return np.tanh(sines @ weight.reshape(-1).astype(np.float64) + float(bias.reshape(-1)[0])).astype(np.float32)


def hann() -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def stft(signal: np.ndarray) -> np.ndarray:
    """torch.stft(n_fft=16, hop=4, hann, center=True, reflect) as [18, frames] (real 0..8 | imag 0..8)."""
    padded = np.pad(signal.astype(np.float64), N_FFT // 2, mode="reflect")
    frames = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None] + HOP * np.arange(frames)[:, None]
    spectrum = np.fft.rfft(padded[index] * hann(), axis=1)  # [frames, 9]
    return np.concatenate([spectrum.real.T, spectrum.imag.T]).astype(np.float32)


def istft(head: np.ndarray) -> np.ndarray:
    """HiFT's tail: exp/sin of conv_post [18, frames], torch.istft(center=True), clamp to +-0.99."""
    magnitude = np.minimum(np.exp(head[:9].astype(np.float64)), 1e2)
    phase = np.sin(head[9:].astype(np.float64))
    frames = head.shape[1]
    window = hann().astype(np.float64)
    chunks = np.fft.irfft((magnitude * np.cos(phase) + 1j * magnitude * np.sin(phase)).T, n=N_FFT, axis=1) * window
    length = N_FFT + HOP * (frames - 1)
    out, norm = np.zeros(length), np.zeros(length)
    for f in range(frames):
        out[f * HOP : f * HOP + N_FFT] += chunks[f]
        norm[f * HOP : f * HOP + N_FFT] += window**2
    out = out[N_FFT // 2 : length - N_FFT // 2] / np.maximum(norm[N_FFT // 2 : length - N_FFT // 2], 1e-11)
    return np.clip(out, -0.99, 0.99).astype(np.float32)


class Token2wavHost:
    """Phone-side token2wav streaming, mirroring stepaudio2 Token2wav.stream with fixed graph shapes.

    encoder(tokens[int64 L], pad, cnn, att) -> (h [1,2L',80], cnn, att) runs the conformer (CPU),
    dit(inputs) one DiT step, f0(mel[1,80,58]) and decode(mel, s_stft) the two HiFT graphs.
    The estimator cache is kept as the reference keeps it: newest frames first; once it exceeds
    prompt+100 frames it is cut to the newest `prompt` frames plus its 100 oldest.
    """

    def __init__(self, constants: dict, encoder, dit, f0, decode, seed: int = 0) -> None:
        self.c, self.encoder, self.dit, self.f0, self.decode = constants, encoder, dit, f0, decode
        self.prompt = int(constants["prompt_frames"])
        self.past = self.prompt + 100
        self.dts = timesteps()[1]
        self.mods = [[*constants["mod_blocks"][s], constants["mod_final"][s]] for s in range(STEPS)]
        self.noise = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        c = self.c
        self.conformer_cnn, self.conformer_att = c["conformer_cnn_cache"], c["conformer_att_cache"]
        self.est_cnn = c["estimator_cnn_cache"].copy()  # [10,16,2,1024,2]
        self.est_att = c["estimator_att_cache"].copy()  # [10,16,2,8,S,128], newest first
        self.mel_cache = np.zeros((MEL, 0), np.float32)
        self.source_cache = np.zeros(0, np.float32)
        self.speech_cache = np.zeros(0, np.float32)

    def estimator(self, mu: np.ndarray) -> np.ndarray:
        n = mu.shape[1]
        if n > T:
            raise ValueError(f"chunk of {n} mel frames exceeds the graph's {T}")
        s = self.est_att.shape[4]
        x = self.c["rand_noise"][:, s : s + n].astype(np.float32)  # [80, n]
        cond = np.zeros((2, 240, T), np.float32)
        cond[0, :MEL, :n] = mu
        cond[0, MEL : 2 * MEL, :n] = self.c["spk"][:, None]
        mask = np.full((1, 1, T, self.past + T), MASK_MIN, np.float32)
        mask[..., :s] = 0
        mask[..., self.past : self.past + n] = 0
        select = np.zeros((1, T + 2, 2), np.float32)
        select[0, n, 0] = select[0, n + 1, 1] = 1
        new_att = np.zeros((STEPS, DIT_BLOCKS, 2, DIT_HEADS, n + s, 128), np.float32)
        for step in range(STEPS):
            xin = np.zeros((2, MEL, T), np.float32)
            xin[:, :, :n] = x
            caches = []
            for b in range(DIT_BLOCKS):
                att = self.est_att[step, b]  # [2,8,s,128]
                key = np.zeros((2 * DIT_HEADS, 1, DIT_HEAD_DIM, self.past), np.float32)
                value = np.zeros((2 * DIT_HEADS, 1, self.past, DIT_HEAD_DIM), np.float32)
                key[:, 0, :, :s] = att[..., :64].reshape(2 * DIT_HEADS, s, 64).transpose(0, 2, 1)
                value[:, 0, :s] = att[..., 64:].reshape(2 * DIT_HEADS, s, 64)
                caches += [key, value, self.est_cnn[step, b]]
            outputs = self.dit([xin, cond, mask, select, *self.mods[step], *caches])
            velocity = outputs[0][:, :, :n]
            x = x + self.dts[step] * ((1 + CFG) * velocity[0] - CFG * velocity[1])
            for b in range(DIT_BLOCKS):
                key, value, cnn = outputs[1 + 3 * b : 4 + 3 * b]
                fresh = np.concatenate([key[:, 0, :, :n].transpose(0, 2, 1), value[:, 0, :n]], axis=-1)
                new_att[step, b, :, :, :n] = fresh.reshape(2, DIT_HEADS, n, 128)
                new_att[step, b, :, :, n:] = self.est_att[step, b]
                self.est_cnn[step, b] = cnn
        self.est_att = new_att
        return x

    def stream(self, tokens: list[int], last_chunk: bool = False) -> np.ndarray:
        pad = LOOKAHEAD if last_chunk else 0
        h, self.conformer_cnn, self.conformer_att = self.encoder(np.asarray(tokens, np.int64), pad, self.conformer_cnn, self.conformer_att)
        mel = self.estimator(h[0].T)  # [80, n]
        if self.est_att.shape[4] > self.past:
            self.est_att = np.concatenate([self.est_att[..., : self.prompt, :], self.est_att[..., -100:, :]], axis=4)
        if self.conformer_att.shape[3] > self.past:
            self.conformer_att = np.concatenate([self.conformer_att[:, :, :, : self.prompt], self.conformer_att[:, :, :, -100:]], axis=3)
        mel = np.concatenate([self.mel_cache, mel], axis=1)
        speech, src = self.vocode(mel)
        if len(self.speech_cache):
            window = np.hamming(2 * SOURCE_CACHE)
            speech[:SOURCE_CACHE] = speech[:SOURCE_CACHE] * window[:SOURCE_CACHE] + self.speech_cache * window[SOURCE_CACHE:]
        first = len(self.speech_cache) == 0
        self.mel_cache, self.source_cache, self.speech_cache = mel[:, -MEL_CACHE:], src[-SOURCE_CACHE:], speech[-SOURCE_CACHE:]
        if not last_chunk:
            speech = np.concatenate([np.zeros(SOURCE_CACHE, np.float32), speech[:-SOURCE_CACHE]]) if first else speech[:-SOURCE_CACHE]
        return speech

    def vocode(self, mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = mel.shape[1]
        padded = np.zeros((1, MEL, HIFT_FRAMES), np.float32)
        if frames > HIFT_FRAMES:
            raise ValueError(f"HiFT chunk of {frames} mel frames exceeds the graph's {HIFT_FRAMES}")
        padded[0, :, :frames] = mel
        f0 = self.f0(padded)[0, :frames]
        src = source(f0, self.noise, (self.c["source_weight"], self.c["source_bias"]))
        src[: len(self.source_cache)] = self.source_cache
        s_stft = np.zeros((1, 2 * (N_FFT // 2 + 1), HIFT_FRAMES * 480 // HOP + 1), np.float32)
        spec = stft(src)
        s_stft[0, :, : spec.shape[1]] = spec
        head = self.decode(padded, s_stft)[0, :, : frames * 480 // HOP + 1]
        return istft(head), src
