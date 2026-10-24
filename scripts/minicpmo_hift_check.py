# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Checks the CPU half of HiFT in minicpmo_flow.py (NSF source, STFT, iSTFT) against stepaudio2's
HiFTGenerator on random inputs, with its Gaussian noise switched off on both sides; needs the
minicpmo env. Also writes OUT/hift_vectors.npz for the phone runner's host test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator

from minicpmo_flow import istft, source, stft


class NoNoise:
    def standard_normal(self, shape):
        return np.zeros(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    hift = HiFTGenerator()
    state = torch.load(args.model_dir / "assets" / "token2wav" / "hift.pt", map_location="cpu", weights_only=True)
    hift.load_state_dict({k.replace("generator.", ""): v for k, v in state.items()})
    hift.eval()
    rng = np.random.default_rng(0)
    f0 = np.where(rng.random(58) < 0.7, rng.uniform(80, 300, 58), 0.0).astype(np.float32)
    real_randn = torch.randn_like
    torch.randn_like = lambda x, **kw: torch.zeros_like(x)
    try:
        with torch.no_grad():
            s = hift.f0_upsamp(torch.from_numpy(f0)[None, None]).transpose(1, 2)
            reference_source = hift.m_source(s)[0][0, :, 0].numpy()
    finally:
        torch.randn_like = real_randn
    weight, bias = hift.m_source.l_linear.weight.detach().numpy(), hift.m_source.l_linear.bias.detach().numpy()
    ours = source(f0, NoNoise(), (weight, bias))
    print(f"NSF source: max abs error {np.abs(ours - reference_source).max():.2e}", flush=True)

    signal = rng.standard_normal(58 * 480).astype(np.float32) * 0.1
    real, imag = hift._stft(torch.from_numpy(signal)[None])
    reference_stft = torch.cat([real, imag], dim=1)[0].numpy()
    print(f"STFT: max abs error {np.abs(stft(signal) - reference_stft).max():.2e}", flush=True)

    head = rng.standard_normal((18, 58 * 120 + 1)).astype(np.float32)
    magnitude = torch.exp(torch.from_numpy(head[None, :9]))
    phase = torch.sin(torch.from_numpy(head[None, 9:]))
    reference_wave = torch.clamp(hift._istft(magnitude, phase), -0.99, 0.99)[0].numpy()
    print(f"iSTFT: max abs error {np.abs(istft(head) - reference_wave).max():.2e}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "hift_vectors.npz", f0=f0, source=ours, weight=weight.reshape(-1), bias=bias.reshape(-1),
             signal=signal, stft=stft(signal), head=head, wave=istft(head))


if __name__ == "__main__":
    main()
