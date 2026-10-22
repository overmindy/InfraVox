# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run BF16 Moshi (streaming Mimi + greedy LMGen) on LibriSpeech and record Temporal boundaries.

For every frame, saves the residual stream entering layers 0/8/16/24 and leaving layer 31
(the four 8-layer part boundaries). Clips are the first 10-16 s utterance of each of the first
8 speakers (sorted) in train-clean-100, plus 3 s of silence that covers Moshi's replies.
Run with NO_CUDA_GRAPH=1: the boundary hooks copy to the host, which CUDA graph capture forbids.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.lm import (
    LMGen,
)
from qai_hub_models.models.templates.moshi.model import (
    FRAME_SAMPLES,
    resolve_moshi_checkpoint,
)

SAMPLE_RATE = 24000
BOUNDARIES = (0, 8, 16, 24)
SILENCE_FRAMES = 37
CLIPS = 8


def duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def pick(root: Path) -> list[Path]:
    chosen = []
    for speaker in sorted(p for p in root.iterdir() if p.is_dir()):
        clip = next(
            (f for f in sorted(speaker.rglob("*.flac")) if 10 <= duration(f) <= 16),
            None,
        )
        chosen += [clip] if clip else []
        if len(chosen) == CLIPS:
            return chosen
    raise SystemExit("Not enough speakers")


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    audio = audio[: len(audio) // FRAME_SAMPLES * FRAME_SAMPLES]
    return np.concatenate(
        [audio, np.zeros(SILENCE_FRAMES * FRAME_SAMPLES, dtype=np.float32)]
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--librispeech-dir", type=Path, required=True, help="train-clean-100"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = resolve_moshi_checkpoint(model_dir=args.model_dir)
    mimi = checkpoint.get_mimi(device=device)
    lm = checkpoint.get_moshi(device=device, dtype=torch.bfloat16)
    captured: list[torch.Tensor] = []
    for index in BOUNDARIES:
        lm.transformer.layers[index].register_forward_pre_hook(
            lambda _m, inputs: captured.append(inputs[0][0, -1].float().cpu())
        )
    lm.transformer.layers[-1].register_forward_hook(
        lambda _m, _i, output: captured.append(output[0, -1].float().cpu())
    )

    manifest = []
    for path in pick(args.librispeech_dir):
        audio = torch.from_numpy(decode(path)).to(device)
        hidden = []
        lm_gen = LMGen(lm, use_sampling=False)
        with mimi.streaming(1), lm_gen.streaming(1):
            for t in range(audio.numel() // FRAME_SAMPLES):
                captured.clear()
                lm_gen.step(
                    mimi.encode(
                        audio[t * FRAME_SAMPLES : (t + 1) * FRAME_SAMPLES].reshape(
                            1, 1, -1
                        )
                    )
                )
                if len(captured) != len(BOUNDARIES) + 1:
                    raise RuntimeError(
                        f"Expected one Temporal step per frame, captured {len(captured)}"
                    )
                hidden.append(torch.stack(captured).numpy())
        name = f"calibration_{path.stem}"
        np.savez(args.output_dir / f"{name}.npz", hidden=np.stack(hidden))
        manifest.append(
            {
                "name": name,
                "split": "calibration",
                "source": str(path),
                "frames": len(hidden),
            }
        )
        print(json.dumps(manifest[-1]), flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
