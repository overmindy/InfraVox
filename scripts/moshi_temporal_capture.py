"""Run BF16 Moshi (streaming Mimi + greedy LMGen) on LibriSpeech and record Temporal boundaries.

For every frame, saves the residual stream entering layers 0/8/16/24 and leaving layer 31
(the four 8-layer part boundaries), plus generated tokens. Calibration and held-out clips
come from disjoint speakers; 3 s of silence after each utterance covers Moshi's replies.
Run with NO_CUDA_GRAPH=1: the boundary hooks copy to the host, which CUDA graph capture forbids.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.lm import LMGen
from qai_hub_models.models.templates.moshi.model import resolve_moshi_checkpoint

SAMPLE_RATE, FRAME = 24000, 1920
BOUNDARIES = (0, 8, 16, 24)
SILENCE_FRAMES = 37


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def pick(root: Path, speakers: int, low: float, high: float, skip: set[str]) -> list[Path]:
    """First utterance of low..high seconds for each of the first `speakers` speakers (sorted)."""
    chosen = []
    for speaker in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in skip):
        for flac in sorted(speaker.rglob("*.flac")):
            if low <= duration(flac) <= high:
                chosen.append(flac)
                break
        if len(chosen) == speakers:
            break
    return chosen


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    audio = audio[: len(audio) // FRAME * FRAME]
    return np.concatenate([audio, np.zeros(SILENCE_FRAMES * FRAME, dtype=np.float32)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-roots", type=Path, nargs=2, required=True, help="train-clean-100 train-other-500")
    parser.add_argument("--test-roots", type=Path, nargs=2, required=True, help="test-clean test-other")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    calibration = pick(args.train_roots[0], 16, 10, 16, set()) + pick(args.train_roots[1], 8, 10, 16, set())
    calibration_speakers = {p.parts[-3] for p in calibration}
    held_out = (pick(args.test_roots[0], 4, 10, 16, calibration_speakers)
                + pick(args.test_roots[1], 4, 10, 16, calibration_speakers))
    clips = [("calibration", p) for p in calibration] + [("held_out", p) for p in held_out]

    device = torch.device(args.device)
    checkpoint = resolve_moshi_checkpoint(model_dir=args.model_dir)
    mimi = checkpoint.get_mimi(device=device)
    lm = checkpoint.get_moshi(device=device, dtype=torch.bfloat16)
    captured: list[torch.Tensor] = []
    for index in BOUNDARIES:
        lm.transformer.layers[index].register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0][0, -1].float().cpu()))
    lm.transformer.layers[-1].register_forward_hook(lambda _m, _i, output: captured.append(output[0, -1].float().cpu()))

    manifest = []
    for split, path in clips:
        audio = torch.from_numpy(decode(path)).to(device)
        frames = audio.numel() // FRAME
        hidden, tokens = [], []
        lm_gen = LMGen(lm, use_sampling=False)
        with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
            for t in range(frames):
                captured.clear()
                codes = mimi.encode(audio[t * FRAME : (t + 1) * FRAME].reshape(1, 1, FRAME))
                out = lm_gen.step(codes)
                if len(captured) != len(BOUNDARIES) + 1:
                    raise RuntimeError(f"Expected one Temporal step per frame, captured {len(captured)}")
                hidden.append(torch.stack(captured).numpy())
                tokens.append(out[0, :, 0].cpu().numpy() if out is not None else np.full(9, -1))
        name = f"{split}_{path.stem}"
        np.savez(args.output_dir / f"{name}.npz", hidden=np.stack(hidden), tokens=np.stack(tokens))
        manifest.append({"name": name, "split": split, "source": str(path), "frames": frames,
                         "speech_frames": frames - SILENCE_FRAMES})
        print(json.dumps(manifest[-1]), flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
