"""BF16 reference reply for listening comparisons: streaming Mimi + greedy LMGen on a WAV.

Matches the phone pipeline: 24 kHz mono input in 80 ms frames, greedy decoding, and the
generated 8 audio codebooks decoded back to PCM (frames before the first generated one are
dropped, as on the phone).
"""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.lm import LMGen
from qai_hub_models.models.templates.moshi.model import resolve_moshi_checkpoint

SAMPLE_RATE, FRAME = 24000, 1920


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0, help="0 = every complete frame")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.input), "-f", "f32le", "-ac", "1", "-ar",
                          str(SAMPLE_RATE), "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    frames = args.frames or len(audio) // FRAME
    device = torch.device(args.device)
    checkpoint = resolve_moshi_checkpoint(model_dir=args.model_dir)
    mimi = checkpoint.get_mimi(device=device)
    lm = checkpoint.get_moshi(device=device, dtype=torch.bfloat16)
    lm_gen = LMGen(lm, use_sampling=False)
    pcm, texts = [], []
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for t in range(frames):
            chunk = torch.from_numpy(audio[t * FRAME : (t + 1) * FRAME].copy()).to(device).reshape(1, 1, FRAME)
            tokens = lm_gen.step(mimi.encode(chunk))
            if tokens is None:
                continue
            texts.append(int(tokens[0, 0, 0]))
            pcm.append(mimi.decode(tokens[:, 1:]).reshape(-1).float().cpu().numpy())
    samples = (np.clip(np.concatenate(pcm), -1.0, 32767 / 32768) * 32768).round().astype(np.int16)
    with wave.open(str(args.output), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())
    args.output.with_suffix(".text_tokens.txt").write_text("\n".join(map(str, texts)) + "\n")
    print(f"frames={frames} emitted={len(pcm)} seconds={len(samples) / SAMPLE_RATE:.2f}")


if __name__ == "__main__":
    main()
