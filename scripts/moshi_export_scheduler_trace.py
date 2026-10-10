#!/usr/bin/env python3
"""Export captured greedy LMGen data for the native delay-ring regression test."""
import argparse
from pathlib import Path


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tensors = {
        name: torch.load(args.trace_dir / (name + ".pt"), map_location="cpu", weights_only=True)
        for name in ("mimi_codes", "temporal_sequence", "text_logits", "depformer_logits",
                     "text_tokens", "audio_codes")
    }
    frames = tensors["mimi_codes"].shape[-1]
    shapes = {
        "mimi_codes": (1, 8, frames), "temporal_sequence": (1, 17, frames),
        "text_logits": (1, 1, frames, 32000),
        "depformer_logits": (1, frames, 8, 1, 2048),
        "text_tokens": (1, 1, frames - 1), "audio_codes": (1, 8, frames - 1),
    }
    for name, shape in shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name}: expected {shape}, got {tuple(tensors[name].shape)}")
    for name in ("text_logits", "depformer_logits"):
        if not torch.isfinite(tensors[name]).all():
            raise ValueError(f"Nonfinite captured {name}")
    text = tensors["text_logits"].argmax(-1)[0, 0]
    audio = tensors["depformer_logits"].argmax(-1)[0, :, :, 0]
    rows = [str(frames)]
    for frame in range(frames):
        emitted = [-1] * 9 if frame == 0 else (
            [int(tensors["text_tokens"][0, 0, frame - 1])]
            + tensors["audio_codes"][0, :, frame - 1].tolist())
        values = (tensors["mimi_codes"][0, :, frame].tolist()
                  + [int(text[frame])] + audio[frame].tolist()
                  + tensors["temporal_sequence"][0, :, frame].tolist() + emitted)
        rows.append(" ".join(str(int(value)) for value in values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n")
    print(f"Exported {frames} captured frames to {args.output}")


if __name__ == "__main__":
    main()
