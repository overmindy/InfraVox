"""Pack real consecutive calibration hidden tensors for a native recurrent probe."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; do not overwrite captured inputs")
    frames = []
    for position, sample in enumerate(args.samples):
        with np.load(sample) as data:
            value = data["hidden"]
            if int(data["position"].item()) != position or value.shape != (1, 1, 4096) or not np.isfinite(value).all():
                raise ValueError("Expected consecutive real frames starting at zero")
            frames.append(value.astype("<f4"))
    np.concatenate(frames, axis=1).tofile(args.output)
    print(f"PASS packed {len(frames)} real hidden frames into {args.output}")


if __name__ == "__main__":
    main()
