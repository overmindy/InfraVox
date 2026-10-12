"""Compare native captured-hidden replay dumps from two real DLC variants.

Both runs must use identical hidden inputs and zero initial caches. Each run
feeds its OWN new cache slot back into its next frame; this is not a complete
LM chain or a comparison to the BF16 reference. No third-party modules needed.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def compare(reference, actual):
    left, right = reference.read_bytes(), actual.read_bytes()
    if not left or len(left) != len(right) or len(left) % 4:
        raise ValueError(f"Missing/unequal float32 dump sizes: {reference}, {actual}")
    values = list(zip(struct.iter_unpack("<f", left), struct.iter_unpack("<f", right)))
    finite = all(math.isfinite(a[0]) and math.isfinite(b[0]) for a, b in values)
    if not finite:
        return {"finite": False, "byte_identical": left == right}
    squared_error = math.fsum((a[0] - b[0]) ** 2 for a, b in values)
    squared_reference = math.fsum(a[0] ** 2 for a, _ in values)
    rmse = math.sqrt(squared_error / len(values))
    return {"finite": True, "byte_identical": left == right,
            "elements": len(values), "rmse": rmse,
            "relative_rms": rmse / max(math.sqrt(squared_reference / len(values)), 1e-12),
            "max_abs": max(abs(a[0] - b[0]) for a, b in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--actual-prefix", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--max-relative-rms", type=float, default=0.02)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.first_layer <= 30 or args.first_layer % 2 or not 1 <= args.frames <= 20:
        raise ValueError("Expected a two-layer Temporal shard and 1..20 frames")
    if not 0 < args.max_relative_rms <= 0.02:
        raise ValueError("Device checkpoint must be in (0, 0.02]")
    names = ["hidden", *(f"output_layer_{layer}_{kind}"
              for layer in (args.first_layer, args.first_layer + 1) for kind in ("key", "value"))]
    report = {"format": "moshi-device-recurrent-dump-comparison-v1",
              "reference_prefix": str(args.reference_prefix), "actual_prefix": str(args.actual_prefix),
              "max_relative_rms_checkpoint": args.max_relative_rms, "frames": [],
              "complete_lm_chain_verified": False}
    for frame in range(args.frames):
        outputs = {}
        for name in names:
            suffix = f".frame_{frame}.{name}.raw"
            metrics = compare(Path(str(args.reference_prefix) + suffix), Path(str(args.actual_prefix) + suffix))
            metrics["passed"] = metrics["finite"] and metrics["relative_rms"] <= args.max_relative_rms
            outputs[name] = metrics
        report["frames"].append({"frame": frame, "outputs": outputs})
    report["passed"] = all(m["passed"] for f in report["frames"] for m in f["outputs"].values())
    report["all_byte_identical"] = all(m["byte_identical"] for f in report["frames"] for m in f["outputs"].values())
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
