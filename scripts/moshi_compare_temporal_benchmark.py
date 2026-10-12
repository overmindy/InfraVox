"""Compare small frame-0 HTP benchmark dumps to the existing guarded QDQ ONNX."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--dump-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-relative-rms", type=float, default=0.02)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    with np.load(args.sample) as data:
        feed = {v.name: data[v.name] for v in session.get_inputs()}
    if int(feed["position"].item()) != 0 or any(np.count_nonzero(v) for k, v in feed.items() if k.startswith("layer_")):
        raise ValueError("Benchmark dump uses frame-0 position with zero incoming caches")
    report = {"sample": str(args.sample), "reference": str(args.model), "outputs": {},
              "max_relative_rms_threshold": args.max_relative_rms}
    for output, expected in zip(session.get_outputs(), session.run(None, feed)):
        suffix = "hidden" if output.name == "output_hidden" else output.name
        if output.name.startswith("output_layer_"):
            expected = expected[:, :, :1, :]
        actual = np.fromfile(str(args.dump_prefix) + "." + suffix + ".raw", dtype="<f4")
        if actual.size != expected.size:
            raise ValueError(f"Dump shape mismatch: {output.name}")
        actual = actual.reshape(expected.shape).astype(np.float64)
        expected = expected.astype(np.float64)
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        error = actual - expected
        rmse = float(np.sqrt(np.mean(error * error)))
        rms = float(np.sqrt(np.mean(expected * expected)))
        relative = rmse / max(rms, 1e-12)
        report["outputs"][output.name] = {"finite": finite, "rmse": rmse,
            "relative_rms": relative, "max_abs": float(np.max(np.abs(error))),
            "passed": finite and relative <= args.max_relative_rms}
    report["passed"] = all(v["passed"] for v in report["outputs"].values())
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
