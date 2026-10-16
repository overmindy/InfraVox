"""Compare recurrent HTP Temporal dumps with original and DLC-derived CPU outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--reference-directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = args.directory
    reference_directory = args.reference_directory or root/"reference"
    reference = json.loads((reference_directory / "phone_reference.json").read_text())
    report = {"format": "moshi-phone-reference-comparison-v1", "candidate_dlc_sha256": reference["candidate_dlc_sha256"],
              "candidate_vs_dlc_cpu_limit": 0.02, "candidate_vs_original_limit": 0.10,
              "baseline_vs_original_limit": 0.02, "frames": [],
              "scope": "Three captured upstream hidden inputs; each implementation updates its own KV"}
    for record in reference["frames"]:
        frame = record["frame"]
        for field in ("original", "quantized"):
            path = reference_directory / record[field+"_file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[field+"_sha256"]:
                raise ValueError("CPU reference file changed")
        with np.load(reference_directory / record["original_file"]) as orig, np.load(reference_directory / record["quantized_file"]) as quant:
            row = {"frame": frame, "outputs": {}}
            for name in orig.files:
                suffix = "hidden" if name == "output_hidden" else name
                source, simulated = orig[name].astype(np.float64), quant[name].astype(np.float64)
                actual = {}
                for variant in ("baseline", "candidate"):
                    path = root / "single" / f"{variant}.csv.frame_{frame}.{suffix}.raw"
                    actual[variant] = np.fromfile(path, dtype="<f4").reshape(source.shape).astype(np.float64)
                def rel(a, b):
                    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))
                errors = {"baseline_vs_original": rel(actual["baseline"], source),
                          "candidate_vs_original": rel(actual["candidate"], source),
                          "candidate_vs_dlc_cpu": rel(actual["candidate"], simulated)}
                finite = all(np.isfinite(v).all() for v in actual.values())
                nonzero = all(np.any(v) or not np.any(source) for v in actual.values())
                passed = finite and nonzero and all(value <= report[key+"_limit"] for key, value in errors.items())
                row["outputs"][name] = dict(errors, finite=finite, nonzero=nonzero, passed=bool(passed))
            report["frames"].append(row)
    report["passed"] = all(v["passed"] for row in report["frames"] for v in row["outputs"].values())
    report["max_errors"] = {key: max(v[key] for row in report["frames"] for v in row["outputs"].values())
                            for key in ("baseline_vs_original", "candidate_vs_original", "candidate_vs_dlc_cpu")}
    (args.output or root / "single_comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "max_errors": report["max_errors"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
