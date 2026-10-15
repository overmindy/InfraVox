"""Lower a bounded number of measured projections to W4. Keep the rest W8."""
import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-onnx", "sensitivity-dir", "calibration-dir", "w4-minmax-encodings", "w4-mse-policy", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--w4-budget", type=int, default=1)
    p.add_argument("--local-w4-error-cap", type=float, default=0.15)
    args = p.parse_args()
    if not 1 <= args.w4_budget <= 8 or not 0 < args.local_w4_error_cap < 1:
        p.error("Use a W4 budget from one to eight and a local cap between zero and one")
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from moshi_quantize_temporal_offline import sha256, write

    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.sensitivity_dir / "RESULT.json").read_text())
    if sha256(args.source_onnx) != report["source_onnx_sha256"]:
        raise ValueError("Sensitivity report belongs to a different source")
    eligible = [v for v in report["layers"] if v["bitwidth"] == 8
                and v["w4"]["max_relative_rms"] <= args.local_w4_error_cap]
    selected = sorted(eligible, key=lambda v: v["w4"]["max_relative_rms"])[:args.w4_budget]
    if len(selected) != args.w4_budget:
        raise ValueError("Not enough measured projections satisfy the local error cap")
    policy = json.loads((args.sensitivity_dir / "mixed_policy.json").read_text())
    source = onnx.load(str(args.source_onnx), load_external_data=False)
    calibration = json.loads((args.calibration_dir / "calibration.json").read_text())
    if calibration["source_sha256"] != report["source_onnx_sha256"]:
        raise ValueError("Calibration source mismatch")
    samples = []
    for item in calibration["projection_samples"]:
        path = args.calibration_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Projection sample changed")
        with np.load(path) as data:
            samples.append({name: data[name] for name in data.files})
    all_alternatives = [json.loads(path.read_text())["param_encodings"]
                        for path in (args.w4_minmax_encodings, args.w4_mse_policy)]
    for layer in selected:
        node = next(n for n in source.graph.node if n.name == layer["node"])
        tensor = next(t for t in source.graph.initializer if t.name == layer["name"])
        weight = numpy_helper.to_array(tensor, base_dir=str(args.source_onnx.parent))
        x = np.concatenate([v[node.input[0]].reshape(-1, weight.shape[0]) for v in samples])
        expected = np.concatenate([v[node.output[0]].reshape(-1, weight.shape[1]) for v in samples])
        alternatives = [v[layer["name"]] for v in all_alternatives]
        errors = []
        for values in alternatives:
            scale = np.asarray([v["scale"] for v in values], np.float32)[None, :]
            predicted = x @ (np.clip(np.rint(weight / scale), -8, 7) * scale)
            errors.append(np.mean((predicted.astype(np.float64)-expected)**2, axis=0))
        prefer_mse = errors[1] < errors[0]
        policy["param_encodings"][layer["name"]] = [copy.deepcopy(alternatives[int(value)][i])
                                                   for i, value in enumerate(prefer_mse)]
        print(layer["node"], "W4", flush=True)
        del weight
    selected_names = {v["name"] for v in selected}
    for row in report["layers"]:
        if row["name"] in selected_names:
            row["bitwidth"] = 4
    count = len(selected)
    total = sum(v["parameter_count"] for v in report["layers"])
    report.update(status="bounded_w4_policy_selected", w4_projection_budget=args.w4_budget,
                  selection_reason="Lower measured low-sensitivity projections together, then require the whole-shard gate.",
                  local_w4_error_cap=args.local_w4_error_cap, whole_shard_error_limit=0.10,
                  script_sha256=sha256(__file__), selected_w4=[v["node"] for v in selected],
                  w4_parameter_fraction=sum(v["parameter_count"] for v in selected)/total,
                  selected_precision={"4": count, "8": 8-count, "32": 0})
    write(root / "RESULT.json", report)
    write(root / "mixed_policy.json", policy)
    write(root / "weight_statistics.json", report["layers"])
    print(count, "projections W4; all other projections W8", flush=True)


if __name__ == "__main__":
    main()
