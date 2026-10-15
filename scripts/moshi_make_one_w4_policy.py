"""Lower the least sensitive projection to W4, retaining the W8 control elsewhere."""
import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-onnx", "sensitivity-dir", "calibration-dir", "w4-minmax-encodings", "w4-mse-policy", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from moshi_quantize_temporal_offline import sha256, write

    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True)
    report = json.loads((args.sensitivity_dir / "RESULT.json").read_text())
    if sha256(args.source_onnx) != report["source_onnx_sha256"]:
        raise ValueError("Sensitivity report belongs to a different source")
    eligible = [v for v in report["layers"] if v["bitwidth"] == 8 and v["w4"]["max_relative_rms"] <= 0.15]
    selected = min(eligible, key=lambda v: v["w4"]["max_relative_rms"])
    policy = json.loads((args.sensitivity_dir / "mixed_policy.json").read_text())
    source = onnx.load(str(args.source_onnx), load_external_data=False)
    node = next(n for n in source.graph.node if n.name == selected["node"])
    tensor = next(t for t in source.graph.initializer if t.name == selected["name"])
    weight = numpy_helper.to_array(tensor, base_dir=str(args.source_onnx.parent))
    calibration = json.loads((args.calibration_dir / "calibration.json").read_text())
    if calibration["source_sha256"] != report["source_onnx_sha256"]:
        raise ValueError("Calibration source mismatch")
    inputs, outputs = [], []
    for item in calibration["projection_samples"]:
        path = args.calibration_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Projection sample changed")
        with np.load(path) as data:
            inputs.append(data[node.input[0]].reshape(-1, weight.shape[0]))
            outputs.append(data[node.output[0]].reshape(-1, weight.shape[1]))
    x, expected = np.concatenate(inputs), np.concatenate(outputs)
    alternatives = [json.loads(path.read_text())["param_encodings"][selected["name"]]
                    for path in (args.w4_minmax_encodings, args.w4_mse_policy)]
    errors = []
    for values in alternatives:
        scale = np.asarray([v["scale"] for v in values], np.float32)[None, :]
        predicted = x @ (np.clip(np.rint(weight / scale), -8, 7) * scale)
        errors.append(np.mean((predicted.astype(np.float64)-expected)**2, axis=0))
    prefer_mse = errors[1] < errors[0]
    policy["param_encodings"][selected["name"]] = [copy.deepcopy(alternatives[int(value)][i])
                                                 for i, value in enumerate(prefer_mse)]
    for row in report["layers"]:
        if row["name"] == selected["name"]:
            row["bitwidth"] = 4
    report.update(status="one_projection_w4_policy_selected", w4_projection_budget=1,
                  selection_reason="Lowest measured W4 error. Only this projection is lowered before one whole-shard gate.",
                  local_w4_error_cap=0.15, whole_shard_error_limit=0.10,
                  script_sha256=sha256(__file__), selected_w4=selected["node"],
                  selected_precision={"4": 1, "8": 7, "32": 0})
    write(root / "RESULT.json", report)
    write(root / "mixed_policy.json", policy)
    write(root / "weight_statistics.json", report["layers"])
    print(selected["node"], "W4; all other projections W8", flush=True)


if __name__ == "__main__":
    main()
