"""Select W4/W8/float by isolated projection error on retained calibration data."""
import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-onnx", "calibration-dir", "w4-minmax-encodings", "w4-mse-policy", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--relative-rms-limit", type=float, default=0.10)
    args = p.parse_args()
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from moshi_quantize_temporal_offline import sha256, write

    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh sensitivity directory")
    root.mkdir(parents=True)
    provenance = json.loads((args.calibration_dir / "calibration.json").read_text())
    source_hash = sha256(args.source_onnx)
    if source_hash != provenance["source_sha256"]:
        raise ValueError("Projection samples have a different source")
    samples = []
    for item in provenance["projection_samples"]:
        path = args.calibration_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Projection sample changed")
        with np.load(path) as data:
            samples.append({name: data[name] for name in data.files})
    if len(samples) != 16:
        raise ValueError("Expected the existing sixteen calibration samples")
    model = onnx.load(str(args.source_onnx))
    initializers = {t.name: t for t in model.graph.initializer}
    minmax = json.loads(args.w4_minmax_encodings.read_text())["param_encodings"]
    mse = json.loads(args.w4_mse_policy.read_text())["param_encodings"]
    policy = {"version": "0.6.1", "activation_encodings": {}, "param_encodings": {}}
    report = {"source_onnx_sha256": source_hash, "script_sha256": sha256(__file__),
              "relative_rms_limit": args.relative_rms_limit,
              "method": "Choose W4 scales per output channel from two existing encodings. Raise sensitive projections to W8 or float.",
              "calibration_samples": provenance["projection_samples"], "quality_test": False,
              "layers": [], "device_verified": False}

    def metrics(actual, expected, groups):
        rows = []
        for index in range(groups):
            a = np.array_split(actual, groups, axis=1)[index].astype(np.float64)
            b = np.array_split(expected, groups, axis=1)[index].astype(np.float64)
            per_sample = np.sqrt(np.mean((a-b)**2, axis=1)) / np.maximum(np.sqrt(np.mean(b*b, axis=1)), 1e-12)
            rows.append({"group": index, "max_relative_rms": float(per_sample.max()),
                         "per_sample_relative_rms": per_sample.tolist()})
        return {"groups": rows, "max_relative_rms": max(v["max_relative_rms"] for v in rows),
                "finite": bool(np.isfinite(actual).all())}

    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in initializers:
            continue
        name = node.input[1]
        weight = numpy_helper.to_array(initializers[name])
        x = np.concatenate([v[node.input[0]].reshape(-1, weight.shape[0]) for v in samples])
        expected = np.concatenate([v[node.output[0]].reshape(-1, weight.shape[1]) for v in samples])
        baseline = x @ weight
        baseline_error = float(np.linalg.norm(baseline-expected) / max(np.linalg.norm(expected), 1e-12))
        if baseline_error > 1e-4:
            raise ValueError("Projection replay does not match the source reference")
        candidates = []
        for encodings in (minmax[name], mse[name]):
            scale = np.asarray([v["scale"] for v in encodings], np.float32)[None, :]
            reconstructed = np.clip(np.rint(weight / scale), -8, 7) * scale
            candidates.append(x @ reconstructed)
            del reconstructed
        error0 = np.mean((candidates[0].astype(np.float64)-expected)**2, axis=0)
        error1 = np.mean((candidates[1].astype(np.float64)-expected)**2, axis=0)
        prefer_mse = error1 < error0
        w4_output = np.where(prefer_mse[None, :], candidates[1], candidates[0])
        groups = 3 if node.name.startswith("/in_projs") else (2 if weight.shape[1] == 22528 else 1)
        w4 = metrics(w4_output, expected, groups)
        scale8 = np.maximum(np.maximum(-weight.min(axis=0)/128, weight.max(axis=0)/127), np.finfo(np.float32).tiny)
        reconstructed8 = np.clip(np.rint(weight / scale8[None, :]), -128, 127) * scale8[None, :]
        w8 = metrics(x @ reconstructed8, expected, groups)
        del reconstructed8
        if w4["finite"] and w4["max_relative_rms"] <= args.relative_rms_limit:
            bits = 4
            selected = [copy.deepcopy(mse[name][i] if use_mse else minmax[name][i])
                        for i, use_mse in enumerate(prefer_mse)]
        elif w8["finite"] and w8["max_relative_rms"] <= args.relative_rms_limit:
            bits = 8
            selected = [{"bitwidth": 8, "is_symmetric": "true", "min": float(-128*s),
                         "max": float(127*s), "scale": float(s), "offset": 0} for s in scale8]
        else:
            bits = 32
            selected = [{"bitwidth": 32, "dtype": "float"}]
        policy["param_encodings"][name] = selected
        layer = {"name": name, "node": node.name, "shape": list(weight.shape), "bitwidth": bits,
                 "source_replay_relative_rms": baseline_error, "w4": w4, "w8": w8,
                 "w4_channels_using_mse": int(prefer_mse.sum()), "parameter_count": int(weight.size)}
        report["layers"].append(layer)
        print(node.name, "W4 worst", w4["max_relative_rms"], "W8 worst", w8["max_relative_rms"], "selected", bits, flush=True)
        write(root / "RESULT.json", report)
    if len(report["layers"]) != 8:
        raise ValueError("Expected eight projections")
    write(root / "mixed_policy.json", policy)
    write(root / "weight_statistics.json", report["layers"])
    report["selected_precision"] = {str(bits): sum(v["bitwidth"] == bits for v in report["layers"]) for bits in (4,8,32)}
    report["status"] = "sensitivity_policy_selected"
    write(root / "RESULT.json", report)


if __name__ == "__main__":
    main()
