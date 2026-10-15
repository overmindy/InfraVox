"""Isolate W4 weight error without rebuilding weights or repeating calibration."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    from moshi_quantize_temporal_offline import write

    d = args.directory
    report_path = d / "cpu_weights_only.json"
    if report_path.exists():
        raise ValueError("Diagnostic exists. Do not repeat it")
    model = onnx.load(str(d / "cpu_simulation.onnx"), load_external_data=False)
    replacements = {n.output[0].removesuffix("_scaled") + "_out": n.input[0]
                    for n in model.graph.node if n.name.startswith("moshi_cpu_qdq_") and n.op_type == "Div"}
    nodes = []
    for node in model.graph.node:
        if node.name.startswith("moshi_cpu_qdq_"):
            continue
        for index, value in enumerate(node.input):
            node.input[index] = replacements.get(value, value)
        nodes.append(node)
    for value in model.graph.output:
        value.name = replacements.get(value.name, value.name)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    path = d / "cpu_weights_only.onnx"
    # Keep the exact dequantized DLC weights in the existing external file.
    onnx.save_model(model, str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    state = json.loads((d / "quantization.json").read_text())
    reference = ort.InferenceSession(state["source_onnx"], options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    rows = []
    for sample in state["samples"][:2]:
        source = Path(state["calibration_manifest"]).parent / sample["file"]
        with np.load(source) as data:
            feed = {v.name: data[v.name] for v in reference.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:, :, :1000, :].copy()
        expected, actual = reference.run(None, feed), candidate.run(None, feed)
        row = {"frame": sample["frame"], "outputs": {}}
        for output, got, target in zip(reference.get_outputs(), actual, expected):
            error = got.astype(np.float64) - target.astype(np.float64)
            row["outputs"][output.name] = {"finite": bool(np.isfinite(got).all()),
                "relative_rms": float(np.sqrt(np.mean(error**2)) /
                                      max(float(np.sqrt(np.mean(target.astype(np.float64)**2))), 1e-12))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    write(report_path, {"diagnostic_only": True, "activation_precision": "source float",
                       "candidate_dlc_sha256": state["output_sha256"], "samples": rows,
                       "actual_dlc_execution": False, "device_verified": False})


if __name__ == "__main__":
    main()
