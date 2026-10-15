"""Check DLC-derived W4 weights and mapped A16 encodings on ONNX CPU.

This simulator is not an execution of the compiled DLC. Record every encoding
that does not map to the source graph. Preserve the source graph and cache ABI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--sdk-root", required=True, type=Path)
    p.add_argument("--numpy-dir", required=True, type=Path)
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root / "lib/python")]
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    from qti.aisw.dlc_utils import modeltools

    d = args.directory
    if (d / "cpu_simulation.json").exists():
        raise ValueError("CPU checkpoint already exists")
    state = json.loads((d / "quantization.json").read_text())
    encodings = json.loads((d / "model.w4a16_encoding.json").read_text())
    source = Path(state["source_onnx"])
    model = onnx.load(str(source))
    initializers = {t.name: i for i, t in enumerate(model.graph.initializer)}
    reader = modeltools.IrDlcReader()
    reader.open(state["output_dlc"])
    graph = reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    weights = []
    for op in graph.get_ops():
        if op.type != "FullyConnected":
            continue
        t = op.inputs()[1]
        info = t.get_encoding().axisEncInfo
        values = list(info.encInfos)
        if info.axis != 0 or any(v.bw != 4 for v in values):
            raise ValueError("Unexpected projection encoding")
        raw = np.asarray(t.get_data()).astype(np.int16)
        signed = np.where(raw >= 8, raw - 16, raw).astype(np.float32)
        scales = np.asarray([v.scale for v in values], np.float32)[:, None]
        if any(v.offset != 0 for v in values):
            raise ValueError("Unexpected signed weight offset")
        dequantized = np.ascontiguousarray((signed * scales).T)
        original = numpy_helper.to_array(model.graph.initializer[initializers[t.name()]])
        expected_codes = np.clip(np.rint(original.T / scales), -8, 7)
        if not np.array_equal(expected_codes, signed):
            # QAIRT rounds half-way values away from zero on some paths.
            mismatches = expected_codes != signed
            if np.any(np.abs(original.T / scales - signed)[mismatches] > 0.5001):
                raise ValueError("DLC nibble decoding does not match original weight rounding")
        weights.append({"name": t.name(), "shape": list(dequantized.shape), "bitwidth": 4,
                        "relative_rms": float(np.linalg.norm(dequantized - original) /
                                              max(np.linalg.norm(original), 1e-12))})
        model.graph.initializer[initializers[t.name()]].CopyFrom(numpy_helper.from_array(dequantized, t.name()))
    if len(weights) != 8:
        raise ValueError("Expected eight static projection weights")
    # QNN may introduce extra reshape/convert tensors. Record their absence.
    available = {v.name for v in model.graph.input} | {o for n in model.graph.node for o in n.output}
    mapped = {}
    missing = []
    for name, values in encodings["activation_encodings"].items():
        if name not in available:
            missing.append(name)
        elif len(values) != 1 or values[0]["bitwidth"] != 16:
            raise ValueError("Expected scalar A16 encoding")
        else:
            mapped[name] = values[0]
    original_nodes = list(model.graph.node)
    replacements = {name: f"moshi_cpu_qdq_{i}_out" for i, name in enumerate(mapped)}
    nodes = []

    def qdq(name):
        index = list(mapped).index(name)
        prefix = f"moshi_cpu_qdq_{index}"
        e = mapped[name]
        for suffix, value in [("scale", e["scale"]), ("offset", e["offset"]), ("low", 0), ("high", 65535)]:
            model.graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), prefix + "_" + suffix))
        operations = [("Div", [name, prefix + "_scale"], "scaled"),
                      ("Round", [prefix + "_scaled"], "rounded"),
                      ("Sub", [prefix + "_rounded", prefix + "_offset"], "unsigned"),
                      ("Clip", [prefix + "_unsigned", prefix + "_low", prefix + "_high"], "clipped"),
                      ("Add", [prefix + "_clipped", prefix + "_offset"], "shifted"),
                      ("Mul", [prefix + "_shifted", prefix + "_scale"], "out")]
        return [helper.make_node(op, inputs, [prefix + "_" + output], name=prefix + "_" + str(i))
                for i, (op, inputs, output) in enumerate(operations)]

    for value in model.graph.input:
        if value.name in mapped:
            nodes.extend(qdq(value.name))
    for node in original_nodes:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
        nodes.append(node)
        for name in node.output:
            if name in mapped:
                nodes.extend(qdq(name))
    names = [v.name for v in model.graph.output]
    for value in model.graph.output:
        if value.name in replacements:
            value.name = replacements[value.name]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    simulation = d / "cpu_simulation.onnx"
    onnx.save_model(model, str(simulation), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="cpu_simulation.data", size_threshold=1024)
    del model, graph, reader
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(simulation), options, providers=["CPUExecutionProvider"])
    report = {"format": "moshi-dlc-derived-onnx-cpu-check-v1", "weights": weights,
              "mapped_a16_encodings": sorted(mapped), "unmapped_encodings": missing,
              "actual_dlc_execution": False, "device_verified": False,
              "source_onnx_sha256": sha256(source), "candidate_dlc_sha256": state["output_sha256"],
              "relative_rms_limit": 0.10, "samples": [],
              "scope": "Two teacher-forced source-graph simulations. QNN-only transforms are not reproduced."}
    for sample in state["samples"][:2]:
        path = Path(state["calibration_manifest"]).parent / sample["file"]
        with np.load(path) as arrays:
            feed = {v.name: arrays[v.name] for v in reference.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:, :, :1000, :].copy()
        expected = reference.run(None, feed)
        actual = candidate.run(None, feed)
        row = {"source_id": sample["source_id"], "frame": sample["frame"], "outputs": {}}
        for name, got, target in zip(names, actual, expected):
            difference = got.astype(np.float64) - target.astype(np.float64)
            relative = float(np.sqrt(np.mean(difference**2)) /
                             max(float(np.sqrt(np.mean(target.astype(np.float64)**2))), 1e-12))
            row["outputs"][name] = {"finite": bool(np.isfinite(got).all()), "relative_rms": relative,
                "max_abs": float(np.max(np.abs(difference))), "nonzero": int(np.count_nonzero(got)),
                "passed": bool(np.isfinite(got).all() and relative <= report["relative_rms_limit"]
                               and (np.any(got) or not np.any(target)))}
        report["samples"].append(row)
        print(json.dumps(row), flush=True)
    report["passed"] = all(v["passed"] for row in report["samples"] for v in row["outputs"].values())
    write(d / "cpu_simulation.json", report)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
