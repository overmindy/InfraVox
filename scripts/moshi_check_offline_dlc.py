"""Audit real DLC precision and compare two CPU outputs to its source ONNX."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from moshi_quantize_temporal_offline import sha256, write


def audit(path):
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.dlc_utils.snpe_dlc_utils import is_axis_scale_offset_encoding, is_scale_offset_encoding
    reader = modeltools.IrDlcReader()
    reader.open(str(path))
    graphs = []
    for name in reader.get_ir_graph_names():
        graph = reader.get_ir_graph(name)

        def tensor(t):
            encoding = t.get_encoding()
            values = []
            axis = None
            block = {}
            if is_scale_offset_encoding(encoding.type):
                values = [encoding.encInfo]
            elif is_axis_scale_offset_encoding(encoding.type):
                values = list(encoding.axisEncInfo.encInfos)
                axis = encoding.axisEncInfo.axis
            elif "BLOCKWISE_EXPANSION" in str(encoding.type):
                info = encoding.lpbqEncInfo
                values = list(info.encInfos)
                axis = info.axis
                block = {"block_size":info.blockSize,"blocks_per_channel":info.numBlocksPerAxis,
                         "compressed_bitwidth":values[0].bw-info.blockScaleBitwidth,
                         "block_scale_bitwidth":info.blockScaleBitwidth}
            return {"name": t.name(), "dtype": t.data_type_string(), "shape": t.dims(),
                    "static": t.is_static(), "data_bytes": t.get_data_size() if t.is_static() else None,
                    "encoding": str(encoding.type), "axis": axis,
                    "bitwidths": sorted(set(int(v.bw) for v in values)),
                    "encoding_count": len(values), **block}

        rows = [{"name": op.name, "type": op.type,
                 "inputs": [tensor(t) for t in op.inputs()],
                 "outputs": [tensor(t) for t in op.outputs()]}
                for op in graph.get_ops() if op.type in ("FullyConnected", "MatMul", "Conv2d")]
        state_updates = [{"name": op.name, "type": op.type,
                          "inputs": [tensor(t) for t in op.inputs()],
                          "outputs": [tensor(t) for t in op.outputs()]}
                         for op in graph.get_ops() if "Scatter" in op.type]
        normalization_ops = [{"name": op.name, "type": op.type,
                              "input_dtypes": [t.data_type_string() for t in op.inputs()],
                              "output_dtypes": [t.data_type_string() for t in op.outputs()]}
                             for op in graph.get_ops() if op.name.startswith("/norm") or any(t.name().startswith("/norm") for t in op.outputs())]
        graphs.append({"name": name, "projections": rows, "state_updates": state_updates,
                       "normalization_ops": normalization_ops})
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "graphs": graphs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--sdk-root", type=Path, required=True)
    p.add_argument("--numpy-dir", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root / "lib/python")]
    import numpy as np
    import onnxruntime as ort
    state = json.loads((args.directory / "quantization.json").read_text())
    if state["status"] != "quantized_not_verified":
        raise ValueError("Quantization did not complete")
    candidate = Path(state["output_dlc"])
    report = {"source": audit(Path(state["source_dlc"])), "candidate": audit(candidate),
              "cpu_verified": False, "device_verified": False}
    projections = [row for graph in report["candidate"]["graphs"] for row in graph["projections"]
                   if len(row["inputs"]) > 1 and row["inputs"][1]["static"]]
    report["all_projection_weights_w4"] = len(projections) == 8 and all(
        row["inputs"][1]["bitwidths"] == [4] for row in projections)
    report["projection_count"] = len(projections)
    write(args.directory / "precision_audit.json", report)
    print("Real DLC W4 projection audit:", report["all_projection_weights_w4"], flush=True)
    if args.audit_only:
        return
    if (args.directory / "cpu_check.json").exists():
        raise ValueError("CPU checkpoint exists. Inspect it instead of repeating")
    env = dict(os.environ)
    lib = args.sdk_root / "lib/x86_64-linux-clang"
    env["LD_LIBRARY_PATH"] = str(lib)
    command = [str(args.sdk_root / "bin/x86_64-linux-clang/qnn-net-run"),
               "--model", str(lib / "libQnnModelDlc.so"), "--dlc_path", str(candidate),
               "--backend", str(lib / "libQnnCpu.so"),
               "--input_list", str(args.directory / "check_input_list.txt"),
               "--use_native_input_files", "--output_dir", str(args.directory / "cpu_output"),
               "--log_level", "warn"]
    with (args.directory / "cpu_run.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    checkpoint = {"command": command, "returncode": result.returncode,
                  "passed": False, "device_verified": False, "samples": [],
                  "relative_rms_limit": 0.10,
                  "scope": "Two teacher-forced CPU samples. No recurrent or audio quality proof."}
    if result.returncode:
        write(args.directory / "cpu_check.json", checkpoint)
        raise SystemExit(result.returncode)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(state["source_onnx"], options, providers=["CPUExecutionProvider"])
    for index, sample in enumerate(state["samples"][:2]):
        source = Path(state["calibration_manifest"]).parent / sample["file"]
        with np.load(source) as arrays:
            feed = {v.name: arrays[v.name] for v in session.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:, :, :1000, :].copy()
        expected = session.run(None, feed)
        row = {"sample": sample, "outputs": {}}
        for output, reference in zip(session.get_outputs(), expected):
            path = args.directory / "cpu_output" / f"Result_{index}" / (output.name + ".raw")
            actual = np.fromfile(path, dtype=np.float32).reshape(reference.shape)
            difference = actual.astype(np.float64) - reference.astype(np.float64)
            relative = float(np.sqrt(np.mean(difference**2)) /
                             max(float(np.sqrt(np.mean(reference.astype(np.float64)**2))), 1e-12))
            row["outputs"][output.name] = {"finite": bool(np.isfinite(actual).all()),
                "relative_rms": relative, "max_abs": float(np.max(np.abs(difference))),
                "nonzero": int(np.count_nonzero(actual)), "shape": list(actual.shape),
                "passed": bool(np.isfinite(actual).all() and relative <= checkpoint["relative_rms_limit"]
                               and (np.any(actual) or not np.any(reference)))}
        row["passed"] = all(v["passed"] for v in row["outputs"].values())
        checkpoint["samples"].append(row)
    checkpoint["passed"] = report["all_projection_weights_w4"] and all(v["passed"] for v in checkpoint["samples"])
    write(args.directory / "cpu_check.json", checkpoint)
    print(json.dumps(checkpoint), flush=True)
    if not checkpoint["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
