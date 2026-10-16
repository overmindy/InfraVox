"""Reuse screened LPBQ weights with equivalent pointwise Conv projections.

Only static projection MatMuls with singleton leading dimensions are rewritten.
The source FP32 export, quantization grids, cache and public I/O stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-directory", "output-dir", "sdk-root", "numpy-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root / "lib/python")]
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    from moshi_check_offline_dlc import audit
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.converters.common import encodings_json_serializer

    root, parent = args.output_dir, args.parent_directory
    root.mkdir()
    previous = json.loads((parent / "RESULT.json").read_text())
    state = json.loads((parent / "candidate/quantization.json").read_text())
    parent_validation = json.loads((parent / "candidate/calibration_validation.json").read_text())
    if not parent_validation["passed"] or sha256(state["output_dlc"]) != parent_validation["candidate_dlc_sha256"]:
        raise ValueError("Require the numerically screened parent DLC")
    source = Path(state["source_onnx"])
    if sha256(source) != state["source_onnx_sha256"]:
        raise ValueError("Parent source changed")
    for name, digest in state["source_external_sha256"].items():
        if sha256(source.parent / name) != digest:
            raise ValueError("Parent external weights changed")
    model = onnx.load(str(source))
    original_abi = [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    tensors = {v.name: v for v in model.graph.initializer}
    policy = json.loads((parent / "quantization_overrides.json").read_text())
    activation = {v["name"]: v for v in policy["activation_encodings"]}
    replacements, rewrites = [], []
    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in state["expected_projection_bits"]:
            replacements.append(node)
            continue
        name = node.input[1]
        weight = numpy_helper.to_array(tensors[name])
        ni, no = weight.shape
        input_shape = shapes[node.input[0]]
        output_shape = shapes[node.output[0]]
        if input_shape[-1] != ni or output_shape[-1] != no or any(v != 1 for v in input_shape[:-1]):
            raise ValueError("Only single-token singleton-batch projections can be rewritten")
        x, y = name + "_pointwise_input", name + "_pointwise_output"
        xs, ys = name + "_pointwise_input_shape", name + "_pointwise_output_shape"
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array([1, ni, 1, 1], np.int64), xs),
            numpy_helper.from_array(np.array(output_shape, np.int64), ys),
        ])
        tensors[name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weight.T[:, :, None, None]), name))
        replacements.extend([
            helper.make_node("Reshape", [node.input[0], xs], [x], name=node.name + "/to_pointwise"),
            helper.make_node("Conv", [x, name], [y], name=node.name, kernel_shape=[1, 1]),
            helper.make_node("Reshape", [y, ys], [node.output[0]], name=node.name + "/from_pointwise"),
        ])
        for old, new in ((node.input[0], x), (node.output[0], y)):
            entry = copy.deepcopy(activation[old])
            entry["name"] = new
            policy["activation_encodings"].append(entry)
        rewrites.append({"node": node.name, "weight": name, "input_shape": input_shape,
                         "output_shape": output_shape, "bitwidth": state["expected_projection_bits"][name]})
    if len(rewrites) != 8:
        raise ValueError("Expected eight projection rewrites")
    del model.graph.node[:]
    model.graph.node.extend(replacements)
    if [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)] != original_abi:
        raise ValueError("Public I/O changed")
    out_source = root / "source/model.onnx"
    out_source.parent.mkdir()
    onnx.save_model(model, str(out_source), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.data", size_threshold=1024)
    write(root / "quantization_overrides.json", policy)
    del model, inferred, tensors, weight, replacements
    result = {"format": "moshi-lpbq-pointwise-v1", "status": "checking_float_equivalence",
              "script_sha256": sha256(__file__), "parent_directory": str(parent),
              "parent_dlc_sha256": state["output_sha256"], "rewrites": rewrites,
              "expected_projection_bits": state["expected_projection_bits"],
              "w4_parameter_fraction": previous["w4_parameter_fraction"],
              "selected_precision": previous["selected_precision"],
              "quantization_refitted": False, "public_io_identical": True,
              "device_verified": False, "default_bundle_replaced": False}
    write(root / "RESULT.json", result)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    before = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    after = ort.InferenceSession(str(out_source), options, providers=["CPUExecutionProvider"])
    equivalence = []
    for sample in state["samples"][:2]:
        path = Path(state["calibration_manifest"]).parent / sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Calibration input changed")
        with np.load(path) as data:
            feed = {v.name: data[v.name] for v in before.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:, :, :1000, :].copy()
        expected, got = before.run(None, feed), after.run(None, feed)
        errors = {v.name: float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64)) /
                                  max(np.linalg.norm(b.astype(np.float64)), 1e-12))
                  for v, a, b in zip(before.get_outputs(), got, expected)}
        equivalence.append({"sample": sample, "relative_rms": errors})
        if not all(np.isfinite(v) and v <= 1e-5 for v in errors.values()):
            raise ValueError("Pointwise rewrite is not numerically equivalent")
    del before, after
    result["float_equivalence"] = equivalence
    candidate = root / "candidate"
    candidate.mkdir()
    scripts = root / "runtime_scripts"
    scripts.mkdir()
    for name in ("moshi_rewrite_lpbq_pointwise.py", "moshi_guard_qairt_converter.py", "moshi_check_offline_dlc.py",
                 "moshi_quantize_temporal_offline.py", "moshi_simulate_offline_dlc.py",
                 "moshi_validate_temporal_calibration.py", "moshi_compile_temporal_candidate.py"):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    common = ["--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir)]
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.sdk_root / "lib/x86_64-linux-clang"), OPENBLAS_NUM_THREADS="4")
    dlc = candidate / "model.w4a16.dlc"
    command = [sys.executable, str(scripts / "moshi_guard_qairt_converter.py"), *common,
               "--receipt", str(candidate / "conversion.json"), "--", "--input_network", str(out_source),
               "--output_path", str(dlc), "--quantization_overrides", str(root / "quantization_overrides.json"),
               "--preserve_onnx_output_order", "--onnx_skip_simplification", "--source_model_input_datatype", "position", "int32",
               "--preserve_io_datatype", *[v.name for v in (*onnx.load(str(source), load_external_data=False).graph.input,
                                                          *onnx.load(str(source), load_external_data=False).graph.output)
                                          if v.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]]
    result.update(status="converting", conversion_command=command)
    write(root / "RESULT.json", result)
    with (root / "conversion.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    actual = audit(dlc)
    write(candidate / "precision_audit.json", actual)
    projections = [v for g in actual["graphs"] for v in g["projections"] if v["type"] == "Conv2d"]
    result["actual_precision_passed"] = len(projections) == 8 and all(
        v["inputs"][1].get("compressed_bitwidth", v["inputs"][1]["bitwidths"][0]) == state["expected_projection_bits"][v["inputs"][1]["name"]]
        and v["inputs"][0]["dtype"] == "uFxp_16" and v["outputs"][0]["dtype"] == "uFxp_16" for v in projections)
    if not result["actual_precision_passed"]:
        raise ValueError("Pointwise DLC precision differs from the selected policy")
    reader = modeltools.IrDlcReader()
    reader.open(str(dlc))
    graph = reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    serializer = encodings_json_serializer.IrEncodingsJsonSerializer(str(candidate / "model.w4a16_encoding.json"), True)
    serializer.serialize(graph)
    (candidate / "model.w4a16_encoding.json").write_text(serializer.get_graph_json())
    state.update(source_onnx=str(out_source), source_onnx_sha256=sha256(out_source),
                 source_external_sha256={"model.data": sha256(out_source.parent / "model.data")},
                 output_dlc=str(dlc), output_sha256=sha256(dlc), output_bytes=dlc.stat().st_size,
                 quantization_overrides=str(root / "quantization_overrides.json"),
                 method="Unchanged sensitivity-selected LPBQ W4/W8 A16; equivalent pointwise Conv projections")
    write(candidate / "quantization.json", state)
    write(root / "RESULT.json", result)
    for script, leaf, extra in (("moshi_simulate_offline_dlc.py", "cpu_simulation", common),
                                ("moshi_validate_temporal_calibration.py", "calibration_validation", []),
                                ("moshi_compile_temporal_candidate.py", "htp_compile", ["--sdk-root", str(args.sdk_root)])):
        target = root if leaf == "htp_compile" else candidate
        started = time.monotonic()
        with (root / (leaf + ".log")).open("w") as log:
            run = subprocess.run([sys.executable, str(scripts / script), "--directory", str(target), *extra],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        print(leaf, run.returncode, "seconds", time.monotonic()-started, flush=True)
        if leaf != "htp_compile":
            result[leaf + "_passed"] = run.returncode == 0
            if leaf == "cpu_simulation":
                result["cpu_screen_passed"] = run.returncode == 0
            result["status"] = leaf + ("_passed" if run.returncode == 0 else "_failed")
            write(root / "RESULT.json", result)
        if run.returncode:
            raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
