"""Author bounded rowwise W4 MSE encodings, then apply them with guarded QAIRT.

Reuse the existing 16-sample activation calibration. Preserve norm internals
and attention masks as float. This is one range-search algorithm, not a model
parameter sweep. Save the actual DLC encodings before the two-sample CPU gate.
"""
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
    for name in ("source-onnx", "reference-directory", "output-root", "sdk-root", "numpy-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--reuse-weight-policy", type=Path)
    p.add_argument("--activation-encodings", type=Path)
    args = p.parse_args()
    root = args.output_root
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    scripts = root / "runtime_scripts"
    scripts.mkdir()
    for name in ("moshi_author_w4_encodings.py", "moshi_quantize_temporal_offline.py",
                 "moshi_guard_qairt_converter.py", "moshi_check_offline_dlc.py", "moshi_simulate_offline_dlc.py"):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root / "lib/python")]
    import numpy as np
    import onnx
    from onnx import numpy_helper

    prior = json.loads((args.reference_directory / "quantization.json").read_text())
    if sha256(args.source_onnx) != prior["source_onnx_sha256"]:
        raise ValueError("Activation calibration belongs to a different source")
    model = onnx.load(str(args.source_onnx), load_external_data=not bool(args.reuse_weight_policy))
    if any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in model.graph.node):
        raise ValueError("Refusing repeated quantization of QDQ input")
    initializers = {t.name: t for t in model.graph.initializer}
    available = {v.name for v in model.graph.input} | {o for n in model.graph.node for o in n.output}
    old = json.loads((args.reference_directory / "model.w4a16_encoding.json").read_text())
    policy = {"version": "0.6.1", "activation_encodings": {
        name: copy.deepcopy(value) for name, value in old["activation_encodings"].items() if name in available},
        "param_encodings": copy.deepcopy(old["param_encodings"])}
    float_value = [{"bitwidth": 32, "dtype": "float"}]
    fresh_activations = None
    if args.activation_encodings:
        provenance = json.loads((args.activation_encodings.parent / "calibration.json").read_text())
        if provenance["source_sha256"] != prior["source_onnx_sha256"]:
            raise ValueError("Projection calibration has a different source")
        fresh_activations = json.loads(args.activation_encodings.read_text())["activation_encodings"]
        policy["activation_encodings"] = copy.deepcopy(fresh_activations)
        policy["param_encodings"] = {name: float_value for name in old["param_encodings"]
                                     if not name.startswith("onnx::MatMul_")}
    for node in model.graph.node:
        norm = node.name.startswith(("/norm1/", "/norm1_1/", "/norm2/", "/norm2_1/"))
        if (norm and not node.name.endswith("/Mul_1")) or node.name in ("/Add_1", "/Add_5", "/Where_3", "/Where_7"):
            if node.op_type == "Constant":
                value = next((numpy_helper.to_array(a.t) for a in node.attribute if a.name == "value"), None)
                if value is not None and not np.issubdtype(value.dtype, np.floating):
                    continue
            for name in node.output:
                policy["activation_encodings"][name] = float_value
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    value = numpy_helper.to_array(a.t)
                    if np.issubdtype(value.dtype, np.floating) and np.any(np.abs(value) > 1e30):
                        for name in node.output:
                            policy["param_encodings"][name] = float_value
    for name in policy["param_encodings"]:
        if "norm" in name:
            policy["param_encodings"][name] = float_value
    if fresh_activations:
        # Cast/reshape aliases must not turn the projection boundary into float.
        policy["activation_encodings"].update(copy.deepcopy(fresh_activations))
    ratios = np.linspace(1.0, 0.2, 17, dtype=np.float32)
    statistics = []
    started = time.monotonic()
    result = {"format": "moshi-authored-w4-mse-v1", "status": "authoring_weight_encodings",
              "source_onnx": str(args.source_onnx), "source_onnx_sha256": prior["source_onnx_sha256"],
              "script_sha256": sha256(__file__), "ratios": ratios.tolist(), "row_chunk": 64,
              "activation_calibration_reused_from": str(args.reference_directory),
              "device_verified": False, "default_bundle_replaced": False}
    write(root / "RESULT.json", result)
    reused_policy = None
    if args.reuse_weight_policy:
        provenance = json.loads((args.reuse_weight_policy.parent / "RESULT.json").read_text())
        if provenance["source_onnx_sha256"] != prior["source_onnx_sha256"]:
            raise ValueError("Reused weights refer to a different source")
        reused_policy = json.loads(args.reuse_weight_policy.read_text())["param_encodings"]
        prior_stats = {v["name"]: v for v in json.loads((args.reuse_weight_policy.parent / "weight_statistics.json").read_text())}
    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in initializers:
            continue
        name = node.input[1]
        if reused_policy is not None:
            encodings = reused_policy[name]
            float_weight = len(encodings) == 1 and encodings[0].get("dtype") == "float"
            if not float_weight and (len(encodings) != initializers[name].dims[1]
                    or len({v["bitwidth"] for v in encodings}) != 1
                    or encodings[0]["bitwidth"] not in (4, 8)):
                raise ValueError("Reused weight encoding has an unexpected shape or bitwidth")
            policy["param_encodings"][name] = copy.deepcopy(encodings)
            if float_weight:
                policy["activation_encodings"][node.input[0]] = float_value
                policy["activation_encodings"][node.output[0]] = float_value
            statistics.append(prior_stats[name])
            continue
        original = numpy_helper.to_array(initializers[name])
        rows = np.ascontiguousarray(original.T)
        encodings = []
        error_total = 0.0
        base_total = 0.0
        energy = 0.0
        for start in range(0, len(rows), 64):
            values = rows[start:start + 64]
            base = np.maximum(-values.min(axis=1) / 8, values.max(axis=1) / 7)
            base = np.maximum(base, np.finfo(np.float32).tiny)
            best_error = np.full(len(values), np.inf, np.float64)
            best_scale = base.copy()
            for index, ratio in enumerate(ratios):
                scale = base * ratio
                codes = np.clip(np.rint(values / scale[:, None]), -8, 7)
                error = np.mean(np.square(codes * scale[:, None] - values), axis=1, dtype=np.float64)
                if index == 0:
                    base_total += float(error.sum()) * values.shape[1]
                selected = error < best_error
                best_error[selected] = error[selected]
                best_scale[selected] = scale[selected]
            error_total += float(best_error.sum()) * values.shape[1]
            energy += float(np.sum(np.square(values), dtype=np.float64))
            for scale in best_scale:
                encodings.append({"bitwidth": 4, "is_symmetric": "true", "min": float(-8 * scale),
                                  "max": float(7 * scale), "scale": float(scale), "offset": 0})
        policy["param_encodings"][name] = encodings
        statistics.append({"name": name, "shape": list(original.shape), "encodings": len(encodings),
                           "minmax_relative_rms": (base_total / max(energy, 1e-30))**0.5,
                           "mse_relative_rms": (error_total / max(energy, 1e-30))**0.5})
        print(json.dumps(statistics[-1]), flush=True)
        write(root / "weight_statistics.json", statistics)
        del rows
    if len(statistics) != 8:
        raise ValueError("Expected eight projection weights")
    write(root / "weight_statistics.json", statistics)
    if args.reuse_weight_policy:
        result["reused_weight_policy_sha256"] = sha256(args.reuse_weight_policy)
    if args.activation_encodings:
        result["activation_encodings_sha256"] = sha256(args.activation_encodings)
    result["weight_authoring_s"] = time.monotonic() - started
    policy_path = root / "quantization_overrides.json"
    write(policy_path, policy)
    del model, initializers
    candidate = root / "candidate"
    candidate.mkdir()
    output = candidate / "model.w4a16.dlc"
    receipt = candidate / "conversion.json"
    metadata_model = onnx.load(str(args.source_onnx), load_external_data=False)
    preserve = [v.name for v in (*metadata_model.graph.input, *metadata_model.graph.output)
                if v.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.sdk_root / "lib/x86_64-linux-clang"), OPENBLAS_NUM_THREADS="4")
    common = ["--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir)]
    command = [sys.executable, str(scripts / "moshi_guard_qairt_converter.py"), *common,
               "--receipt", str(receipt), "--", "--input_network", str(args.source_onnx),
               "--output_path", str(output), "--quantization_overrides", str(policy_path),
               "--preserve_onnx_output_order", "--onnx_skip_simplification",
               "--source_model_input_datatype", "position", "int32", "--preserve_io_datatype", *preserve]
    result["status"] = "applying_authored_encodings"
    result["policy_sha256"] = sha256(policy_path)
    write(root / "RESULT.json", result)
    with (root / "conversion.log").open("w") as log:
        run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        result.update(status="conversion_failed", returncode=run.returncode)
        write(root / "RESULT.json", result)
        raise SystemExit(run.returncode)
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.converters.common import encodings_json_serializer
    from moshi_check_offline_dlc import audit
    actual = audit(output)
    write(candidate / "precision_audit.json", actual)
    expected_bits = {name: values[0]["bitwidth"] for name, values in policy["param_encodings"].items()
                     if name.startswith("onnx::MatMul_")}
    actual_projections = [row for g in actual["graphs"] for row in g["projections"] if row["type"] == "FullyConnected"]
    precision_passed = len(actual_projections) == 8 and all(
        (expected_bits[row["inputs"][1]["name"]] == 32
         and row["inputs"][1]["dtype"] == "Float_32") or (
        row["inputs"][1]["bitwidths"] == [expected_bits[row["inputs"][1]["name"]]]
        and row["inputs"][0]["dtype"] == "uFxp_16" and row["outputs"][0]["dtype"] == "uFxp_16")
        for row in actual_projections)
    result["expected_projection_bits"] = expected_bits
    result["actual_precision_passed"] = precision_passed
    if not precision_passed:
        result["status"] = "precision_gate_failed"
        write(root / "RESULT.json", result)
        return
    reader = modeltools.IrDlcReader()
    reader.open(str(output))
    graph = reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    target = candidate / "model.w4a16_encoding.json"
    serializer = encodings_json_serializer.IrEncodingsJsonSerializer(str(target), True)
    serializer.serialize(graph)
    target.write_text(serializer.get_graph_json())
    q = dict(prior)
    q.update(status="quantized_not_verified", output_dlc=str(output), output_sha256=sha256(output),
             output_bytes=output.stat().st_size, source_onnx=str(args.source_onnx),
             script_sha256=sha256(__file__), method="offline numpy rowwise MSE, 17 bounded ranges",
             activation_calibration_reused_from=str(args.reference_directory), quantization_overrides=str(policy_path),
             quantization_overrides_sha256=sha256(policy_path), converter_receipt_sha256=sha256(receipt),
             command=command, elapsed_s=time.monotonic() - started)
    q["expected_projection_bits"] = expected_bits
    if args.activation_encodings:
        q["activation_calibration_manifest"] = str(args.activation_encodings.parent / "calibration.json")
        q["activation_encodings_sha256"] = sha256(args.activation_encodings)
    write(candidate / "quantization.json", q)
    result.update(status="cpu_simulation", candidate=str(candidate), output_bytes=q["output_bytes"])
    write(root / "RESULT.json", result)
    with (root / "cpu_simulation.log").open("w") as log:
        run = subprocess.run([sys.executable, str(scripts / "moshi_simulate_offline_dlc.py"), *common,
                              "--directory", str(candidate)], env=env, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads((candidate / "cpu_simulation.json").read_text())
    result.update(status="cpu_screen_passed_not_device_verified" if run.returncode == 0 else "rejected_by_cpu_screen",
                  cpu_screen_passed=report["passed"], samples=report["samples"], weights=statistics)
    write(root / "RESULT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
