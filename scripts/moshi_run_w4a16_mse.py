"""Run one W4/MSE candidate with explicit float norm and mask exclusions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--sdk-root", type=Path, required=True)
    p.add_argument("--numpy-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.numpy_dir))
    import numpy as np
    import onnx
    from onnx import numpy_helper

    source = args.baseline / "candidate/model.onnx"
    if sha256(args.baseline / "model.dlc") != "456c05a979b4d283c4796906c1bbbb947324358df8096487401db82c8b002082":
        raise ValueError("Unexpected baseline DLC")
    model = onnx.load(str(source), load_external_data=False)
    activation_names = set()
    param_names = set()
    for node in model.graph.node:
        if node.name.startswith(("/norm1/", "/norm1_1/", "/norm2/", "/norm2_1/")):
            if node.name.endswith("/Mul_1"):
                continue  # A16 projection input boundary remains quantized.
            activation_names.update(node.output)
        if node.name in ("/Add_1", "/Add_5", "/Where_3", "/Where_7"):
            activation_names.update(node.output)
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    value = numpy_helper.to_array(attribute.t)
                    if np.issubdtype(value.dtype, np.floating) and np.any(np.abs(value) > 1e30):
                        param_names.update(node.output)
    policy = {"version": "0.6.1",
              "activation_encodings": {name: [{"bitwidth": 32, "dtype": "float"}]
                                       for name in sorted(activation_names)},
              "param_encodings": {name: [{"bitwidth": 32, "dtype": "float"}]
                                  for name in sorted(param_names)}}
    policy_path = root / "float_exclusions.json"
    write(policy_path, policy)
    scripts = root / "runtime_scripts"
    scripts.mkdir()
    for name in ("moshi_run_w4a16_mse.py", "moshi_quantize_temporal_offline.py",
                 "moshi_guard_qairt_converter.py", "moshi_check_offline_dlc.py", "moshi_simulate_offline_dlc.py"):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(args.sdk_root / "lib/x86_64-linux-clang")
    env["OPENBLAS_NUM_THREADS"] = "4"
    conversion = root / "conversion"
    conversion.mkdir()
    dlc = conversion / "model.dlc"
    receipt = conversion / "conversion.json"
    preserve = [v.name for v in (*model.graph.input, *model.graph.output)
                if v.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]
    result = {"format": "moshi-w4a16-mse-experiment-v1", "status": "converting_float_exclusions",
              "source_onnx": str(source), "source_onnx_sha256": sha256(source),
              "policy_sha256": sha256(policy_path), "weight_calibration": "mse",
              "device_verified": False, "default_bundle_replaced": False}
    write(root / "RESULT.json", result)

    def run(command, log):
        with (root / log).open("w") as handle:
            handle.write(repr(command) + "\n")
            handle.flush()
            completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        return completed.returncode

    try:
        common = ["--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir)]
        command = [sys.executable, str(scripts / "moshi_guard_qairt_converter.py"), *common,
                   "--receipt", str(receipt), "--", "--input_network", str(source),
                   "--output_path", str(dlc), "--quantization_overrides", str(policy_path),
                   "--preserve_onnx_output_order", "--onnx_skip_simplification",
                   "--source_model_input_datatype", "position", "int32", "--preserve_io_datatype", *preserve]
        if run(command, "conversion.log"):
            raise RuntimeError("Guarded conversion failed")
        candidate = root / "candidate"
        result["status"] = "quantizing"
        write(root / "RESULT.json", result)
        command = [sys.executable, str(scripts / "moshi_quantize_temporal_offline.py"), *common,
                   "--source-onnx", str(source), "--source-dlc", str(dlc),
                   "--expected-dlc-sha256", sha256(dlc), "--source-conversion-receipt", str(receipt),
                   "--calibration-dir", str(args.calibration_dir), "--graph", "temporal_layers_2_3",
                   "--param-calibration", "mse", "--output-dir", str(candidate)]
        if run(command, "quantization.log"):
            raise RuntimeError("Offline MSE quantization failed")
        command = [sys.executable, str(scripts / "moshi_check_offline_dlc.py"), *common,
                   "--directory", str(candidate), "--audit-only"]
        if run(command, "audit.log"):
            raise RuntimeError("DLC precision audit failed")
        result["status"] = "cpu_simulation"
        write(root / "RESULT.json", result)
        command = [sys.executable, str(scripts / "moshi_simulate_offline_dlc.py"), *common,
                   "--directory", str(candidate)]
        code = run(command, "cpu_simulation.log")
        report = json.loads((candidate / "cpu_simulation.json").read_text())
        result.update(status="cpu_screen_passed_not_device_verified" if code == 0 else "rejected_by_cpu_screen",
                      cpu_screen_passed=report["passed"], samples=report["samples"], candidate=str(candidate))
    except Exception as error:
        result.update(status="failed", error=str(error))
    write(root / "RESULT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
