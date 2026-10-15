"""Quantize one existing float cache1000 DLC locally with QAIRT.

Reuse the exact guarded graph and calibration archive. Refuse QDQ sources.
Keep all outputs separate. A completed quantizer is not an HTP accuracy pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--source-dlc", type=Path, required=True)
    parser.add_argument("--expected-dlc-sha256", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--numpy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.numpy_dir))
    import numpy as np
    import onnx

    model = onnx.load(str(args.source_onnx), load_external_data=False)
    if any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in model.graph.node):
        raise ValueError("Source contains QDQ. Refusing repeated quantization")
    actual_hash = sha256(args.source_dlc)
    if actual_hash != args.expected_dlc_sha256:
        raise ValueError("DLC hash does not match the selected baseline")
    inputs = {v.name: v for v in model.graph.input}
    shapes = {k: [d.dim_value for d in v.type.tensor_type.shape.dim] for k, v in inputs.items()}
    if any(shape != [1, 32, 1000, 128] for k, shape in shapes.items() if k.startswith("layer_")):
        raise ValueError("Expected the existing 1000-slot cache ABI")
    if any([d.dim_value for d in v.type.tensor_type.shape.dim] != [1, 32, 1, 128]
           for v in model.graph.output if v.name.startswith("output_layer_")):
        raise ValueError("Expected delta KV outputs")
    manifest = args.calibration_dir / "manifest.json"
    calibration = json.loads(manifest.read_text())["graphs"][args.graph]
    raw = args.output_dir / "calibration_raw"
    raw.mkdir()
    samples = []
    lines = []
    dtypes = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16,
              onnx.TensorProto.INT32: np.int32, onnx.TensorProto.INT64: np.int64}
    for index, item in enumerate(calibration["samples"]):
        source = args.calibration_dir / item["file"]
        row = []
        with np.load(source) as arrays:
            for name, value in inputs.items():
                array = arrays[name]
                if name.startswith("layer_"):
                    array = array[:, :, :1000, :]
                # The verified converter overrides ONNX int64 position to DLC int32.
                dtype = np.int32 if name == "position" else dtypes[value.type.tensor_type.elem_type]
                array = np.ascontiguousarray(array, dtype=dtype)
                if list(array.shape) != shapes[name] or not np.isfinite(array).all():
                    raise ValueError(f"Invalid calibration input: {source}, {name}")
                target = raw / f"{index:03d}_{name}.raw"
                array.tofile(target)
                row.append(f"{name}:={target}")
        samples.append({**item, "sha256": sha256(source)})
        lines.append(" ".join(row))
    input_list = args.output_dir / "input_list.txt"
    input_list.write_text("\n".join(lines) + "\n")
    (args.output_dir / "check_input_list.txt").write_text("\n".join(lines[:2]) + "\n")
    external = {}
    for tensor in model.graph.initializer:
        meta = {v.key: v.value for v in tensor.external_data}
        if "location" in meta:
            relative = Path(meta["location"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe external initializer path")
            external[str(relative)] = None
    for name in external:
        external[name] = sha256(args.source_onnx.parent / name)
    output = args.output_dir / "model.w4a16.dlc"
    command = [sys.executable, str(args.sdk_root / "bin/x86_64-linux-clang/qairt-quantizer"),
               "--input_dlc", str(args.source_dlc), "--output_dlc", str(output),
               "--input_list", str(input_list), "--weights_bitwidth", "4",
               "--act_bitwidth", "16", "--bias_bitwidth", "32",
               "--use_per_row_quantization", "--param_quantizer_schema", "symmetric",
               "--param_quantizer_calibration", "min-max", "--act_quantizer_calibration", "min-max",
               "--preserve_io_datatype", "--use_native_input_files", "--dump_encoding_json"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(args.numpy_dir), str(args.sdk_root / "lib/python")])
    env["LD_LIBRARY_PATH"] = str(args.sdk_root / "lib/x86_64-linux-clang")
    env["OMP_NUM_THREADS"] = "4"
    report = {"format": "moshi-offline-w4a16-v1", "graph": args.graph,
              "source_onnx": str(args.source_onnx), "source_onnx_sha256": sha256(args.source_onnx),
              "source_external_sha256": external, "source_dlc": str(args.source_dlc),
              "source_dlc_sha256": actual_hash, "source_dlc_bytes": args.source_dlc.stat().st_size,
              "calibration_manifest": str(manifest), "calibration_manifest_sha256": sha256(manifest),
              "samples": samples, "sdk_root": str(args.sdk_root),
              "script_sha256": sha256(__file__), "command": command, "status": "quantizing",
              "input_shapes": shapes, "cpu_verified": False, "device_verified": False,
              "default_bundle_replaced": False, "started_at": time.time()}
    write(args.output_dir / "quantization.json", report)
    print(f"Quantizing {args.graph}, calibration samples={len(samples)}", flush=True)
    started = time.monotonic()
    with (args.output_dir / "quantizer.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    report.update(status="quantized_not_verified" if result.returncode == 0 else "failed",
                  returncode=result.returncode, elapsed_s=time.monotonic() - started)
    if result.returncode == 0:
        report.update(output_dlc=str(output), output_sha256=sha256(output), output_bytes=output.stat().st_size)
    write(args.output_dir / "quantization.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "command")} ), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
