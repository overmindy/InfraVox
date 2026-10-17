"""Calibrate only eight projection boundaries on the unchanged source graph."""
import argparse
import errno
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-onnx", type=Path, required=True)
    p.add_argument("--reference-directory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--save-projections", action="store_true")
    args = p.parse_args()
    import numpy as np
    import onnx
    from onnx import helper
    import onnxruntime as ort
    from moshi_quantize_temporal_offline import sha256, write

    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh calibration directory")
    root.mkdir(parents=True)
    prior = json.loads((args.reference_directory / "quantization.json").read_text())
    if sha256(args.source_onnx) != prior["source_onnx_sha256"]:
        raise ValueError("Source differs from the calibration provenance")
    model = onnx.load(str(args.source_onnx), load_external_data=False)
    initializers = {t.name for t in model.graph.initializer}
    names = list(dict.fromkeys(name for n in model.graph.node
                              if n.op_type == "MatMul" and n.input[1] in initializers
                              for name in (n.input[0], n.output[0])))
    existing = {v.name for v in model.graph.output}
    original_output_names = [v.name for v in model.graph.output]
    for name in names:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    for t in model.graph.initializer:
        meta = {v.key: v.value for v in t.external_data}
        if "location" in meta:
            relative = Path(meta["location"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source data location")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source_data = (args.source_onnx.parent / relative).resolve()
                try:
                    os.link(source_data, target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    target.symlink_to(source_data)
    probe = root / "projection_probe.onnx"
    # Metadata-only diagnostic view. Existing external weights are hard-linked.
    onnx.save_model(model, str(probe))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(probe), opts, providers=["CPUExecutionProvider"])
    ranges = {name: [0.0, 0.0] for name in names}
    output_ranges = {name: [0.0, 0.0] for name in original_output_names}
    saved = []
    if args.save_projections:
        (root / "projection_samples").mkdir()
    for index, sample in enumerate(prior["samples"]):
        path = Path(prior["calibration_manifest"]).parent / sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Calibration sample changed")
        with np.load(path) as data:
            feed = {v.name: data[v.name] for v in session.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:, :, :1000, :].copy()
        values = session.run(None, feed)
        if args.save_projections:
            target_path = root / "projection_samples" / f"{index:03d}.npz"
            np.savez_compressed(target_path, **{v.name: a for v, a in zip(session.get_outputs(), values) if v.name in names})
            saved.append({"file": str(target_path.relative_to(root)), "source_id": sample["source_id"],
                          "frame": sample["frame"], "sha256": sha256(target_path)})
        for output, value in zip(session.get_outputs(), values):
            if not np.isfinite(value).all():
                raise ValueError("Source calibration produced nonfinite values")
            target = ranges.get(output.name, output_ranges.get(output.name))
            if target is not None:
                target[0] = min(target[0], float(value.min()))
                target[1] = max(target[1], float(value.max()))
        print(sample["source_id"], sample["frame"], flush=True)
    encodings = {}
    for name, (low, high) in ranges.items():
        scale = max((high - low) / 65535, 1e-12)
        offset = -int(np.clip(np.rint(-low / scale), 0, 65535))
        encodings[name] = [{"bitwidth": 16, "is_symmetric": "false", "scale": scale,
                            "offset": offset, "min": offset * scale, "max": (65535 + offset) * scale}]
    write(root / "activation_encodings.json", {"version": "0.6.1", "activation_encodings": encodings,
                                               "param_encodings": {}})
    write(root / "calibration.json", {"source_sha256": prior["source_onnx_sha256"], "samples": prior["samples"],
                                      "ranges": ranges, "source_output_ranges": output_ranges,
                                      "projection_samples": saved,
                                      "script_sha256": sha256(__file__), "quality_test": False})


if __name__ == "__main__":
    main()
