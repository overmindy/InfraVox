"""Screen selective Mimi W8A16 PTQ from existing explicit-state exports.

This experiment does not change the deployed codec. It quantizes internal
linear weights, while retaining codebooks, normalization and convolutions.
Convolution quantization is an optional diagnostic. Four streaming frames calibrate A16
ranges. Four later frames provide a small, separate numerical screen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "audio-wav", "output-dir", "sdk-root", "numpy-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--components", nargs="+", default=["encoder", "decoder"])
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--calibration-frames", type=int, default=4)
    p.add_argument("--quantize-convolutions", action="store_true",
                   help="Diagnostic only; retain bias-free and boundary convolutions")
    args = p.parse_args()
    sys.path[:0] = [str(args.numpy_dir), str(args.sdk_root / "lib/python")]
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort

    assert 0 < args.calibration_frames < args.frames
    with wave.open(str(args.audio_wav), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        audio = np.frombuffer(w.readframes(args.frames * 1920), "<i2").astype(np.float32) / 32768
    assert audio.size == args.frames * 1920
    frames = list(audio.reshape(args.frames, 1, 1, 1920))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    def session(model):
        return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])

    def initial(model):
        return {v.name: np.zeros([d.dim_value for d in v.type.tensor_type.shape.dim],
                                np.int64 if v.type.tensor_type.elem_type == 7 else np.float32)
                for v in model.graph.input if v.name not in ("audio", "codes")}

    def run_chain(model, data, names=None):
        engine = session(model)
        state = initial(model)
        outputs = []
        for value in data:
            result = engine.run(names, {engine.get_inputs()[0].name: value, **state})
            outputs.append(result)
            state = dict(zip(("position", "kv_cache", "conv_state"), result[1:4]))
        return outputs

    encoder = onnx.load(str(args.source_dir / "mimi_streaming_encoder.onnx"))
    reference_codes = [r[0] for r in run_chain(encoder, frames)]

    def metrics(actual, expected):
        a, b = np.asarray(actual).astype(np.float64), np.asarray(expected).astype(np.float64)
        return {"finite": bool(np.isfinite(a).all()),
                "relative_rms": float(np.sqrt(np.mean((a-b)**2)) / max(np.sqrt(np.mean(b*b)), 1e-12)),
                "normalized_max": float(np.max(np.abs(a-b)) / max(np.max(np.abs(b)), 1e-12))}

    for comp in args.components:
        root = args.output_dir / comp
        root.mkdir(exist_ok=False)
        source = args.source_dir / f"mimi_streaming_{comp}.onnx"
        model = onnx.load(str(source))
        weights = {t.name: t for t in model.graph.initializer}
        convs = [n for n in model.graph.node if n.op_type == "Conv"]
        excluded_conv = {convs[0].name, convs[-1].name}
        selected = [n for n in model.graph.node
                    if len(n.input) > 1 and n.input[1] in weights
                    and (n.op_type == "MatMul" or args.quantize_convolutions and n.op_type == "Conv"
                         and len(n.input) > 2
                         and n.name not in excluded_conv
                         and "quantizer" not in n.input[1])]
        boundaries = sorted({name for n in selected for name in (n.input[0], n.output[0])})
        probe = copy.deepcopy(model)
        for name in boundaries:
            probe.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        data = frames if comp == "encoder" else reference_codes
        probe_session = session(probe)
        state = initial(model)
        bounds = {name: [0., 0.] for name in boundaries}
        reference = []
        for index, value in enumerate(data):
            result = probe_session.run(None, {probe_session.get_inputs()[0].name: value, **state})
            reference.append(result[:4])
            state = dict(zip(("position", "kv_cache", "conv_state"), result[1:4]))
            if index < args.calibration_frames:
                for name, tensor in zip(boundaries, result[4:]):
                    assert np.isfinite(tensor).all(), name
                    bounds[name][0] = min(bounds[name][0], float(tensor.min()))
                    bounds[name][1] = max(bounds[name][1], float(tensor.max()))
        del probe_session, probe
        policy = {"version": "1.0.0", "activation_encodings": [], "param_encodings": []}
        candidate = copy.deepcopy(model)
        cweights = {t.name: t for t in candidate.graph.initializer}
        weight_records = []
        selected_weights = {n.input[1] for n in selected}
        selected_biases = {n.input[2] for n in selected if len(n.input) > 2}
        for t in model.graph.initializer:
            if t.data_type in (1, 10) and t.name not in selected_weights | selected_biases:
                policy["param_encodings"].append({"name": t.name, "dtype": "FLOAT", "bw": 32})
        for node in selected:
            w = numpy_helper.to_array(weights[node.input[1]]).astype(np.float32)
            axis = 1 if node.op_type == "MatMul" else 0
            axes = tuple(i for i in range(w.ndim) if i != axis)
            scale = np.maximum(np.max(np.abs(w), axis=axes), 1e-8) / 127
            shape = [1] * w.ndim
            shape[axis] = w.shape[axis]
            grid = scale.reshape(shape)
            decoded = np.clip(np.rint(w / grid), -128, 127) * grid
            cweights[node.input[1]].CopyFrom(numpy_helper.from_array(decoded, node.input[1]))
            policy["param_encodings"].append({"name": node.input[1], "dtype": "INT", "bw": 8,
                "enc_type": "PER_CHANNEL", "is_sym": True, "scale": scale.tolist(), "offset": [0]*len(scale)})
            weight_records.append({"node": node.name, "name": node.input[1], "axis": axis,
                                   "shape": list(w.shape), "parameters": int(w.size)})
        ranges = {}
        for name, (lo, hi) in bounds.items():
            # A small fixed guard absorbs interpolation outside the four fitting frames.
            lo, hi = lo * 1.05, hi * 1.05
            scale = max((hi-lo)/65535, 1e-10)
            offset = int(np.clip(np.rint(lo/scale), -65535, 0))
            ranges[name] = (scale, offset)
            policy["activation_encodings"].append({"name": name, "dtype": "INT", "bw": 16,
                "enc_type": "PER_TENSOR", "is_sym": False, "scale": [scale], "offset": [offset]})
        save(root / "quantization_overrides.json", policy)
        # ONNX arithmetic emulates the authored W8/A16 grids. This is not QNN CPU execution.
        for index, name in enumerate(boundaries):
            scale, offset = ranges[name]
            tag = f"mimi_ptq_{index}"
            for n in candidate.graph.node:
                for i, value in enumerate(n.input):
                    if value == name:
                        n.input[i] = tag + "_decoded"
            for suffix, value in (("scale", scale), ("lo", offset), ("hi", offset+65535)):
                candidate.graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), tag+"_"+suffix))
            new = [helper.make_node("Div", [name, tag+"_scale"], [tag+"_scaled"]),
                   helper.make_node("Round", [tag+"_scaled"], [tag+"_rounded"]),
                   helper.make_node("Clip", [tag+"_rounded", tag+"_lo", tag+"_hi"], [tag+"_clipped"]),
                   helper.make_node("Mul", [tag+"_clipped", tag+"_scale"], [tag+"_decoded"])]
            producer = next((i for i, n in enumerate(candidate.graph.node) if name in n.output), -1)
            for j, n in enumerate(new):
                candidate.graph.node.insert(producer+1+j, n)
        actual = run_chain(candidate, data)
        scores = []
        for index, (a, b) in enumerate(zip(actual, reference)):
            score = {"frame": index, "split": "calibration" if index < args.calibration_frames else "held_out"}
            if comp == "encoder":
                score["code_agreement"] = float(np.mean(a[0] == b[0]))
            else:
                score["audio"] = metrics(a[0], b[0])
            score["position_exact"] = bool(np.array_equal(a[1], b[1]))
            score["kv"] = metrics(a[2], b[2])
            score["conv_state"] = metrics(a[3], b[3])
            scores.append(score)
        # Initial diagnostic gates; code identity alone is not an audio quality gate.
        passed = all(x["position_exact"] and x["kv"]["finite"] and x["conv_state"]["finite"]
                     and x["kv"]["relative_rms"] <= .1 and x["conv_state"]["relative_rms"] <= .1
                     and (comp == "encoder" or x["audio"]["relative_rms"] <= .1) for x in scores)
        state = {"format": "mimi-selective-w8a16-v1", "component": comp,
                 "source_onnx": str(source), "source_sha256": sha(source),
                 "audio_wav": str(args.audio_wav), "audio_sha256": sha(args.audio_wav),
                 "script_sha256": sha(__file__), "selected_weights": weight_records,
                 "retained": ["codebook embeddings/search", "normalization",
                              "all convolutions unless diagnostic flag is set", "public float32 state ABI"],
                 "quantize_convolutions": args.quantize_convolutions,
                 "calibration_frames": args.calibration_frames, "screen_frames": args.frames,
                 "cpu_screen": {"passed": passed, "scores": scores, "kind": "authored encoding ONNX simulation",
                                "actual_dlc_cpu_execution": False},
                 "default_bundle_replaced": False, "device_verified": False}
        save(root / "RESULT.json", state)
        print(comp, "CPU screen", passed, flush=True)
        if not passed:
            continue
        command = [sys.executable, str(Path(__file__).with_name("moshi_guard_qairt_converter.py")),
            "--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir),
            "--receipt", str(root/"conversion.json"), "--", "--input_network", str(source),
            "--output_path", str(root/"model.dlc"), "--quantization_overrides", str(root/"quantization_overrides.json"),
            "--preserve_onnx_output_order", "--onnx_skip_simplification",
            "--source_model_input_datatype", "position", "int32", "--preserve_io_datatype",
            *[v.name for v in (*model.graph.input, *model.graph.output) if v.type.tensor_type.elem_type == 1]]
        env = dict(os.environ, LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),
                   OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        started = time.monotonic()
        with (root/"conversion.log").open("w") as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        state["conversion"] = {"command": command, "returncode": run.returncode, "seconds": time.monotonic()-started}
        if run.returncode == 0:
            state["dlc"] = {"path": str(root/"model.dlc"), "bytes": (root/"model.dlc").stat().st_size,
                            "sha256": sha(root/"model.dlc")}
        save(root/"RESULT.json", state)
        print(comp, "conversion returncode", run.returncode, flush=True)


if __name__ == "__main__":
    main()
