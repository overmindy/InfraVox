"""Search activation-aware projection scales and select a bounded mixed policy.

Reuse the source graph and its captured inputs. Do not export the PyTorch model.
The AWQ-style scale search uses source projection snapshots. A small whole-shard
calibration search then selects W4/W8. This is not an independent quality test.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-onnx", "reference-directory", "calibration-dir", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--whole-shard-limit", type=float, default=0.10)
    p.add_argument("--prepare-only", action="store_true", help="Prepare scales and A16 probe for a separate LPBQ policy search")
    args = p.parse_args()
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    prior = json.loads((args.reference_directory / "quantization.json").read_text())
    provenance = json.loads((args.calibration_dir / "calibration.json").read_text())
    source_hash = sha256(args.source_onnx)
    if source_hash != prior["source_onnx_sha256"] or source_hash != provenance["source_sha256"]:
        raise ValueError("Source and calibration hashes differ")
    # Every source external file must match the retained conversion provenance.
    for name, digest in prior["source_external_sha256"].items():
        if sha256(args.source_onnx.parent / name) != digest:
            raise ValueError("Source weights changed")
    snapshots = []
    for item in provenance["projection_samples"]:
        path = args.calibration_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Projection sample changed")
        with np.load(path) as data:
            snapshots.append({name: data[name] for name in data.files})
    if len(snapshots) != 16:
        raise ValueError("Expected the retained sixteen samples")
    # These four snapshots do not select per-channel ranges or input scales.
    held_out = [2, 5, 10, 13]
    training = [i for i in range(16) if i not in held_out]
    model = onnx.load(str(args.source_onnx))
    originals = {t.name: t for t in model.graph.initializer}
    projections = [n for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in originals]
    if len(projections) != 8:
        raise ValueError("Expected eight static projections")
    float_value = [{"bitwidth": 32, "dtype": "float"}]
    policy = {"version": "0.6.1", "activation_encodings": {}, "param_encodings": {}}
    rows, inverse_scales, candidates, transformed = [], {}, {}, {}
    result = {"format": "moshi-awq-style-mixed-v1", "status": "searching_projection_scales",
              "original_source_onnx": str(args.source_onnx), "original_source_sha256": source_hash,
              "original_source_external_sha256": copy.deepcopy(prior["source_external_sha256"]),
              "script_sha256": sha256(__file__), "torch_version": torch.__version__, "device": args.device,
              "algorithm": "AWQ-style input scaling plus per-output-channel activation MSE range search",
              "algorithm_reference": "https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py",
              "alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "clipping_ratios": [1.0, 0.95, 0.90, 0.85, 0.80],
              "weight_search_train_indices": training, "weight_search_holdout_indices": held_out,
              "independent_quality_test": False, "activation_ranges_use_all_sixteen": True,
              "whole_shard_relative_rms_limit": args.whole_shard_limit,
              "layers": rows, "device_verified": False, "default_bundle_replaced": False}
    started = time.monotonic()
    write(root / "RESULT.json", result)

    def local_metrics(got, target, groups, indices):
        return max(float(np.max(np.linalg.norm(a[indices]-b[indices], axis=1) /
                                np.maximum(np.linalg.norm(b[indices], axis=1), 1e-12)))
                   for a, b in zip(np.array_split(got, groups, axis=1), np.array_split(target, groups, axis=1)))

    def activation(values):
        low, high = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
        scale = max((high-low)/65535, 1e-12)
        offset = -int(np.clip(np.rint(-low/scale), 0, 65535))
        return [{"bitwidth": 16, "is_symmetric": "false", "scale": scale, "offset": offset,
                 "min": offset*scale, "max": (65535+offset)*scale}]

    def encodings(scale, bits):
        low, high = -(2**(bits-1)), 2**(bits-1)-1
        return [{"bitwidth": bits, "is_symmetric": "true", "scale": float(s), "offset": 0,
                 "min": float(low*s), "max": float(high*s)} for s in scale]

    with torch.inference_mode():
        for node in projections:
            name = node.input[1]
            weight = np.array(numpy_helper.to_array(originals[name]), copy=True)
            x = np.concatenate([v[node.input[0]].reshape(-1, weight.shape[0]) for v in snapshots])
            y = np.concatenate([v[node.output[0]].reshape(-1, weight.shape[1]) for v in snapshots])
            if len(x) != 16:
                raise ValueError("Each snapshot must contain one decode vector")
            groups = 3 if node.name.startswith("/in_projs") else (2 if weight.shape[1] == 22528 else 1)
            w = torch.as_tensor(weight, device=args.device)
            xt = torch.as_tensor(x, device=args.device)
            yt = torch.as_tensor(y, device=args.device)
            replay = local_metrics((xt @ w).cpu().numpy(), y, groups, list(range(16)))
            if replay > 1e-4:
                raise ValueError("Projection replay differs from saved source output")
            # Normalize each sample's contribution, so large frame-zero outputs do not hide later errors.
            norm = torch.sqrt(torch.mean(yt[training]**2, dim=1)).clamp(min=1e-12)[:, None]
            magnitude = xt[training].abs().mean(dim=0).clamp(min=1e-6)
            best, trials = None, []
            for alpha in result["alphas"]:
                scaling = magnitude.pow(alpha)
                scaling /= torch.sqrt(scaling.max()*scaling.min())
                sw = w * scaling[:, None]
                sx = xt / scaling[None, :]
                base = torch.maximum(-sw.amin(dim=0)/8, sw.amax(dim=0)/7).clamp(min=1e-12)
                channel_error = torch.full_like(base, float("inf"))
                chosen = base.clone()
                for ratio in result["clipping_ratios"]:
                    scale = base * ratio
                    q = torch.clamp(torch.round(sw/scale), -8, 7)*scale
                    loss = torch.mean(((sx[training] @ q-yt[training])/norm)**2, dim=0)
                    replace = loss < channel_error
                    channel_error = torch.minimum(channel_error, loss)
                    chosen = torch.where(replace, scale, chosen)
                q = torch.clamp(torch.round(sw/chosen), -8, 7)*chosen
                predicted = (sx @ q).cpu().numpy()
                train_error = local_metrics(predicted, y, groups, training)
                holdout_error = local_metrics(predicted, y, groups, held_out)
                trials.append({"alpha": alpha, "train_worst_relative_rms": train_error,
                               "holdout_worst_relative_rms": holdout_error})
                if best is None or train_error < best["train"]:
                    best = {"alpha": alpha, "train": train_error, "holdout": holdout_error,
                            "scaling": scaling.cpu().numpy().copy(), "scale4": chosen.cpu().numpy().copy()}
            s = best["scaling"]
            sw = np.ascontiguousarray(weight*s[:, None])
            scale4 = best["scale4"]
            scale8 = np.maximum(np.maximum(-sw.min(axis=0)/128, sw.max(axis=0)/127), 1e-12)
            candidates[name] = {4: np.ascontiguousarray(np.clip(np.rint(sw/scale4), -8, 7)*scale4),
                                8: np.ascontiguousarray(np.clip(np.rint(sw/scale8), -128, 127)*scale8)}
            transformed[name] = sw
            inverse_scales[name] = np.ascontiguousarray(1/s)
            scaled_name = name + "_awq_input"
            policy["activation_encodings"][node.input[0]] = float_value
            policy["activation_encodings"][scaled_name] = activation(x/s)
            policy["activation_encodings"][node.output[0]] = activation(y)
            row = {"name": name, "node": node.name, "shape": list(weight.shape),
                   "parameter_count": int(weight.size), "bitwidth": 8, "selected_alpha": best["alpha"],
                   "train_worst_relative_rms": best["train"], "holdout_worst_relative_rms": best["holdout"],
                   "trials": trials, "scale4": scale4.tolist(), "scale8": scale8.tolist(),
                   "input_scale_min": float(s.min()), "input_scale_max": float(s.max())}
            rows.append(row)
            print(node.name, "alpha", best["alpha"], "W4 train/holdout", best["train"], best["holdout"], flush=True)
            write(root / "RESULT.json", result)
            del w, xt, yt, sw, q, sx, scaling, weight

    # Preserve every original nonprojection node. Add one inverse scale before each projection.
    replacement = []
    for node in model.graph.node:
        if node in projections:
            name = node.input[1]
            prefix = name + "_awq"
            model.graph.initializer.append(numpy_helper.from_array(inverse_scales[name], prefix + "_inverse"))
            replacement.append(helper.make_node("Mul", [node.input[0], prefix + "_inverse"],
                                                 [name + "_awq_input"], name=prefix + "_scale"))
            node.input[0] = name + "_awq_input"
            originals[name].CopyFrom(numpy_helper.from_array(transformed[name], name))
        replacement.append(copy.deepcopy(node))
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    source_dir = root / "source"
    source_dir.mkdir()
    new_source = source_dir / "model.onnx"
    onnx.save_model(model, str(new_source), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.data", size_threshold=1024)
    new_hash = sha256(new_source)
    result.update(source_onnx_sha256=new_hash, source_external_sha256={"model.data": sha256(source_dir/"model.data")},
                  projection_search_s=time.monotonic()-started, status="whole_shard_policy_search")
    write(root / "RESULT.json", result)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(args.source_onnx), opts, providers=["CPUExecutionProvider"])
    # Four selection samples cover both clips, the second frame, and late context.
    selection_indices = [1, 9, 15, 0]
    feeds, expected = [], []
    for index in selection_indices:
        sample = prior["samples"][index]
        path = Path(prior["calibration_manifest"]).parent/sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Captured calibration input changed")
        with np.load(path) as arrays:
            feed = {v.name: arrays[v.name] for v in reference.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:, :, :1000, :].copy()
        feeds.append(feed)
        expected.append(reference.run(None, feed))
    # First verify the equivalent float transform against the original graph.
    equivalent = ort.InferenceSession(str(new_source), opts, providers=["CPUExecutionProvider"])
    equivalence = []
    for feed, targets in zip(feeds, expected):
        got = equivalent.run(None, feed)
        errors = [float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12)) for a,b in zip(got,targets)]
        if max(errors) > 1e-4 or not all(np.isfinite(a).all() for a in got):
            raise ValueError("Equivalent float transformation failed whole-shard parity")
        equivalence.append(errors)
    result["float_equivalence_relative_rms"] = equivalence
    del reference, equivalent
    # A probe graph accepts weight arrays. It does not create a model per policy.
    dynamic = onnx.load(str(new_source), load_external_data=False)
    weight_names = {row["name"] for row in rows}
    keep = [copy.deepcopy(t) for t in dynamic.graph.initializer if t.name not in weight_names]
    del dynamic.graph.initializer[:]
    dynamic.graph.initializer.extend(keep)
    for row in rows:
        dynamic.graph.input.append(helper.make_tensor_value_info(row["name"], onnx.TensorProto.FLOAT, row["shape"]))
    mapped = {name: values[0] for name, values in policy["activation_encodings"].items()
              if values[0].get("bitwidth") == 16}
    replacements = {name: f"moshi_policy_a16_{i}_out" for i, name in enumerate(mapped)}
    qdq_nodes = []
    for node in dynamic.graph.node:
        node = copy.deepcopy(node)
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
        qdq_nodes.append(node)
        for name in node.output:
            if name not in mapped:
                continue
            prefix = replacements[name][:-4]
            e = mapped[name]
            for suffix, value in [("scale", e["scale"]), ("offset", e["offset"]), ("low", 0), ("high", 65535)]:
                dynamic.graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), prefix+"_"+suffix))
            operations = [("Div", [name, prefix+"_scale"], "scaled"),
                          ("Round", [prefix+"_scaled"], "rounded"),
                          ("Sub", [prefix+"_rounded", prefix+"_offset"], "unsigned"),
                          ("Clip", [prefix+"_unsigned", prefix+"_low", prefix+"_high"], "clipped"),
                          ("Add", [prefix+"_clipped", prefix+"_offset"], "shifted"),
                          ("Mul", [prefix+"_shifted", prefix+"_scale"], "out")]
            qdq_nodes.extend(helper.make_node(op, inputs, [prefix+"_"+output], name=prefix+"_"+str(i))
                             for i,(op,inputs,output) in enumerate(operations))
    for value in dynamic.graph.output:
        if value.name in replacements:
            value.name = replacements[value.name]
    del dynamic.graph.node[:]
    dynamic.graph.node.extend(qdq_nodes)
    probe_path = source_dir / "policy_probe.onnx"
    onnx.save_model(dynamic, str(probe_path))
    session = ort.InferenceSession(str(probe_path), opts, providers=["CPUExecutionProvider"])
    del model, dynamic, originals, transformed
    output_names = [v.name for v in session.get_outputs()]
    current = {row["name"]: 8 for row in rows}

    def evaluate(bits):
        records = []
        for index, feed, targets in zip(selection_indices, feeds, expected):
            inputs = dict(feed)
            for name, bit in bits.items():
                inputs[name] = candidates[name][bit]
            actual = session.run(None, inputs)
            errors = {name: float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))
                      for name,a,b in zip(output_names, actual, targets)}
            passed = max(errors.values()) <= args.whole_shard_limit and all(np.isfinite(a).all() for a in actual)
            records.append({"sample_index": index, "relative_rms": errors, "passed": passed})
            if not passed:
                return {"passed": False, "samples": records}
        return {"passed": True, "samples": records}

    # Authored A16 boundaries guide selection. The real DLC still requires its own check.
    result["policy_search_activation_quantization"] = True
    baseline = evaluate(current) if not args.prepare_only else {"passed": None, "skipped": "Separate LPBQ gate evaluates W8 and W4"}
    result["w8_control"] = baseline
    if not args.prepare_only and not baseline["passed"]:
        result["status"] = "w8_control_rejected"
        write(root / "RESULT.json", result)
        raise SystemExit(2)
    result["policy_trials"] = []
    # Largest matrices first. Each accepted change must preserve the shard output gate.
    for row in ([] if args.prepare_only else sorted(rows, key=lambda value: -value["parameter_count"])):
        attempt = dict(current)
        attempt[row["name"]] = 4
        check = evaluate(attempt)
        result["policy_trials"].append({"node": row["node"], "weight_bits": attempt, **check})
        if check["passed"]:
            current = attempt
        print("whole shard", row["node"], "W4 accepted", check["passed"], flush=True)
        write(root / "RESULT.json", result)
    for row in rows:
        row["bitwidth"] = current[row["name"]]
        policy["param_encodings"][row["name"]] = encodings(row["scale"+str(row["bitwidth"])], row["bitwidth"])
    result.update(status="activation_aware_prepared_not_policy_verified" if args.prepare_only else "activation_aware_policy_selected", expected_projection_bits=current,
                  selected_precision={str(bits): sum(v == bits for v in current.values()) for bits in (4,8)},
                  w4_parameter_fraction=sum(v["parameter_count"] for v in rows if v["bitwidth"] == 4)/sum(v["parameter_count"] for v in rows),
                  elapsed_s=time.monotonic()-started)
    write(root / "RESULT.json", result)
    write(root / "mixed_policy.json", policy)
    write(root / "weight_statistics.json", rows)
    calibration_dir = root / "calibration"
    calibration_dir.mkdir()
    write(calibration_dir / "activation_encodings.json", {"version": "0.6.1", "param_encodings": {},
                                                           "activation_encodings": policy["activation_encodings"]})
    write(calibration_dir / "calibration.json", {"source_sha256": new_hash, "original_source_sha256": source_hash,
                                                 "samples": prior["samples"], "quality_test": False,
                                                 "original_projection_calibration": str(args.calibration_dir)})
    ref = root / "reference"
    ref.mkdir()
    prior.update(source_onnx=str(new_source), source_onnx_sha256=new_hash,
                 source_external_sha256=result["source_external_sha256"], reference_onnx=str(args.source_onnx),
                 reference_external_sha256=result["original_source_external_sha256"],
                 reference_onnx_sha256=source_hash, equivalent_transform_report=str(root/"RESULT.json"))
    write(ref / "quantization.json", prior)
    # The authoring tool only needs this file for its nonprojection exclusions.
    write(ref / "model.w4a16_encoding.json", json.loads((args.reference_directory/"model.w4a16_encoding.json").read_text()))
    print("Prepared for a separate LPBQ gate" if args.prepare_only else "Selected",
          result["selected_precision"], "W4 fraction", result["w4_parameter_fraction"], flush=True)


if __name__ == "__main__":
    main()
