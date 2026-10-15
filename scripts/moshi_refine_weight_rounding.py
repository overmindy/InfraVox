"""Refine retained W8 projections with GPTQ-style W4 error compensation.

Use a damped input covariance to compensate quantization error in later input
channels. Keep a change only when the shard calibration gate passes. Rejected
weights remain at the previously selected precision. No PyTorch export is used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from moshi_quantize_temporal_offline import sha256, write


def compensated_rounding(weight, inputs, scale, torch, device, damping_fraction=0.01):
    """Implement the second-order compensation described in arXiv:2210.17323."""
    with torch.inference_mode():
        x = torch.as_tensor(inputs, dtype=torch.float64, device=device)
        # Normalize samples before the covariance. This gives later frames weight.
        x = x / x.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-12)
        covariance = x.T @ x / len(x)
        diagonal = covariance.diagonal()
        order = torch.argsort(diagonal, descending=True)
        damping = damping_fraction * diagonal.mean().clamp(min=1e-12)
        covariance.diagonal().add_(damping)
        covariance = covariance[order][:, order]
        chol = torch.linalg.cholesky(covariance)
        inverse = torch.cholesky_inverse(chol)
        upper = torch.linalg.cholesky(inverse, upper=True).float()
        del covariance, chol, inverse, x
        work = torch.as_tensor(weight.T.copy(), device=device)[:, order].contiguous()
        scale = torch.as_tensor(scale, device=device)
        if scale.ndim == 2:
            scale = scale[:,order]
        quantized = torch.empty_like(work)
        # Block updates reduce launches and retain fixed, per-output-channel scales.
        for start in range(0, work.shape[1], 128):
            end = min(start+128, work.shape[1])
            block = work[:, start:end].clone()
            errors = torch.empty_like(block)
            for column in range(end-start):
                index = start+column
                value = block[:, column].clone()
                column_scale = scale if scale.ndim==1 else scale[:,index]
                rounded = torch.clamp(torch.round(value/column_scale), -8, 7)*column_scale
                quantized[:, index] = rounded
                correction = (value-rounded)/upper[index, index]
                errors[:, column] = correction
                block[:, column:].addcmul_(correction[:, None], upper[index, index:end][None, :], value=-1)
            if end < work.shape[1]:
                work[:, end:] -= errors @ upper[start:end, end:]
        restore = torch.argsort(order)
        output = quantized[:, restore].T.contiguous().cpu().numpy()
        if not torch.isfinite(quantized).all():
            raise ValueError("Covariance compensation produced nonfinite weights")
        return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--awq-directory", required=True, type=Path)
    p.add_argument("--projection-calibration-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    old = args.awq_directory
    parent = json.loads((old/"RESULT.json").read_text())
    if parent["status"] != "activation_aware_policy_selected":
        raise ValueError("Require a completed scale/policy search")
    prior = json.loads((old/"reference/quantization.json").read_text())
    policy = json.loads((old/"mixed_policy.json").read_text())
    source = old/"source/model.onnx"
    if sha256(source) != parent["source_onnx_sha256"]:
        raise ValueError("Parent source changed")
    for name, digest in parent["source_external_sha256"].items():
        if sha256(source.parent/name) != digest:
            raise ValueError("Parent weights changed")
    projection_manifest = json.loads((args.projection_calibration_dir/"calibration.json").read_text())
    if projection_manifest["source_sha256"] != parent["original_source_sha256"]:
        raise ValueError("Projection samples have a different original source")
    snapshots = []
    for item in projection_manifest["projection_samples"]:
        path = args.projection_calibration_dir/item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Projection snapshot changed")
        with np.load(path) as data:
            snapshots.append({name: data[name] for name in data.files})
    model = onnx.load(str(source))
    tensors = {t.name: t for t in model.graph.initializer}
    original = onnx.load(parent["original_source_onnx"], load_external_data=False)
    original_nodes = {n.name: n for n in original.graph.node}
    rows = copy.deepcopy(parent["layers"])
    current = dict(parent["expected_projection_bits"])
    weights = {row["name"]: np.array(numpy_helper.to_array(tensors[row["name"]]), copy=True) for row in rows}
    inputs = {row["name"]: np.ascontiguousarray(np.clip(np.rint(weights[row["name"]]/np.asarray(row["scale"+str(current[row["name"]])],np.float32)),
                                                      -(2**(current[row["name"]]-1)), 2**(current[row["name"]]-1)-1)
                                               * np.asarray(row["scale"+str(current[row["name"]])],np.float32)) for row in rows}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    probe = ort.InferenceSession(str(old/"source/policy_probe.onnx"), opts, providers=["CPUExecutionProvider"])
    reference = ort.InferenceSession(parent["original_source_onnx"], opts, providers=["CPUExecutionProvider"])
    selection_indices = [1,9,15,0]
    feeds, expected = [], []
    for index in selection_indices:
        sample = prior["samples"][index]
        path = Path(prior["calibration_manifest"]).parent/sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Captured shard input changed")
        with np.load(path) as data:
            feed = {v.name: data[v.name] for v in reference.get_inputs()}
        for name in feed:
            if name.startswith("layer_"):
                feed[name] = feed[name][:,:,:1000,:].copy()
        feeds.append(feed)
        expected.append(reference.run(None, feed))
    output_names = [v.name for v in reference.get_outputs()]
    result = {"format": "moshi-compensated-rounding-v1", "status": "refining_weights",
              "script_sha256": sha256(__file__), "algorithm_reference": "https://arxiv.org/abs/2210.17323",
              "algorithm": "GPTQ-style covariance compensation, sample normalization, fixed per-output-channel W4 grids",
              "damping_fraction": 0.01, "block_size": 128, "activation_order": True,
              "parent_result_sha256": sha256(old/"RESULT.json"), "parent_directory": str(old),
              "original_source_sha256": parent["original_source_sha256"], "device_verified": False,
              "independent_quality_test": False, "trials": [], "whole_shard_relative_rms_limit": 0.10}

    def evaluate():
        records = []
        for index, feed, target in zip(selection_indices, feeds, expected):
            actual = probe.run(None, dict(feed, **inputs))
            errors = {name: float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
                      for name,a,b in zip(output_names,actual,target)}
            passed = max(errors.values()) <= 0.10 and all(np.isfinite(a).all() for a in actual)
            records.append({"sample_index": index, "relative_rms": errors, "passed": passed})
            if not passed:
                return {"passed": False, "samples": records}
        return {"passed": True, "samples": records}

    result["parent_check"] = evaluate()
    if not result["parent_check"]["passed"]:
        raise ValueError("Parent policy replay failed")
    start = time.monotonic()
    for row in sorted(rows, key=lambda value: -value["parameter_count"]):
        name = row["name"]
        if current[name] != 8:
            continue
        node = original_nodes[row["node"]]
        inverse = numpy_helper.to_array(tensors[name+"_awq_inverse"])
        x = np.concatenate([v[node.input[0]].reshape(-1,row["shape"][0]) for v in snapshots])*inverse
        y = np.concatenate([v[node.output[0]].reshape(-1,row["shape"][1]) for v in snapshots])
        fit = parent["weight_search_train_indices"]
        heldout = parent["weight_search_holdout_indices"]
        scale4 = np.asarray(row["scale4"], np.float32)
        tick = time.monotonic()
        rounded = compensated_rounding(weights[name], x[fit], scale4, torch, args.device)
        predicted = x @ rounded
        def worst(indices):
            return float(np.max(np.linalg.norm(predicted[indices]-y[indices],axis=1)/
                                np.maximum(np.linalg.norm(y[indices],axis=1),1e-12)))
        before = inputs[name]
        inputs[name] = rounded
        check = evaluate()
        trial = {"name": name, "node": row["node"], "elapsed_s": time.monotonic()-tick,
                 "train_relative_rms": worst(fit), "holdout_relative_rms": worst(heldout), **check}
        if check["passed"]:
            current[name] = 4
            tensors[name].CopyFrom(numpy_helper.from_array(rounded, name))
            row["bitwidth"] = 4
            row["rounding_method"] = "covariance_compensated"
            policy["param_encodings"][name] = [{"bitwidth":4,"is_symmetric":"true","scale":float(s),"offset":0,
                                                "min":float(-8*s),"max":float(7*s)} for s in scale4]
        else:
            inputs[name] = before
        result["trials"].append(trial)
        print(row["node"], "compensated W4 accepted", check["passed"], "seconds", trial["elapsed_s"], flush=True)
        write(root/"RESULT.json", result)
    new_source = root/"source/model.onnx"
    new_source.parent.mkdir()
    onnx.save_model(model, str(new_source), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.data", size_threshold=1024)
    source_hash = sha256(new_source)
    result.update(status="compensated_policy_selected", source_onnx_sha256=source_hash,
                  source_external_sha256={"model.data":sha256(new_source.parent/"model.data")},
                  selected_precision={str(bits):sum(v==bits for v in current.values()) for bits in (4,8)},
                  expected_projection_bits=current, elapsed_s=time.monotonic()-start,
                  w4_parameter_fraction=sum(v["parameter_count"] for v in rows if v["bitwidth"]==4)/sum(v["parameter_count"] for v in rows))
    write(root/"RESULT.json", result)
    write(root/"mixed_policy.json", policy)
    write(root/"weight_statistics.json", rows)
    calibration = root/"calibration"
    calibration.mkdir()
    write(calibration/"activation_encodings.json", json.loads((old/"calibration/activation_encodings.json").read_text()))
    cal = json.loads((old/"calibration/calibration.json").read_text())
    cal["source_sha256"] = source_hash
    write(calibration/"calibration.json", cal)
    ref = root/"reference"
    ref.mkdir()
    prior.update(source_onnx=str(new_source), source_onnx_sha256=source_hash,
                 source_external_sha256=result["source_external_sha256"], weight_optimization_report=str(root/"RESULT.json"))
    write(ref/"quantization.json", prior)
    write(ref/"model.w4a16_encoding.json", json.loads((old/"reference/model.w4a16_encoding.json").read_text()))
    print("Selected", result["selected_precision"], "W4 fraction", result["w4_parameter_fraction"], flush=True)


if __name__ == "__main__":
    main()
