"""Select compensated W4 weights with a full calibration and short recurrence gate.

The gate uses the retained sixteen inputs and three contiguous frames per clip.
It remains calibration evidence. It does not establish independent audio quality.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from moshi_quantize_temporal_offline import sha256, write
from moshi_refine_weight_rounding import compensated_rounding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("awq-directory", "compensated-directory", "projection-calibration-dir", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--damping-fraction", type=float, default=32.0)
    p.add_argument("--selection-limit", type=float, default=0.095)
    p.add_argument("--prefer-node",action="append",default=[],help="Prioritize a projection identified by prior whole-shard sensitivity")
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
    awq = args.awq_directory
    parent = json.loads((awq/"RESULT.json").read_text())
    previous = json.loads((args.compensated_directory/"RESULT.json").read_text())
    if parent["status"] != "activation_aware_policy_selected" or previous["status"] != "compensated_policy_selected":
        raise ValueError("Require completed scale and compensation results")
    for directory, provenance in ((awq,parent),(args.compensated_directory,previous)):
        if sha256(directory/"source/model.onnx") != provenance["source_onnx_sha256"]:
            raise ValueError("Source graph changed")
        for name,digest in provenance["source_external_sha256"].items():
            if sha256(directory/"source"/name) != digest:
                raise ValueError("Source weights changed")
    prior = json.loads((awq/"reference/quantization.json").read_text())
    model = onnx.load(str(awq/"source/model.onnx"))
    previous_model = onnx.load(str(args.compensated_directory/"source/model.onnx"))
    tensors = {t.name:t for t in model.graph.initializer}
    previous_tensors = {t.name:t for t in previous_model.graph.initializer}
    original = onnx.load(parent["original_source_onnx"],load_external_data=False)
    original_nodes = {n.name:n for n in original.graph.node}
    rows = copy.deepcopy(parent["layers"])
    policy = json.loads((awq/"mixed_policy.json").read_text())
    calibration = json.loads((args.projection_calibration_dir/"calibration.json").read_text())
    if calibration["source_sha256"] != parent["original_source_sha256"]:
        raise ValueError("Projection calibration source differs")
    snapshots = []
    for item in calibration["projection_samples"]:
        path = args.projection_calibration_dir/item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Projection sample changed")
        with np.load(path) as data:
            snapshots.append({name:data[name] for name in data.files})
    candidates, inputs, alternatives = {}, {}, {}
    result = {"format":"moshi-regularized-calibration-policy-v1", "status":"regularizing_rounding",
              "script_sha256":sha256(__file__), "device":args.device, "damping_fraction":args.damping_fraction,
              "parent_directory":str(awq), "compensated_directory":str(args.compensated_directory),
              "original_source_sha256":parent["original_source_sha256"], "independent_quality_test":False,
              "fit_sample_count":12, "range_and_rounding_selection_sample_count":16,
              "selection_relative_rms_limit":args.selection_limit, "final_cpu_relative_rms_limit":0.10,
              "preferred_nodes":args.prefer_node,
              "projection_trials":[], "policy_trials":[], "device_verified":False,
              "algorithm":"Per-channel rounding selection from AWQ, weak covariance compensation, and strong covariance compensation; gated on shard outputs and recurrence"}
    started = time.monotonic()
    for row in rows:
        name = row["name"]
        weight = np.array(numpy_helper.to_array(tensors[name]),copy=True)
        inverse = numpy_helper.to_array(tensors[name+"_awq_inverse"])
        node = original_nodes[row["node"]]
        x = np.concatenate([v[node.input[0]].reshape(-1,weight.shape[0]) for v in snapshots])*inverse
        y = np.concatenate([v[node.output[0]].reshape(-1,weight.shape[1]) for v in snapshots])
        scale = np.asarray(row["scale4"],np.float32)
        variants = [np.clip(np.rint(weight/scale),-8,7)*scale]
        labels = ["awq_nearest"]
        if previous["expected_projection_bits"][name] == 4 and parent["expected_projection_bits"][name] == 8:
            variants.append(np.array(numpy_helper.to_array(previous_tensors[name]),copy=True))
            labels.append("weak_compensation")
        variants.append(compensated_rounding(weight,x[parent["weight_search_train_indices"]],scale,torch,args.device,args.damping_fraction))
        labels.append("strong_compensation")
        norms = np.maximum(np.sqrt(np.mean(y.astype(np.float64)**2,axis=1))[:,None],1e-12)
        errors = [np.mean(((x@v-y).astype(np.float64)/norms)**2,axis=0) for v in variants]
        choices = np.argmin(np.stack(errors),axis=0)
        selected = variants[0].copy()
        for index,variant in enumerate(variants[1:],1):
            selected[:,choices==index] = variant[:,choices==index]
        candidates[name] = np.ascontiguousarray(selected)
        alternatives[name] = [("per_channel_selection", candidates[name]), *list(zip(labels,variants))]
        scale8 = np.asarray(row["scale8"],np.float32)
        inputs[name] = np.ascontiguousarray(np.clip(np.rint(weight/scale8),-128,127)*scale8)
        result["projection_trials"].append({"node":row["node"],"choice_counts":{
            label:int(np.sum(choices==index)) for index,label in enumerate(labels)},
            "mean_normalized_output_mse":float(np.mean(np.min(np.stack(errors),axis=0)))})
        print("rounding",row["node"],result["projection_trials"][-1]["choice_counts"],flush=True)
        write(root/"RESULT.json",result)
    del previous_model,previous_tensors
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    reference = ort.InferenceSession(parent["original_source_onnx"],opts,providers=["CPUExecutionProvider"])
    probe = ort.InferenceSession(str(awq/"source/policy_probe.onnx"),opts,providers=["CPUExecutionProvider"])
    names = [v.name for v in reference.get_outputs()]
    cache_names = [v.name for v in reference.get_inputs() if v.name.startswith("layer_")]
    feeds,targets = [],[]
    for sample in prior["samples"]:
        path = Path(prior["calibration_manifest"]).parent/sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Captured shard input changed")
        with np.load(path) as data:
            feed = {v.name:data[v.name] for v in reference.get_inputs()}
        for name in cache_names:
            feed[name] = feed[name][:,:,:1000,:].copy()
        feeds.append(feed)
        targets.append(reference.run(None,feed))
    # Reference recurrence uses its own K/V. Upstream hidden remains captured.
    recurrent_targets = {}
    for source_id in dict.fromkeys(v["source_id"] for v in prior["samples"]):
        indices = [next(i for i,v in enumerate(prior["samples"]) if v["source_id"]==source_id and v["frame"]==f) for f in range(3)]
        states = {n:feeds[indices[0]][n].copy() for n in cache_names}
        for frame,index in enumerate(indices):
            got = reference.run(None,dict(feeds[index],**states))
            recurrent_targets[index] = got
            for name in cache_names:
                states[name][:,:,frame:frame+1,:] = got[names.index("output_"+name)]

    def compare(actual,expected):
        errors = {n:float(np.linalg.norm(a.astype(np.float64)-b)/max(np.linalg.norm(b.astype(np.float64)),1e-12))
                  for n,a,b in zip(names,actual,expected)}
        passed = max(errors.values()) <= args.selection_limit and all(np.isfinite(a).all() for a in actual)
        return errors,passed

    def evaluate():
        records = []
        for index in [1,9,2,10,15,7,5,13,0,8,3,4,6,11,12,14]:
            actual = probe.run(None,dict(feeds[index],**inputs))
            errors,passed = compare(actual,targets[index])
            records.append({"mode":"teacher", "sample_index":index,"relative_rms":errors,"passed":passed})
            if not passed:
                return {"passed":False,"samples":records}
        for source_id in dict.fromkeys(v["source_id"] for v in prior["samples"]):
            indices = [next(i for i,v in enumerate(prior["samples"]) if v["source_id"]==source_id and v["frame"]==f) for f in range(3)]
            states = {n:feeds[indices[0]][n].copy() for n in cache_names}
            for frame,index in enumerate(indices):
                actual = probe.run(None,dict(feeds[index],**states,**inputs))
                errors,passed = compare(actual,recurrent_targets[index])
                records.append({"mode":"recurrent","sample_index":index,"relative_rms":errors,"passed":passed})
                if not passed:
                    return {"passed":False,"samples":records}
                for name in cache_names:
                    states[name][:,:,frame:frame+1,:] = actual[names.index("output_"+name)]
        return {"passed":True,"samples":records}

    result["w8_control"] = evaluate()
    if not result["w8_control"]["passed"]:
        result["status"] = "w8_control_rejected"
        write(root/"RESULT.json",result)
        raise SystemExit(2)
    current = {row["name"]:8 for row in rows}
    selected_methods = {}
    if any(name not in {row["node"] for row in rows} for name in args.prefer_node):
        raise ValueError("Preferred projection is absent from the source")
    order=sorted(rows,key=lambda v:(args.prefer_node.index(v["node"]) if v["node"] in args.prefer_node else len(args.prefer_node),-v["parameter_count"]))
    for row in order:
        name = row["name"]
        before = inputs[name]
        best, trials = None, []
        for method,weights in alternatives[name]:
            inputs[name] = weights
            check = evaluate()
            maximum = max(v for sample in check["samples"] for v in sample["relative_rms"].values())
            trials.append({"rounding_method":method,"max_relative_rms":maximum,**check})
            if check["passed"] and (best is None or maximum < best["error"]):
                best = {"method":method,"weights":weights,"error":maximum}
            # Do not compare further rounding choices after a comfortable pass.
            if best and best["error"] <= 0.05:
                break
        if best:
            current[name] = 4
            inputs[name] = candidates[name] = best["weights"]
            selected_methods[name] = best["method"]
        else:
            inputs[name] = before
        result["policy_trials"].append({"node":row["node"],"passed":best is not None,"rounding_trials":trials})
        print("gated",row["node"],"W4",best is not None,"method",best["method"] if best else None,flush=True)
        write(root/"RESULT.json",result)
    for row in rows:
        name = row["name"]
        bits = current[name]
        row["bitwidth"] = bits
        row["rounding_method"] = selected_methods.get(name,"nearest")
        scale = np.asarray(row["scale"+str(bits)],np.float32)
        policy["param_encodings"][name] = [{"bitwidth":bits,"is_symmetric":"true","scale":float(s),"offset":0,
                                            "min":float(-(2**(bits-1))*s),"max":float((2**(bits-1)-1)*s)} for s in scale]
        if bits==4:
            tensors[name].CopyFrom(numpy_helper.from_array(candidates[name],name))
    new_source = root/"source/model.onnx"
    new_source.parent.mkdir()
    onnx.save_model(model,str(new_source),save_as_external_data=True,all_tensors_to_one_file=True,location="model.data",size_threshold=1024)
    source_hash = sha256(new_source)
    result.update(status="regularized_policy_selected",source_onnx_sha256=source_hash,
                  source_external_sha256={"model.data":sha256(new_source.parent/"model.data")},expected_projection_bits=current,
                  selected_precision={str(bits):sum(v==bits for v in current.values()) for bits in (4,8)},elapsed_s=time.monotonic()-started,
                  w4_parameter_fraction=sum(v["parameter_count"] for v in rows if v["bitwidth"]==4)/sum(v["parameter_count"] for v in rows))
    write(root/"RESULT.json",result)
    write(root/"mixed_policy.json",policy)
    write(root/"weight_statistics.json",rows)
    cal = root/"calibration"
    cal.mkdir()
    write(cal/"activation_encodings.json",json.loads((awq/"calibration/activation_encodings.json").read_text()))
    cal_receipt = json.loads((awq/"calibration/calibration.json").read_text())
    cal_receipt["source_sha256"] = source_hash
    write(cal/"calibration.json",cal_receipt)
    ref = root/"reference"
    ref.mkdir()
    prior.update(source_onnx=str(new_source),source_onnx_sha256=source_hash,source_external_sha256=result["source_external_sha256"],
                 weight_optimization_report=str(root/"RESULT.json"))
    write(ref/"quantization.json",prior)
    write(ref/"model.w4a16_encoding.json",json.loads((awq/"reference/model.w4a16_encoding.json").read_text()))
    print("Selected",result["selected_precision"],"W4 fraction",result["w4_parameter_fraction"],flush=True)


if __name__ == "__main__":
    main()
