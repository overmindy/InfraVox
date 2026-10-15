"""Select ordinary per-channel W4/W8 with calibrated output-bias correction.

Use twelve projection snapshots to estimate each correction. Select precision
with all sixteen retained samples and short shard recurrence. This is calibration
evidence, not independent quality evidence. Keep the original model untouched.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

from moshi_calibration_gate import CalibrationGate
from moshi_quantize_temporal_offline import sha256,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("awq-directory","projection-calibration-dir","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import onnx
    from onnx import helper,numpy_helper
    root=args.output_dir;root.mkdir()
    awq=args.awq_directory
    parent=json.loads((awq/"RESULT.json").read_text())
    prior=json.loads((awq/"reference/quantization.json").read_text())
    calibration=json.loads((args.projection_calibration_dir/"calibration.json").read_text())
    if calibration["source_sha256"]!=parent["original_source_sha256"]:
        raise ValueError("Projection source differs")
    for name,digest in parent["source_external_sha256"].items():
        if sha256(awq/"source"/name)!=digest:
            raise ValueError("AWQ weights changed")
    snapshots=[]
    for item in calibration["projection_samples"]:
        path=args.projection_calibration_dir/item["file"]
        if sha256(path)!=item["sha256"]:
            raise ValueError("Projection snapshot changed")
        with np.load(path) as data:
            snapshots.append({name:data[name] for name in data.files})
    model=onnx.load(str(awq/"source/model.onnx"))
    tensors={t.name:t for t in model.graph.initializer}
    original=onnx.load(parent["original_source_onnx"],load_external_data=False)
    nodes={n.name:n for n in original.graph.node}
    rows=copy.deepcopy(parent["layers"])
    inputs,candidates,corrections={},{},{}
    result={"format":"moshi-bias-corrected-policy-v1","status":"estimating_bias_corrections",
            "script_sha256":sha256(__file__),"original_source_sha256":parent["original_source_sha256"],
            "independent_quality_test":False,"fit_indices":parent["weight_search_train_indices"],
            "local_holdout_indices":parent["weight_search_holdout_indices"],"selection_sample_count":16,
            "selection_recurrent_frame_count":6,"selection_relative_rms_limit":0.095,
            "final_cpu_relative_rms_limit":0.10,"layers":rows,"policy_trials":[],"device_verified":False}
    started=time.monotonic()
    fit=parent["weight_search_train_indices"]
    heldout=parent["weight_search_holdout_indices"]
    for row in rows:
        name=row["name"]
        weight=numpy_helper.to_array(tensors[name])
        node=nodes[row["node"]]
        inverse=numpy_helper.to_array(tensors[name+"_awq_inverse"])
        x=np.concatenate([v[node.input[0]].reshape(-1,weight.shape[0]) for v in snapshots])*inverse
        y=np.concatenate([v[node.output[0]].reshape(-1,weight.shape[1]) for v in snapshots])
        s4=np.asarray(row["scale4"],np.float32)
        s8=np.asarray(row["scale8"],np.float32)
        candidates[name]=np.ascontiguousarray(np.clip(np.rint(weight/s4),-8,7)*s4)
        inputs[name]=np.ascontiguousarray(np.clip(np.rint(weight/s8),-128,127)*s8)
        inputs[name+"_bias_correction"]=np.zeros(weight.shape[1],np.float32)
        predicted=x@candidates[name]
        groups=3 if row["node"].startswith("/in_projs") else (2 if weight.shape[1]==22528 else 1)
        bias=np.zeros(weight.shape[1],np.float32)
        for columns in np.array_split(np.arange(weight.shape[1]),groups):
            target=y[:,columns].astype(np.float64)
            energy=np.mean(target[fit]**2,axis=1)
            weights=1/np.maximum(energy,max(float(np.median(energy))*0.01,1e-12))
            weights/=weights.sum()
            bias[columns]=np.sum((target[fit]-predicted[fit][:,columns])*weights[:,None],axis=0)
        corrections[name]=bias
        def worst(values,indices):
            return max(float(np.max(np.linalg.norm(a[indices]-b[indices],axis=1)/np.maximum(np.linalg.norm(b[indices],axis=1),1e-12)))
                       for a,b in zip(np.array_split(values,groups,axis=1),np.array_split(y,groups,axis=1)))
        row["bias_correction"]={"fit_before":worst(predicted,fit),"fit_after":worst(predicted+bias,fit),
                                "holdout_before":worst(predicted,heldout),"holdout_after":worst(predicted+bias,heldout),
                                "max_abs_bias":float(np.max(np.abs(bias)))}
        print(row["node"],row["bias_correction"],flush=True)
    # A probe accepts weight and bias arrays. QDQ remains after the added bias.
    probe=onnx.load(str(awq/"source/policy_probe.onnx"),load_external_data=False)
    weight_names={row["name"] for row in rows}
    replacement=[]
    for node in probe.graph.node:
        node=copy.deepcopy(node)
        if node.op_type=="MatMul" and node.input[1] in weight_names:
            old_output=node.output[0]
            node.output[0]=old_output+"_before_bias"
            replacement.append(node)
            replacement.append(helper.make_node("Add",[node.output[0],node.input[1]+"_bias_correction"],[old_output],name=node.name+"_bias_correction"))
        else:
            replacement.append(node)
    del probe.graph.node[:];probe.graph.node.extend(replacement)
    for row in rows:
        probe.graph.input.append(helper.make_tensor_value_info(row["name"]+"_bias_correction",onnx.TensorProto.FLOAT,[row["shape"][1]]))
    # Preserve existing external tensor offsets without another weight export.
    for tensor in probe.graph.initializer:
        for entry in tensor.external_data:
            if entry.key=="location":entry.value=str((awq/"source"/entry.value).resolve())
    probe_path=root/"policy_probe.onnx";onnx.save_model(probe,str(probe_path))
    gate=CalibrationGate(parent["original_source_onnx"],probe_path,prior)
    result["w8_control"]=gate(inputs)
    if not result["w8_control"]["passed"]:
        result["status"]="w8_control_rejected";write(root/"RESULT.json",result);raise SystemExit(2)
    current={row["name"]:8 for row in rows}
    selected_biases={}
    for row in sorted(rows,key=lambda v:-v["parameter_count"]):
        name=row["name"]
        before=inputs[name]
        inputs[name]=candidates[name]
        best,trials=None,[]
        for corrected in (True,False):
            bias=corrections[name] if corrected else np.zeros_like(corrections[name])
            inputs[name+"_bias_correction"]=bias
            check=gate(inputs)
            maximum=max(v for s in check["samples"] for v in s["relative_rms"].values())
            trials.append({"bias_corrected":corrected,"max_relative_rms":maximum,**check})
            if check["passed"] and (best is None or maximum<best["error"]):
                best={"bias":bias,"error":maximum,"corrected":corrected}
            if best and best["error"]<=0.05:break
        if best:
            current[name]=4;selected_biases[name]=best["bias"]
            inputs[name+"_bias_correction"]=best["bias"]
        else:
            inputs[name]=before;inputs[name+"_bias_correction"]=np.zeros_like(corrections[name])
        result["policy_trials"].append({"node":row["node"],"passed":best is not None,"rounding_trials":trials})
        print("bias gated",row["node"],"W4",best is not None,flush=True)
        write(root/"RESULT.json",result)
    replacement=[]
    for node in model.graph.node:
        node=copy.deepcopy(node)
        if node.op_type=="MatMul" and node.input[1] in selected_biases:
            old_output=node.output[0]
            node.output[0]=old_output+"_before_bias"
            bias_name=node.input[1]+"_bias_correction"
            model.graph.initializer.append(numpy_helper.from_array(selected_biases[node.input[1]],bias_name))
            replacement.append(node)
            replacement.append(helper.make_node("Add",[node.output[0],bias_name],[old_output],name=node.name+"_bias_correction"))
        else:replacement.append(node)
    del model.graph.node[:];model.graph.node.extend(replacement)
    source=root/"source/model.onnx";source.parent.mkdir()
    onnx.save_model(model,str(source),save_as_external_data=True,all_tensors_to_one_file=True,location="model.data",size_threshold=1024)
    policy=json.loads((awq/"mixed_policy.json").read_text())
    for row in rows:
        name=row["name"];bits=current[name];row["bitwidth"]=bits
        scales=row["scale"+str(bits)]
        policy["param_encodings"][name]=[{"bitwidth":bits,"is_symmetric":"true","scale":s,"offset":0,
                                           "min":-(2**(bits-1))*s,"max":(2**(bits-1)-1)*s} for s in scales]
    result.update(status="bias_corrected_policy_selected",source_onnx_sha256=sha256(source),
                  source_external_sha256={"model.data":sha256(source.parent/"model.data")},expected_projection_bits=current,
                  selected_precision={str(b):sum(v==b for v in current.values()) for b in (4,8)},elapsed_s=time.monotonic()-started,
                  w4_parameter_fraction=sum(v["parameter_count"] for v in rows if v["bitwidth"]==4)/sum(v["parameter_count"] for v in rows))
    write(root/"RESULT.json",result);write(root/"mixed_policy.json",policy);write(root/"weight_statistics.json",rows)
    cal=root/"calibration";cal.mkdir()
    write(cal/"activation_encodings.json",json.loads((awq/"calibration/activation_encodings.json").read_text()))
    receipt=json.loads((awq/"calibration/calibration.json").read_text());receipt["source_sha256"]=result["source_onnx_sha256"]
    write(cal/"calibration.json",receipt)
    ref=root/"reference";ref.mkdir()
    prior.update(source_onnx=str(source),source_onnx_sha256=result["source_onnx_sha256"],source_external_sha256=result["source_external_sha256"],
                 bias_correction_report=str(root/"RESULT.json"))
    write(ref/"quantization.json",prior)
    write(ref/"model.w4a16_encoding.json",json.loads((awq/"reference/model.w4a16_encoding.json").read_text()))
    print("Selected",result["selected_precision"],"W4 fraction",result["w4_parameter_fraction"],flush=True)


if __name__=="__main__":main()
