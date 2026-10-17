"""Build a sensitivity-selected LPBQ W4/A16 candidate from the retained Moshi graph."""
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

from moshi_calibration_gate import CalibrationGate
from moshi_quantize_temporal_offline import sha256,write
from moshi_refine_weight_rounding import compensated_rounding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("awq-directory","awq-candidate-directory","projection-calibration-dir","output-dir","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--block-size",type=int,default=32)
    p.add_argument("--pointwise",action=argparse.BooleanOptionalAction,default=True,
                   help="Default to the HTP-validated Conv layout; --no-pointwise retains the diagnostic FC path")
    p.add_argument("--rounding",choices=("compensated","nearest"),default="compensated")
    p.add_argument("--selection-limit",type=float,default=0.095)
    p.add_argument("--first-passing-rounding",action="store_true",help="Stop alternatives when the current W4 policy passes the full gate")
    args=p.parse_args()
    sys.path[:0]=[str(args.numpy_dir),str(args.sdk_root/"lib/python")]
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=args.output_dir
    root.mkdir()
    awq=args.awq_directory
    parent=json.loads((awq/"RESULT.json").read_text())
    prior=json.loads((awq/"reference/quantization.json").read_text())
    calibration=json.loads((args.projection_calibration_dir/"calibration.json").read_text())
    if calibration["source_sha256"]!=parent["original_source_sha256"]:
        raise ValueError("Calibration source differs")
    if sha256(awq/"source/model.onnx")!=parent["source_onnx_sha256"]:
        raise ValueError("AWQ source graph changed")
    for name,digest in parent["source_external_sha256"].items():
        if sha256(awq/"source"/name)!=digest:
            raise ValueError("AWQ weights changed")
    snapshots=[]
    for item in calibration["projection_samples"]:
        path=args.projection_calibration_dir/item["file"]
        if sha256(path)!=item["sha256"]:
            raise ValueError("Projection sample changed")
        with np.load(path) as data:
            snapshots.append({n:data[n] for n in data.files})
    model=onnx.load(str(awq/"source/model.onnx"))
    tensors={t.name:t for t in model.graph.initializer}
    original=onnx.load(parent["original_source_onnx"],load_external_data=False)
    original_nodes={n.name:n for n in original.graph.node}
    rows=copy.deepcopy(parent["layers"])
    result={"format":"moshi-lpbq-mixed-candidate-v1","status":"searching_lpbq_weights",
            "script_sha256":sha256(__file__),"block_size":args.block_size,"damping_fraction":32.0,
            "original_source_onnx":parent["original_source_onnx"],"original_source_sha256":parent["original_source_sha256"],
            "selection_relative_rms_limit":args.selection_limit,"final_cpu_relative_rms_limit":0.10,
            "pointwise":args.pointwise,"rounding":args.rounding,
            "fit_samples":12,"selection_samples":16,"selection_recurrent_frames":6,
            "independent_quality_test":False,"device_verified":False,"default_bundle_replaced":False,
            "layers":rows,"policy_trials":[]}
    reference_external=copy.deepcopy(parent.get("original_source_external_sha256", {}))
    if not reference_external:
        reference_external={"model.data":sha256(Path(parent["original_source_onnx"]).parent/"model.data")}
    result["reference_external_sha256"]=reference_external
    started=time.monotonic()
    alternatives,inputs,lpbq={}, {}, {}
    for row in rows:
        name=row["name"]
        weight=np.array(numpy_helper.to_array(tensors[name]),copy=True)
        nin,nout=weight.shape
        if nin%args.block_size:
            raise ValueError("Input channels are not divisible by block size")
        blocked=np.ascontiguousarray(weight.T).reshape(nout,nin//args.block_size,args.block_size)
        block_steps=np.maximum(-blocked.min(axis=2)/8,blocked.max(axis=2)/7)
        channel_steps=np.maximum(block_steps.max(axis=1)/16,1e-12).astype(np.float32)
        expansion=np.clip(np.rint(block_steps/channel_steps[:,None]),1,16).astype(np.uint8)
        scales=channel_steps[:,None]*np.repeat(expansion,args.block_size,axis=1)
        nearest=np.ascontiguousarray((np.clip(np.rint(weight.T/scales),-8,7)*scales).T)
        node=original_nodes[row["node"]]
        inverse=numpy_helper.to_array(tensors[name+"_awq_inverse"])
        x=np.concatenate([v[node.input[0]].reshape(-1,nin) for v in snapshots])*inverse
        y=np.concatenate([v[node.output[0]].reshape(-1,nout) for v in snapshots])
        refined=(compensated_rounding(weight,x[parent["weight_search_train_indices"]],scales,torch,args.device,32.0)
                 if args.rounding=="compensated" else nearest)
        norm=np.maximum(np.sqrt(np.mean(y.astype(np.float64)**2,axis=1))[:,None],1e-12)
        err0=np.mean(((x@nearest-y).astype(np.float64)/norm)**2,axis=0)
        err1=np.mean(((x@refined-y).astype(np.float64)/norm)**2,axis=0)
        prefer=err1<err0
        selected=np.where(prefer[None,:],refined,nearest)
        alternatives[name]=[("per_channel_rounding",np.ascontiguousarray(selected)),("nearest",nearest),("strong_compensation",refined)]
        scale8=np.asarray(row["scale8"],np.float32)
        inputs[name]=np.ascontiguousarray(np.clip(np.rint(weight/scale8),-128,127)*scale8)
        lpbq[name]={"name":name,"dtype":"INT","enc_type":"LPBQ","bw":8,"compressed_bw":4,"is_sym":True,
                    "scale":channel_steps.tolist(),"offset":[0]*nout,"block_size":args.block_size,
                    "per_block_int_scale":expansion.flatten().tolist()}
        row["lpbq_refined_channels"]=int(prefer.sum())
        print("LPBQ prepared",row["node"],"refined channels",int(prefer.sum()),flush=True)
        write(root/"RESULT.json",result)
    gate=CalibrationGate(parent["original_source_onnx"],awq/"source/policy_probe.onnx",prior,limit=args.selection_limit)
    result["w8_control"]=gate(inputs)
    if not result["w8_control"]["passed"]:
        result["status"]="w8_control_rejected"
        write(root/"RESULT.json",result)
        raise SystemExit(2)
    current={row["name"]:8 for row in rows}
    selected_methods={}
    for row in sorted(rows,key=lambda v:-v["parameter_count"]):
        name=row["name"]
        before=inputs[name]
        best,trials=None,[]
        for method,weight in alternatives[name]:
            inputs[name]=weight
            check=gate(inputs)
            maximum=max(v for sample in check["samples"] for v in sample["relative_rms"].values())
            trials.append({"rounding_method":method,"max_relative_rms":maximum,**check})
            if check["passed"] and (best is None or maximum<best["error"]):
                best={"method":method,"weight":weight,"error":maximum}
            if best and (best["error"]<=0.05 or args.first_passing_rounding):
                break
        if best:
            current[name]=4
            inputs[name]=best["weight"]
            selected_methods[name]=best["method"]
        else:
            inputs[name]=before
        result["policy_trials"].append({"node":row["node"],"passed":best is not None,"rounding_trials":trials})
        print("LPBQ gated",row["node"],"W4",best is not None,flush=True)
        write(root/"RESULT.json",result)
    base_policy=json.loads((args.awq_candidate_directory/"quantization_overrides.json").read_text())
    overrides={"version":"1.0.0","activation_encodings":[],"param_encodings":[]}
    for name,values in base_policy["activation_encodings"].items():
        e=values[0]
        entry={"name":name,"bw":e["bitwidth"],"dtype":"FLOAT" if e.get("dtype")=="float" else "INT"}
        if entry["dtype"]=="INT":
            entry.update(enc_type="PER_TENSOR",is_sym=False,scale=[e["scale"]],offset=[e["offset"]])
        overrides["activation_encodings"].append(entry)
    for name,values in base_policy["param_encodings"].items():
        if name not in current:
            if values[0].get("dtype")!="float":
                raise ValueError("Expected retained float nonprojection parameter")
            overrides["param_encodings"].append({"name":name,"dtype":"FLOAT","bw":32})
    for row in rows:
        name=row["name"]
        row["bitwidth"]=current[name]
        row["rounding_method"]=selected_methods.get(name,"nearest")
        if current[name]==4:
            tensors[name].CopyFrom(numpy_helper.from_array(inputs[name],name))
            overrides["param_encodings"].append(lpbq[name])
        else:
            overrides["param_encodings"].append({"name":name,"dtype":"INT","enc_type":"PER_CHANNEL","is_sym":True,
                                                "bw":8,"scale":row["scale8"],"offset":[0]*row["shape"][1]})
    if args.pointwise:
        from moshi_projection_layout import rewrite_pointwise_projections
        result["pointwise_rewrites"]=rewrite_pointwise_projections(model,overrides,current)
    source=root/"source/model.onnx"
    source.parent.mkdir()
    onnx.save_model(model,str(source),save_as_external_data=True,all_tensors_to_one_file=True,location="model.data",size_threshold=1024)
    write(root/"quantization_overrides.json",overrides)
    result.update(status="converting_lpbq_policy",expected_projection_bits=current,
                  selected_precision={str(bits):sum(v==bits for v in current.values()) for bits in (4,8)},
                  w4_parameter_fraction=sum(v["parameter_count"] for v in rows if v["bitwidth"]==4)/sum(v["parameter_count"] for v in rows),
                  source_onnx_sha256=sha256(source),source_external_sha256={"model.data":sha256(source.parent/"model.data")})
    write(root/"RESULT.json",result)
    candidate=root/"candidate";candidate.mkdir()
    scripts=root/"runtime_scripts";scripts.mkdir()
    for name in ("moshi_build_lpbq_candidate.py","moshi_calibration_gate.py","moshi_refine_weight_rounding.py",
                 "moshi_quantize_temporal_offline.py","moshi_guard_qairt_converter.py","moshi_check_offline_dlc.py",
                 "moshi_simulate_offline_dlc.py","moshi_validate_temporal_calibration.py","moshi_compile_temporal_candidate.py",
                 "moshi_projection_layout.py"):
        shutil.copy2(Path(__file__).with_name(name),scripts/name)
    common=["--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir)]
    command=[sys.executable,str(scripts/"moshi_guard_qairt_converter.py"),*common,"--receipt",str(candidate/"conversion.json"),"--",
             "--input_network",str(source),"--output_path",str(candidate/"model.w4a16.dlc"),
             "--quantization_overrides",str(root/"quantization_overrides.json"),"--preserve_onnx_output_order","--onnx_skip_simplification",
             "--source_model_input_datatype","position","int32","--preserve_io_datatype",*[v.name for v in (*model.graph.input,*model.graph.output) if v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT]]
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),OPENBLAS_NUM_THREADS="4")
    with (root/"conversion.log").open("w") as log:
        run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        result.update(status="conversion_failed",returncode=run.returncode);write(root/"RESULT.json",result)
        raise SystemExit(run.returncode)
    sys.path[:0]=[str(args.numpy_dir),str(args.sdk_root/"lib/python")]
    from moshi_check_offline_dlc import audit
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.converters.common import encodings_json_serializer
    dlc=candidate/"model.w4a16.dlc"
    actual=audit(dlc)
    write(candidate/"precision_audit.json",actual)
    ops=[v for g in actual["graphs"] for v in g["projections"] if v["type"]==("Conv2d" if args.pointwise else "FullyConnected")]
    result["actual_precision_passed"]=len(ops)==8 and all(
        op["inputs"][1].get("compressed_bitwidth",op["inputs"][1]["bitwidths"][0])==current[op["inputs"][1]["name"]]
        and op["inputs"][0]["dtype"]=="uFxp_16" and op["outputs"][0]["dtype"]=="uFxp_16" for op in ops)
    norms=[op for g in actual["graphs"] for op in g["normalization_ops"]]
    result["normalization_precision_passed"]=len(norms)>=24 and all(
        dtype=="Float_32" for op in norms for dtype in (*op["input_dtypes"],*op["output_dtypes"]))
    result["actual_precision_passed"] &= result["normalization_precision_passed"]
    if not result["actual_precision_passed"]:
        result["status"]="precision_gate_failed";write(root/"RESULT.json",result);raise SystemExit(2)
    reader=modeltools.IrDlcReader();reader.open(str(dlc));graph=reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    target=candidate/"model.w4a16_encoding.json"
    serializer=encodings_json_serializer.IrEncodingsJsonSerializer(str(target),True);serializer.serialize(graph)
    target.write_text(serializer.get_graph_json())
    prior.update(status="quantized_not_verified",source_onnx=str(source),source_onnx_sha256=result["source_onnx_sha256"],
                 source_external_sha256=result["source_external_sha256"],output_dlc=str(dlc),output_sha256=sha256(dlc),output_bytes=dlc.stat().st_size,
                 method=f"AWQ-style scales and sensitivity-selected LPBQ W4/W8 A16; block {args.block_size}; pointwise {args.pointwise}",
                 expected_projection_bits=current,script_sha256=sha256(__file__),quantization_overrides=str(root/"quantization_overrides.json"))
    prior["reference_external_sha256"]=reference_external
    write(candidate/"quantization.json",prior)
    write(root/"RESULT.json",result)
    for script,leaf,extra in (("moshi_simulate_offline_dlc.py","cpu_simulation",common),
                              ("moshi_validate_temporal_calibration.py","calibration_validation",[])):
        with (root/(leaf+".log")).open("w") as log:
            run=subprocess.run([sys.executable,str(scripts/script),"--directory",str(candidate),*extra],env=env,stdout=log,stderr=subprocess.STDOUT)
        if not (candidate/(leaf+".json")).is_file():
            result.update(status=leaf+"_tool_failed",returncode=run.returncode,diagnostic_log=str(root/(leaf+".log")))
            write(root/"RESULT.json",result)
            raise SystemExit(run.returncode or 2)
        check=json.loads((candidate/(leaf+".json")).read_text())
        result[leaf+"_passed"]=check["passed"]
        if leaf=="cpu_simulation":result["cpu_screen_passed"]=check["passed"]
        result["status"]="cpu_screen_passed_not_device_verified" if run.returncode==0 else "rejected_by_"+leaf
        write(root/"RESULT.json",result)
        if run.returncode:raise SystemExit(run.returncode)
    with (root/"compile.log").open("w") as log:
        run=subprocess.run([sys.executable,str(scripts/"moshi_compile_temporal_candidate.py"),"--directory",str(root),"--sdk-root",str(args.sdk_root)],env=env,stdout=log,stderr=subprocess.STDOUT)
    print("LPBQ workflow finished",run.returncode,"seconds",time.monotonic()-started,flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)


if __name__=="__main__":
    main()
