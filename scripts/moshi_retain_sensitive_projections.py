"""Repair float tensor boundaries or retain original FCs after a phone failure.

Reuse all other selected LPBQ/W8 weights. Do not repeat the weight search.
Rebuild and check a separate candidate; never overwrite the failed experiment.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from moshi_quantize_temporal_offline import sha256,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("directory","output-dir","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--retain-node",action="append",default=[])
    p.add_argument("--float-tensor-prefix",action="append",default=[])
    p.add_argument("--stabilize-rmsnorm",action="store_true",help="Use a bounded square and restore denominator units before reciprocal")
    p.add_argument("--rms-scale-floor",type=float,default=1.0)
    args=p.parse_args()
    if not(args.retain_node or args.float_tensor_prefix or args.stabilize_rmsnorm):
        p.error("Select a float boundary repair, a retained projection, or stable RMSNorm")
    sys.path[:0]=[str(args.numpy_dir),str(args.sdk_root/"lib/python")]
    import numpy as np
    import onnx
    from onnx import numpy_helper
    parent=json.loads((args.directory/"RESULT.json").read_text())
    state=json.loads((args.directory/"candidate/quantization.json").read_text())
    source=Path(state["source_onnx"])
    reference=Path(state["reference_onnx"])
    if sha256(source)!=state["source_onnx_sha256"] or sha256(reference)!=state["reference_onnx_sha256"]:
        raise ValueError("Parent source or reference changed")
    for name,digest in state["source_external_sha256"].items():
        if sha256(source.parent/name)!=digest:
            raise ValueError("Parent external weights changed")
    model=onnx.load(str(source),load_external_data=bool(args.retain_node))
    original=onnx.load(str(reference)) if args.retain_node else None
    originals={v.name:v for v in original.graph.initializer} if original else {}
    original_nodes={v.name:v for v in original.graph.node} if original else {}
    tensors={v.name:v for v in model.graph.initializer}
    policy=json.loads((args.directory/"quantization_overrides.json").read_text())
    entries={v["name"]:v for v in policy["activation_encodings"]}
    params={v["name"]:v for v in policy["param_encodings"]}
    bits=copy.deepcopy(state["expected_projection_bits"])
    retained=[]
    remove=set()
    stabilized=[]
    if args.stabilize_rmsnorm:
        from moshi_stabilize_rmsnorm import stabilize
        stabilized=stabilize(model,args.rms_scale_floor)
        args.float_tensor_prefix.extend(["/norm","moshi_stable_rms_"])
    for node in model.graph.node:
        for value in node.output:
            if any(value.startswith(prefix) for prefix in args.float_tensor_prefix):
                entries[value]={"name":value,"bw":32,"dtype":"FLOAT"}
    for name in args.retain_node:
        node=original_nodes[name]
        weight=node.input[1]
        if node.op_type!="MatMul" or weight not in bits:
            raise ValueError("Retention target is not an original projection")
        bits[weight]=32
        tensors[weight].CopyFrom(originals[weight])
        for suffix in ("/to_pointwise","/from_pointwise"):
            remove.add(name+suffix)
        remove.add(weight+"_awq_scale")
        for value in (node.input[0],node.output[0]):
            entries[value]={"name":value,"bw":32,"dtype":"FLOAT"}
        entries.pop(weight+"_pointwise_input",None)
        entries.pop(weight+"_pointwise_output",None)
        entries.pop(weight+"_awq_input",None)
        params[weight]={"name":weight,"bw":32,"dtype":"FLOAT"}
        retained.append({"node":name,"weight":weight,"shape":list(originals[weight].dims),"reason":"HTP precision failure; restore the baseline float FC path"})
    nodes=[]
    for node in model.graph.node:
        if node.name in remove:
            continue
        nodes.append(copy.deepcopy(original_nodes[node.name]) if node.name in args.retain_node else node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used={name for n in nodes for name in n.input}
    keep=[copy.deepcopy(t) for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    policy["activation_encodings"]=list(entries.values())
    policy["param_encodings"]=[v for name,v in params.items() if name in used]
    root=args.output_dir
    root.mkdir(parents=True)
    (root/"source").mkdir()
    path=root/"source/model.onnx"
    if args.retain_node:
        onnx.save_model(model,str(path),save_as_external_data=True,all_tensors_to_one_file=True,location="model.data",size_threshold=1024)
    else:
        # A precision or stable-math repair does not alter projection weights.
        # Keep their original external storage and save only the new graph.
        for name in state["source_external_sha256"]:
            (path.parent/name).symlink_to((source.parent/name).resolve())
        onnx.save_model(model,str(path))
    write(root/"quantization_overrides.json",policy)
    candidate=root/"candidate";candidate.mkdir()
    dlc=candidate/"model.w4a16.dlc"
    scripts=root/"runtime_scripts";scripts.mkdir()
    for name in ("moshi_retain_sensitive_projections.py","moshi_guard_qairt_converter.py","moshi_quantize_temporal_offline.py",
                 "moshi_check_offline_dlc.py","moshi_simulate_offline_dlc.py","moshi_validate_temporal_calibration.py",
                 "moshi_compile_temporal_candidate.py","moshi_resume_offline_checks.py","moshi_stabilize_rmsnorm.py"):
        shutil.copy2(Path(__file__).with_name(name),scripts/name)
    common=["--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir)]
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),OPENBLAS_NUM_THREADS="4")
    command=[sys.executable,str(scripts/"moshi_guard_qairt_converter.py"),*common,"--receipt",str(candidate/"conversion.json"),"--",
             "--input_network",str(path),"--output_path",str(dlc),"--quantization_overrides",str(root/"quantization_overrides.json"),
             "--preserve_onnx_output_order","--onnx_skip_simplification","--source_model_input_datatype","position","int32",
             "--preserve_io_datatype",*[v.name for v in (*model.graph.input,*model.graph.output) if v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT]]
    with (root/"conversion.log").open("w") as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from moshi_check_offline_dlc import audit
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.converters.common import encodings_json_serializer
    actual=audit(dlc)
    write(candidate/"precision_audit.json",actual)
    ops=[op for g in actual["graphs"] for op in g["projections"] if op["type"] in ("FullyConnected","Conv2d")]
    passed=len(ops)==8 and all((op["inputs"][1]["dtype"]=="Float_32" if bits[op["inputs"][1]["name"]]==32 else
            op["inputs"][1].get("compressed_bitwidth",op["inputs"][1]["bitwidths"][0])==bits[op["inputs"][1]["name"]]) for op in ops)
    if not passed:
        raise ValueError("Actual mixed float/integer projection audit failed")
    norms=[op for g in actual["graphs"]for op in g["normalization_ops"]]
    if len(norms)<24 or any(dtype!="Float_32" for op in norms for dtype in (*op["input_dtypes"],*op["output_dtypes"])):
        raise ValueError("RMSNorm contains a lower-precision arithmetic boundary")
    reader=modeltools.IrDlcReader();reader.open(str(dlc));graph=reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    encoding=candidate/"model.w4a16_encoding.json"
    serializer=encodings_json_serializer.IrEncodingsJsonSerializer(str(encoding),True);serializer.serialize(graph)
    encoding.write_text(serializer.get_graph_json())
    state.update(source_onnx=str(path),source_onnx_sha256=sha256(path),source_external_sha256={"model.data":sha256(path.parent/"model.data")},
                 output_dlc=str(dlc),output_sha256=sha256(dlc),output_bytes=dlc.stat().st_size,expected_projection_bits=bits,
                 status="quantized_not_verified",retained_float_projections=retained)
    write(candidate/"quantization.json",state)
    result=copy.deepcopy(parent)
    for field in ("htp_offline_compile","cpu_simulation_passed","cpu_screen_passed","calibration_validation_passed"):
        result.pop(field,None)
    for row in result["layers"]:
        row["bitwidth"]=bits[row["name"]]
    result.update(status="precision_passed_new_float_retention",actual_precision_passed=True,
                  expected_projection_bits=bits,retained_float_projections=retained,
                  selected_precision={str(b):sum(v==b for v in bits.values())for b in (4,8,32)},
                  w4_parameter_fraction=sum(v["parameter_count"]for v in result["layers"]if v["bitwidth"]==4)/sum(v["parameter_count"]for v in result["layers"]),
                  source_onnx_sha256=state["source_onnx_sha256"],source_external_sha256=state["source_external_sha256"],
                  parent_candidate_sha256=parent["htp_offline_compile"]["candidate_dlc_sha256"],device_verified=False,default_bundle_replaced=False)
    result["float_tensor_prefix_repairs"]=args.float_tensor_prefix
    result["stable_rmsnorm_rewrites"]=stabilized
    write(root/"RESULT.json",result)
    subprocess.run([sys.executable,str(scripts/"moshi_resume_offline_checks.py"),"--directory",str(root),*common],env=env,check=True)


if __name__=="__main__":
    main()
