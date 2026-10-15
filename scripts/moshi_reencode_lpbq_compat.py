"""Expand the two 11264-input LPBQ projections to equivalent W8 for HTP compatibility.

Keep the remaining LPBQ projections. Expansion multiplies each signed W4 code
by its integer block factor. The existing per-channel float scale is unchanged.
This changes storage representation and does not fit or search new weights.
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


def verify_equivalent_expansion(parent_dlc,candidate_dlc,names):
    """Require exact integer-code and channel-scale equivalence for both weights."""
    import numpy as np
    from qti.aisw.dlc_utils import modeltools
    readers=[]
    graphs=[]
    for path in (parent_dlc,candidate_dlc):
        reader=modeltools.IrDlcReader();reader.open(str(path));readers.append(reader)
        graphs.append(reader.get_ir_graph(next(iter(reader.get_ir_graph_names()))))
    tensors=[{op.inputs()[1].name():op.inputs()[1] for op in g.get_ops() if op.type=="FullyConnected"} for g in graphs]
    rows=[]
    for name in names:
        old,new=tensors[0][name],tensors[1][name]
        info=old.get_encoding().lpbqEncInfo
        new_info=new.get_encoding().axisEncInfo
        raw=np.asarray(old.get_data()).astype(np.int16)
        signed=np.where(raw>7,raw-16,raw)
        expansion=np.asarray(info.blocksScale8 or info.blocksScale16,np.int16).reshape(len(info.encInfos),info.numBlocksPerAxis)
        expected=signed*np.repeat(expansion,info.blockSize,axis=1)
        actual=np.asarray(new.get_data()).astype(np.int16)
        actual=np.where(actual>127,actual-256,actual)
        mismatches=int(np.count_nonzero(expected!=actual))
        same_scales=bool(np.array_equal(np.asarray([v.scale for v in info.encInfos],np.float32),
                                        np.asarray([v.scale for v in new_info.encInfos],np.float32)))
        if mismatches or not same_scales or any(v.bw!=8 or v.offset!=0 for v in new_info.encInfos):
            raise ValueError("LPBQ-to-W8 expansion changed the encoded function")
        rows.append({"name":name,"weight_count":int(expected.size),"code_mismatches":mismatches,
                     "channel_scales_identical":same_scales,"passed":True})
    return {"passed":True,"parent_dlc_sha256":sha256(parent_dlc),"candidate_dlc_sha256":sha256(candidate_dlc),"weights":rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input-directory","output-directory","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    sys.path[:0]=[str(args.numpy_dir),str(args.sdk_root/"lib/python")]
    old,root=args.input_directory,args.output_directory
    validation=json.loads((old/"candidate/calibration_validation.json").read_text())
    if not validation["passed"]:
        raise ValueError("Require passed source-candidate numerical checks")
    prior=json.loads((old/"candidate/quantization.json").read_text())
    result=json.loads((old/"RESULT.json").read_text())
    overrides=json.loads((old/"quantization_overrides.json").read_text())
    selected={v["name"] for v in result["layers"] if v["shape"][0]==11264 and v["bitwidth"]==4}
    if len(selected)!=2:
        raise ValueError("Expected exactly the two FFN down projections")
    root.mkdir()
    source=root/"source/model.onnx"
    source.parent.mkdir()
    if sha256(prior["source_onnx"])!=prior["source_onnx_sha256"]:
        raise ValueError("Source graph changed")
    os.link(prior["source_onnx"],source)
    for name,digest in prior["source_external_sha256"].items():
        old_file=Path(prior["source_onnx"]).parent/name
        if sha256(old_file)!=digest:
            raise ValueError("Source weights changed")
        os.link(old_file,source.parent/name)
    for encoding in overrides["param_encodings"]:
        if encoding["name"] in selected:
            if encoding["enc_type"]!="LPBQ" or encoding["bw"]!=8 or encoding["compressed_bw"]!=4:
                raise ValueError("Unexpected source LPBQ representation")
            encoding["enc_type"]="PER_CHANNEL"
            for key in ("compressed_bw","per_block_int_scale","block_size"):
                del encoding[key]
    write(root/"quantization_overrides.json",overrides)
    previous_compile=result.pop("htp_offline_compile",None)
    result.update(status="reencoding_lpbq_compatibility",parent_directory=str(old),parent_dlc_sha256=prior["output_sha256"],
                  previous_compile_failure=previous_compile,compatibility_script_sha256=sha256(__file__),
                  compatibility_expanded_weights=sorted(selected))
    for row in result["layers"]:
        if row["name"] in selected:
            row.update(bitwidth=8,rounding_method="equivalent_LPBQ_expansion_to_W8")
            result["expected_projection_bits"][row["name"]]=8
    result["selected_precision"]={str(bits):sum(v==bits for v in result["expected_projection_bits"].values()) for bits in (4,8)}
    result["w4_parameter_fraction"]=sum(v["parameter_count"] for v in result["layers"] if v["bitwidth"]==4)/sum(v["parameter_count"] for v in result["layers"])
    write(root/"RESULT.json",result)
    scripts=root/"runtime_scripts";scripts.mkdir()
    for name in ("moshi_reencode_lpbq_compat.py","moshi_quantize_temporal_offline.py","moshi_guard_qairt_converter.py",
                 "moshi_check_offline_dlc.py","moshi_simulate_offline_dlc.py","moshi_validate_temporal_calibration.py",
                 "moshi_compile_temporal_candidate.py","moshi_resume_offline_checks.py"):
        shutil.copy2(Path(__file__).with_name(name),scripts/name)
    candidate=root/"candidate";candidate.mkdir()
    command=json.loads((old/"candidate/conversion.json").read_text())["converter_args"]
    command[command.index("--input_network")+1]=str(source)
    command[command.index("--output_path")+1]=str(candidate/"model.w4a16.dlc")
    command[command.index("--quantization_overrides")+1]=str(root/"quantization_overrides.json")
    converter=[sys.executable,str(scripts/"moshi_guard_qairt_converter.py"),"--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir),
               "--receipt",str(candidate/"conversion.json"),"--",*command]
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),OPENBLAS_NUM_THREADS="4")
    with (root/"conversion.log").open("w") as log:
        run=subprocess.run(converter,env=env,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        result.update(status="compatibility_conversion_failed",returncode=run.returncode);write(root/"RESULT.json",result)
        raise SystemExit(run.returncode)
    from moshi_check_offline_dlc import audit
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.converters.common import encodings_json_serializer
    dlc=candidate/"model.w4a16.dlc"
    actual=audit(dlc);write(candidate/"precision_audit.json",actual)
    ops=[op for g in actual["graphs"] for op in g["projections"] if op["type"]=="FullyConnected"]
    result["actual_precision_passed"]=len(ops)==8 and all(
        op["inputs"][1].get("compressed_bitwidth",op["inputs"][1]["bitwidths"][0])==result["expected_projection_bits"][op["inputs"][1]["name"]]
        and op["inputs"][0]["dtype"]=="uFxp_16" and op["outputs"][0]["dtype"]=="uFxp_16" for op in ops)
    if not result["actual_precision_passed"]:
        result["status"]="precision_gate_failed";write(root/"RESULT.json",result);raise SystemExit(2)
    reader=modeltools.IrDlcReader();reader.open(str(dlc));graph=reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    target=candidate/"model.w4a16_encoding.json"
    serializer=encodings_json_serializer.IrEncodingsJsonSerializer(str(target),True);serializer.serialize(graph)
    target.write_text(serializer.get_graph_json())
    write(candidate/"equivalent_expansion.json",verify_equivalent_expansion(Path(prior["output_dlc"]),dlc,selected))
    prior.update(source_onnx=str(source),output_dlc=str(dlc),output_sha256=sha256(dlc),output_bytes=dlc.stat().st_size,
                 expected_projection_bits=result["expected_projection_bits"],
                 method="LPBQ W4/A16 with two equivalent per-channel W8 expansions for HTP compatibility",
                 quantization_overrides=str(root/"quantization_overrides.json"),source_dlc_role="Original FP32 baseline reference")
    write(candidate/"quantization.json",prior)
    result["status"]="compatibility_numerical_checks"
    write(root/"RESULT.json",result)
    with (root/"workflow.log").open("w") as log:
        run=subprocess.run([sys.executable,str(scripts/"moshi_resume_offline_checks.py"),"--directory",str(root),
                            "--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir)],env=env,stdout=log,stderr=subprocess.STDOUT)
    print("Compatibility workflow finished",run.returncode,flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)


if __name__=="__main__":
    main()
