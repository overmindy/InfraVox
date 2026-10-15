"""Check a small LPBQ W4/A16 FullyConnected conversion and HTP v81 compile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from moshi_quantize_temporal_offline import sha256, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--input-width",type=int,default=128)
    p.add_argument("--output-width",type=int,default=256)
    p.add_argument("--input-rank",type=int,choices=(2,3),default=2)
    p.add_argument("--block-size",type=int,default=32)
    args=p.parse_args()
    root=args.output_dir
    root.mkdir()
    sys.path[:0]=[str(args.numpy_dir),str(args.sdk_root/"lib/python")]
    import numpy as np
    import onnx
    from onnx import helper,numpy_helper
    rng=np.random.default_rng(7341)
    ni,no,bs=args.input_width,args.output_width,args.block_size
    if ni%bs:
        raise ValueError("Input width must be divisible by block size")
    weight=rng.normal(0,0.1,(no,ni)).astype(np.float32)
    blocks=weight.reshape(no,ni//bs,bs)
    scale=np.maximum(-blocks.min(axis=2)/8,blocks.max(axis=2)/7)
    per_channel=np.maximum(scale.max(axis=1)/16,1e-12)
    expansion=np.clip(np.rint(scale/per_channel[:,None]),1,16).astype(int)
    model=helper.make_model(helper.make_graph([helper.make_node("MatMul",["x","weight"],["y"],name="projection")],
                            "lpbq_probe",[helper.make_tensor_value_info("x",onnx.TensorProto.FLOAT,[1]*(args.input_rank-1)+[ni])],
                            [helper.make_tensor_value_info("y",onnx.TensorProto.FLOAT,[1]*(args.input_rank-1)+[no])],
                            [numpy_helper.from_array(weight.T.copy(),"weight")]),opset_imports=[helper.make_opsetid("",17)])
    model.ir_version=9
    onnx.save_model(model,str(root/"model.onnx"))
    overrides={"version":"1.0.0","activation_encodings":[{
        "name":name,"dtype":"INT","enc_type":"PER_TENSOR","bw":16,"is_sym":False,
        "scale":[0.001],"offset":[-32768]} for name in ("x","y")],
        "param_encodings":[{"name":"weight","dtype":"INT","enc_type":"LPBQ","bw":8,
                            "compressed_bw":4,"is_sym":True,"scale":per_channel.tolist(),"offset":[0]*no,
                            "block_size":bs,"per_block_int_scale":expansion.flatten().tolist()}]}
    write(root/"overrides.json",overrides)
    converter=[sys.executable,str(Path(__file__).with_name("moshi_guard_qairt_converter.py")),
               "--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir),"--receipt",str(root/"conversion.json"),
               "--","--input_network",str(root/"model.onnx"),"--output_path",str(root/"model.dlc"),
               "--quantization_overrides",str(root/"overrides.json"),"--preserve_io_datatype","x","y",
               "--onnx_skip_simplification"]
    lib=args.sdk_root/"lib/x86_64-linux-clang"
    env=dict(os.environ,LD_LIBRARY_PATH=str(lib))
    result={"format":"moshi-lpbq-support-v1","device_verified":False,"conversion_command":converter,
            "input_width":ni,"output_width":no,"input_rank":args.input_rank,"block_size":bs}
    with (root/"conversion.log").open("w") as log:
        run=subprocess.run(converter,env=env,stdout=log,stderr=subprocess.STDOUT)
    result["conversion_returncode"]=run.returncode
    if run.returncode:
        result["status"]="conversion_rejected"
        write(root/"RESULT.json",result)
        raise SystemExit(run.returncode)
    from qti.aisw.dlc_utils import modeltools
    reader=modeltools.IrDlcReader();reader.open(str(root/"model.dlc"))
    graph=reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    ops=[op for op in graph.get_ops() if op.type=="FullyConnected"]
    if len(ops)!=1:
        raise ValueError("Expected one FullyConnected")
    op=ops[0]
    result["actual_precision"]={"input":op.inputs()[0].data_type_string(),"output":op.outputs()[0].data_type_string(),
                                "weight_dtype":op.inputs()[1].data_type_string(),"weight_encoding":str(op.inputs()[1].get_encoding().type)}
    write(root/"backend.json",{"devices":[{"dsp_arch":"v81","soc_model":87}]})
    write(root/"config.json",{"backend_extensions":{"shared_library_path":str(lib/"libQnnHtpNetRunExtensions.so"),
                                                     "config_file_path":str(root/"backend.json")}})
    command=[str(args.sdk_root/"bin/x86_64-linux-clang/qnn-context-binary-generator"),"--backend",str(lib/"libQnnHtp.so"),
             "--model",str(lib/"libQnnModelDlc.so"),"--dlc_path",str(root/"model.dlc"),"--config_file",str(root/"config.json"),
             "--binary_file","lpbq_probe","--output_dir",str(root),"--log_level","warn"]
    with (root/"compile.log").open("w") as log:
        run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    result.update(compile_command=command,compile_returncode=run.returncode,
                  status="htp_offline_compile_passed_not_device_verified" if run.returncode==0 else "htp_offline_compile_rejected")
    if run.returncode==0:
        result.update(binary_bytes=(root/"lpbq_probe.bin").stat().st_size,binary_sha256=sha256(root/"lpbq_probe.bin"))
    write(root/"RESULT.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
