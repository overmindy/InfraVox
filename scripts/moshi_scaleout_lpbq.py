"""Run independent Temporal LPBQ jobs with bounded GPU and disk use.

Reuse approved FP32 exports and captured calibration. Each shard has its own
directory and must pass precision, DLC-derived CPU simulation, and HTP compile.
No phone default is changed. Failed jobs retain all diagnostic artifacts.
"""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from moshi_quantize_temporal_offline import sha256, write


def run_shard(layer, gpu, args):
    from moshi_restore_float_source import restore
    import onnx
    graph = f"temporal_layers_{layer}_{layer+1}"
    root = args.output_root/graph
    if root.exists():
        receipt=root/"WORKFLOW.json"
        if receipt.exists():
            prior=json.loads(receipt.read_text())
            return {"graph":graph,"status":"reused_existing_job","existing_status":prior["status"],"workflow":str(receipt)}
        return {"graph":graph,"status":"already_claimed_during_preparation","directory":str(root)}
    root.mkdir()
    inventory = json.loads(args.inventory.read_text())["graphs"][graph]
    dlc = Path(inventory["source_dlc"])
    if sha256(dlc) != inventory["sha256"]:
        raise ValueError("Baseline DLC differs from selected inventory")
    source = Path(inventory["cpu_report"]).parent/"model.onnx"
    restored = root/"reference_source/model.onnx"
    restoration = restore(source, args.export_root/(graph+".onnx"), restored)
    ref = root/"reference"
    ref.mkdir()
    manifest = json.loads(args.calibration_manifest.read_text())
    samples = copy.deepcopy(manifest["graphs"][graph]["samples"])
    for sample in samples:
        sample["sha256"] = sha256(args.calibration_manifest.parent/sample["file"])
    prior = {"format":"moshi-restored-fp32-reference-v1", "graph":graph, "source_onnx":str(restored),
             "source_onnx_sha256":sha256(restored), "source_external_sha256":restoration["external_sha256"],
             "source_dlc":str(dlc), "source_dlc_sha256":sha256(dlc), "source_dlc_bytes":dlc.stat().st_size,
             "calibration_manifest":str(args.calibration_manifest), "calibration_manifest_sha256":sha256(args.calibration_manifest),
             "samples":samples, "sdk_root":str(args.sdk_root), "device_verified":False,
             "default_bundle_replaced":False, "cache_update":inventory["update"], "capacity":1000,
             "restoration_receipt":str(restored.parent/"restoration.json")}
    write(ref/"quantization.json", prior)
    # AWQ preparation reads only the reference provenance. The new policy below
    # explicitly retains all nonprojection floating parameters; baseline W8
    # ranges must not be reused as encodings of the restored FP32 graph.
    write(ref/"model.w4a16_encoding.json", {"version":"0.6.1", "activation_encodings":{}, "param_encodings":{}})
    receipt = {"graph":graph, "status":"running", "gpu":gpu, "stages":[], "default_bundle_replaced":False}
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),
               OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
    scripts = Path(__file__).parent

    def run(name, script, extra):
        command = [sys.executable,str(scripts/script),*map(str,extra)]
        started = time.monotonic()
        with (root/(name+".log")).open("w") as log:
            child = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt["stages"].append({"name":name, "command":command, "script_sha256":sha256(scripts/script),
                                  "returncode":child.returncode, "elapsed_s":time.monotonic()-started})
        receipt["status"] = "running" if not child.returncode else "failed_"+name
        write(root/"WORKFLOW.json",receipt)
        if child.returncode:
            raise RuntimeError(f"{graph} {name} exited {child.returncode}")
        print(graph,name,"passed",flush=True)

    write(root/"WORKFLOW.json",receipt)
    run("calibration", "moshi_calibrate_fc_boundaries.py", ["--source-onnx",restored,"--reference-directory",ref,
        "--output-dir",root/"calibration","--save-projections"])
    run("awq", "moshi_optimize_activation_aware.py", ["--source-onnx",restored,"--reference-directory",ref,
        "--calibration-dir",root/"calibration","--output-dir",root/"awq","--device",f"cuda:{gpu}","--prepare-only"])
    base = json.loads((root/"awq/mixed_policy.json").read_text())
    model = onnx.load(str(root/"awq/source/model.onnx"),load_external_data=False)
    projections = set(base["param_encodings"])
    float_value = [{"bitwidth":32,"dtype":"float"}]
    for tensor in model.graph.initializer:
        if tensor.data_type in (onnx.TensorProto.FLOAT,onnx.TensorProto.FLOAT16) and tensor.name not in projections:
            base["param_encodings"][tensor.name] = copy.deepcopy(float_value)
    for node in model.graph.node:
        if (node.name.startswith(("/norm1/","/norm1_1/","/norm2/","/norm2_1/")) or any(v.startswith("/norm") for v in node.output)) and not node.name.endswith("/Mul_1"):
            for name in node.output:
                base["activation_encodings"][name] = copy.deepcopy(float_value)
    (root/"base_policy").mkdir()
    write(root/"base_policy/quantization_overrides.json",base)
    run("lpbq", "moshi_build_lpbq_candidate.py", ["--awq-directory",root/"awq","--awq-candidate-directory",root/"base_policy",
        "--projection-calibration-dir",root/"calibration","--output-dir",root/"lpbq",
        "--sdk-root",args.sdk_root,"--numpy-dir",args.numpy_dir,"--pointwise","--first-passing-rounding","--device",f"cuda:{gpu}"])
    result = json.loads((root/"lpbq/RESULT.json").read_text())
    if not result.get("htp_offline_compile",{}).get("status") == "success":
        raise RuntimeError("No successful HTP context receipt")
    run("phone_reference", "moshi_prepare_phone_reference.py", ["--candidate-directory",root/"lpbq/candidate",
        "--reuse-simulation-directory",root/"lpbq/candidate","--output-dir",root/"lpbq/phone_reference",
        "--sdk-root",args.sdk_root,"--numpy-dir",args.numpy_dir])
    # Only delete reproducible derived weights after every offline gate passes.
    # Keep the final quantizer input, DLC, context, calibration, hashes and logs.
    removed = []
    for relative in ("calibration/projection_probe.onnx", "awq/source/model.data", "lpbq/candidate/cpu_simulation.data", "lpbq/phone_reference/cpu_simulation.data"):
        path = root/relative
        if path.is_file() and not path.is_symlink():
            record = {"file":relative,"bytes":path.stat().st_size,"sha256":sha256(path),
                      "reason":"Reproducible intermediate; final input and actual DLC retained"}
            path.unlink()
            removed.append(record)
    receipt.update(status="offline_passed_phone_pending", cleanup=removed,
                   w4_parameter_fraction=result["w4_parameter_fraction"],
                   selected_precision=result["selected_precision"],
                   context_bytes=result["htp_offline_compile"]["binary_bytes"])
    write(root/"WORKFLOW.json",receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output-root","inventory","export-root","calibration-manifest","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--layers",type=int,nargs="+",required=True)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--first-gpu",type=int,default=1)
    args = p.parse_args()
    sys.path.insert(0,str(args.numpy_dir))
    args.output_root.mkdir(parents=True,exist_ok=True)
    # Each lane keeps one GPU, avoiding overlapping covariance jobs on it.
    lanes = [args.layers[i::args.workers] for i in range(args.workers)]
    def lane(index, layers):
        rows = []
        for layer in layers:
            if shutil.disk_usage(args.output_root).free < 14*(1<<30):
                rows.append({"layer":layer,"status":"disk_budget_stop"})
                break
            try:
                rows.append(run_shard(layer,args.first_gpu+index,args))
            except Exception as error:
                rows.append({"layer":layer,"status":"failed","error":str(error)})
                print("FAILED",layer,str(error),flush=True)
        return rows
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(lane,i,layers) for i,layers in enumerate(lanes) if layers]
        for job in concurrent.futures.as_completed(jobs):
            rows.extend(job.result())
            write(args.output_root/f"BATCH_{os.getpid()}.json",{"jobs":rows,"default_bundle_replaced":False})


if __name__ == "__main__":
    main()
