"""Resume missing offline checks without repeating quantization or passed checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from moshi_quantize_temporal_offline import sha256,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("directory","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    root=args.directory
    scripts=root/"runtime_scripts"
    result=json.loads((root/"RESULT.json").read_text())
    if not result["actual_precision_passed"]:
        raise ValueError("Require a passed actual-precision gate")
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),OPENBLAS_NUM_THREADS="4")
    common=["--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir)]
    result["offline_resume_script_sha256"]=sha256(__file__)
    result["resumed_simulator_sha256"]=sha256(scripts/"moshi_simulate_offline_dlc.py")
    for name,extra in (("cpu_simulation",common),("calibration_validation",[])):
        receipt=root/"candidate"/(name+".json")
        if receipt.exists():
            if not json.loads(receipt.read_text())["passed"]:
                raise ValueError("Do not repeat a failed numerical checkpoint")
        else:
            script="moshi_simulate_offline_dlc.py" if name=="cpu_simulation" else "moshi_validate_temporal_calibration.py"
            result["status"]="running_"+name
            write(root/"RESULT.json",result)
            log_path=root/(name+".log")
            if log_path.exists():
                log_path.rename(root/(name+".before_resume.log"))
            with log_path.open("w") as log:
                run=subprocess.run([sys.executable,str(scripts/script),"--directory",str(root/"candidate"),*extra],env=env,stdout=log,stderr=subprocess.STDOUT)
            if run.returncode or not receipt.exists():
                result.update(status="rejected_or_failed_"+name,returncode=run.returncode)
                write(root/"RESULT.json",result)
                raise SystemExit(run.returncode or 2)
        result[name+"_passed"]=True
        if name=="cpu_simulation":result["cpu_screen_passed"]=True
        write(root/"RESULT.json",result)
    if not (root/"htp_offline/compile.json").exists():
        with (root/"compile.log").open("w") as log:
            run=subprocess.run([sys.executable,str(scripts/"moshi_compile_temporal_candidate.py"),"--directory",str(root),"--sdk-root",str(args.sdk_root)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:raise SystemExit(run.returncode)
    elif json.loads((root/"htp_offline/compile.json").read_text())["status"]!="success":
        raise ValueError("Existing HTP compile failed. Inspect it before a changed-model retry")
    print(json.loads((root/"RESULT.json").read_text())["status"],flush=True)


if __name__=="__main__":
    main()
