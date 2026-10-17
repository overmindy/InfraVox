"""Apply the verified stable RMS rewrite to a list of existing LPBQ candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","sdk-root","numpy-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--layers",type=int,nargs="+",required=True)
    args=p.parse_args()
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.sdk_root/"lib/x86_64-linux-clang"),OPENBLAS_NUM_THREADS="4")
    scripts=Path(__file__).parent
    common=["--sdk-root",str(args.sdk_root),"--numpy-dir",str(args.numpy_dir)]
    def run(layer):
        graph=f"temporal_layers_{layer}_{layer+1}"
        out=args.root/"stable_rms"/graph/"lpbq"
        command=[sys.executable,str(scripts/"moshi_retain_sensitive_projections.py"),"--directory",str(args.root/"expanded"/graph/"lpbq"),
                 "--output-dir",str(out),"--stabilize-rmsnorm",*common]
        if layer==0:command += ["--rms-scale-floor","0.0009765625"]
        out.parent.mkdir(parents=True,exist_ok=True)
        with(out.parent/"repair.log").open("w")as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        print(layer,"offline",r.returncode,flush=True)
        if r.returncode:return
        command=[sys.executable,str(scripts/"moshi_prepare_phone_reference.py"),"--candidate-directory",str(out/"candidate"),
                 "--reuse-simulation-directory",str(out/"candidate"),"--output-dir",str(out/"phone_reference"),*common]
        with(out.parent/"reference.log").open("w")as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        print(layer,"reference",r.returncode,flush=True)
        if not r.returncode:
            (out/"candidate/cpu_simulation.data").unlink()
            (out/"phone_reference/cpu_simulation.data").unlink()
    with ThreadPoolExecutor(min(4,len(args.layers)))as pool:list(pool.map(run,args.layers))


if __name__=="__main__":
    main()
