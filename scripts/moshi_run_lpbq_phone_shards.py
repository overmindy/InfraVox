"""Run one three-frame HTP gate per exported LPBQ shard, outside app defaults."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tempfile

from moshi_quantize_temporal_offline import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--adb",type=Path,required=True)
    p.add_argument("--serial",required=True)
    p.add_argument("--phone-root",required=True)
    p.add_argument("--layers",nargs="+",type=int,required=True)
    p.add_argument("--context-only",action="store_true",help="Upload compiled contexts only; reject any DLC fallback")
    args=p.parse_args()
    lock=(Path(tempfile.gettempdir())/("moshi-adb-"+args.serial+".lock")).open("w")
    fcntl.flock(lock,fcntl.LOCK_EX)
    adb=[str(args.adb),"-s",args.serial]
    commands=[]
    def run(*arguments):
        t=time.monotonic()
        r=subprocess.run(adb+list(arguments),capture_output=True,text=True)
        commands.append({"arguments":arguments,"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr,"elapsed_s":time.monotonic()-t})
        if r.returncode:
            raise RuntimeError(r.stderr+r.stdout)
        return r.stdout.strip()
    def shell(command):
        return run("shell",command)
    base="/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models/lm-cache1000-v1"
    cache="/sdcard/Android/data/com.overmindy.mimicodecdemo/files/qnn-context-v1/cache1000-v1"
    apk=run("shell","pm path com.overmindy.mimicodecdemo").split("package:",1)[1]
    lib=str(Path(apk).parent/"lib/arm64")
    for layer in args.layers:
        graph=f"temporal_layers_{layer}_{layer+1}"
        candidate=args.directory/graph
        local=args.directory/"phone"/graph
        local.mkdir(parents=True)
        single=local/"single"
        single.mkdir()
        remote=args.phone_root+"/"+graph
        shell(f"test ! -e {remote} && mkdir -p {remote}/single")
        result=json.loads((candidate/"RESULT.json").read_text())
        if result["htp_offline_compile"]["status"] != "success" or not result["calibration_validation_passed"]:
            raise ValueError("Offline gates have not passed")
        dlc=candidate/"candidate/model.w4a16.dlc"
        context=next((candidate/"htp_offline").glob("*.bin"))
        hidden=candidate/"phone_reference/hidden_recurrent.raw"
        reference=json.loads((candidate/"phone_reference/phone_reference.json").read_text())
        metadata=json.loads((candidate/"candidate/quantization.json").read_text())
        actual_dlc_hash=metadata["output_sha256"] if args.context_only else sha256(dlc)
        if actual_dlc_hash != reference["candidate_dlc_sha256"] or actual_dlc_hash != result["htp_offline_compile"]["candidate_dlc_sha256"] or sha256(context) != result["htp_offline_compile"]["binary_sha256"] or sha256(hidden) != reference["hidden_sha256"]:
            raise ValueError("Exported files differ from validated hashes")
        uploads=[(context,"context.bin"),(hidden,"hidden.raw")]
        if not args.context_only:
            uploads.append((dlc,"model.dlc"))
        for path,name in uploads:
            run("push",str(path),remote+"/"+name)
            if shell(f"sha256sum {remote}/{name}").split()[0] != sha256(path):
                raise ValueError("Phone upload changed")
        baseline_context=shell(f"ls {cache}/{graph}-*.bin").splitlines()
        if len(baseline_context)!=1:
            raise ValueError("Expected a unique current baseline context")
        candidate_model=base+"/dlc/"+graph+".dlc" if args.context_only else remote+"/model.dlc"
        for variant,model,binary in (("baseline",base+"/dlc/"+graph+".dlc",baseline_context[0]),("candidate",candidate_model,remote+"/context.bin")):
            command=["env","MOSHI_BENCH_DUMP=1","MOSHI_BENCH_RECURRENT=1","MOSHI_HTP_PERFORMANCE=1",
                     "LD_LIBRARY_PATH="+lib,"ADSP_LIBRARY_PATH="+lib,"/data/local/tmp/moshi_qnn_benchmark",
                     "--benchmark-temporal",lib,model,binary,remote+"/hidden.raw",remote+"/single/"+variant+".csv","3"]
            shell(shlex.join(command)+f" > {remote}/single/{variant}.log 2>&1")
        run("pull",remote+"/single/.",str(single))
        phases=list(csv.DictReader((single/"candidate.csv").open()))
        if sum(row["phase"]=="restore_context" for row in phases)!=1 or any(row["phase"] in ("open_dlc","compose","finalize") for row in phases):
            raise ValueError("Require candidate binary restoration without any DLC fallback")
        if shell(f"sha256sum {remote}/context.bin").split()[0] != sha256(context):
            raise ValueError("Phone context changed during the probe")
        comparison=[sys.executable,str(Path(__file__).with_name("moshi_compare_phone_reference.py")),"--directory",str(local),
                    "--reference-directory",str(candidate/"phone_reference")]
        child=subprocess.run(comparison,capture_output=True,text=True)
        report=json.loads((local/"single_comparison.json").read_text())
        report["primary_candidate_10_percent_gate_passed"]=report["max_errors"]["candidate_vs_original"]<=0.10 and all(
            v["finite"] and v["nonzero"] for row in report["frames"] for v in row["outputs"].values())
        report.update(comparison_returncode=child.returncode,phone_root=remote,commands=commands.copy(),
                      context_only_phone_deployment=args.context_only, candidate_context_sha256=sha256(context),
                      candidate_cpu_alignment_2_percent_passed=report["max_errors"]["candidate_vs_dlc_cpu"]<=0.02,
                      strict_alignment_diagnostic_passed=report["passed"])
        (local/"PHONE_GATE.json").write_text(json.dumps(report,indent=2)+"\n")
        print(graph,report["max_errors"],"primary passed",report["primary_candidate_10_percent_gate_passed"],flush=True)
        if not report["primary_candidate_10_percent_gate_passed"]:
            raise SystemExit(2)


if __name__=="__main__":
    main()
