"""Run a paired phone chain from isolated, hash-verified compiled contexts.

The manifest names the actual compiled candidate contexts. Existing DLC paths
provide runtime graph labels and cache keys. Any DLC compose/finalize fallback
invalidates the run. This does not install an app bundle or change its defaults.
"""
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

from moshi_quantize_temporal_offline import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--frames",type=int,choices=(4,38),required=True)
    p.add_argument("--phone-root",required=True)
    p.add_argument("--adb",required=True)
    p.add_argument("--serial",required=True)
    args=p.parse_args()
    lock=(Path(tempfile.gettempdir())/("moshi-adb-"+args.serial+".lock")).open("w")
    fcntl.flock(lock,fcntl.LOCK_EX)
    manifest=json.loads(args.manifest.read_text())
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=True)
    adb=[args.adb,"-s",args.serial]
    commands=[]
    def run(*arguments):
        r=subprocess.run(adb+list(arguments),capture_output=True,text=True)
        commands.append({"args":arguments,"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr})
        if r.returncode:
            raise RuntimeError(r.stdout+r.stderr)
        return r.stdout.strip()
    def shell(command):
        return run("shell",command)
    base=manifest["baseline_root"]
    cache=manifest["baseline_cache"]
    remote=args.phone_root
    setup=root/"phone_setup.json"
    if not setup.exists():
        shell(f"test ! -e {remote} && mkdir -p {remote}/bundle/dlc {remote}/cache_baseline {remote}/cache_candidate {remote}/results")
        for source in shell(f"ls {base}/dlc/*.dlc").splitlines():
            shell(f"ln -s {source} {remote}/bundle/dlc/{Path(source).name}")
        actual={}
        for graph,item in manifest["candidates"].items():
            digest=shell("sha256sum "+shlex.quote(item["context"])).split()[0]
            if digest!=item["context_sha256"] or not item["primary_single_gate_passed"]:
                raise ValueError("Candidate context or phone single gate differs")
            actual[graph]=digest
        caches=shell(f"ls {cache}/*.bin").splitlines()
        mapped=set()
        for path in caches:
            name=Path(path).name
            graph=name.split("-",1)[0]
            shell(f"ln -s {path} {remote}/cache_baseline/{name}")
            source=manifest["candidates"][graph]["context"] if graph in manifest["candidates"] else path
            shell(f"ln -s {source} {remote}/cache_candidate/{name}")
            if graph in manifest["candidates"]:
                mapped.add(graph)
        if mapped!=set(manifest["candidates"]):
            raise ValueError("Some candidate contexts lack a runtime cache key")
        apk=run("shell","pm path com.overmindy.mimicodecdemo").split("package:",1)[1]
        write(setup,{"phone_root":remote,"library_directory":str(Path(apk).parent/"lib/arm64"),
                     "candidate_context_sha256":actual,"manifest":manifest,"commands":commands,
                     "default_inventory_before":shell(f"sha256sum {base}/dlc_inventory.json")})
    state=json.loads(setup.read_text())
    if state["manifest"]!=manifest:
        raise ValueError("Do not change an existing experiment manifest")
    if args.frames==38:
        for variant in ("baseline","candidate"):
            check=json.loads((root/f"f4/{variant}/analysis.json").read_text())
            if not check["first_warm_wav_byte_equal"] or any(v["audio"]["samples"]!=5760 for v in check["runs"].values()):
                raise ValueError("Four-frame gate has not passed")
    lib=state["library_directory"]
    for variant in ("baseline","candidate"):
        out=root/f"f{args.frames}"/variant
        out.mkdir(parents=True)
        prefix=f"{variant}_f{args.frames}"
        target=remote+"/results/"+prefix
        model=base if variant=="baseline" else remote+"/bundle"
        command=["env","LD_LIBRARY_PATH="+lib,"ADSP_LIBRARY_PATH="+lib,"/data/local/tmp/moshi_reply_latency_probe",lib,
                 base+"/mimi_encoder.dlc",base+"/mimi_decoder.dlc",model,remote+"/cache_"+variant,
                 manifest["input_wav"],target,str(args.frames)]
        script=f"test ! -e {target}.latency.json || exit 2; dumpsys battery > {target}.battery_before.txt; "+shlex.join(command)+f" > {target}.log 2>&1"
        write(out/"command.json",{"command":command,"shell":script,"context_only_candidate":True})
        started=time.monotonic()
        child=subprocess.Popen(adb+["shell",script],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        memory_captured=False
        with (out/(prefix+".memory.txt")).open("w") as memory:
            while child.poll() is None:
                sample=run("shell","task_pid=$(pidof moshi_reply_latency_probe); echo SAMPLE $(cat /proc/uptime); if test -n \"$task_pid\"; then grep -E '^(VmRSS|VmHWM|RssAnon|RssFile):' /proc/$task_pid/status; fi; grep '^MemAvailable:' /proc/meminfo")
                memory.write(sample+"\n");memory.flush()
                # Use the short gate for one owned DMA snapshot. Keep the long
                # paired timing run free of DMA/SMAPS inspections.
                if args.frames==4 and not memory_captured:
                    ready=run("shell",f"if grep -q 'LATENCY_RESULT run=first' {target}.log; then pidof moshi_reply_latency_probe; fi")
                    if ready:
                        write(out/"memory_snapshot.json",{"pid":ready,"phase":"after first complete chain; warm may be executing",
                             "dmabuf":run("shell","dmabuf_dump "+ready),
                             "smaps_rollup":run("shell","cat /proc/"+ready+"/smaps_rollup")})
                        memory_captured=True
                time.sleep(0.5)
            stdout,stderr=child.communicate()
        write(out/"driver.json",{"returncode":child.returncode,"elapsed_s":time.monotonic()-started,"stdout":stdout,"stderr":stderr})
        shell(f"dumpsys battery > {target}.battery_after.txt")
        for path in shell(f"ls {target}.*").splitlines():
            run("pull",path,str(out/Path(path).name))
        if child.returncode:
            raise RuntimeError((out/(prefix+".log")).read_text()[-4000:])
        for mode in ("first","warm"):
            rows=list(csv.DictReader((out/f"{prefix}.{mode}.wav.performance.csv").open()))
            if any(v["phase"] in ("open_dlc","compose","finalize") for v in rows):
                raise ValueError("DLC fallback invalidates the context-only experiment")
            if mode=="first" and variant=="candidate":
                for graph in manifest["candidates"]:
                    if not any(v["phase"]=="restore_context" and v["graph"]==graph+".dlc" for v in rows):
                        raise ValueError("Candidate graph was not restored")
        subprocess.run([sys.executable,str(Path(__file__).with_name("moshi_analyze_lpbq_phone_chain.py")),"--directory",str(out),
                        "--prefix",prefix,"--output",str(out/"analysis.json")],check=True,stdout=(out/"analysis.log").open("w"))
        print(variant,args.frames,"completed",flush=True)
    for graph,item in manifest["candidates"].items():
        if shell("sha256sum "+shlex.quote(item["context"])).split()[0]!=item["context_sha256"]:
            raise ValueError("Candidate binary changed during the chain")
    after=shell(f"sha256sum {base}/dlc_inventory.json")
    if after!=state["default_inventory_before"]:
        raise ValueError("Default model inventory changed during the experiment")
    write(root/f"f{args.frames}/INTEGRITY.json",{"passed":True,"compiled_context_only":True,"dlc_fallback_occurred":False,
          "default_inventory_after":after,"commands":commands})


if __name__=="__main__":
    main()
