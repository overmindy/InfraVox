"""Collect ready context candidates, then run serial phone shard gates."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from moshi_quantize_temporal_offline import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("ssh-host","remote-root","phone-root","adb","serial"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--ssh-port",type=int,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--layers",type=int,nargs="+",required=True)
    args=p.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    pending=set(args.layers)
    downloading={}
    ready=[]
    result={"jobs":{},"default_bundle_replaced":False}
    for layer in list(pending):
        graph=f"temporal_layers_{layer}_{layer+1}"
        gate=args.directory/"phone"/graph/"PHONE_GATE.json"
        source=args.directory/graph/"RESULT.json"
        if gate.exists() and source.exists():
            previous=json.loads(gate.read_text())
            candidate=json.loads(source.read_text())
            if previous["candidate_dlc_sha256"]!=candidate["htp_offline_compile"]["candidate_dlc_sha256"]:
                raise ValueError("Existing phone gate belongs to another candidate")
            result["jobs"][str(layer)]={"status":"reused_phone_gate","passed":previous["primary_candidate_10_percent_gate_passed"]}
            pending.remove(layer)
    def snapshot():
        code="""import json
from pathlib import Path
base=Path(ROOT)
rows={}
for layer in LAYERS:
 p=base/f'temporal_layers_{layer}_{layer+1}'
 r=p/'lpbq/RESULT.json'
 if r.exists():
  a=json.loads(r.read_text());rows[str(layer)]={'status':a['status'],'reference_ready':(p/'lpbq/phone_reference/phone_reference.json').exists()}
 else:
  w=p/'WORKFLOW.json'
  rows[str(layer)]={'status':json.loads(w.read_text())['status'] if w.exists() else 'pending','reference_ready':False}
print(json.dumps(rows))
""".replace("ROOT",repr(args.remote_root)).replace("LAYERS",repr(sorted(pending)))
        return json.loads(subprocess.run(["ssh","-p",str(args.ssh_port),args.ssh_host,"python3 -c "+shlex.quote(code)],capture_output=True,text=True,check=True).stdout)
    def collect(layer):
        graph=f"temporal_layers_{layer}_{layer+1}"
        command=["rsync","-az","--include=RESULT.json","--include=candidate/","--include=quantization.json",
                 "--include=precision_audit.json","--include=calibration_validation.json","--include=cpu_simulation.json",
                 "--include=htp_offline/","--include=*.bin","--include=compile.json","--include=phone_reference/",
                 "--include=phone_reference.json","--include=*.npz","--include=*.raw","--exclude=*",
                 "-e",f"ssh -p {args.ssh_port}",args.ssh_host+":"+args.remote_root+"/"+graph+"/lpbq/",str(args.directory/graph)+"/"]
        subprocess.run(command,check=True,capture_output=True)
        return layer
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        while pending or downloading or ready:
            if pending:
                state=snapshot()
                for text_layer,entry in state.items():
                    layer=int(text_layer)
                    if entry["reference_ready"] and entry["status"].startswith("offline_calibration_recurrence_and_htp_compile_passed"):
                        downloading[layer]=pool.submit(collect,layer)
                        pending.remove(layer)
                        result["jobs"][text_layer]={"status":"downloading"}
                        print("Collecting",layer,flush=True)
                    elif "failed" in entry["status"] or "rejected" in entry["status"]:
                        pending.remove(layer)
                        result["jobs"][text_layer]=entry
                        print("Offline candidate rejected",layer,entry["status"],flush=True)
            for layer,future in list(downloading.items()):
                if future.done():
                    future.result()
                    ready.append(layer)
                    del downloading[layer]
            if ready:
                layer=ready.pop(0)
                command=[sys.executable,str(Path(__file__).with_name("moshi_run_lpbq_phone_shards.py")),
                         "--directory",str(args.directory),"--adb",args.adb,"--serial",args.serial,
                         "--phone-root",args.phone_root,"--layers",str(layer),"--context-only"]
                child=subprocess.run(command)
                result["jobs"][str(layer)]={"status":"phone_gate_complete" if child.returncode==0 else "phone_gate_rejected",
                                            "returncode":child.returncode}
            write(args.directory/"COLLECTION.json",result)
            if pending or downloading:
                time.sleep(15)


if __name__=="__main__":
    main()
