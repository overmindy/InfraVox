"""Summarize per-shard gates and paired phone chains without hiding warnings."""
import argparse
import csv
import json
from pathlib import Path
import re
import wave

from moshi_quantize_temporal_offline import sha256,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--chain-directory",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--baseline-context-inventory",type=Path)
    args=p.parse_args()
    import numpy as np
    manifest=json.loads(args.manifest.read_text())
    rows=[]
    for graph,item in manifest["candidates"].items():
        folder=Path(item["artifact_directory"])
        result=json.loads((folder/"RESULT.json").read_text())
        precision=json.loads((folder/"candidate/precision_audit.json").read_text())
        norms=[op for g in precision["graphs"]for op in g["normalization_ops"]]
        norm_passed=len(norms)>=24 and all(t=="Float_32"for op in norms for t in (*op["input_dtypes"],*op["output_dtypes"]))
        if not norm_passed:
            raise ValueError("Selected shard contains a lower-precision norm: "+graph)
        validation=json.loads((folder/"candidate/calibration_validation.json").read_text())
        maximum=max(v["relative_rms"]for mode in ("teacher_forced","recurrent")for sample in validation[mode]for v in sample["outputs"].values())
        phone=json.loads(Path(item["phone_gate_file"]).read_text())
        rows.append({"graph":graph,"selected_precision":result["selected_precision"],"w4_parameter_fraction":result["w4_parameter_fraction"],
                     "context_bytes":result["htp_offline_compile"]["binary_bytes"],"baseline_context_bytes":item["baseline_context_bytes"],
                     "context_sha256":item["context_sha256"],"candidate_dlc_sha256":item["candidate_dlc_sha256"],
                     "cpu_calibration_max_relative_rms":maximum,"cpu_calibration_passed":validation["passed"],
                     "phone_max_errors":phone["max_errors"],"candidate_cpu_alignment_2_percent_passed":phone["max_errors"]["candidate_vs_dlc_cpu"]<=.02,
                     "phone_primary_10_percent_passed":item["primary_single_gate_passed"],"normalization_precision_passed":norm_passed})
    report={"format":"moshi-lpbq-scaleout-evidence-v1","shards":rows,"chain":{},"default_bundle_replaced":False,
            "independent_quality_test":False,"actual_dlc_cpu_execution":False,"phone_execution":True,
            "manifest_sha256":sha256(args.manifest),"phone_context_deployment":"Actual compiled candidate binary; zero DLC fallback required"}
    report["context_bytes"]={"baseline":sum(v["baseline_context_bytes"]for v in rows),"candidate":sum(v["context_bytes"]for v in rows)}
    report["context_bytes"]["reduction"]=1-report["context_bytes"]["candidate"]/report["context_bytes"]["baseline"]
    report["projection_counts"]={str(bits):sum(row["selected_precision"].get(str(bits),0)for row in rows)for bits in (4,8,32)}
    # Every two-layer Temporal shard has the same eight matrix shapes.
    report["w4_parameter_fraction"]=sum(v["w4_parameter_fraction"]for v in rows)/len(rows)
    report["w4_parameter_fraction_all_temporal"]=sum(v["w4_parameter_fraction"]for v in rows)/16
    if args.baseline_context_inventory:
        inventory=json.loads(args.baseline_context_inventory.read_text())
        if len(inventory)!=16:
            raise ValueError("Require all sixteen baseline Temporal context sizes")
        baseline=sum(v["bytes"]for v in inventory.values())
        for row in rows:
            if inventory[row["graph"]]["bytes"]!=row["baseline_context_bytes"]:
                raise ValueError("Baseline context inventory differs from the experiment")
        candidate=baseline-report["context_bytes"]["baseline"]+report["context_bytes"]["candidate"]
        report["all_temporal_context_bytes"]={"baseline":baseline,"candidate":candidate,"reduction":1-candidate/baseline}
    for frames in (4,38):
        folder=args.chain_directory/f"f{frames}"
        integrity=json.loads((folder/"INTEGRITY.json").read_text())
        setup=json.loads((args.chain_directory/"phone_setup.json").read_text())
        if not integrity["passed"] or integrity["dlc_fallback_occurred"]:
            raise ValueError("Compiled context chain integrity failed")
        if integrity["default_inventory_after"]!=setup["default_inventory_before"]:
            raise ValueError("Default model inventory differs from the recorded baseline")
        variants={}
        for variant in ("baseline","candidate"):
            d=folder/variant;prefix=f"{variant}_f{frames}"
            analysis=json.loads((d/"analysis.json").read_text())
            memory=(d/(prefix+".memory.txt")).read_text()
            item={"deterministic_first_warm_wav":analysis["first_warm_wav_byte_equal"],"runs":{}}
            for key in ("VmRSS","VmHWM"):
                values=[int(v)for v in re.findall(key+r":\s+(\d+)\s+kB",memory)]
                item[key+"_peak_kib"]=max(values)if values else None
            available=[int(v)for v in re.findall(r"MemAvailable:\s+(\d+)\s+kB",memory)]
            item["minimum_system_available_kib"]=min(available)if available else None
            snap=d/"memory_snapshot.json"
            if snap.exists():
                data=json.loads(snap.read_text());match=re.search(r"PROCESS TOTAL\s+(\d+) kB\s+(\d+) kB",data["dmabuf"])
                if match:item["owned_dmabuf_pss_kib"]=int(match[2])
            for run,time_row in zip(("first","warm"),analysis["latency"]["runs"]):
                data=analysis["runs"][run]
                events=list(csv.DictReader((d/f"{prefix}.{run}.wav.performance.csv").open()))
                executes={}
                for event in events:
                    if event["phase"]=="execute":executes.setdefault(event["graph"],[]).append(float(event["milliseconds"]))
                item["runs"][run]={"total_ms":time_row["total_ms"],"first_pcm_ms":time_row["first_pcm_ms"],
                    "first_signal_chunk_ms":data["first_chunk_above_minus40_dbfs_ready_ms"],"output_rtf":data["output_rtf"],
                    "planned_temporal_count":data["planned_temporal_count"],"retained_temporal_count":data["retained_temporal_count"],
                    "context_load_count":sum(v["phase"]=="restore_context"for v in events),
                    "temporal_context_load_count":sum(v["phase"]=="restore_context"and v["graph"].startswith("temporal_layers_")for v in events),
                    "temporal_execute_ms":sum(sum(values)for name,values in executes.items()if name.startswith("temporal_layers_")),
                    "graph_execute_mean_ms":{name:sum(values)/len(values)for name,values in executes.items()},
                    "audio":data["audio"],"text_token_ids":data["generated_text_token_ids"]}
            variants[variant]=item
        def file(variant,suffix):
            return folder/variant/f"{variant}_f{frames}.warm.wav{suffix}"
        def agreement(suffix):
            values=[[int(x)for line in file(v,suffix).read_text().splitlines()for x in line.split(',')[1:]]for v in ("baseline","candidate")]
            a,b=values
            return {"count":max(len(a),len(b)),"matching":sum(x==y for x,y in zip(a,b))}
        pcm=[]
        for variant in ("baseline","candidate"):
            with wave.open(str(file(variant,"")))as stream:pcm.append(np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(np.float64))
        a,b=pcm
        comparison={"warm_total_reduction":1-variants["candidate"]["runs"]["warm"]["total_ms"]/variants["baseline"]["runs"]["warm"]["total_ms"],
                    "audio_codes":agreement(".codes.txt"),"text_tokens":agreement(".text_tokens.txt"),
                    "wav_byte_identical":sha256(file("baseline",""))==sha256(file("candidate","")),
                    "pcm_relative_rms":float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-12)),"pcm_correlation":float(np.corrcoef(a,b)[0,1])}
        report["chain"][str(frames)]={"variants":variants,"comparison":comparison}
    report["basic_phone_checks_passed"]=all(v["deterministic_first_warm_wav"] and all(r["audio"]["samples"]==(int(frames)-1)*1920 and r["audio"]["clipped_fraction"]==0 for r in v["runs"].values())for frames,data in report["chain"].items()for v in data["variants"].values())
    report["cpu_htp_alignment_warnings"]=[v["graph"]for v in rows if not v["candidate_cpu_alignment_2_percent_passed"]]
    report["primary_single_shard_gates_passed"]=all(v["phone_primary_10_percent_passed"]for v in rows)
    report["status"]=("device_profile_complete_with_alignment_warning" if report["cpu_htp_alignment_warnings"] else "device_profile_complete") if report["basic_phone_checks_passed"] and report["primary_single_shard_gates_passed"] else "phone_gate_failed"
    write(args.output,report)
    print(json.dumps({"shards":len(rows),"projections":report["projection_counts"],"w4_parameter_fraction":report["w4_parameter_fraction"],
                      "contexts":report["context_bytes"],"chain":{k:v["comparison"]for k,v in report["chain"].items()}},indent=2))


if __name__=="__main__":
    main()
