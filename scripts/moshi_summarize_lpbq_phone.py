"""Summarize isolated-shard and paired real-phone Moshi chain evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    single = json.loads((root/"single_comparison.json").read_text())
    summary = {"format":"moshi-lpbq-phone-summary-v1", "phone_execution":True,
               "candidate_dlc_sha256":single["candidate_dlc_sha256"],
               "single_original_10_percent_gate_passed":single["max_errors"]["candidate_vs_original"]<=0.10,
               "single_cpu_htp_2_percent_alignment_passed":single["passed"],
               "single_max_errors":single["max_errors"], "chain":{},
               "default_bundle_replaced":False, "production_quality_validated":False,
               "notes":["Timing phase totals can overlap.",
                        "Process RSS/HWM excludes unaccounted DSP/driver allocations.",
                        "MemAvailable also reflects other device processes.",
                        "The first 4-frame baseline wrapper exited 143 after both completed outputs; those outputs were retained.",
                        "The CPU/HTP alignment warning remains unresolved; its threshold was not relaxed."]}
    completed = True
    for frames in (4,38):
        variants = {}
        for variant in ("baseline","candidate"):
            folder = root/f"f{frames}"/variant
            analysis = json.loads((folder/"analysis.json").read_text())
            prefix = f"{variant}_f{frames}"
            memory = (folder/(prefix+".memory.txt")).read_text()
            def peak(key):
                values = [int(v) for v in re.findall(rf"{key}:\s+(\d+)\s+kB",memory)]
                return max(values) if values else None
            available = [int(v) for v in re.findall(r"MemAvailable:\s+(\d+)\s+kB",memory)]
            item = {"deterministic_first_warm_wav":analysis["first_warm_wav_byte_equal"],
                    "process_sampled_peak_rss_kib":peak("VmRSS"),"process_peak_hwm_kib":peak("VmHWM"),
                    "sampled_min_system_available_kib":min(available) if available else None,"runs":{}}
            for run, measured in zip(("first","warm"),analysis["latency"]["runs"]):
                data = analysis["runs"][run]
                phases,counts = {},{}
                executions = {}
                temporal_load_count = 0
                temporal_load_ms = 0
                with (folder/f"{prefix}.{run}.wav.performance.csv").open() as f:
                    for row in csv.DictReader(f):
                        phase,value = row["phase"],float(row["milliseconds"])
                        phases[phase] = phases.get(phase,0)+value
                        counts[phase] = counts.get(phase,0)+1
                        if phase=="execute":
                            values = executions.setdefault(row["graph"],[]);values.append(value)
                        if phase=="restore_context" and row["graph"].startswith("temporal_layers_"):
                            temporal_load_count += 1;temporal_load_ms += value
                item["runs"][run] = {"total_ms":measured["total_ms"],"first_pcm_ms":measured["first_pcm_ms"],
                    "first_signal_chunk_ms":data["first_chunk_above_minus40_dbfs_ready_ms"],
                    "output_rtf":data["output_rtf"],"planned_temporal_count":data["planned_temporal_count"],
                    "retained_temporal_count":data["retained_temporal_count"],
                    "temporal_context_load_count":temporal_load_count,"temporal_context_load_ms":temporal_load_ms,
                    "all_context_load_count":counts.get("restore_context",0),"phase_totals_ms":phases,
                    "graph_execution":{name:{"count":len(values),"total_ms":sum(values),"mean_ms":sum(values)/len(values)}
                                       for name,values in executions.items()},"audio":data["audio"],
                    "text_token_ids":data["generated_text_token_ids"]}
                completed &= data["audio"]["samples"] == (frames-1)*1920
            completed &= item["deterministic_first_warm_wav"]
            variants[variant] = item
        def file_for(variant,suffix):
            return root/f"f{frames}"/variant/f"{variant}_f{frames}.warm.wav{suffix}"
        def code_values(variant,suffix):
            return [int(v) for line in file_for(variant,suffix).read_text().splitlines() for v in line.split(',')[1:]]
        text_a,text_b = code_values("baseline",".text_tokens.txt"),code_values("candidate",".text_tokens.txt")
        audio_a,audio_b = code_values("baseline",".codes.txt"),code_values("candidate",".codes.txt")
        def agreement(a,b):
            return {"baseline_count":len(a),"candidate_count":len(b),"matching_count":sum(x==y for x,y in zip(a,b)),
                    "match_fraction":sum(x==y for x,y in zip(a,b))/max(len(a),len(b))}
        comparison = {"wav_byte_identical":digest(file_for("baseline",""))==digest(file_for("candidate","")),
                      "audio_code_agreement":agreement(audio_a,audio_b),"text_token_agreement":agreement(text_a,text_b),
                      "warm_total_reduction":1-variants["candidate"]["runs"]["warm"]["total_ms"]/variants["baseline"]["runs"]["warm"]["total_ms"],
                      "same_warm_resident_count":variants["baseline"]["runs"]["warm"]["retained_temporal_count"]==variants["candidate"]["runs"]["warm"]["retained_temporal_count"]}
        summary["chain"][str(frames)] = {"variants":variants,"comparison":comparison}
    summary["basic_chain_checks_passed"] = bool(completed)
    memory = {}
    for variant in ("baseline", "candidate"):
        folder = root/"shard_memory"/variant
        if (folder/"dmabuf.txt").is_file():
            text = (folder/"dmabuf.txt").read_text()
            total = re.search(r"PROCESS TOTAL\s+(\d+) kB\s+(\d+) kB", text)
            if not total:
                raise ValueError("Cannot read isolated DMA-BUF process totals")
            memory[variant] = {"process_dmabuf_rss_kib":int(total[1]),"process_dmabuf_pss_kib":int(total[2]),
                               "global_exported_buffer_stats_available": "Could not get total exported" not in text}
    if len(memory)==2:
        memory["saved_kib"] = memory["baseline"]["process_dmabuf_pss_kib"]-memory["candidate"]["process_dmabuf_pss_kib"]
        memory["reduction"] = memory["saved_kib"]/memory["baseline"]["process_dmabuf_pss_kib"]
        memory["scope"] = "One isolated Temporal context and its probe buffers, not whole LM peak memory"
        summary["isolated_shard_memory"] = memory
    if (root/"decoded_text.json").is_file():
        summary["decoded_text"] = json.loads((root/"decoded_text.json").read_text())
    summary["status"] = "device_runs_complete_with_cpu_alignment_warning" if completed else "device_chain_check_failed"
    (root/"RESULT.json").write_text(json.dumps(summary,indent=2)+"\n")
    for frames,data in summary["chain"].items():
        print(frames,"frames",json.dumps(data["comparison"]))
        for variant,item in data["variants"].items():
            warm=item["runs"]["warm"]
            print(variant,"warm_ms",warm["total_ms"],"resident",warm["retained_temporal_count"],
                  "context_loads",warm["temporal_context_load_count"],"HWM_KiB",item["process_peak_hwm_kib"])


if __name__ == "__main__":
    main()
