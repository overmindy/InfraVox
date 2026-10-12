"""Summarize real native phase CSV and optional QNN/memory captures.

wall_total is the enclosing duration, never another phase to add to the rest.
Preparation/restore phases can enclose I/O allocation, so phase totals are
reported separately rather than presented as an additive timing breakdown.
QNN units: 1=microseconds, 3=cycles, 4=count (QnnProfile.h).
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--qnn", type=Path)
    parser.add_argument("--memory-log", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    phases = collections.defaultdict(float)
    graphs = collections.defaultdict(lambda: collections.defaultdict(list))
    with args.timing.open() as handle:
        for row in csv.DictReader(handle):
            milliseconds = float(row["milliseconds"])
            phases[row["phase"]] += milliseconds
            graphs[row["graph"]][row["phase"]].append(milliseconds)
    wall = phases.get("wall_total", 0)
    if wall <= 0:
        raise ValueError("No completed wall_total in timing CSV")
    report = {"timing": str(args.timing), "wall_ms": wall,
              "phase_totals_are_not_additive": True,
              "phases": {name: {"ms": value, "fraction_of_wall": value / wall}
                         for name, value in phases.items() if name != "wall_total"},
              "graphs": {name: {phase: {"count": len(values), "total_ms": sum(values),
                                       "mean_ms": sum(values) / len(values)}
                                for phase, values in items.items()} for name, items in graphs.items()}}
    if args.qnn:
        counters = collections.defaultdict(lambda: {"count": 0, "sum": 0})
        with args.qnn.open() as handle:
            for row in csv.DictReader(handle):
                key = (row["event_type"], row["event_unit"], row["identifier"])
                counters[key]["count"] += 1
                counters[key]["sum"] += int(row["event_value"])
        report["qnn_event_totals"] = [{"type": key[0], "unit": key[1], "identifier": key[2], **value}
                                       for key, value in counters.items()]
        # RPC and accelerator times overlap. Cycles cannot be converted to
        # utilization without an independently verified clock/time interval.
        report["qnn_events_are_not_additive"] = True
    if args.memory_log:
        text = args.memory_log.read_text()
        for key in ("VmRSS", "VmHWM"):
            values = [int(v) for v in re.findall(rf"{key}:\s+(\d+)\s+kB", text)]
            report["sampled_peak_" + key + "_kib"] = max(values) if values else None
        values = [int(v) for v in re.findall(r"MemAvailable:\s+(\d+)\s+kB", text)]
        report["sampled_min_available_kib"] = min(values) if values else None
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wall_total: {wall:.3f} ms")
    for phase, value in sorted(report["phases"].items(), key=lambda item: -item[1]["ms"]):
        print(f"{phase}: {value['ms']:.3f} ms ({value['fraction_of_wall']:.1%} of wall; may overlap)")
    for name in sorted(graphs, key=lambda g: -sum(graphs[g].get("restore_context", [])))[:8]:
        values = graphs[name].get("restore_context", [])
        if values:
            print(f"restore {name}: {len(values)} calls, {sum(values):.3f} ms")


if __name__ == "__main__":
    main()
