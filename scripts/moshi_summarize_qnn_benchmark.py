#!/usr/bin/env python3
"""Summarize resident QNN timings and cycle hotspots without double counting.

Input: CSVs from mimi_qnn_runner --benchmark-temporal. Repetitions use the
same captured frame-0 hidden and zero initial caches, not a recurrent chain.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(timing_path, nodes_path):
    with timing_path.open(newline="") as handle:
        timing = list(csv.DictReader(handle))
    execute = [float(row["milliseconds"]) for row in timing if row["phase"] == "execute"]
    restore = [float(row["milliseconds"]) for row in timing if row["phase"] == "restore_context"]
    print(f"\n{timing_path.name}")
    print(f"  restore context: {restore} ms")
    print(f"  execute: {execute} ms")
    if len(execute) > 1:
        print(f"  warm median (exclude first): {statistics.median(execute[1:]):.3f} ms")
    if nodes_path is None:
        return
    nodes = defaultdict(list)
    threads = set()
    with nodes_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] == "8001":
                threads.add(row["event_value"])
            # 404 = node execution; unit 3 = processor cycles. Other profile
            # events can be inclusive parents: never sum them as wall time.
            if row["event_type"] == "404" and row["event_unit"] == "3":
                nodes[row["identifier"]].append(int(row["event_value"]))
    print(f"  reported HVX threads: {sorted(threads)} (not HMX utilization)")
    ranked = sorted(((statistics.median(values[1:] or values), name, len(values))
                     for name, values in nodes.items()), reverse=True)
    print("  top node cycle medians (exclude first; not additive wall percentages):")
    for cycles, name, count in ranked[:15]:
        print(f"    {cycles:12,.0f}  {name}  [{count} calls]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--nodes", type=Path)
    args = parser.parse_args()
    summarize(args.timing, args.nodes)


if __name__ == "__main__":
    main()
