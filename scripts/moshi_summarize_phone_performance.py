#!/usr/bin/env python3
"""Summarize the native phase CSV without confusing preparation with execution."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def summarize(path):
    totals = defaultdict(float)
    graphs = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            phase = row["phase"]
            milliseconds = float(row["milliseconds"])
            if milliseconds < 0:
                raise ValueError("Negative duration")
            totals[phase] += milliseconds
            if phase not in ("wall_total", "wall_failed"):
                graphs[row["graph"]][phase] += milliseconds
            counts[phase] += 1
    print(f"\n{path}")
    print("phase                      calls    total_seconds")
    for phase, duration in sorted(totals.items(), key=lambda pair: -pair[1]):
        print(f"{phase:26} {counts[phase]:5} {duration / 1000:16.3f}")
    print("\nSlowest graphs (sum of recorded non-overlapping phases):")
    for graph, phases in sorted(graphs.items(), key=lambda pair: -sum(pair[1].values()))[:10]:
        print(f"  {graph}: {sum(phases.values()) / 1000:.3f}s; execute={phases.get('execute', 0) / 1000:.3f}s")
    print("wall_total/wall_failed is end-to-end native time and overlaps other rows; do NOT add it to phase totals.")
    print("Other phases exclude teardown, some host copies/audio I/O and UI. This is not NPU utilization.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.csv:
        summarize(path)


if __name__ == "__main__":
    main()
