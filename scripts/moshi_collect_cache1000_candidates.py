"""Collect strictly CPU-checked Temporal DLCs without copying large weights.

Run remotely. The collection is a candidate bundle, not an App-ready validated
inventory. Its symlinks can be downloaded with rsync -L --partial. Successful
baseline/frontend/head/DepFormer artifacts are deliberately left untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, action="append", required=True,
                        help="Preferred build first; retain only successfully checked graphs")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selected = {}
    for layer in range(0, 32, 2):
        name = f"temporal_layers_{layer}_{layer + 1}"
        for root in args.build_root:
            folder = root / "graphs" / name
            state_file = folder / "state.json"
            if not state_file.is_file():
                continue
            state = json.loads(state_file.read_text())
            if state.get("status") != "converted_not_device_verified":
                continue
            candidate = Path(state["candidate"])
            report_file = candidate.parent / "patch_report.json"
            report = json.loads(report_file.read_text())
            if (report.get("passed") is not True or len(report.get("samples", [])) != 2
                    or report.get("capacity") != 1000 or report.get("cache_output") != "delta"
                    or report.get("max_relative_rms_checkpoint") is not None):
                raise ValueError(f"Candidate lacks strict two-frame checkpoint: {name}")
            dlc = Path(state["dlc"])
            receipt = folder / "conversion.json"
            if json.loads(receipt.read_text()).get("status") != "success" or not dlc.stat().st_size:
                raise ValueError(f"Missing successful conversion: {name}")
            selected[name] = {"file": f"dlc/{name}.dlc", "source_dlc": str(dlc.resolve()),
                              "bytes": dlc.stat().st_size, "sha256": sha256(dlc),
                              "update": report["update"], "capacity": 1000, "cache_output": "delta",
                              "cpu_report": str(report_file), "conversion_receipt": str(receipt),
                              "device_verified": False}
            break
        if name not in selected:
            raise ValueError(f"No eligible candidate for {name}")
    inventory = {"format": "moshi-temporal-cache1000-candidate-set-v1",
                 "all_temporal_converted": True, "all_graphs_verified": False,
                 "device_verified": False, "graphs": selected}
    destination = args.output_root / "temporal_candidate_inventory.json"
    if destination.is_file() and json.loads(destination.read_text()) != inventory:
        raise ValueError("Existing collection differs; use a new output root")
    (args.output_root / "dlc").mkdir(parents=True, exist_ok=True)
    for name, item in selected.items():
        link = args.output_root / item["file"]
        target = Path(item["source_dlc"])
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target:
                raise ValueError(f"Existing artifact differs: {link}")
        else:
            link.symlink_to(target)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(inventory, indent=2) + "\n")
    temporary.replace(destination)
    print(f"Collected {len(selected)} CPU-checked candidate DLCs; device verification pending.")


if __name__ == "__main__":
    main()
