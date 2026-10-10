"""Verify the local 26-DLC delivery without loading or running any model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def verify(bundle):
    inventory = json.loads((bundle / "dlc_inventory.json").read_text())
    manifest = json.loads((bundle / "quantization_manifest.json").read_text())
    names = {"frontend", "head"}
    names.update(f"temporal_layers_{i}_{i+1}" for i in range(0, 32, 2))
    names.update(f"depformer_codebook_{i}" for i in range(8))
    if inventory.get("all_graphs_verified") is not True or set(inventory["graphs"]) != names:
        raise ValueError("Expected a complete verified 26-graph inventory")
    if set(manifest["graphs"]) != names:
        raise ValueError("Manifest graph set differs")
    checksums = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        checksum, filename = line.split(None, 1)
        filename = filename.strip()
        if filename in checksums:
            raise ValueError("Duplicate checksum entry")
        checksums[filename] = checksum
    if set(checksums) != {f"dlc/{name}.dlc" for name in names}:
        raise ValueError("Checksum file does not cover exactly 26 DLCs")
    rows = []
    for name in sorted(names):
        record = inventory["graphs"][name]
        filename = f"dlc/{name}.dlc"
        if record["file"] != filename or checksums[filename] != record["sha256"]:
            raise ValueError(f"Inventory/checksum mismatch: {name}")
        if manifest["graphs"][name]["compiled_model_id"] != record["compiled_model_id"]:
            raise ValueError(f"Cloud model ID mismatch: {name}")
        path = bundle / filename
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/nonregular DLC: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Downloaded DLC hash mismatch: {name}")
        rows.append({"graph": name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
        print(f"DOWNLOAD VERIFIED {name}", flush=True)
    return {"format": "moshi-local-lm-bundle-v1", "download_verified": True,
            "module_validation": "remote captured/recurrent probes; not full-chain proof",
            "graphs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--wait-seconds", type=int, default=14400)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    if args.wait_for_pid:
        print(f"Waiting for existing rsync PID {args.wait_for_pid}", flush=True)
        while True:
            try:
                os.kill(args.wait_for_pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Download still running; not a completed bundle")
            time.sleep(15)
    result = verify(args.bundle_dir)
    destination = args.bundle_dir / "DOWNLOAD_VERIFIED.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
