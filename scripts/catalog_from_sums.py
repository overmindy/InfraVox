# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fill a catalog variant's file list from a bundle listing (`sha256 size path` lines, e.g. from
`sha256sum` joined with `stat` on the server), for development deploys before hf_publish.py runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "android" / "models" / "catalog.json")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    variant = next((v for m in catalog["models"] for v in m["variants"] if v["id"] == args.variant), None)
    if variant is None:
        sys.exit(f"Variant {args.variant} is not in {args.catalog}")
    files = []
    for line in args.listing.read_text().splitlines():
        digest, size, path = line.split(maxsplit=2)
        files.append({"path": path, "size": int(size), "sha256": digest})
    variant["files"] = files
    args.catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.variant}: {len(files)} files, {sum(f['size'] for f in files) / (1 << 30):.2f} GiB")


if __name__ == "__main__":
    main()
