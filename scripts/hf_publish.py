# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Publish one on-device model variant to the shared Hugging Face model repo (main branch).

Repo layout, read by the Android app through /resolve/main/:
    catalog.json              every model and variant, with per-file size and sha256
    README.md                 model card generated from the catalog
    <variant path>/...        the variant's files, e.g. moshi/moshiko-lm-v4-lpbq4-sm8850/dlc/head.bin

android/models/catalog.json is the source of truth: this script refreshes the variant's file list
from the bundle, writes it back there (commit it), and uploads the bundle, catalog and README.
Uploads are resumable; rerun the same command after an interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "android" / "models" / "catalog.json"
SKIPPED = {"SHA256SUMS", "CONTEXT_ONLY"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def find_variant(catalog: dict, variant_id: str) -> tuple[dict, dict]:
    for model in catalog["models"]:
        for variant in model["variants"]:
            if variant["id"] == variant_id:
                return model, variant
    sys.exit(f"Variant {variant_id} is not in the catalog; add its metadata there first.")


def scan_bundle(bundle: Path, extra: list[Path], previous: list[dict]) -> list[dict]:
    files = {
        p.relative_to(bundle).as_posix(): p
        for p in bundle.rglob("*")
        if p.is_file() and p.name not in SKIPPED and not any(part.startswith(".") for part in p.relative_to(bundle).parts)
    }
    for path in extra:
        files[path.name] = path
    expected = {}
    if (bundle / "SHA256SUMS").is_file():
        for line in (bundle / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest
    order = {entry["path"]: i for i, entry in enumerate(previous)}
    entries = []
    for name in sorted(files, key=lambda n: (order.get(n, len(order)), n)):
        digest = sha256(files[name])
        if name in expected and expected[name] != digest:
            sys.exit(f"{name}: sha256 {digest} does not match the bundle's SHA256SUMS")
        entries.append({"path": name, "size": files[name].stat().st_size, "sha256": digest})
        print(f"  {name}  {entries[-1]['size']:>12}  {digest[:16]}")
    return entries


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size and target.stat().st_ino == source.stat().st_ino:
        return
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def model_card(catalog: dict, repo: str) -> str:
    lines = [
        "---",
        "tags: [qai-hub-models, full-duplex, speech-to-speech, qnn, snapdragon]",
        "license: other",
        "---",
        "",
        f"# {repo}",
        "",
        "Quantized full-duplex speech models compiled to Qualcomm QNN HTP context binaries for the "
        "Duplex Android app. The app reads `catalog.json` and downloads from the `main` branch, either "
        "from huggingface.co or a mirror such as hf-mirror.com.",
        "",
        "Context binaries are tied to the listed SoC and QAIRT version and will not load elsewhere.",
        "",
    ]
    for model in catalog["models"]:
        lines += [f"## {model['name']}", "", model.get("summary", ""), ""]
        if model.get("upstream"):
            lines.append(
                f"Converted from [{model['upstream']}](https://huggingface.co/{model['upstream']}) "
                f"by {model.get('publisher', 'its authors')}, license {model.get('license', 'see upstream')}."
            )
            lines.append("")
        lines += ["| Variant | Path | SoC | QAIRT | Size |", "| --- | --- | --- | --- | --- |"]
        for v in model["variants"]:
            size = sum(f["size"] for f in v["files"]) / (1 << 30)
            lines.append(f"| {v['title']} | `{v['path']}` | {'/'.join(v.get('socs', []))} | {v.get('qairt', '')} | {size:.2f} GB |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Hugging Face model repo, e.g. user/duplex-voice-models")
    parser.add_argument("--variant", required=True, help="Variant id from catalog.json")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Local copy of the variant's files")
    parser.add_argument("--add", type=Path, action="append", default=[], help="Extra file placed at the variant root")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--stage-dir", type=Path, help="Upload staging dir (default: <bundle-dir>.hf-stage)")
    parser.add_argument("--private", action="store_true", help="Create the repo as private (the app then needs a token)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel uploads; keep low behind a proxy")
    parser.add_argument("--dry-run", action="store_true", help="Update catalog and stage, but do not upload")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    model, variant = find_variant(catalog, args.variant)
    print(f"== hashing {args.bundle_dir}")
    variant["files"] = scan_bundle(args.bundle_dir, args.add, variant.get("files", []))
    args.catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(f"== updated {args.catalog}")

    stage = args.stage_dir or args.bundle_dir.with_name(args.bundle_dir.name + ".hf-stage")
    extra = {p.name: p for p in args.add}
    for entry in variant["files"]:
        source = extra.get(entry["path"], args.bundle_dir / entry["path"])
        link_or_copy(source, stage / variant["path"] / entry["path"])
    (stage / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    (stage / "README.md").write_text(model_card(catalog, args.repo))
    print(f"== staged {stage}")
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo, folder_path=stage, repo_type="model", revision="main", num_workers=args.workers
    )

    remote = {f.path: f.size for f in api.list_repo_tree(args.repo, recursive=True, revision="main") if hasattr(f, "size")}
    missing = [e["path"] for e in variant["files"] if remote.get(f"{variant['path']}/{e['path']}") != e["size"]]
    if missing:
        sys.exit(f"Upload incomplete, rerun to resume: {missing}")
    print(f"== https://huggingface.co/{args.repo}/tree/main/{variant['path']} ({model['name']}) is complete")


if __name__ == "__main__":
    main()
