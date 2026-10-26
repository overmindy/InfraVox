# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Publish a signed InfraVox APK so the app's update check finds it.

The app reads app/latest.json from the model repo's main branch through the user's download source
(hf-mirror.com or huggingface.co), and falls back to the GitHub Releases API. This script uploads
    app/infravox-<versionName>.apk
    app/latest.json           {"versionCode", "versionName", "notes", "apk": {path, size, sha256}, "github"}
to the Hugging Face repo, and with --github creates the GitHub release v<versionName> with the same APK.
The version comes from android/demo/AndroidManifest.xml; bump it before building the APK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "android" / "demo" / "AndroidManifest.xml"


def manifest_version() -> tuple[int, str]:
    text = MANIFEST.read_text()
    code = re.search(r'android:versionCode="(\d+)"', text)
    name = re.search(r'android:versionName="([^"]+)"', text)
    if not code or not name:
        sys.exit(f"No versionCode/versionName in {MANIFEST}")
    return int(code.group(1)), name.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apk", type=Path, required=True, help="Signed APK (android/demo/build.sh output)")
    parser.add_argument("--repo", default="ELAINAy/duplex-voice-models", help="Hugging Face repo the app reads")
    parser.add_argument("--github-repo", default="overmindy/InfraVox")
    parser.add_argument("--notes", default="", help="Release notes shown in the update dialog")
    parser.add_argument("--notes-file", type=Path, help="Read the release notes from a file")
    parser.add_argument("--github", action="store_true", help="Also create the GitHub release with gh")
    parser.add_argument("--dry-run", action="store_true", help="Print latest.json without uploading")
    args = parser.parse_args()

    code, name = manifest_version()
    notes = args.notes_file.read_text().strip() if args.notes_file else args.notes
    asset = f"infravox-{name}.apk"
    digest = hashlib.sha256(args.apk.read_bytes()).hexdigest()
    latest = {
        "versionCode": code,
        "versionName": name,
        "notes": notes,
        "apk": {"path": f"app/{asset}", "size": args.apk.stat().st_size, "sha256": digest},
        "github": f"https://github.com/{args.github_repo}/releases/download/v{name}/{asset}",
    }
    print(json.dumps(latest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    if args.github:
        with tempfile.TemporaryDirectory() as tmp:
            renamed = Path(tmp) / asset
            renamed.write_bytes(args.apk.read_bytes())
            subprocess.run(
                ["gh", "release", "create", f"v{name}", str(renamed), "--repo", args.github_repo,
                 "--title", f"InfraVox {name}", "--notes", notes or f"InfraVox {name}"],
                check=True,
            )

    api = HfApi()
    api.upload_file(path_or_fileobj=str(args.apk), path_in_repo=f"app/{asset}", repo_id=args.repo, repo_type="model")
    # latest.json goes last: the app must never see a manifest whose APK is not uploaded yet.
    api.upload_file(
        path_or_fileobj=(json.dumps(latest, ensure_ascii=False, indent=2) + "\n").encode(),
        path_in_repo="app/latest.json", repo_id=args.repo, repo_type="model",
        commit_message=f"InfraVox {name}",
    )
    print(f"== https://huggingface.co/{args.repo}/blob/main/app/latest.json -> {name} ({code})")


if __name__ == "__main__":
    main()
