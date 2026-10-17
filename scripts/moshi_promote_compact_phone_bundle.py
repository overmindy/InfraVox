"""Promote the verified compact bundle; retain the old bundle until App checks."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from moshi_quantize_mimi_offline import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--apk", type=Path, required=True)
    p.add_argument("--adb", required=True)
    p.add_argument("--serial", default="ATWV026602002286")
    args = p.parse_args()
    root = args.directory
    checks = json.loads((root/"COMPONENT_VERIFICATION.json").read_text())
    assert checks["passed"]
    inventory = json.loads((root/"COMPACT_INVENTORY.json").read_text())
    stage = json.loads((root/"STAGE.json").read_text())["phone_bundle"]
    assert inventory["device_serial"] == args.serial
    adb = [args.adb, "-s", args.serial]
    commands = []

    def run(*cmd):
        r = subprocess.run(adb+list(cmd), capture_output=True, text=True)
        commands.append({"arguments": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
        save(root/"PROMOTE_COMMANDS.json", commands)
        if r.returncode:
            raise RuntimeError(r.stdout+r.stderr)
        return r.stdout.strip()

    def shell(cmd):
        return run("shell", cmd)

    q = shlex.quote
    assert shell("getprop ro.build.fingerprint") == inventory["device_fingerprint"]
    inventory["all_graphs_verified"] = True
    inventory["verification_scope"].update(full_chain_frames=38, component_verification=checks,
                                           installed_app_verified=False)
    inventory["mimi_precision"] = {"encoder": "32 W8/A16 linear projections; float convolution/codebooks",
                                    "decoder": "32 LPBQ W4/A16 projections, block32; float convolution/codebooks"}
    save(root/"COMPACT_INVENTORY.json", inventory)
    run("push", str(root/"COMPACT_INVENTORY.json"), stage+"/dlc_inventory.json")
    shell("chmod 644 "+q(stage+"/dlc_inventory.json"))
    parent = "/sdcard/Android/data/com.overmindy.mimicodecdemo/files/models"
    default, backup = parent+"/lm-cache1000-v1", parent+"/lm-before-compact-20261018"
    shell("am force-stop com.overmindy.mimicodecdemo")
    shell("test -d "+q(stage)+" && test -d "+q(default)+" && test ! -e "+q(backup)+" && mv "+q(default)+" "+q(backup))
    try:
        shell("mv "+q(stage)+" "+q(default))
        run("install", "-r", str(args.apk))
    except Exception:
        shell("test ! -d "+q(stage)+" && mv "+q(default)+" "+q(stage)+"; mv "+q(backup)+" "+q(default))
        raise
    installed = shell("dumpsys package com.overmindy.mimicodecdemo | grep versionName")
    assert "2.6.0-compact-lpbq" in installed
    shell("am start -n com.overmindy.mimicodecdemo/.MainActivity")
    save(root/"PROMOTION.json", {"phone_bundle": default, "old_bundle_pending_app_verification": backup,
                                "apk": str(args.apk), "apk_sha256": sha(args.apk), "inventory_sha256": sha(root/"COMPACT_INVENTORY.json"),
                                "installed_version": installed, "app_ui_verified": False, "old_models_deleted": False})
    print("Compact bundle installed; old models retained until App verification")


if __name__ == "__main__":
    main()
