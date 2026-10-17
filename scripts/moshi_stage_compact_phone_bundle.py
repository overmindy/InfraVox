"""Stage one device-specific bundle containing 26 LM and two Mimi contexts.

Copy the already-verified models on the phone. Do not download DLCs or alter
the existing default. Promotion requires a separate full-chain result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from moshi_quantize_mimi_offline import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--lm-directory", type=Path, required=True)
    p.add_argument("--adb", required=True)
    p.add_argument("--serial", default="ATWV026602002286")
    p.add_argument("--mimi-phone-root", default="/data/local/tmp/moshi-mimi-w4a16-20261018-v2")
    args = p.parse_args()
    root = args.directory
    gate = json.loads((root/"phone/PHONE_RESULT.json").read_text())
    assert gate["short_numerical_gate_passed"]
    baseline = json.loads((args.lm_directory/"BASELINE_PHONE_MODEL_INVENTORY.json").read_text())
    lm = json.loads((args.lm_directory/"LOW_MEMORY_MANIFEST.json").read_text())
    app = "/sdcard/Android/data/com.overmindy.mimicodecdemo/files"
    target = app+"/models/lm-compact-incoming-v1"
    adb = [args.adb, "-s", args.serial]
    calls = []

    def run(*command):
        r = subprocess.run(adb+list(command), capture_output=True, text=True)
        calls.append({"arguments": command, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
        save(root/"STAGE_COMMANDS.json", calls)
        assert r.returncode == 0, r.stdout+r.stderr
        return r.stdout.strip()

    def shell(command):
        return run("shell", command)

    q = shlex.quote
    sources = {}
    for name, metadata in baseline["graphs"].items():
        selected = lm["candidates"].get(name)
        sources[name] = {"source": selected["context"] if selected else lm["baseline_cache"]+"/"+metadata["device_context_file"],
                         "file": "dlc/"+name+".bin", "expected_sha256": selected["context_sha256"] if selected else None,
                         "source_dlc_sha256": selected["candidate_dlc_sha256"] if selected else metadata["sha256"]}
    for comp in ("encoder", "decoder"):
        result = json.loads((root/"selected"/comp/"RESULT.json").read_text())
        name = "mimi_"+comp
        sources[name] = {"source": args.mimi_phone_root+"/"+name+".bin", "file": name+".bin",
                         "expected_sha256": result["htp_compile"]["sha256"], "source_dlc_sha256": result["dlc"]["sha256"]}
    shell("test ! -e "+q(target)+" && mkdir -p "+q(target+"/dlc")+
          " && chmod 755 "+q(target)+" "+q(target+"/dlc"))
    hashes = shell("sha256sum "+" ".join(q(v["source"]) for v in sources.values()))
    actual = {}
    for line in hashes.splitlines():
        digest, path = line.split(None, 1)
        actual[path.strip()] = digest
    inventory = {"format": "moshi-context-only-app-inventory-v1", "runtime_preset": "cache1000-shared-v1",
                 "all_graphs_verified": False, "device_serial": args.serial,
                 "device_fingerprint": shell("getprop ro.build.fingerprint"),
                 "qnn_core_api": "2.34.0", "qnn_backend_api": "5.45.0",
                 "temporal_capacity": 1000, "temporal_profile": "15 validated LPBQ shards; deployed 0_1 retained",
                 "graphs": {}, "mimi": {},
                 "verification_scope": {"artifact_sha256": True, "full_chain_frames": 0,
                                        "independent_audio_quality": False, "long_context_wrap": False}}
    for name, row in sources.items():
        digest = actual[row["source"]]
        assert row["expected_sha256"] is None or digest == row["expected_sha256"], name
        path = target+"/"+row["file"]
        shell("cp -p "+q(row["source"])+" "+q(path)+" && chmod 644 "+q(path))
        size = int(shell("stat -c %s "+q(path)))
        entry = {"file": row["file"], "sha256": digest, "bytes": size, "source_dlc_sha256": row["source_dlc_sha256"]}
        inventory["mimi" if name.startswith("mimi_") else "graphs"][name] = entry
        print("Staged", name, size, flush=True)
    sums = "\n".join(v["sha256"]+"  "+v["file"] for section in ("graphs", "mimi") for v in inventory[section].values())+"\n"
    (root/"PHONE_SHA256SUMS").write_text(sums)
    run("push", str(root/"PHONE_SHA256SUMS"), target+"/SHA256SUMS")
    shell("cd "+q(target)+" && sha256sum -c SHA256SUMS")
    save(root/"COMPACT_INVENTORY.json", inventory)
    run("push", str(root/"COMPACT_INVENTORY.json"), target+"/dlc_inventory.json")
    marker = root/"CONTEXT_ONLY"
    marker.write_text("moshi-context-only-app-inventory-v1\n")
    run("push", str(marker), target+"/CONTEXT_ONLY")
    shell("chmod 644 "+q(target+"/dlc_inventory.json")+" "+q(target+"/CONTEXT_ONLY"))
    save(root/"STAGE.json", {"phone_bundle": target, "inventory_sha256": hashlib.sha256((root/"COMPACT_INVENTORY.json").read_bytes()).hexdigest(),
                              "files": 28, "model_bytes": sum(v["bytes"] for section in ("graphs", "mimi") for v in inventory[section].values()),
                              "default_replaced": False})


if __name__ == "__main__":
    main()
