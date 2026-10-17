"""Check one compact model bundle before replacing the phone default."""
import argparse
import csv
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import wave

import numpy as np

from moshi_quantize_mimi_offline import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--reference-directory", type=Path, required=True)
    p.add_argument("--adb", required=True)
    p.add_argument("--serial", default="ATWV026602002286")
    p.add_argument("--frames", choices=(4, 38), type=int, required=True)
    args = p.parse_args()
    root = args.directory
    out = root/("chain_f"+str(args.frames))
    out.mkdir(exist_ok=False)
    stage = json.loads((root/"STAGE.json").read_text())["phone_bundle"]
    if (root/"PROMOTION.json").exists():
        stage = json.loads((root/"PROMOTION.json").read_text())["phone_bundle"]
    if args.frames == 38:
        assert json.loads((root/"chain_f4/RESULT.json").read_text())["passed"]
    adb = [args.adb, "-s", args.serial]
    commands = []

    def run(*cmd):
        r = subprocess.run(adb+list(cmd), capture_output=True, text=True)
        commands.append({"arguments": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
        save(out/"commands.json", commands)
        assert r.returncode == 0, r.stdout+r.stderr
        return r.stdout.strip()

    def shell(cmd):
        return run("shell", cmd)

    q = shlex.quote
    phone = "/data/local/tmp/moshi-compact-gates-20261018-v1/f"+str(args.frames)
    shell("test ! -e "+q(phone)+" && mkdir -p "+q(phone))
    run("push", str(root/"moshi_reply_latency_probe"), phone+"/probe")
    shell("chmod 755 "+q(phone+"/probe"))
    apk = shell("pm path com.overmindy.mimicodecdemo").split("package:", 1)[1]
    lib = str(Path(apk).parent/"lib/arm64")
    prefix = phone+"/reply"
    input_wav = (stage+"/validation_input.wav" if (root/"PROMOTION.json").exists()
                 else "/data/local/tmp/moshi-lpbq-phone-20261016-v1/input.wav")
    command = ["env", "LD_LIBRARY_PATH="+lib, "ADSP_LIBRARY_PATH="+lib, phone+"/probe", lib,
               stage+"/mimi_encoder.bin", stage+"/mimi_decoder.bin", stage, stage, input_wav, prefix, str(args.frames)]
    started = time.monotonic()
    shell(shlex.join(command)+" > "+q(prefix+".log")+" 2>&1")
    elapsed = time.monotonic()-started
    for path in shell("ls "+q(prefix)+".*").splitlines():
        run("pull", path, str(out/Path(path).name))
    log = (out/"reply.log").read_text()
    assert not re.search(r"phase=(open_dlc|compose|finalize)\b|cache rejected", log)
    sources = set(re.findall(r"serialized_model_source=(\S+)", log))
    assert len(sources) == 28
    latency = json.loads((out/"reply.latency.json").read_text())
    configuration = json.loads((out/"reply.warm.wav.runtime.json").read_text())
    assert configuration["lm"]["context_only_models"]
    assert len(configuration["lm"]["retained_temporal"]) == 16
    profile = list(csv.DictReader((out/"reply.warm.wav.performance.csv").open()))
    assert not any(x["phase"] == "restore_context" for x in profile)
    assert sha(out/"reply.first.wav") == sha(out/"reply.warm.wav")

    def read_audio(path):
        with wave.open(str(path), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, (args.frames-1)*1920)
            return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)/32768

    ref = args.reference_directory/("candidate_f"+str(args.frames)+".warm.wav")
    a, b = read_audio(out/"reply.warm.wav"), read_audio(ref)
    difference = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))
    correlation = float(np.corrcoef(a, b)[0, 1])
    clipped = float(np.mean(np.abs(a) >= 32767/32768))
    text = np.loadtxt(out/"reply.warm.wav.text_tokens.txt", delimiter=",", dtype=np.int32)
    original_text = np.loadtxt(str(ref)+".text_tokens.txt", delimiter=",", dtype=np.int32)
    report = {"frames": args.frames, "input_wav_sha256": shell("sha256sum "+q(input_wav)).split()[0],
              "latency": latency, "elapsed_adb_s": elapsed, "context_only_sources": sorted(sources),
              "retained_temporal_count": 16, "warm_context_restore_count": 0,
              "first_warm_wav_identical": True, "reference_wav": str(ref), "reference_wav_sha256": sha(ref),
              "reference_is_previously_verified_low_memory_lm_with_float_mimi": True,
              "pcm_relative_rms": difference, "pcm_correlation": correlation, "clipped_fraction": clipped,
              "text_token_agreement": float(np.mean(text[:, 1] == original_text[:, 1])),
              "independent_quality_test": False, "default_replaced": False,
              "passed": bool(np.isfinite(a).all() and difference <= .1 and correlation >= .99 and clipped == 0)}
    save(out/"RESULT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("latency", "context_only_sources")}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
