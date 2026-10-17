"""Run one short baseline/candidate Mimi loopback in isolated context caches."""
import argparse
import csv
import json
import re
from pathlib import Path
import shlex
import subprocess
import time
import wave

import numpy as np

from moshi_quantize_mimi_offline import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--adb", required=True)
    p.add_argument("--serial", default="ATWV026602002286")
    p.add_argument("--phone-root", default="/data/local/tmp/moshi-mimi-w8a16-20261018-v1")
    p.add_argument("--reuse-uploaded-contexts", action="store_true")
    p.add_argument("--candidate-subdir", default="fc_only")
    args = p.parse_args()
    root, remote = args.directory, args.phone_root
    out = root/"phone"
    out.mkdir(exist_ok=False)
    adb = [args.adb, "-s", args.serial]
    commands = []

    def run(*cmd):
        r = subprocess.run(adb+list(cmd), capture_output=True, text=True)
        commands.append({"arguments": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
        save(out/"commands.json", commands)
        if r.returncode:
            raise RuntimeError(r.stdout+r.stderr)
        return r.stdout.strip()

    def shell(cmd):
        return run("shell", cmd)

    q = shlex.quote
    app = "/sdcard/Android/data/com.overmindy.mimicodecdemo/files"
    # Use the deployed bundle paths, whose mtimes identify the existing caches.
    # The separate models/mimi copies have identical bytes but different mtimes.
    models = app+"/models/lm-cache1000-v1"
    cache = app+"/qnn-context-v1/cache1000-v1"
    compact = shell("if test -f "+models+"/CONTEXT_ONLY; then echo yes; else echo no; fi") == "yes"
    suffix = ".bin" if compact else ".dlc"
    baseline_hashes = shell("sha256sum "+models+"/mimi_*"+suffix)
    apk = shell("pm path com.overmindy.mimicodecdemo").split("package:", 1)[1]
    lib = str(Path(apk).parent/"lib/arm64")
    if not args.reuse_uploaded_contexts:
        shell("test ! -e "+q(remote))
    shell("mkdir -p "+q(remote)+"/cache_baseline "+q(remote)+"/cache_candidate "+q(remote)+"/results")
    runner = root/"mimi_quant_probe"
    run("push", str(runner), remote+"/runner")
    shell("chmod 755 "+q(remote+"/runner"))
    run("push", str(root/"input.wav"), remote+"/input.wav")
    for comp in ("encoder", "decoder"):
        state = json.loads((root/args.candidate_subdir/comp/"RESULT.json").read_text())
        assert state["actual_precision_passed"] and state["htp_compile"]["returncode"] == 0
        binary = Path(state["htp_compile"].get("local_binary",
                      str(root/args.candidate_subdir/comp/"htp"/("mimi_"+comp+".bin"))))
        assert sha(binary) == state["htp_compile"]["sha256"]
        target = remote+"/mimi_"+comp+".bin"
        if not args.reuse_uploaded_contexts:
            run("push", str(binary), target)
        assert shell("sha256sum "+q(target)).split()[0] == sha(binary)
        if compact:
            continue
        paths = shell("ls "+cache+"/mimi_"+comp+"-*.bin").splitlines()
        assert len(paths) == 1
        base = paths[0]
        name = Path(base).name
        size_mtime = shell("stat -c '%s-%Y' "+q(models+"/mimi_"+comp+".dlc"))
        assert name.startswith("mimi_"+comp+"-"+size_mtime+"-"), (name, size_mtime)
        for variant, source in (("baseline", base), ("candidate", target)):
            link = remote+"/cache_"+variant+"/"+name
            if args.reuse_uploaded_contexts:
                assert shell("readlink "+q(link)) == source
            else:
                shell("ln -s "+q(source)+" "+q(link))
    results = {}
    for variant in ("baseline", "candidate"):
        target = remote+"/results/"+variant+".wav"
        cmd = ["env", "LD_LIBRARY_PATH="+lib, "ADSP_LIBRARY_PATH="+lib,
               "MOSHI_QNN_PROFILE=1", "MOSHI_CODEC_CONTEXT_CACHE_DIR="+remote+"/cache_"+variant,
               remote+"/runner", lib,
               remote+"/mimi_encoder.bin" if compact and variant == "candidate" else models+"/mimi_encoder"+suffix,
               remote+"/mimi_decoder.bin" if compact and variant == "candidate" else models+"/mimi_decoder"+suffix,
               remote+"/input.wav", target, "4"]
        start = time.monotonic()
        shell("test ! -e "+q(target)+" && "+shlex.join(cmd)+" > "+q(remote+"/results/"+variant+".log")+" 2>&1")
        elapsed = time.monotonic()-start
        local = out/variant
        local.mkdir()
        for path in shell("ls "+q(remote+"/results/"+variant)+".*").splitlines():
            run("pull", path, str(local/Path(path).name))
        profiles = list(csv.DictReader((local/(variant+".wav.performance.csv")).open()))
        # Constructor timings precede run(), which clears the CSV timing list.
        # Require the restore evidence from the full native log as well.
        log = (local/(variant+".log")).read_text()
        restores = re.findall(r"moshi_timing graph=\S+ phase=restore_context ms=\S+", log)
        assert len(restores) == 2 and not re.search(r"phase=(open_dlc|compose|finalize)\b|cache rejected", log)
        results[variant] = {"elapsed_adb_s": elapsed, "profile": profiles,
                            "context_restores": len(restores), "dlc_fallback": False,
                            "runner_sha256": sha(runner)}
    assert shell("sha256sum "+models+"/mimi_*"+suffix) == baseline_hashes

    def audio(variant):
        with wave.open(str(out/variant/(variant+".wav")), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 7680)
            return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)/32768

    a, b = audio("candidate"), audio("baseline")
    codes = {}
    for variant in ("baseline", "candidate"):
        codes[variant] = np.loadtxt(out/variant/(variant+".wav.codes.txt"), delimiter=",", dtype=np.int32)
    agreement = float(np.mean(codes["baseline"][:, 1:] == codes["candidate"][:, 1:]))
    delta = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))
    summary = {"format": "mimi-selective-ptq-phone-short-v1", "frames": 4,
               "code_agreement": agreement, "pcm_relative_rms": delta,
               "pcm_correlation": float(np.corrcoef(a, b)[0, 1]),
               "clipped_fraction": float(np.mean(np.abs(a) >= 32767/32768)),
               "results": results, "default_models_unchanged": True,
               "audio_quality_independently_verified": False,
               "short_numerical_gate_passed": bool(delta <= .1 and np.isfinite(a).all())}
    save(out/"PHONE_RESULT.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    if not summary["short_numerical_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
