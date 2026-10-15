"""Compile a numerically screened Temporal candidate for SM8850 HTP v81 offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--sdk-root", type=Path, required=True)
    args = p.parse_args()
    root, sdk = args.directory, args.sdk_root
    result = json.loads((root/"RESULT.json").read_text())
    state = json.loads((root/"candidate/quantization.json").read_text())
    screen = json.loads((root/"candidate/cpu_simulation.json").read_text())
    validation = json.loads((root/"candidate/calibration_validation.json").read_text())
    dlc = Path(state["output_dlc"])
    if not result["actual_precision_passed"] or not screen["passed"] or not validation["passed"]:
        raise ValueError("Require precision, initial CPU, and calibration gates")
    if sha256(dlc) != validation["candidate_dlc_sha256"] or state["output_sha256"] != validation["candidate_dlc_sha256"]:
        raise ValueError("DLC changed after numerical checks")
    output = root/"htp_offline"
    output.mkdir()
    lib = sdk/"lib/x86_64-linux-clang"
    write(output/"backend.json", {"devices":[{"dsp_arch":"v81", "soc_model":87}]})
    write(output/"config.json", {"backend_extensions":{
        "shared_library_path":str(lib/"libQnnHtpNetRunExtensions.so"), "config_file_path":str(output/"backend.json")}})
    name = state["graph"]+"_mixed"
    command = [str(sdk/"bin/x86_64-linux-clang/qnn-context-binary-generator"),
               "--backend",str(lib/"libQnnHtp.so"), "--model",str(lib/"libQnnModelDlc.so"),
               "--dlc_path",str(dlc), "--config_file",str(output/"config.json"),
               "--binary_file",name, "--output_dir",str(output), "--log_level","warn"]
    receipt = {"status":"running", "script_sha256":sha256(__file__), "command":command,
               "candidate_dlc_sha256":sha256(dlc), "soc_model":87, "dsp_arch":"v81", "phone_execution":False}
    write(output/"compile.json",receipt)
    started = time.monotonic()
    with (output/"compile.log").open("w") as log:
        run = subprocess.run(command, env=dict(os.environ,LD_LIBRARY_PATH=str(lib)), stdout=log,stderr=subprocess.STDOUT)
    binary = output/(name+".bin")
    passed = run.returncode == 0 and binary.is_file() and binary.stat().st_size > 0
    receipt.update(status="success" if passed else "failed",returncode=run.returncode,elapsed_s=time.monotonic()-started)
    if passed:
        receipt.update(binary=str(binary),binary_bytes=binary.stat().st_size,binary_sha256=sha256(binary))
    write(output/"compile.json",receipt)
    result.update(status="offline_calibration_recurrence_and_htp_compile_passed_not_device_verified" if passed else "htp_offline_compile_failed",
                  htp_offline_compile=receipt, actual_dlc_cpu_execution=False, phone_execution=False,
                  device_verified=False, default_bundle_replaced=False, calibration_validation_passed=True,
                  phone_peak_memory_bytes=None,phone_latency_ms=None,actual_resident_contexts=None,context_load_count=None)
    write(root/"RESULT.json",result)
    print(json.dumps(receipt),flush=True)
    if not passed:
        raise SystemExit(run.returncode or 2)


if __name__ == "__main__":
    main()
