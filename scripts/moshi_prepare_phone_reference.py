"""Export a bounded recurrent Temporal reference for the Android HTP probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from moshi_quantize_temporal_offline import sha256, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-directory", "output-dir", "sdk-root", "numpy-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 2 or args.frames > 4:
        raise ValueError("Use two to four captured recurrent frames")
    root = args.output_dir
    root.mkdir()
    for name in ("quantization.json", "model.w4a16_encoding.json"):
        shutil.copy2(args.candidate_directory / name, root / name)
    with (root / "simulation.log").open("w") as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name("moshi_simulate_offline_dlc.py")),
                        "--directory", str(root), "--sdk-root", str(args.sdk_root),
                        "--numpy-dir", str(args.numpy_dir)], stdout=log, stderr=subprocess.STDOUT, check=True)
    import numpy as np
    import onnxruntime as ort
    state = json.loads((root / "quantization.json").read_text())
    screen = json.loads((root / "cpu_simulation.json").read_text())
    if not screen["passed"]:
        raise ValueError("DLC-derived CPU screen failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    reference = ort.InferenceSession(state["reference_onnx"], options, providers=["CPUExecutionProvider"])
    quantized = ort.InferenceSession(str(root / "cpu_simulation.onnx"), options, providers=["CPUExecutionProvider"])
    names = [v.name for v in reference.get_outputs()]
    cache_names = [v.name for v in reference.get_inputs() if v.name.startswith("layer_")]
    clips = {v["source_id"] for v in state["samples"]}
    clip = "clean-000" if "clean-000" in clips else sorted(clips)[0]
    samples = {v["frame"]: v for v in state["samples"] if v["source_id"] == clip}
    ref_cache, quant_cache, hidden, records = {}, {}, [], []
    for frame in range(args.frames):
        sample = samples[frame]
        path = Path(state["calibration_manifest"]).parent / sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Captured source changed")
        with np.load(path) as data:
            feed = {v.name: data[v.name].copy() for v in reference.get_inputs()}
        if int(feed["position"].item()) != frame:
            raise ValueError("Expected contiguous absolute positions from zero")
        if frame == 0:
            for name in cache_names:
                value = feed[name][:, :, :1000, :].copy()
                if np.any(value):
                    raise ValueError("Benchmark requires zero initial cache")
                ref_cache[name], quant_cache[name] = value.copy(), value.copy()
        hidden.append(feed["hidden"].astype("<f4").tobytes())
        expected = reference.run(None, dict(feed, **ref_cache))
        simulated = quantized.run(None, dict(feed, **quant_cache))
        original_file, quantized_file = root / f"reference_frame_{frame}.npz", root / f"quantized_frame_{frame}.npz"
        np.savez(original_file, **dict(zip(names, expected)))
        np.savez(quantized_file, **dict(zip(names, simulated)))
        for cache, outputs in ((ref_cache, expected), (quant_cache, simulated)):
            for name in cache_names:
                cache[name][:, :, frame:frame+1, :] = outputs[names.index("output_" + name)]
        records.append({"frame": frame, "sample": sample, "original_file": original_file.name,
                        "quantized_file": quantized_file.name, "original_sha256": sha256(original_file),
                        "quantized_sha256": sha256(quantized_file)})
    target = root / "hidden_recurrent.raw"
    target.write_bytes(b"".join(hidden))
    write(root / "phone_reference.json", {"format": "moshi-phone-reference-v1", "script_sha256": sha256(__file__),
          "candidate_dlc_sha256": state["output_sha256"], "source_id": clip, "frames": records,
          "hidden_file": target.name, "hidden_sha256": sha256(target),
          "scope": "Captured upstream hidden with each implementation's own recurrent K/V, not full LM"})
    print("Phone reference ready", len(records), flush=True)


if __name__ == "__main__":
    main()
