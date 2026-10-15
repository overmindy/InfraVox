"""Extend a passed DLC-derived simulation to the retained calibration set.

Check the remaining teacher-forced samples and three contiguous frames per clip.
The recurrent check updates only this shard's host K/V. It is not a full LM run.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from moshi_quantize_temporal_offline import sha256, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import onnxruntime as ort

    d = args.directory
    target_path = d / "calibration_validation.json"
    if target_path.exists():
        raise ValueError("Validation receipt exists")
    state = json.loads((d / "quantization.json").read_text())
    screen = json.loads((d / "cpu_simulation.json").read_text())
    if not screen["passed"] or sha256(state["output_dlc"]) != screen["candidate_dlc_sha256"]:
        raise ValueError("Require the same candidate and a passed initial screen")
    source = Path(state.get("reference_onnx", state["source_onnx"]))
    if sha256(source) != screen.get("reference_onnx_sha256", screen["source_onnx_sha256"]):
        raise ValueError("Reference source changed")
    simulation = d / "cpu_simulation.onnx"
    if not simulation.is_file() or not (d / "cpu_simulation.data").is_file():
        raise ValueError("Retain the initial DLC-derived simulation for this check")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(simulation), options, providers=["CPUExecutionProvider"])
    names = [v.name for v in reference.get_outputs()]
    input_names = [v.name for v in reference.get_inputs()]
    cache_names = [name for name in input_names if name.startswith("layer_")]
    if len(cache_names) != 4 or any("output_"+name not in names for name in cache_names):
        raise ValueError("Expected the existing four-cache delta ABI")
    limit = screen["relative_rms_limit"]
    report = {"format": "moshi-temporal-calibration-validation-v1", "script_sha256": sha256(__file__),
              "candidate_dlc_sha256": screen["candidate_dlc_sha256"], "reference_onnx_sha256": sha256(source),
              "simulation_onnx_sha256": sha256(simulation), "simulation_data_sha256": sha256(d/"cpu_simulation.data"),
              "relative_rms_limit": limit, "actual_dlc_execution": False, "device_verified": False,
              "independent_quality_test": False, "teacher_forced": copy.deepcopy(screen["samples"]),
              "recurrent": [], "status": "running",
              "scope": "Existing calibration clips only. Recurrent shard K/V with captured upstream hidden, not the full LM."}
    done = {(v["source_id"], v["frame"]) for v in report["teacher_forced"]}

    def load(sample):
        path = Path(state["calibration_manifest"]).parent/sample["file"]
        if sha256(path) != sample["sha256"]:
            raise ValueError("Captured calibration input changed")
        with np.load(path) as arrays:
            feed = {name: arrays[name] for name in input_names}
        for name in cache_names:
            feed[name] = feed[name][:, :, :1000, :].copy()
        return feed

    def compare(actual, expected):
        outputs = {}
        for name, got, target in zip(names, actual, expected):
            delta = got.astype(np.float64)-target.astype(np.float64)
            error = float(np.linalg.norm(delta)/max(np.linalg.norm(target.astype(np.float64)), 1e-12))
            finite = bool(np.isfinite(got).all())
            nonzero = int(np.count_nonzero(got))
            outputs[name] = {"relative_rms": error, "max_abs": float(np.max(np.abs(delta))),
                             "finite": finite, "nonzero": nonzero,
                             "passed": finite and error <= limit and (nonzero > 0 or not np.any(target))}
        return outputs

    for sample in state["samples"]:
        if (sample["source_id"], sample["frame"]) in done:
            continue
        feed = load(sample)
        outputs = compare(candidate.run(None, feed), reference.run(None, feed))
        report["teacher_forced"].append({"source_id": sample["source_id"], "frame": sample["frame"], "outputs": outputs})
        print("teacher", sample["source_id"], sample["frame"], max(v["relative_rms"] for v in outputs.values()), flush=True)
        write(target_path, report)
    for source_id in dict.fromkeys(v["source_id"] for v in state["samples"]):
        samples = {v["frame"]: v for v in state["samples"] if v["source_id"] == source_id}
        if not all(frame in samples for frame in (0,1,2)):
            raise ValueError("Require captured contiguous frames zero, one, and two")
        ref_cache, got_cache = {}, {}
        for frame in range(3):
            feed = load(samples[frame])
            if frame == 0:
                ref_cache = {name: feed[name].copy() for name in cache_names}
                got_cache = {name: feed[name].copy() for name in cache_names}
            ref_feed, got_feed = dict(feed, **ref_cache), dict(feed, **got_cache)
            expected, actual = reference.run(None, ref_feed), candidate.run(None, got_feed)
            outputs = compare(actual, expected)
            position = int(np.asarray(feed["position"]).reshape(-1)[0])
            if position != frame:
                raise ValueError("Captured absolute position differs from frame")
            slot = position % 1000
            preserved = True
            for states, values in ((ref_cache, expected), (got_cache, actual)):
                for name in cache_names:
                    value = values[names.index("output_"+name)]
                    if value.shape != (1,32,1,128):
                        raise ValueError("Cache output is not the existing delta shape")
                    before = states[name].copy()
                    states[name][:, :, slot:slot+1, :] = value
                    preserved &= bool(np.array_equal(states[name][:,:,:slot,:], before[:,:,:slot,:])
                                      and np.array_equal(states[name][:,:,slot+1:,:], before[:,:,slot+1:,:]))
            report["recurrent"].append({"source_id": source_id, "frame": frame, "slot": slot,
                                       "host_retained_slots_unchanged": preserved, "outputs": outputs})
            print("recurrent", source_id, frame, max(v["relative_rms"] for v in outputs.values()), flush=True)
            write(target_path, report)
    report["passed"] = all(v["passed"] for key in ("teacher_forced", "recurrent")
                           for row in report[key] for v in row["outputs"].values()) and all(
                               row["host_retained_slots_unchanged"] for row in report["recurrent"])
    report["status"] = "passed_not_device_verified" if report["passed"] else "rejected_by_calibration_check"
    report["teacher_forced_count"] = len(report["teacher_forced"])
    report["recurrent_count"] = len(report["recurrent"])
    write(target_path, report)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
