"""Calibrate one 8-layer Moshi Temporal part on real speech and gate it on held-out speakers.

Inputs come from moshi_temporal_capture.py (BF16 residual stream at the part boundaries).
Calibration samples are AR1 steps of the float part replayed over calibration clips, so the
KV history is realistic. SeqMSE picks LPBQ weight encodings, then activations are calibrated.

Gate on held-out clips, float part vs quantized simulation, each feeding back its own KV:
  local:   input = BF16 boundary hidden            (this part's error alone)
  chained: input = previous part's quantized output (accumulated error; saved for the next part)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from aimet_onnx import apply_seq_mse

from moshi_temporal_part import PAST, HostRing, TemporalPart, export_onnx, relative_rms, rope_tables
from moshi_temporal_quantize import WEIGHTS, create_sim, cuda_providers
from qai_hub_models.models.templates.moshi.model import load_moshi_lm

LAYERS_PER_PART = 8


def load_clips(capture: Path, split: str) -> list[tuple[str, np.ndarray]]:
    manifest = json.loads((capture / "manifest.json").read_text())
    return [(c["name"], np.load(capture / f"{c['name']}.npz")["hidden"]) for c in manifest if c["split"] == split]


@torch.no_grad()
def float_samples(part: TemporalPart, clips: list[np.ndarray], total: int, device: torch.device) -> list[dict]:
    """AR1 inputs at the first steps and evenly spaced later steps, with the float part's own KV.

    Step 0 (all-initial-token frame) has its own activation range and acts as an attention
    sink for every later step; leaving it out clips its K/V and corrupts the whole clip.
    """
    names, _ = part.names()
    per_clip = math.ceil(total / len(clips))
    samples = []
    for hidden in clips:
        steps = len(hidden)
        keep = {0, 1, 2, 3} | set(np.linspace(4, steps - 1, max(per_clip - 4, 1)).round().astype(int).tolist())
        ring = HostRing(len(part.layers), device)
        cos, sin = rope_tables(torch.arange(steps))
        for t in range(steps):
            mask, past = ring.inputs(t, device)
            x = torch.from_numpy(hidden[t]).to(device).reshape(1, 1, -1)
            inputs = (x, mask, cos[t].to(device), sin[t].to(device), *past)
            if t in keep:
                samples.append({n: v.cpu().numpy() for n, v in zip(names, inputs)})
            ring.write(t, part(*inputs)[1:])
    return samples


@torch.no_grad()
def run_float(part: TemporalPart, hidden: np.ndarray, device: torch.device) -> np.ndarray:
    ring = HostRing(len(part.layers), device)
    cos, sin = rope_tables(torch.arange(len(hidden)))
    outputs = []
    for t in range(len(hidden)):
        mask, past = ring.inputs(t, device)
        x = torch.from_numpy(hidden[t]).to(device).reshape(1, 1, -1)
        result = part(x, mask, cos[t].to(device), sin[t].to(device), *past)
        ring.write(t, result[1:])
        outputs.append(result[0].reshape(-1).cpu().numpy())
    return np.stack(outputs)


def run_sim(session, names: list[str], layers: int, hidden: np.ndarray) -> np.ndarray:
    """Quantized simulation with its own (quantize-dequantized) KV fed back through a host ring."""
    cpu = torch.device("cpu")
    ring = HostRing(layers, cpu)
    cos, sin = rope_tables(torch.arange(len(hidden)))
    outputs = []
    for t in range(len(hidden)):
        mask, past = ring.inputs(t, cpu)
        values = (hidden[t].reshape(1, 1, -1).astype(np.float32), mask.numpy(), cos[t].numpy(), sin[t].numpy(),
                  *[p.numpy() for p in past])
        result = session.run(None, dict(zip(names, values)))
        ring.write(t, tuple(torch.from_numpy(r) for r in result[1:]))
        outputs.append(result[0].reshape(-1))
    return np.stack(outputs)


def summarize(errors: list[float]) -> dict:
    a = np.asarray(errors)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "steps": len(a)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--part", type=int, required=True, choices=range(4))
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--seqmse-samples", type=int, default=32, help="0 disables SeqMSE")
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--seqmse-scope", choices=("large", "all"), default="large",
                        help="large: output/gate/up/down only; per-head q/k/v (768 tiny convs) keep min-max")
    parser.add_argument("--weights", default="lpbq4_w8vd", choices=WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    part_dir = args.work_dir / f"part{args.part}"
    part_dir.mkdir(parents=True, exist_ok=False)
    first = args.part * LAYERS_PER_PART
    report: dict = {"part": args.part, "layers": [first, first + LAYERS_PER_PART - 1], "weights": args.weights,
                    "past": PAST, "seqmse_samples": args.seqmse_samples, "num_candidates": args.num_candidates,
                    "seqmse_scope": args.seqmse_scope}
    timer = time.monotonic()

    lm = load_moshi_lm(model_dir=args.model_dir, device="cpu", dtype=torch.bfloat16)
    part = TemporalPart(lm, first, LAYERS_PER_PART).to(device).eval()
    del lm
    calibration = [h[:, args.part] for _, h in load_clips(args.capture_dir, "calibration")]
    held_out = load_clips(args.capture_dir, "held_out")
    samples = float_samples(part, calibration, args.calibration_samples, device)
    report["calibration"] = {"clips": len(calibration), "samples": len(samples)}

    onnx_path = part_dir / "onnx" / "model.onnx"
    report["onnx_bytes"] = export_onnx(part, onnx_path)
    part = part.to(device)
    sim = create_sim(onnx.load(str(onnx_path)), args.weights, "htp_v81", cuda_providers())
    if args.seqmse_samples:
        start = time.monotonic()
        exclude = [op.name for op in sim.connected_graph.ordered_ops
                   if args.seqmse_scope == "large" and re.search(r"/(query|key|value)\.\d+/", op.name)]
        apply_seq_mse(sim, samples[: args.seqmse_samples], num_candidates=args.num_candidates,
                      nodes_to_exclude=exclude)
        report["seqmse_s"] = time.monotonic() - start
        report["seqmse_excluded_ops"] = len(exclude)
    start = time.monotonic()
    sim.compute_encodings(samples)
    report["compute_encodings_s"] = time.monotonic() - start
    name = f"temporal_part_{args.part}"
    sim.export(str(part_dir), name, export_model=False)
    del samples

    names, _ = part.names()
    previous = args.work_dir / f"part{args.part - 1}"
    gate = {"local": [], "chained": [], "float_vs_bf16": []}
    per_clip = []
    for clip, hidden in held_out:
        local_in = hidden[:, args.part]
        chained_in = np.load(previous / f"chained_{clip}.npy") if args.part else local_in
        reference_next = torch.from_numpy(hidden[:, args.part + 1])
        float_local = torch.from_numpy(run_float(part, local_in, device))
        sim_local = torch.from_numpy(run_sim(sim.session, names, LAYERS_PER_PART, local_in))
        float_chained = float_local if not args.part else torch.from_numpy(run_float(part, chained_in, device))
        sim_chained = sim_local if not args.part else torch.from_numpy(run_sim(sim.session, names, LAYERS_PER_PART, chained_in))
        np.save(part_dir / f"chained_{clip}.npy", sim_chained.numpy())
        local = [relative_rms(sim_local[t], float_local[t]) for t in range(len(local_in))]
        # Chained error is measured against the float model fed the BF16 input, i.e. what the
        # unquantized Temporal stack would have produced at this boundary.
        chained = [relative_rms(sim_chained[t], float_local[t]) for t in range(len(local_in))]
        floor = [relative_rms(float_local[t], reference_next[t]) for t in range(len(local_in))]
        gate["local"] += local
        gate["chained"] += chained
        gate["float_vs_bf16"] += floor
        per_clip.append({"clip": clip, "local": summarize(local), "chained": summarize(chained),
                         "float_vs_bf16": summarize(floor)})
        print(json.dumps(per_clip[-1]), flush=True)
    report["gate"] = {k: summarize(v) for k, v in gate.items()}
    report["per_clip"] = per_clip
    report["elapsed_s"] = time.monotonic() - timer
    (part_dir / "calibrate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"part": args.part, "gate": report["gate"]}), flush=True)


if __name__ == "__main__":
    main()
