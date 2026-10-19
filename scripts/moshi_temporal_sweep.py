"""Fast quantization screening on a small Temporal part: load once, try many configs.

One process loads the model, replays float calibration samples, exports a 2-layer ONNX
and float references once, then for each variant only rebuilds QuantSim + encodings
(no SeqMSE) and reports held-out error with weights-only / activations-only splits.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from aimet_onnx import apply_seq_mse

from moshi_temporal_calibrate import float_samples, load_clips, run_float, run_sim
from moshi_temporal_part import TemporalPart, export_onnx, relative_rms
from moshi_temporal_quantize import create_sim, cuda_providers, hold_at_int8
from qai_hub_models.models.templates.llm._utils import _get_kv_io_map
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


def variant_sim(model: onnx.ModelProto, name: str, samples: list[dict]):
    weights = {"w8": "pc8", "pc4": "pc4", "w8_value_down": "lpbq4_w8vd"}.get(name, "lpbq4")
    sim = create_sim(model, weights, "htp_v81", cuda_providers(), block_size=32 if name == "lpbq32" else 64)
    held = {"w8_down": r"down", "w8_value": r"value\.\d+",
            "w8_qk": r"(query|key)\.\d+", "w8_qkv": r"(query|key|value)\.\d+"}
    if name in held:
        hold_at_int8(sim, held[name])
    if name == "seqmse":
        exclude = [op.name for op in sim.connected_graph.ordered_ops if re.search(r"/(query|key|value)\.\d+/", op.name)]
        apply_seq_mse(sim, samples[:32], num_candidates=20, nodes_to_exclude=exclude)
    if name == "kv16":
        kv = _get_kv_io_map(sim)
        for tensor in list(kv) + list(kv.values()):
            quantizer = sim._get_enabled_quantizer(tensor)
            if quantizer:
                quantizer.set_bitwidth(16)
                quantizer.use_symmetric_encodings = False
    if name == "matmul16":
        for op in sim.connected_graph.ordered_ops:
            if op.type == "MatMul":
                quantizer = sim.qc_quantize_op_dict.get(op.inputs[1].name)
                if quantizer and quantizer.enabled and quantizer.bitwidth == 8:
                    quantizer.set_bitwidth(16)
                    quantizer.use_symmetric_encodings = False
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--calibration-clips", type=int, default=8)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--gate-clips", type=int, default=2)
    parser.add_argument("--gate-steps", type=int, default=64)
    parser.add_argument("--variants", default="base,kv16,matmul16,w8")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    lm = load_moshi_lm(model_dir=args.model_dir, device="cpu", dtype=torch.bfloat16)
    part = TemporalPart(lm, args.first_layer, args.layers).to(device).eval()
    del lm
    boundary = args.first_layer // 8  # Captured inputs exist at multiples of 8 layers.
    if args.first_layer % 8:
        raise SystemExit("first-layer must be a captured boundary (0, 8, 16, 24)")
    calibration = [h[:, boundary] for _, h in load_clips(args.capture_dir, "calibration")[: args.calibration_clips]]
    held_out = [h[: args.gate_steps, boundary] for _, h in load_clips(args.capture_dir, "held_out")[: args.gate_clips]]
    samples = float_samples(part, calibration, args.calibration_samples, device)
    references = [torch.from_numpy(run_float(part, h, device)) for h in held_out]
    names, _ = part.names()
    onnx_path = args.work_dir / f"sweep_l{args.first_layer}_{args.layers}" / "model.onnx"
    if not onnx_path.exists():
        export_onnx(part, onnx_path)
    print(json.dumps({"setup_s": round(time.monotonic() - start), "samples": len(samples)}), flush=True)

    for name in args.variants.split(","):
        t0 = time.monotonic()
        sim = variant_sim(onnx.load(str(onnx_path)), name, samples)  # QuantSim rewrites the model in place.
        sim.compute_encodings(samples)
        params = set(sim.param_names)
        enabled = {n for n, q in sim.qc_quantize_op_dict.items() if q.enabled}
        result = {"variant": name}
        for label, selected in (("all", enabled), ("weights", enabled & params), ("activations", enabled - params)):
            for n, q in sim.qc_quantize_op_dict.items():
                q.enabled = n in selected
            errors = []
            for hidden, reference in zip(held_out, references):
                out = torch.from_numpy(run_sim(sim.session, names, args.layers, hidden))
                errors += [relative_rms(out[t], reference[t]) for t in range(len(hidden))]
            result[label] = {"median": round(float(np.median(errors)), 4), "p95": round(float(np.percentile(errors, 95)), 4),
                             "step0": round(errors[0], 4)}
        result["seconds"] = round(time.monotonic() - t0)
        print(json.dumps(result), flush=True)
        del sim


if __name__ == "__main__":
    main()
