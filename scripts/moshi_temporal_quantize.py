"""Calibrate one exported Moshi Temporal part with AIMET-ONNX like the official w4a16 LLMs.

int4 LPBQ weights (block 64), int16 activations, 8-bit symmetric KV cache I/O with
input/output quantizers tied, and 8-bit second inputs for 16x8 attention MatMuls.
Writes only the encodings file; the converter reads it next to the float ONNX.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import aimet_onnx.common.quantsim as qs
import numpy as np
import onnx
from aimet_onnx import quantsim
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme, set_lpbq_for_params

from qai_hub_models.models.templates.llm._utils import (
    _get_kv_io_map,
    _set_matmul_second_input_to_8b,
    _set_tensors_to_output_8b_sym,
    _tie_quantizers_for_kv_cache,
)


def cuda_providers() -> list:
    # TF32 matmuls perturb the simulation by ~1e-4 relative; keep full FP32.
    return [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]


def hold_at_int8(sim: QuantizationSimModel, pattern: str) -> int:
    """Official int8 parameter hold: 8-bit symmetric per-channel, no LPBQ blocks."""
    held = 0
    for name in sim.param_names:
        quantizer = sim.qc_quantize_op_dict.get(name)
        if quantizer is None or not re.search(pattern, name):
            continue
        quantizer.set_bitwidth(8)
        quantizer.quant_info.blockSize = 0
        quantizer.quant_info.blockAxis = -1
        quantizer.use_symmetric_encodings = True
        quantizer.enable_per_channel_quantization()
        held += 1
    return held


WEIGHTS = ("lpbq4_w8vd", "lpbq4", "pc4", "pc8")


def create_sim(model: onnx.ModelProto, weights: str, config: str, providers: list,
               block_size: int = 64) -> QuantizationSimModel:
    """QuantSim configured like the official w4a16 LLM recipes, before any calibration.

    lpbq4_w8vd (the chosen layout) holds attention value and MLP down projections at int8:
    teacher-forced audio cb0 agreement 98.4% vs 95.9% for all-LPBQ on held-out speech.
    """
    AimetLogger.set_level_for_all_areas(logging.WARNING)
    quantsim.op_types_to_tie_qtzrs = ["Concat"]
    quantsim._tie_qtzrs = True
    if "Slice" not in quantsim.op_outputs_to_ignore:
        quantsim.op_outputs_to_ignore += ["Slice", "Constant"]
    qs.encoding_version = "1.0.0"
    sim = QuantizationSimModel(model, param_type="int8" if weights == "pc8" else "int4",
                               activation_type="int16", quant_scheme=QuantScheme.min_max,
                               config_file=config, providers=providers)
    if weights.startswith("lpbq4"):
        # Hold int8 weights first; LPBQ then applies only to the remaining 4-bit weights,
        # as in the official recipe (a held LPBQ quantizer cannot export its encodings).
        if weights == "lpbq4_w8vd":
            hold_at_int8(sim, r"(value\.\d+|down)")
        four_bit = set()
        for op in sim.connected_graph.ordered_ops:
            weight = sim.get_op_quantizers(op)[2].get("weight") if op.type in {"Conv", "Gemm", "MatMul"} else None
            if weight is not None and weight.bitwidth == 4:
                four_bit.add(op.name)
        set_lpbq_for_params(sim, bitwidth=4, block_size=block_size, nodes_to_include=four_bit)
    kv_io_map = _get_kv_io_map(sim)
    sim._tie_quantizers_for_op_types(["Concat"])
    sim._rebuild_session()
    _set_tensors_to_output_8b_sym(sim, list(kv_io_map) + list(kv_io_map.values()))
    _tie_quantizers_for_kv_cache(sim, kv_io_map)
    _set_matmul_second_input_to_8b(sim)
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-dir", type=Path, required=True)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--weights", choices=WEIGHTS, default="lpbq4_w8vd",
                        help="LPBQ int4 (+int8 value/down), int4 per-channel, or int8 per-channel weights")
    parser.add_argument("--config", default="htp_v81")
    parser.add_argument("--name", default="model", help="Encodings file prefix")
    args = parser.parse_args()

    start = time.monotonic()
    model = onnx.load(str(args.part_dir / "onnx" / "model.onnx"))
    samples = [dict(np.load(path)) for path in sorted(args.part_dir.glob("calibration_*.npz"))]
    if not samples:
        raise SystemExit("No calibration_*.npz samples found")
    providers = cuda_providers() if args.providers.startswith("CUDA") else args.providers.split(",")
    sim = create_sim(model, args.weights, args.config, providers)
    kv_io_map = _get_kv_io_map(sim)
    sim.compute_encodings(samples)
    sim.export(str(args.part_dir), args.name, export_model=False)
    report = {"samples": len(samples), "kv_pairs": len(kv_io_map), "providers": sim.session.get_providers(),
              "weights": args.weights, "config": args.config, "elapsed_s": time.monotonic() - start,
              "encodings": str(args.part_dir / f"{args.name}.encodings")}
    (args.part_dir / f"{args.name}.quantize.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
