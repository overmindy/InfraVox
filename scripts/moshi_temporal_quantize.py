"""Calibrate one exported Moshi Temporal part with AIMET-ONNX like the official w4a16 LLMs.

int4 LPBQ weights (block 64), int16 activations, 8-bit symmetric KV cache I/O with
input/output quantizers tied, and 8-bit second inputs for 16x8 attention MatMuls.
Writes only the encodings file; the converter reads it next to the float ONNX.
"""

from __future__ import annotations

import argparse
import json
import logging
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-dir", type=Path, required=True)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--weights", choices=("lpbq4", "pc4", "pc8"), default="lpbq4",
                        help="int4 LPBQ (block 64), int4 per-channel, or int8 per-channel weights")
    parser.add_argument("--config", default="htp_v81")
    parser.add_argument("--name", default="model", help="Encodings file prefix")
    args = parser.parse_args()
    AimetLogger.set_level_for_all_areas(logging.WARNING)
    quantsim.op_types_to_tie_qtzrs = ["Concat"]
    quantsim._tie_qtzrs = True
    quantsim.op_outputs_to_ignore.append("Slice")
    quantsim.op_outputs_to_ignore.append("Constant")
    qs.encoding_version = "1.0.0"

    start = time.monotonic()
    model = onnx.load(str(args.part_dir / "onnx" / "model.onnx"))
    samples = [dict(np.load(path)) for path in sorted(args.part_dir.glob("calibration_*.npz"))]
    if not samples:
        raise SystemExit("No calibration_*.npz samples found")
    sim = QuantizationSimModel(model, param_type="int8" if args.weights == "pc8" else "int4",
                               activation_type="int16", quant_scheme=QuantScheme.min_max,
                               config_file=args.config, providers=args.providers.split(","))
    if args.weights == "lpbq4":
        set_lpbq_for_params(sim, bitwidth=4, block_size=64, op_types={"Conv", "Gemm", "MatMul"})
    kv_io_map = _get_kv_io_map(sim)
    sim._tie_quantizers_for_op_types(["Concat"])
    sim._rebuild_session()
    _set_tensors_to_output_8b_sym(sim, list(kv_io_map) + list(kv_io_map.values()))
    _tie_quantizers_for_kv_cache(sim, kv_io_map)
    _set_matmul_second_input_to_8b(sim)
    sim.compute_encodings(samples)
    sim.export(str(args.part_dir), args.name, export_model=False)
    report = {"samples": len(samples), "kv_pairs": len(kv_io_map), "providers": sim.session.get_providers(),
              "weights": args.weights, "config": args.config, "elapsed_s": time.monotonic() - start,
              "encodings": str(args.part_dir / f"{args.name}.encodings")}
    (args.part_dir / f"{args.name}.quantize.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
