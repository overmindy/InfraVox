"""Quantize the 8 DepFormer codebook graphs with AIMET (int16 activations; weights int8 per-channel,
int4 per-channel or int4 LPBQ) and gate them on held-out speech against BF16 greedy tokens.

DepFormer inputs come from moshi_temporal_capture.py: the BF16 layer-31 output goes through the
BF16 head, whose normalized hidden and argmax text token condition the DepFormer. Scores per frame:
  chain:   each codebook is fed the previous codebook's own token and cache (what the phone runs)
  forced:  each codebook is fed the BF16 reference token and the FP32 cache built from it
for FP32 ONNX (floor), the deployed legacy AI Hub QDQ graphs, and the new QuantSim.
Writes WORK/depformer_codebook_{k}/onnx/model.onnx plus .encodings for moshi_quantized_build.sh.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme, load_encodings_to_sim, set_lpbq_for_params

from moshi_export_lm_onnx import _export
from moshi_temporal_calibrate import load_clips
from moshi_temporal_quantize import cuda_providers
from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitDepFormer, ExplicitDepFormerStep, TemporalHead
from qai_hub_models.models.templates.moshi.model import load_moshi_lm

BOOKS = 8
CACHE = [f"layer_{layer}_{kind}" for layer in range(6) for kind in ("key", "value")]
INPUTS = ["previous_token", "temporal", *CACHE]
OUTPUTS = ["audio_token", *(f"output_{name}" for name in CACHE)]


def create_sim(model: onnx.ModelProto, weights: str) -> QuantizationSimModel:
    sim = QuantizationSimModel(model, param_type="int8" if weights == "w8" else "int4", activation_type="int16",
                               quant_scheme=QuantScheme.min_max, config_file="htp_v81", providers=cuda_providers())
    if weights == "lpbq4":
        four_bit = set()
        for op in sim.connected_graph.ordered_ops:
            weight = sim.get_op_quantizers(op)[2].get("weight") if op.type in {"Conv", "Gemm", "MatMul"} else None
            if weight is not None and weight.bitwidth == 4:
                four_bit.add(op.name)
        set_lpbq_for_params(sim, bitwidth=4, block_size=64, nodes_to_include=four_bit)
    return sim


class StepWithoutLogits(torch.nn.Module):
    """The runner only reads the token and caches; dropping logits saves an output copy."""

    def __init__(self, step: ExplicitDepFormerStep) -> None:
        super().__init__()
        self.step = step

    def forward(self, previous_token, temporal, *caches):
        token, _, *updated = self.step(previous_token, temporal, *caches)
        return token, *updated


@torch.no_grad()
def conditions(lm, head, depformer, clips: list[np.ndarray], device) -> list[list[tuple]]:
    """Per frame: (temporal float32 [1,1,4096], BF16 text token, BF16 greedy audio tokens [8])."""
    result = []
    for hidden in clips:
        frames = []
        for t in range(len(hidden)):
            final = torch.from_numpy(hidden[t, 4]).to(device, torch.bfloat16).reshape(1, 1, -1)
            temporal, text_logits = head(final)
            text = text_logits.reshape(-1).argmax().reshape(1)
            audio, _ = depformer(text, temporal)
            frames.append((temporal.float().cpu().numpy(), int(text), audio.reshape(-1).cpu().numpy()))
        result.append(frames)
    return result


def run_chain(run_step, temporal: np.ndarray, text: int, forced: np.ndarray | None = None,
              forced_caches: list | None = None, record: list | None = None) -> np.ndarray:
    """Greedy chain over 8 codebooks; `forced` replaces own tokens and caches with the reference."""
    previous = np.array([text], np.int64)
    caches = [np.zeros((1, 16, 8, 64), np.float32) for _ in CACHE]
    tokens = []
    for book in range(BOOKS):
        if forced is not None:
            previous = np.array([text if book == 0 else forced[book - 1]], np.int64)
            caches = forced_caches[book]
        feed = dict(zip(INPUTS, [previous, temporal, *caches]))
        if record is not None:
            record.append(feed)
        token, *caches = run_step(book, feed)
        tokens.append(int(token.reshape(-1)[0]))
        previous = token.reshape(1).astype(np.int64)
    return np.array(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=Path("/data3/yejialei/moshi-lm-guarded-dlc-set-v2"))
    parser.add_argument("--calibration-clips", type=int, default=8)
    parser.add_argument("--calibration-frames", type=int, default=32, help="Evenly spaced frames per clip")
    parser.add_argument("--held-out-clips", type=int, default=8)
    parser.add_argument("--no-legacy", action="store_true", help="Skip the slow CPU legacy QDQ reference")
    parser.add_argument("--weights", choices=("w8", "pc4", "lpbq4"), default="w8")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    lm = load_moshi_lm(model_dir=args.model_dir, device=device, dtype=torch.bfloat16)
    head, depformer = TemporalHead(lm), ExplicitDepFormer(lm)
    steps = [copy.deepcopy(ExplicitDepFormerStep(depformer, k)).to(device).float().eval() for k in range(BOOKS)]

    def fp32(book, feed):
        with torch.no_grad():
            inputs = [torch.from_numpy(feed[name]).to(device) for name in INPUTS]
            token, _, *caches = steps[book](*inputs)
        return [token.cpu().numpy(), *(c.cpu().numpy() for c in caches)]

    calibration = [h for _, h in load_clips(args.capture_dir, "calibration")][: args.calibration_clips]
    held_out = [h for _, h in load_clips(args.capture_dir, "held_out")][: args.held_out_clips]
    calibration_frames = []
    for frames in conditions(lm, head, depformer, calibration, device):
        picks = np.linspace(0, len(frames) - 1, args.calibration_frames).round().astype(int)
        calibration_frames += [frames[i] for i in picks]
    evaluation = [f for clip in conditions(lm, head, depformer, held_out, device) for f in clip]
    del lm, head, depformer  # The FP32 step copies are independent; frees ~15 GB for the ORT sessions.
    gc.collect()
    torch.cuda.empty_cache()

    samples = [[] for _ in range(BOOKS)]
    for temporal, text, _ in calibration_frames:
        record = []
        run_chain(fp32, temporal, text, record=record)
        for book in range(BOOKS):
            samples[book].append(record[book])

    sessions, legacy = [], []
    for book in range(BOOKS):
        name = f"depformer_codebook_{book}"
        onnx_path = args.work_dir / name / "onnx" / "model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if not onnx_path.exists():
            example = [torch.from_numpy(samples[book][0][n]) for n in INPUTS]
            _export(StepWithoutLogits(copy.deepcopy(steps[book]).cpu()), tuple(example), onnx_path, INPUTS, OUTPUTS)
        sim = create_sim(onnx.load(str(onnx_path)), args.weights)
        encodings = args.work_dir / name / f"{name}.encodings"
        if encodings.exists():
            load_encodings_to_sim(sim, str(encodings))
        else:
            sim.compute_encodings(samples[book])
            sim.export(str(args.work_dir / name), name, export_model=False)
        sessions.append(sim)
        if not args.no_legacy:
            source = json.loads((args.legacy_root / "graphs" / name / "source.json").read_text())["onnx"]
            # The CUDA provider has no opset-21 Squeeze kernel for these AI Hub QDQ graphs.
            legacy.append(ort.InferenceSession(source, providers=["CPUExecutionProvider"]))
        print(json.dumps({"calibrated": name, "samples": len(samples[book]),
                          "seconds": round(time.monotonic() - start)}), flush=True)

    def quantized(book, feed):
        return sessions[book].session.run(None, feed)  # QuantSim renames outputs; order is kept.

    def old(book, feed):
        return legacy[book].run(OUTPUTS, feed)

    keys = ["fp32_chain", "quant_chain", "quant_forced"] + ([] if args.no_legacy else ["legacy_chain", "legacy_forced"])
    results = {key: [] for key in keys}
    references = []
    for temporal, text, reference in evaluation:
        results["fp32_chain"].append(run_chain(fp32, temporal, text))
        forced_caches, previous_caches = [], [np.zeros((1, 16, 8, 64), np.float32) for _ in CACHE]
        for book in range(BOOKS):
            forced_caches.append(previous_caches)
            previous = np.array([text if book == 0 else reference[book - 1]], np.int64)
            _, *previous_caches = fp32(book, dict(zip(INPUTS, [previous, temporal, *previous_caches])))
        results["quant_chain"].append(run_chain(quantized, temporal, text))
        results["quant_forced"].append(run_chain(quantized, temporal, text, reference, forced_caches))
        if not args.no_legacy:
            results["legacy_chain"].append(run_chain(old, temporal, text))
            results["legacy_forced"].append(run_chain(old, temporal, text, reference, forced_caches))
        references.append(reference)
    references = np.stack(references)
    report = {"weights": args.weights, "frames": len(references), "calibration_frames": len(calibration_frames)}
    for key, tokens in results.items():
        match = np.stack(tokens) == references
        report[key] = {"all8": round(float(match.all(axis=1).mean()), 4),
                       "per_codebook": [round(float(m), 4) for m in match.mean(axis=0)]}
    report["seconds"] = round(time.monotonic() - start)
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
