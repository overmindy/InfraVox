"""Teacher-forced token impact of fully quantized Temporal parts (AIMET QuantSim) on held-out speech.

Each listed part is replaced by its QuantSim (weights + int16 activations + 8-bit KV, own KV
ring); the remaining Temporal layers, head and DepFormer run in BF16. The run starts at the
first quantized part's captured BF16 boundary, and the reference is the same computation
with no quantization, so every frame sees identical history.

  each:     one part at a time (isolates each part)
  together: all listed parts chained (accumulated error)

Encodings are reused from --work-dir when present, otherwise calibrated (no SeqMSE) and saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from aimet_onnx.quantsim import load_encodings_to_sim

from moshi_temporal_calibrate import LAYERS_PER_PART, float_samples, load_clips, run_sim
from moshi_temporal_part import HostRing, TemporalPart, export_onnx, rope_tables
from moshi_temporal_quantize import WEIGHTS, create_sim, cuda_providers
from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitDepFormer, TemporalBlock, TemporalHead
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


def build_sim(lm, part: int, args, device: torch.device):
    encodings = args.work_dir / f"temporal_part_{part}.encodings"
    float_part = TemporalPart(lm, part * LAYERS_PER_PART, LAYERS_PER_PART).to(device).eval()
    names, _ = float_part.names()
    samples = None
    if not encodings.exists():
        clips = [h[: args.calibration_steps, part] for _, h in load_clips(args.capture_dir, "calibration")]
        samples = float_samples(float_part, clips[: args.calibration_clips], args.calibration_samples, device)
    onnx_path = args.work_dir / f"segment{part}_{os.getpid()}" / "model.onnx"
    export_onnx(float_part, onnx_path)
    del float_part
    model = onnx.load(str(onnx_path))
    shutil.rmtree(onnx_path.parent)  # Weights now live in memory; keeps peak disk at one part.
    sim = create_sim(model, args.weights, "htp_v81", cuda_providers())
    if samples is None:
        load_encodings_to_sim(sim, str(encodings))
    else:
        sim.compute_encodings(samples)
        sim.export(str(args.work_dir), f"temporal_part_{part}", export_model=False)
    return sim, names


@torch.no_grad()
def run_clip(blocks, head, depformer, sims: dict, captured: np.ndarray, start: int,
             text_tokens: np.ndarray | None, device: torch.device) -> dict:
    caches = {i: blocks[i].empty_cache() for i in range(start * LAYERS_PER_PART, len(blocks))}
    rings = {p: HostRing(LAYERS_PER_PART, torch.device("cpu")) for p in sims}
    cos, sin = rope_tables(torch.arange(len(captured)))
    finals = []
    for t in range(len(captured)):
        hidden = torch.from_numpy(captured[t, start]).to(device, torch.bfloat16).reshape(1, 1, -1)
        position = torch.tensor([t], device=device)
        for part in range(start, len(blocks) // LAYERS_PER_PART):
            if part in sims:
                session, names = sims[part]
                mask, past = rings[part].inputs(t, torch.device("cpu"))
                values = (hidden.float().cpu().numpy(), mask.numpy(), cos[t].numpy(), sin[t].numpy(),
                          *[p.numpy() for p in past])
                result = session.run(None, dict(zip(names, values)))
                rings[part].write(t, tuple(torch.from_numpy(r) for r in result[1:]))
                hidden = torch.from_numpy(result[0]).to(device, torch.bfloat16)
                continue
            for i in range(part * LAYERS_PER_PART, (part + 1) * LAYERS_PER_PART):
                hidden, caches[i] = blocks[i](hidden, caches[i], position)
        finals.append(hidden.reshape(-1))
    return decode_stream(head, depformer, torch.stack(finals), text_tokens, device)


@torch.no_grad()
def run_together(lm, blocks, head, depformer, parts: list[int], args, held_out: list[np.ndarray],
                 references: list[dict], device: torch.device) -> list[dict]:
    """Chain quantized parts one at a time over whole clips, so only one QuantSim is alive."""
    start = min(parts)
    streams = [h[:, start] for h in held_out]
    for part in range(start, len(blocks) // LAYERS_PER_PART):
        if part in parts:
            sim, names = build_sim(lm, part, args, device)
            streams = [run_sim(sim.session, names, LAYERS_PER_PART, s) for s in streams]
            del sim
            gc.collect()  # Frees AIMET's multi-GB TMPDIR model copy before the next export.
            continue
        layers = range(part * LAYERS_PER_PART, (part + 1) * LAYERS_PER_PART)
        streams = [run_float_layers(blocks, layers, s, device) for s in streams]
    return [decode_stream(head, depformer, torch.from_numpy(s).to(device, torch.bfloat16), r["text"], device)
            for s, r in zip(streams, references)]


@torch.no_grad()
def decode_stream(head, depformer, final: torch.Tensor, text_tokens: np.ndarray | None, device: torch.device) -> dict:
    """Text argmax and greedy DepFormer audio tokens for each frame of the final Temporal stream."""
    text, audio, logprobs = [], [], []
    for t in range(len(final)):
        temporal, text_logits = head(final[t].reshape(1, 1, -1))
        token = text_logits.reshape(-1).argmax()
        condition = torch.tensor([text_tokens[t]], device=device) if text_tokens is not None else token.reshape(1)
        audio_tokens, _ = depformer(condition, temporal)
        text.append(int(token))
        audio.append(audio_tokens.reshape(-1).cpu().numpy())
        logprobs.append(torch.log_softmax(text_logits.reshape(-1).float(), dim=-1).cpu())
    return {"text": np.array(text), "audio": np.stack(audio), "logprobs": torch.stack(logprobs)}


@torch.no_grad()
def run_float_layers(blocks, layers: range, stream: np.ndarray, device: torch.device) -> np.ndarray:
    caches = {i: blocks[i].empty_cache() for i in layers}
    out = []
    for t in range(len(stream)):
        hidden = torch.from_numpy(stream[t]).to(device, torch.bfloat16).reshape(1, 1, -1)
        position = torch.tensor([t], device=device)
        for i in layers:
            hidden, caches[i] = blocks[i](hidden, caches[i], position)
        out.append(hidden.reshape(-1).float().cpu().numpy())
    return np.stack(out)


def compare(outputs: list[dict], references: list[dict]) -> dict:
    text = np.concatenate([o["text"] == r["text"] for o, r in zip(outputs, references)])
    cb0 = np.concatenate([o["audio"][:, 0] == r["audio"][:, 0] for o, r in zip(outputs, references)])
    audio = np.concatenate([o["audio"] == r["audio"] for o, r in zip(outputs, references)])
    kl = np.concatenate([(r["logprobs"].exp() * (r["logprobs"] - o["logprobs"])).sum(-1).numpy()
                         for o, r in zip(outputs, references)])
    return {"text_top1": round(float(text.mean()), 4), "audio_cb0": round(float(cb0.mean()), 4),
            "audio_all": round(float(audio.mean()), 4), "text_kl_mean": round(float(kl.mean()), 5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--parts", default="0,1,2,3")
    parser.add_argument("--mode", choices=("each", "together"), default="each")
    parser.add_argument("--weights", choices=WEIGHTS, default="lpbq4")
    parser.add_argument("--held-out-clips", type=int, default=4)
    parser.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th selected clip from i (multi-GPU)")
    parser.add_argument("--calibration-clips", type=int, default=8)
    parser.add_argument("--calibration-steps", type=int, default=120)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    parts = [int(p) for p in args.parts.split(",")]
    lm = load_moshi_lm(model_dir=args.model_dir, device=device, dtype=torch.bfloat16)
    blocks = [TemporalBlock(layer) for layer in lm.transformer.layers]
    head, depformer = TemporalHead(lm), ExplicitDepFormer(lm)
    clips = load_clips(args.capture_dir, "held_out")
    stride = max(1, len(clips) // args.held_out_clips)  # Spread over test-clean and test-other.
    shard, shards = (int(x) for x in args.shard.split("/"))
    held_out = [h for _, h in clips[::stride][: args.held_out_clips]][shard::shards]

    groups = [[p] for p in parts] if args.mode == "each" else [parts]
    for group in groups:
        start_time = time.monotonic()
        start = min(group)
        references = [run_clip(blocks, head, depformer, {}, h, start, None, device) for h in held_out]
        if len(group) == 1:
            sim, names = build_sim(lm, group[0], args, device)
            outputs = [run_clip(blocks, head, depformer, {group[0]: (sim.session, names)}, h, start, r["text"], device)
                       for h, r in zip(held_out, references)]
            del sim
        else:
            outputs = run_together(lm, blocks, head, depformer, group, args, held_out, references, device)
        print(json.dumps({"parts": group, "weights": args.weights, **compare(outputs, references),
                          "frames": int(sum(len(h) for h in held_out)), "seconds": round(time.monotonic() - start_time)}),
              flush=True)


if __name__ == "__main__":
    main()
