"""Score phone Temporal output (moshi_temporal_chain_probe) against the unquantized model.

--export writes each held-out clip's captured layer-0 stream as raw float32 for the phone.
--score decodes the phone's final hidden streams with the BF16 head and DepFormer
(teacher-forced, same metric as moshi_temporal_segment_eval) and compares tokens and
final-hidden relative RMS with the unquantized Temporal run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moshi_temporal_calibrate import load_clips
from moshi_temporal_segment_eval import compare, decode_stream, run_float_layers
from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitDepFormer, TemporalBlock, TemporalHead
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--stream-dir", type=Path, required=True, help="<clip>.in.f32 / <clip>.out.f32")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    clips = load_clips(args.capture_dir, "held_out")
    if args.export:
        args.stream_dir.mkdir(parents=True, exist_ok=True)
        for name, hidden in clips:
            hidden[:, 0].astype(np.float32).tofile(args.stream_dir / f"{name}.in.f32")
        print(json.dumps({"exported": [name for name, _ in clips]}))
        return

    device = torch.device(args.device)
    lm = load_moshi_lm(model_dir=args.model_dir, device=device, dtype=torch.bfloat16)
    blocks = [TemporalBlock(layer) for layer in lm.transformer.layers]
    head, depformer = TemporalHead(lm), ExplicitDepFormer(lm)
    references, outputs, hidden_errors = [], [], []
    for name, hidden in clips:
        phone_path = args.stream_dir / f"{name}.out.f32"
        if not phone_path.exists():
            continue
        phone = np.fromfile(phone_path, dtype=np.float32).reshape(len(hidden), -1)
        float_final = run_float_layers(blocks, range(len(blocks)), hidden[:, 0], device)
        reference = decode_stream(head, depformer, torch.from_numpy(float_final).to(device, torch.bfloat16), None, device)
        output = decode_stream(head, depformer, torch.from_numpy(phone).to(device, torch.bfloat16), reference["text"], device)
        references.append(reference)
        outputs.append(output)
        errors = np.linalg.norm(phone - float_final, axis=1) / np.linalg.norm(float_final, axis=1)
        hidden_errors.append(errors)
        print(json.dumps({"clip": name, **compare([output], [reference]),
                          "final_hidden_rel_rms_median": round(float(np.median(errors)), 4)}), flush=True)
    print(json.dumps({"clips": len(outputs), **compare(outputs, references),
                      "final_hidden_rel_rms_median": round(float(np.median(np.concatenate(hidden_errors))), 4),
                      "frames": int(sum(len(o["text"]) for o in outputs))}), flush=True)


if __name__ == "__main__":
    main()
