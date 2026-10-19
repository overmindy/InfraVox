"""Teacher-forced token screen of Temporal weight quantization on held-out speech (PyTorch only).

Per frame, the 32 Temporal layers (fake-quantized weights: LPBQ int4 block 64 or int8
per-channel, min-max like AIMET) consume the BF16 run's captured layer-0 input, so every
variant sees the same token history. Text argmax and the 8 greedy DepFormer audio tokens
(conditioned on the unquantized text token) are compared with the unquantized run.
Activation quantization is not simulated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitDepFormer, TemporalBlock, TemporalHead
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


def int8_per_channel(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    return (weight / scale).round().clamp(-128, 127) * scale


def lpbq_int4(weight: torch.Tensor, block: int = 64) -> torch.Tensor:
    """int4 blocks whose scales are integer multiples (1..16) of one per-channel scale."""
    out_features, in_features = weight.shape
    blocks = weight.reshape(out_features, in_features // block, block)
    block_scale = blocks.abs().amax(dim=2).clamp_min(1e-12) / 7
    channel_scale = block_scale.amax(dim=1, keepdim=True) / 16
    multiplier = (block_scale / channel_scale).ceil().clamp(1, 16)
    scale = (multiplier * channel_scale)[..., None]
    return ((blocks / scale).round().clamp(-8, 7) * scale).reshape(out_features, in_features)


def projections(layer) -> dict[str, torch.nn.Linear]:
    return {"qkv": layer.self_attn.in_projs[0], "output": layer.self_attn.out_projs[0],
            "gate_up": layer.gating.linear_in, "down": layer.gating.linear_out}


# Per-variant precision of each projection; "qkv" may be split as "q,k,v".
VARIANTS = {
    "none": {},
    "w8": {"qkv": "w8", "output": "w8", "gate_up": "w8", "down": "w8"},
    "lpbq4": {"qkv": "w4", "output": "w4", "gate_up": "w4", "down": "w4"},
    "lpbq4_w8_value_down": {"qkv": "w4,w4,w8", "output": "w4", "gate_up": "w4", "down": "w8"},
    "lpbq4_w8_mlp": {"qkv": "w4", "output": "w4", "gate_up": "w8", "down": "w8"},
    "lpbq4_w8_attn": {"qkv": "w8", "output": "w8", "gate_up": "w4", "down": "w4"},
}


def fake_quantize(weight: torch.Tensor, precision: str) -> torch.Tensor:
    if "," in precision:
        parts = weight.split(weight.shape[0] // 3)
        return torch.cat([fake_quantize(p, q) for p, q in zip(parts, precision.split(","))])
    return {"w8": int8_per_channel, "w4": lpbq_int4}[precision](weight)


def apply_variant(lm, variant: str, originals: dict) -> None:
    for index, layer in enumerate(lm.transformer.layers):
        for name, linear in projections(layer).items():
            weight = originals[(index, name)].to(linear.weight.device, torch.float32)
            precision = VARIANTS[variant].get(name)
            linear.weight.data.copy_((fake_quantize(weight, precision) if precision else weight).to(linear.weight.dtype))


@torch.no_grad()
def run_clip(blocks, head, depformer, layer0: np.ndarray, text_tokens: np.ndarray | None, device) -> dict:
    caches = [block.empty_cache() for block in blocks]
    text, audio, logits = [], [], []
    for t in range(len(layer0)):
        hidden = torch.from_numpy(layer0[t]).to(device, torch.bfloat16).reshape(1, 1, -1)
        position = torch.tensor([t], device=device)
        for i, block in enumerate(blocks):
            hidden, caches[i] = block(hidden, caches[i], position)
        temporal, text_logits = head(hidden)
        token = text_logits.reshape(-1).argmax()
        condition = torch.tensor([text_tokens[t]], device=device) if text_tokens is not None else token.reshape(1)
        audio_tokens, _ = depformer(condition, temporal)
        text.append(int(token))
        audio.append(audio_tokens.reshape(-1).cpu().numpy())
        logits.append(torch.log_softmax(text_logits.reshape(-1).float(), dim=-1).cpu())
    return {"text": np.array(text), "audio": np.stack(audio), "text_logprobs": torch.stack(logits)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--variants", default="w8,lpbq4,lpbq4_w8_value_down,lpbq4_w8_mlp,lpbq4_w8_attn")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    lm = load_moshi_lm(model_dir=args.model_dir, device=device, dtype=torch.bfloat16)
    blocks = [TemporalBlock(layer) for layer in lm.transformer.layers]
    head, depformer = TemporalHead(lm), ExplicitDepFormer(lm)
    originals = {(i, n): l.weight.detach().cpu().clone() for i, layer in enumerate(lm.transformer.layers)
                 for n, l in projections(layer).items()}
    manifest = [c for c in json.loads((args.capture_dir / "manifest.json").read_text()) if c["split"] == "held_out"]
    clips = [np.load(args.capture_dir / f"{c['name']}.npz")["hidden"][:, 0] for c in manifest]

    apply_variant(lm, "none", originals)
    references = [run_clip(blocks, head, depformer, h, None, device) for h in clips]
    for variant in args.variants.split(","):
        apply_variant(lm, variant, originals)
        text_same, cb0_same, audio_same, kl = [], [], [], []
        for layer0, ref in zip(clips, references):
            out = run_clip(blocks, head, depformer, layer0, ref["text"], device)
            text_same.append(out["text"] == ref["text"])
            cb0_same.append(out["audio"][:, 0] == ref["audio"][:, 0])
            audio_same.append(out["audio"] == ref["audio"])
            kl.append((ref["text_logprobs"].exp() * (ref["text_logprobs"] - out["text_logprobs"])).sum(-1).numpy())
        print(json.dumps({"variant": variant,
                          "text_top1": round(float(np.concatenate(text_same).mean()), 4),
                          "audio_cb0": round(float(np.concatenate(cb0_same).mean()), 4),
                          "audio_all": round(float(np.concatenate(audio_same).mean()), 4),
                          "text_kl_mean": round(float(np.concatenate(kl).mean()), 5)}), flush=True)


if __name__ == "__main__":
    main()
