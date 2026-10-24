# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Official token2wav (stepaudio2) side of the MiniCPM-o phone build; needs the minicpmo env.

Writes WORK/token2wav/:
  constants.npz        the fixed voice: prompt caches, speaker projection, DiT timestep modulation,
                       the estimator noise, the NSF source projection
  flow_encoder.onnx    conformer chunk encoder with dynamic token / cache lengths (phone CPU, ORT)
  hift_f0.onnx, hift_decode/onnx/model.onnx   the HiFT graphs at 58 mel frames (phone HTP)
  golden.npz           official Token2wav.stream calls on captured speech tokens (tokens, flags,
                       waveforms) and the estimator I/O of the first call, for parity downstream
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from stepaudio2 import Token2wav
from torch import nn
from torch.nn import functional
from torch.nn.utils import parametrize

from minicpmo_flow import HIFT_FRAMES, LOOKAHEAD, MEL, STEPS, modulations, timesteps


class EncoderChunk(nn.Module):
    """UpsampleConformerEncoderV2.forward_chunk + encoder_proj with explicit caches."""

    def __init__(self, flow) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, tokens, pad, cnn_cache, att_cache):
        enc = self.flow.encoder
        xs, _, _ = enc.embed(self.flow.input_embedding(tokens), None)
        xs = torch.cat([xs, pad], dim=1)
        offset1 = att_cache.shape[3] // 2
        att1, att2 = att_cache[: len(enc.encoders), :, :, :offset1], att_cache[len(enc.encoders) :]
        xs, cnn1 = enc.pre_lookahead_layer.forward_chunk(xs, cache=cnn_cache[:, :, :2])
        pos_emb = enc.embed.position_encoding(offset=None, size=offset1 + xs.shape[1])
        new1 = []
        for i, layer in enumerate(enc.encoders):
            xs, _, cache, _ = layer(xs, torch.zeros((0, 0, 0)), pos_emb, att_cache=att1[i])
            new1.append(cache)
        xs, _, cnn2 = enc.up_layer.forward_chunk(xs.transpose(1, 2), None, cache=cnn_cache[:, :, 2:])
        xs, _, _ = enc.up_embed(xs.transpose(1, 2), None)
        pos_emb = enc.embed.position_encoding(offset=None, size=offset1 * enc.up_layer.stride + xs.shape[1])
        new2 = []
        for i, layer in enumerate(enc.up_encoders):
            xs, _, cache, _ = layer(xs, torch.zeros((0, 0, 0)), pos_emb, att_cache=att2[i])
            new2.append(cache)
        h = self.flow.encoder_proj(enc.after_norm(xs))
        att = torch.cat([torch.stack(new1).repeat(1, 1, 1, 2, 1), torch.stack(new2)], dim=0)
        return h, torch.cat([cnn1, cnn2], dim=2), att


class HiftF0(nn.Module):
    def __init__(self, hift) -> None:
        super().__init__()
        self.hift = hift

    def forward(self, mel):
        return self.hift.f0_predictor(mel)


class HiftDecode(nn.Module):
    """HiFTGenerator.decode without its STFT front and iSTFT tail (both run on the phone CPU)."""

    def __init__(self, hift) -> None:
        super().__init__()
        self.hift = hift

    def forward(self, mel, s_stft):
        g = self.hift
        x = g.conv_pre(mel)
        for i in range(g.num_upsamples):
            x = functional.leaky_relu(x, g.lrelu_slope)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = x + g.source_resblocks[i](g.source_downs[i](s_stft))
            xs = None
            for j in range(g.num_kernels):
                y = g.resblocks[i * g.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / g.num_kernels
        return g.conv_post(functional.leaky_relu(x))


def fold_weight_norm(module: nn.Module) -> None:
    """HTP needs constant conv weights; weight_norm would export them as g * v / |v| ops."""
    for m in module.modules():
        if parametrize.is_parametrized(m):
            for name in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)


def ort(path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def check(name: str, ours: np.ndarray, reference: np.ndarray, tolerance: float) -> None:
    error = float(np.linalg.norm(ours - reference) / max(np.linalg.norm(reference), 1e-12))
    print(f"{name}: relative error {error:.2e}", flush=True)
    if error > tolerance:
        raise SystemExit(f"{name} does not match")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ref-wav", type=Path, required=True)
    parser.add_argument("--golden-sessions", type=int, default=3)
    args = parser.parse_args()
    out = args.work_dir / "token2wav"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tw = Token2wav(str(args.model_dir / "assets" / "token2wav"))
    flow, hift = tw.flow.float(), tw.hift.float()
    fold_weight_norm(hift)
    tw.cache = None
    stream_cache, hift_cache = tw.set_stream_cache(str(args.ref_wav))
    prompt_tokens, _, spk_emb, prompt_mels, _ = tw.cache
    prompt = prompt_mels.shape[1]
    state = {k: v.float().cpu() for k, v in flow.state_dict().items()}
    ts = timesteps()[0]
    mods = [modulations(state, t) for t in ts]
    constants = {
        "prompt_frames": np.asarray(prompt),
        "prompt_tokens": prompt_tokens.cpu().numpy()[0],
        "spk": flow.spk_embed_affine_layer(functional.normalize(spk_emb, dim=1))[0].cpu().numpy(),
        "conformer_cnn_cache": stream_cache["conformer_cnn_cache"].cpu().numpy(),
        "conformer_att_cache": stream_cache["conformer_att_cache"].cpu().numpy(),
        "estimator_cnn_cache": stream_cache["estimator_cnn_cache"][:STEPS].cpu().numpy(),
        "estimator_att_cache": stream_cache["estimator_att_cache"][:STEPS].cpu().numpy(),
        "rand_noise": flow.decoder.rand_noise[0, :, : prompt + 400].cpu().numpy(),
        "mod_blocks": np.stack([np.stack([m.numpy() for m in step[:-1]]) for step in mods]),
        "mod_final": np.stack([step[-1].numpy() for step in mods]),
        "source_weight": hift.m_source.l_linear.weight.cpu().numpy(),
        "source_bias": hift.m_source.l_linear.bias.cpu().numpy(),
    }
    np.savez(out / "constants.npz", **constants)
    torch.save(state, out / "flow_state.pt")
    print("constants:", {k: v.shape for k, v in constants.items()}, flush=True)

    # Conformer encoder for the phone CPU, checked at the prompt cache and after one chunk.
    encoder = EncoderChunk(flow).eval()
    cnn, att = stream_cache["conformer_cnn_cache"], stream_cache["conformer_att_cache"]
    tokens = torch.randint(0, 6561, (1, 28), device=cnn.device)
    pad = torch.zeros(1, 0, 512, device=cnn.device)
    reference = flow.encoder.forward_chunk(flow.input_embedding(tokens), last_chunk=False, cnn_cache=cnn, att_cache=att)
    ours = encoder(tokens, pad, cnn, att)
    check("flow_encoder torch rewrite", ours[0].cpu().numpy(), flow.encoder_proj(reference[0]).cpu().numpy(), 1e-5)
    torch.onnx.export(
        encoder.cpu(), (tokens.cpu(), pad.cpu(), cnn.cpu(), att.cpu()), str(out / "flow_encoder.onnx"),
        input_names=["tokens", "pad", "cnn_cache", "att_cache"], output_names=["h", "cnn_cache_out", "att_cache_out"],
        dynamic_axes={"tokens": {1: "L"}, "pad": {1: "P"}, "att_cache": {3: "S"}, "h": {1: "H"}, "att_cache_out": {3: "S2"}},
        opset_version=17, dynamo=False,
    )
    session = ort(out / "flow_encoder.onnx")
    encoder = encoder.cuda()
    for length, pad_len, cache in ((28, 0, att), (20, 3, ours[2]), (28, 0, ours[2][:, :, :, :300])):
        t = torch.randint(0, 6561, (1, length))
        p = torch.zeros(1, pad_len, 512)
        expected = encoder(t.cuda(), p.cuda(), ours[1], cache)
        got = session.run(None, {"tokens": t.numpy(), "pad": p.numpy(), "cnn_cache": ours[1].cpu().numpy(), "att_cache": cache.cpu().numpy()})
        check(f"flow_encoder ORT L={length} pad={pad_len} S={cache.shape[3]}", got[0], expected[0].cpu().numpy(), 1e-3)

    # HiFT graphs (fixed 58 frames).
    mel = torch.randn(1, MEL, HIFT_FRAMES) * 2 - 6
    f0_module, decode_module = HiftF0(hift).cpu().eval(), HiftDecode(hift).cpu().eval()
    s_stft = torch.randn(1, 18, HIFT_FRAMES * 120 + 1) * 0.1
    (out / "hift_f0" / "onnx").mkdir(parents=True, exist_ok=True)
    (out / "hift_decode" / "onnx").mkdir(parents=True, exist_ok=True)
    torch.onnx.export(f0_module, (mel,), str(out / "hift_f0" / "onnx" / "model.onnx"), input_names=["mel"],
                      output_names=["f0"], opset_version=17, dynamo=False)
    torch.onnx.export(decode_module, (mel, s_stft), str(out / "hift_decode" / "onnx" / "model.onnx"),
                      input_names=["mel", "s_stft"], output_names=["head"], opset_version=17, dynamo=False)
    check("hift_f0 ORT", ort(out / "hift_f0" / "onnx" / "model.onnx").run(None, {"mel": mel.numpy()})[0], f0_module(mel).numpy(), 1e-4)
    check("hift_decode ORT", ort(out / "hift_decode" / "onnx" / "model.onnx").run(None, {"mel": mel.numpy(), "s_stft": s_stft.numpy()})[0],
          decode_module(mel, s_stft).numpy(), 1e-4)
    hift.cuda()

    # Golden streams: the duplex buffering (3 silence tokens, 25-token hops with 3 look-ahead).
    estimator_io = []
    estimator = flow.decoder.estimator
    forward_chunk = estimator.forward_chunk

    def record(x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        result = forward_chunk(x, mu, t, spks, cond, cnn_cache=cnn_cache, att_cache=att_cache)
        if len(estimator_io) < STEPS:
            estimator_io.append([v.cpu().numpy() for v in (x, mu, t, spks, cond, cnn_cache, att_cache, *result)])
        return result

    estimator.forward_chunk = record
    vocoder_mels = []
    hift_forward = hift.forward

    def record_mel(speech_feat, cache_source=torch.zeros(1, 1, 0)):
        vocoder_mels.append(speech_feat[0].cpu().numpy())
        return hift_forward(speech_feat, cache_source)

    hift.forward = record_mel
    golden = {}
    for g, path in enumerate(sorted(args.capture_dir.glob("s*.npz"))[: args.golden_sessions]):
        speech = np.load(path)["speech_tokens"].tolist()[:400]
        tw.stream_cache = {k: v.clone() for k, v in stream_cache.items()}
        tw.hift_cache_dict = {k: v.clone() for k, v in hift_cache.items()}
        buffer, calls = [4218] * LOOKAHEAD + speech, []
        while len(buffer) >= 25 + LOOKAHEAD:
            calls.append((buffer[: 25 + LOOKAHEAD], False))
            buffer = buffer[25:]
        calls.append((buffer, True))
        for c, (tokens, last) in enumerate(calls):
            golden[f"s{g}_c{c}_tokens"] = np.asarray(tokens, np.int64)
            golden[f"s{g}_c{c}_last"] = np.asarray(last)
            golden[f"s{g}_c{c}_wave"] = tw.stream(tokens, str(args.ref_wav), last_chunk=last, return_waveform=True)[0]
            golden[f"s{g}_c{c}_mel"] = vocoder_mels[-1]
        golden[f"s{g}_calls"] = np.asarray(len(calls))
    golden["sessions"] = np.asarray(args.golden_sessions)
    for i, io in enumerate(estimator_io):
        for j, v in enumerate(io):
            golden[f"est{i}_{j}"] = v
    np.savez(out / "golden.npz", **golden)
    print("golden:", args.golden_sessions, "sessions", len(estimator_io), "estimator calls", flush=True)


if __name__ == "__main__":
    main()
