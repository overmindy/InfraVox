# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Phone graphs of MiniCPM-o token2wav, after minicpmo_token2wav_prepare.py wrote WORK/token2wav.

  1. DiTStep (minicpmo_flow.py) against the official estimator I/O of the first golden call.
  2. Token2wavHost with float graphs (DiT in torch, conformer and HiFT through ORT) replays the
     golden stream calls; the vocoder input mel must match the official one.
  3. ONNX of the DiT step and AIMET calibration (W8 / A16 / 8-bit KV) of the DiT and both HiFT
     graphs on inputs recorded during that replay.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from minicpmo_flow import DIT_BLOCKS, STEPS, T, DiTStep, Token2wavHost, dit_names, modulations
from minicpmo_llm import create_sim, export_onnx, write_input_list


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def dit_parity(dit: DiTStep, state: dict, golden, device: torch.device) -> None:
    for step in range(STEPS):
        x, mu, t, spks, cond, cnn, att, out, new_cnn, new_att = (golden[f"est{step}_{j}"] for j in range(10))
        n, s = x.shape[2], att.shape[3]
        pad = lambda a: np.pad(a, ((0, 0), (0, 0), (0, T - n)))  # noqa: E731
        packed = np.concatenate([mu, np.repeat(spks[:, :, None], n, axis=2), cond], axis=1)
        mask = np.full((1, 1, T, s + T), -50.0, np.float32)
        mask[..., :s] = 0
        mask[..., s : s + n] = 0
        select = np.zeros((1, T + 2, 2), np.float32)
        select[0, n, 0] = select[0, n + 1, 1] = 1
        caches = []
        for b in range(DIT_BLOCKS):
            caches += [att[b, ..., :64].reshape(16, s, 64).transpose(0, 2, 1)[:, None], att[b, ..., 64:].reshape(16, s, 64)[:, None], cnn[b]]
        mods = [m.numpy() for m in modulations(state, float(t[0]))]
        inputs = [pad(x), pad(packed), mask, select, *mods, *caches]
        with torch.no_grad():
            result = [r.cpu().numpy() for r in dit(*[torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(device) for v in inputs])]
        errors = [relative(result[0][:, :, :n], out)]
        for b in range(DIT_BLOCKS):
            fresh = np.concatenate([result[1 + 3 * b][:, 0, :, :n].transpose(0, 2, 1), result[2 + 3 * b][:, 0, :n]], axis=-1)
            errors.append(relative(fresh.reshape(2, 8, n, 128), new_att[b, :, :, :n]))
            errors.append(relative(result[3 + 3 * b], new_cnn[b]))
        print(f"dit step {step}: velocity {errors[0]:.2e}, caches max {max(errors[1:]):.2e}", flush=True)
        if max(errors) > 1e-3:
            raise SystemExit("DiTStep does not match the official estimator")


class Recorder:
    def __init__(self, fn, keep: int) -> None:
        self.fn, self.keep, self.samples, self.calls = fn, keep, [], 0

    def __call__(self, *inputs):
        self.calls += 1
        if len(self.samples) < self.keep and self.calls % 3 == 1:
            flat = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], list) else inputs
            self.samples.append([np.ascontiguousarray(v) for v in flat])
        return self.fn(*inputs)


def build(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    work = args.work_dir / "token2wav"
    constants = dict(np.load(work / "constants.npz"))
    golden = np.load(work / "golden.npz")
    state = torch.load(work / "flow_state.pt")
    dit = DiTStep(state).to(device).eval()
    dit_parity(dit, state, golden, device)

    cpu = ["CPUExecutionProvider"]
    encoder = onnxruntime.InferenceSession(str(work / "flow_encoder.onnx"), providers=cpu)
    f0 = onnxruntime.InferenceSession(str(work / "hift_f0" / "onnx" / "model.onnx"), providers=cpu)
    decode = onnxruntime.InferenceSession(str(work / "hift_decode" / "onnx" / "model.onnx"), providers=cpu)

    def run_encoder(tokens, pad, cnn, att):
        return encoder.run(None, {"tokens": tokens[None], "pad": np.zeros((1, pad, 512), np.float32), "cnn_cache": cnn, "att_cache": att})

    @torch.no_grad()
    def run_dit(inputs):
        return [r.cpu().numpy() for r in dit(*[torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(device) for v in inputs])]

    dit_calls = Recorder(lambda inputs: run_dit(inputs), args.samples)
    f0_calls = Recorder(lambda mel: f0.run(None, {"mel": mel})[0], args.samples)
    decode_calls = Recorder(lambda mel, s: decode.run(None, {"mel": mel, "s_stft": s})[0], args.samples)
    host = Token2wavHost(constants, run_encoder, dit_calls, f0_calls, decode_calls)
    vocoder_input = []
    vocode = host.vocode
    host.vocode = lambda mel: (vocoder_input.append(mel), vocode(mel))[1]
    worst = 0.0
    for g in range(int(golden["sessions"])):
        host.reset()
        for c in range(int(golden[f"s{g}_calls"])):
            host.stream(golden[f"s{g}_c{c}_tokens"].tolist(), bool(golden[f"s{g}_c{c}_last"]))
            error = relative(vocoder_input[-1], golden[f"s{g}_c{c}_mel"])
            worst = max(worst, error)
            if c < 3 or error > 1e-3:
                print(f"token2wav session {g} call {c}: vocoder mel relative error {error:.2e}", flush=True)
    print(f"token2wav host: worst vocoder mel error {worst:.2e}", flush=True)
    if worst > 1e-3:
        raise SystemExit("Token2wavHost does not reproduce the official stream")

    names = dit_names()
    sample = dit_calls.samples[0]
    export_onnx(dit, tuple(torch.from_numpy(np.ascontiguousarray(v, np.float32)) for v in sample), names, work / "dit" / "onnx" / "model.onnx")
    del dit
    torch.cuda.empty_cache()
    for name, recorder, input_names, kv in (
        ("dit", dit_calls, names[0], True),
        ("hift_f0", f0_calls, ["mel"], False),
        ("hift_decode", decode_calls, ["mel", "s_stft"], False),
    ):
        sim = create_sim(onnx.load(str(work / name / "onnx" / "model.onnx")), "int8", kv=kv)
        sim.compute_encodings([dict(zip(input_names, s, strict=False)) for s in recorder.samples])
        sim.export(str(work / name), name, export_model=False)
        write_input_list(work / name, input_names, recorder.samples)
        print(f"{name}: {len(recorder.samples)} calibration calls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    start = time.monotonic()
    build(args)
    print(f"token2wav done in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
