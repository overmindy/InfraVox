# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Export the streaming Mimi encoder/decoder frame graphs and author their QAIRT encodings.

Only the Transformer MatMul weights are quantized (encoder W8 per-channel, decoder LPBQ W4
block 32) with A16 ranges on their inputs/outputs from the first 4 frames of --audio-wav;
codebooks, normalization and convolutions stay FP32. The decoder's W4 MatMuls become
equivalent pointwise Convs, the layout HTP accepts for LPBQ. Writes WORK/mimi_{encoder,decoder}/
onnx/model.onnx and <name>.encodings for the converter (no qairt-quantizer pass).
"""

from __future__ import annotations

import argparse
import copy
import json
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

from qai_hub_models.models.templates.moshi.explicit_mimi import (
    ExplicitMimiDecoder,
    ExplicitMimiEncoder,
)
from qai_hub_models.models.templates.moshi.model import (
    FRAME_SAMPLES,
    export_onnx,
    resolve_moshi_checkpoint,
)

STATE = ["position", "kv_cache", "conv_state"]
STATE_OUT = ["position_out", "kv_cache_out", "conv_state_out"]
FRAMES, CALIBRATION_FRAMES = 8, 4
BITS = {"mimi_encoder": 8, "mimi_decoder": 4}
BLOCK = 32


def read_frames(path: Path) -> list[np.ndarray]:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 24000):
            raise SystemExit("Expected 24 kHz mono PCM16 audio")
        audio = (
            np.frombuffer(w.readframes(FRAMES * FRAME_SAMPLES), "<i2").astype(
                np.float32
            )
            / 32768
        )
    return list(audio.reshape(FRAMES, 1, 1, FRAME_SAMPLES))


@torch.no_grad()
def export(model_dir: Path, audio: np.ndarray, work: Path, device: str) -> None:
    mimi = resolve_moshi_checkpoint(model_dir=model_dir).get_mimi(device=device)
    frame = torch.from_numpy(audio).to(device)
    encoder = ExplicitMimiEncoder(mimi)
    codes = encoder(frame, *encoder.initial_state())[0]
    export_onnx(
        encoder,
        (frame, *encoder.initial_state()),
        work / "mimi_encoder/onnx/model.onnx",
        ["audio", *STATE],
        ["codes", *STATE_OUT],
    )
    mimi._stop_streaming()
    decoder = ExplicitMimiDecoder(mimi)
    export_onnx(
        decoder,
        (codes, *decoder.initial_state()),
        work / "mimi_decoder/export.onnx",
        ["codes", *STATE],
        ["audio", *STATE_OUT],
    )


def stream(model: onnx.ModelProto, data: list[np.ndarray]) -> list[list[np.ndarray]]:
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = (
        4,
        1,
    )  # Fixed for reproducible ranges.
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    state = {
        v.name: np.zeros(
            [d.dim_value for d in v.type.tensor_type.shape.dim],
            np.int64
            if v.type.tensor_type.elem_type == onnx.TensorProto.INT64
            else np.float32,
        )
        for v in model.graph.input[1:]
    }
    outputs = []
    for value in data:
        result = session.run(None, {model.graph.input[0].name: value, **state})
        outputs.append(result)
        state = dict(zip(STATE, result[1:4], strict=False))
    return outputs


def weight_encoding(name: str, w: np.ndarray, bits: int) -> tuple[np.ndarray, dict]:
    """Symmetric W8 per output channel, or LPBQ W4: int4 blocks along the input with a per-channel
    float scale and a 4-bit per-block integer multiplier. Returns the dequantized weight too.
    """
    if bits == 8:
        scale = np.maximum(np.max(np.abs(w), axis=0), 1e-8) / 127
        decoded = np.clip(np.rint(w / scale[None]), -128, 127) * scale[None]
        return decoded, {
            "name": name,
            "dtype": "INT",
            "bw": 8,
            "enc_type": "PER_CHANNEL",
            "is_sym": True,
            "scale": scale.tolist(),
            "offset": [0] * len(scale),
        }
    blocked = np.ascontiguousarray(w.T).reshape(w.shape[1], -1, BLOCK)
    step = np.maximum(-blocked.min(axis=2) / 8, blocked.max(axis=2) / 7)
    scale = np.maximum(step.max(axis=1) / 16, 1e-12).astype(np.float32)
    expansion = np.clip(np.rint(step / scale[:, None]), 1, 16).astype(np.uint8)
    grid = (scale[:, None] * np.repeat(expansion, BLOCK, axis=1)).T
    decoded = np.clip(np.rint(w / grid), -8, 7) * grid
    return decoded, {
        "name": name,
        "dtype": "INT",
        "bw": 8,
        "compressed_bw": 4,
        "enc_type": "LPBQ",
        "is_sym": True,
        "scale": scale.tolist(),
        "offset": [0] * len(scale),
        "block_size": BLOCK,
        "per_block_int_scale": expansion.flatten().tolist(),
    }


def to_pointwise(model: onnx.ModelProto, policy: dict, names: set[str]) -> None:
    """Rewrite [.., N, Cin] x [Cin, Cout] MatMuls as Reshape -> 1x1 Conv (+zero bias) -> Reshape."""
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in (
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        )
    }
    tensors = {t.name: t for t in model.graph.initializer}
    activations = {e["name"]: e for e in policy["activation_encodings"]}
    nodes = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in names:
            nodes.append(node)
            continue
        name = node.input[1]
        weight = numpy_helper.to_array(tensors[name])
        before, after = shapes[node.input[0]], shapes[node.output[0]]
        x, y, bias = (
            name + "_pointwise_input",
            name + "_pointwise_output",
            name + "_pointwise_zero_bias",
        )
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(
                    np.array(
                        [int(np.prod(before[:-1])), weight.shape[0], 1, 1], np.int64
                    ),
                    name + "_pointwise_input_shape",
                ),
                numpy_helper.from_array(
                    np.array(after, np.int64), name + "_pointwise_output_shape"
                ),
                numpy_helper.from_array(np.zeros(weight.shape[1], np.float32), bias),
            ]
        )
        tensors[name].CopyFrom(
            numpy_helper.from_array(
                np.ascontiguousarray(weight.T[:, :, None, None]), name
            )
        )
        nodes += [
            helper.make_node(
                "Reshape",
                [node.input[0], name + "_pointwise_input_shape"],
                [x],
                name=node.name + "/to_pointwise",
            ),
            helper.make_node(
                "Conv", [x, name, bias], [y], name=node.name, kernel_shape=[1, 1]
            ),
            helper.make_node(
                "Reshape",
                [y, name + "_pointwise_output_shape"],
                [node.output[0]],
                name=node.name + "/from_pointwise",
            ),
        ]
        for old, new in ((node.input[0], x), (node.output[0], y)):
            policy["activation_encodings"].append(
                {**copy.deepcopy(activations[old]), "name": new}
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def quantize(
    name: str, model: onnx.ModelProto, data: list[np.ndarray], out: Path
) -> None:
    bits = BITS[name]
    weights = {t.name: t for t in model.graph.initializer}
    selected = [
        n
        for n in model.graph.node
        if n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] in weights
    ]
    selected_names = {n.input[1] for n in selected}
    boundaries = sorted({v for n in selected for v in (n.input[0], n.output[0])})
    probe = copy.deepcopy(model)
    probe.graph.output.extend(
        helper.make_tensor_value_info(b, onnx.TensorProto.FLOAT, None)
        for b in boundaries
    )
    bounds = {b: [0.0, 0.0] for b in boundaries}
    for result in stream(probe, data[:CALIBRATION_FRAMES]):
        for b, tensor in zip(boundaries, result[4:], strict=False):
            bounds[b] = [
                min(bounds[b][0], float(tensor.min())),
                max(bounds[b][1], float(tensor.max())),
            ]

    policy = {
        "version": "1.0.0",
        "activation_encodings": [],
        "param_encodings": [
            {"name": t.name, "dtype": "FLOAT", "bw": 32}
            for t in model.graph.initializer
            if t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)
            and t.name not in selected_names
        ],
    }
    for node in selected:
        decoded, encoding = weight_encoding(
            node.input[1],
            numpy_helper.to_array(weights[node.input[1]]).astype(np.float32),
            bits,
        )
        policy["param_encodings"].append(encoding)
        if (
            bits == 4
        ):  # The converter keeps W4 grids exactly only if the source already holds them.
            weights[node.input[1]].CopyFrom(
                numpy_helper.from_array(decoded, node.input[1])
            )
    for b, (lo, hi) in bounds.items():
        # A small fixed guard absorbs interpolation outside the four fitting frames.
        lo, hi = lo * 1.05, hi * 1.05
        scale = max((hi - lo) / 65535, 1e-10)
        policy["activation_encodings"].append(
            {
                "name": b,
                "dtype": "INT",
                "bw": 16,
                "enc_type": "PER_TENSOR",
                "is_sym": False,
                "scale": [scale],
                "offset": [int(np.clip(np.rint(lo / scale), -65535, 0))],
            }
        )
    if bits == 4:
        to_pointwise(model, policy, selected_names)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(out / "onnx/model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )
    (out / f"{name}.encodings").write_text(json.dumps(policy, indent=2) + "\n")
    print(
        f"{name}: {len(selected)} W{bits} MatMuls, {len(boundaries)} A16 ranges",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    frames = read_frames(args.audio_wav)
    export(args.model_dir, frames[0], args.work_dir, args.device)
    encoder = onnx.load(str(args.work_dir / "mimi_encoder/onnx/model.onnx"))
    codes = [r[0] for r in stream(encoder, frames)]
    quantize("mimi_encoder", encoder, frames, args.work_dir / "mimi_encoder")
    decoder_export = args.work_dir / "mimi_decoder/export.onnx"
    quantize(
        "mimi_decoder",
        onnx.load(str(decoder_export)),
        codes,
        args.work_dir / "mimi_decoder",
    )
    decoder_export.unlink()


if __name__ == "__main__":
    main()
