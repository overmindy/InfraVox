# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Host-side assets of the MiniCPM-o phone bundle, as one tensor container OUT/assets.bin.

Format (little endian): "MCPOTEN1", uint32 count, then per tensor: uint16 name length, name,
uint8 dtype (0 f32, 1 f16, 2 i32, 3 u8, 4 i8), uint8 rank, uint32 dims[rank], uint64 offset, uint64 bytes;
data blocks follow, each 64-byte aligned (the phone mmaps the file).

Tensors: LLM and TTS embedding tables, Whisper positions and mel filters, the system prompt
embeddings (text + reference voice), the byte-level BPE pieces for text display, and the fixed
token2wav voice state (prompt caches, speaker projection, DiT timestep modulation, noise).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor

import minicpmo_audio as audio
from minicpmo_flow import timesteps
from minicpmo_llm import Weights

DTYPES = {np.float32: 0, np.float16: 1, np.int32: 2, np.uint8: 3, np.int8: 4}
UNIT_TOKEN = 151683


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte <-> printable-unicode table used by Qwen's byte-level BPE."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    codes = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + n)
            n += 1
    return dict(zip(printable, map(chr, codes), strict=True))


def vocabulary(tokenizer_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    decode = {c: b for b, c in bytes_to_unicode().items()}
    special = {t.content for t in tokenizer.get_added_tokens_decoder().values()}
    pieces = []
    for i in range(size):
        token = tokenizer.id_to_token(i)
        pieces.append(b"" if token is None or token in special else bytes(decode[c] for c in token))
    offsets = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int32)
    return offsets, np.frombuffer(b"".join(pieces), np.uint8)


def write_container(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = bytearray(b"MCPOTEN1" + struct.pack("<I", len(tensors)))
    entries = []
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        entries.append((name.encode(), array))
        header += struct.pack("<H", len(name)) + name.encode() + struct.pack("<BB", DTYPES[array.dtype.type], array.ndim)
        header += struct.pack(f"<{array.ndim}I", *array.shape) + struct.pack("<QQ", 0, 0)
    offset = (len(header) + 63) // 64 * 64
    cursor = 12
    placed = []
    for name, array in entries:
        cursor += 2 + len(name) + 2 + 4 * array.ndim
        struct.pack_into("<QQ", header, cursor, offset, array.nbytes)
        placed.append((offset, array))
        cursor += 16
        offset = (offset + array.nbytes + 63) // 64 * 64
    with open(path, "wb") as f:
        f.write(header)
        for start, array in placed:
            f.seek(start)
            f.write(array.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    w = Weights(args.model_dir)
    embed = w("llm.model.embed_tokens.weight")
    reference = np.load(sorted(args.capture_dir.glob("s*.npz"))[0])["llm_embeds"]
    unit = embed[UNIT_TOKEN].to(torch.bfloat16).float().numpy()
    system = reference[: audio.unit_rows(reference, unit)[0] - 1]
    c = dict(np.load(args.work_dir / "token2wav" / "constants.npz"))
    offsets, pieces = vocabulary(args.model_dir / "tokenizer.json", embed.shape[0])
    tensors = {
        "embed_tokens": embed.numpy().astype(np.float16),
        "tts_emb_text": w("tts.emb_text.weight").numpy().astype(np.float16),
        "tts_emb_code": w("tts.emb_code.0.weight").numpy().astype(np.float16),
        "whisper_positions": w("apm.embed_positions.weight").numpy().astype(np.float16),
        "mel_filters": WhisperFeatureExtractor(feature_size=80).mel_filters.astype(np.float32),
        "system_embeds": system.astype(np.float16),
        "vocab_offsets": offsets,
        "vocab_bytes": pieces,
        "t2w_prompt_frames": np.asarray([int(c["prompt_frames"])], np.int32),
        "t2w_spk": c["spk"].astype(np.float32),
        "t2w_conformer_cnn": c["conformer_cnn_cache"].astype(np.float32),
        "t2w_conformer_att": c["conformer_att_cache"].astype(np.float32),
        "t2w_est_cnn": c["estimator_cnn_cache"].astype(np.float32),
        "t2w_est_att": c["estimator_att_cache"].astype(np.float16),
        "t2w_noise": c["rand_noise"].astype(np.float32),
        "t2w_mod_blocks": c["mod_blocks"].reshape(10, 16, 9, 512).astype(np.float32),
        "t2w_mod_final": c["mod_final"].reshape(10, 2, 512).astype(np.float32),
        "t2w_dts": np.asarray(timesteps()[1], np.float32),
        "t2w_source_weight": c["source_weight"].reshape(-1).astype(np.float32),
        "t2w_source_bias": c["source_bias"].reshape(-1).astype(np.float32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_container(args.output, tensors)
    print({k: (v.dtype.name, v.shape) for k, v in tensors.items()}, flush=True)


if __name__ == "__main__":
    main()
