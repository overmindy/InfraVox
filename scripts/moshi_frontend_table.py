# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Export Moshi's Temporal input embeddings as one raw BF16 table for the on-device CPU frontend.

The frontend only sums one text row and 16 audio rows per frame; on HTP that graph took ~9 ms,
on the CPU it is a 17-row lookup. Layout (little endian):
  "MOSHIEMB" | uint32 version=1 | uint32 dim | uint32 tables | uint32 rows[tables] | BF16 rows
Table 0 is the text embedding (sequence slot 0); table 1+k is audio embedding k (slot 1+k).
"""

import argparse
import struct
from pathlib import Path

import torch

from qai_hub_models.models.templates.moshi.explicit_cache import TemporalFrontend
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lm = load_moshi_lm(
        model_dir=args.model_dir, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    if lm.audio_offset != 1:
        raise SystemExit("Expected the text stream at slot 0 and audio from slot 1")
    tables = [lm.text_emb, *lm.emb]
    for table in tables:
        if (
            table.norm is not None
            or table.low_rank is not None
            or table.demux_second_stream
        ):
            raise SystemExit("Only plain embedding tables are supported")
    weights = [t.weight.detach().to(torch.bfloat16).contiguous() for t in tables]
    dim = weights[0].shape[1]
    with open(args.output, "wb") as out:
        out.write(
            b"MOSHIEMB"
            + struct.pack(
                f"<III{len(weights)}I",
                1,
                dim,
                len(weights),
                *(w.shape[0] for w in weights),
            )
        )
        out.writelines(w.view(torch.int16).numpy().tobytes() for w in weights)

    # Reference check: the BF16 frontend vs an FP32 sum of the exported rows.
    frontend = TemporalFrontend(lm).eval()
    generator = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(16):
        sequence = torch.cat(
            [
                torch.randint(0, 32001, (1, 1, 1), generator=generator),
                torch.randint(0, 2049, (1, 16, 1), generator=generator),
            ],
            dim=1,
        )
        with torch.no_grad():
            expected = frontend(sequence).float().reshape(-1)
        total = sum(w[int(sequence[0, i, 0])].float() for i, w in enumerate(weights))
        worst = max(worst, float((total - expected).abs().max() / expected.abs().max()))
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes, {len(weights)} tables, dim {dim}); "
        f"max relative difference vs BF16 frontend {worst:.2e}"
    )


if __name__ == "__main__":
    main()
