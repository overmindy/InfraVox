# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Write Moshi's SentencePiece pieces, one per line by id, as text_vocab.txt for the app's text decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="kyutai/moshiko-pytorch-bf16 checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    processor = sentencepiece.SentencePieceProcessor(model_file=str(args.model_dir / "tokenizer_spm_32k_3.model"))
    # The app splits on newlines, so backslash, CR and LF inside a piece are escaped.
    lines = [
        processor.id_to_piece(i).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
        for i in range(processor.get_piece_size())
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
