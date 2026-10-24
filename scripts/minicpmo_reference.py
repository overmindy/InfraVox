# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Official BF16 MiniCPM-o 4.5 duplex run on the server GPU: the reference every phone graph is checked against.

Synthesizes spoken user questions with the model's own zero-shot TTS (a different voice than the
assistant's), then plays each through the duplex loop in 1 s chunks followed by silence, and saves
the per-chunk listen/speak log, text and 24 kHz reply audio to OUT/<name>.{json,wav}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

SYSTEM_PROMPT = "Streaming Duplex Conversation! You are a helpful assistant."
QUESTIONS = {
    "user_zh": "你好，请问周末适合去爬山吗？给我一点建议吧。",
    "user_en": "Hi there! Can you tell me a fun fact about the moon?",
}


def load_model(model_dir: Path):
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=True,
    )
    model.eval().cuda()
    model.init_tts()
    return model


def synthesize(model, text: str, voice: Path, out: Path) -> None:
    ref, _ = librosa.load(str(voice), sr=16000, mono=True)
    system = {
        "role": "system",
        "content": ["模仿音频样本的音色并生成新的内容。", ref, "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"],
    }
    user = {"role": "user", "content": ["请朗读以下内容。 " + text]}
    model.chat(
        msgs=[system, user],
        do_sample=True,
        max_new_tokens=512,
        use_tts_template=True,
        generate_audio=True,
        temperature=0.1,
        output_audio_path=str(out),
    )


def duplex(model, user_wav: Path, ref_wav: Path, silence_s: int, seed: int) -> tuple[list[dict], np.ndarray]:
    torch.manual_seed(seed)
    ref, _ = librosa.load(str(ref_wav), sr=16000, mono=True)
    user, _ = librosa.load(str(user_wav), sr=16000, mono=True)
    user = np.concatenate([np.zeros(16000, np.float32), user, np.zeros(16000 * silence_s, np.float32)])
    session = model.as_duplex()
    session.prepare(prefix_system_prompt=SYSTEM_PROMPT, ref_audio=ref, prompt_wav_path=str(ref_wav))
    log, audio = [], []
    for i in range(len(user) // 16000):
        session.streaming_prefill(audio_waveform=user[i * 16000 : (i + 1) * 16000])
        result = session.streaming_generate(prompt_wav_path=str(ref_wav), max_new_speak_tokens_per_chunk=20)
        wave = result["audio_waveform"]
        audio.append(np.zeros(24000, np.float32) if wave is None else wave)
        log.append({k: result[k] for k in ("is_listen", "text", "end_of_turn", "cost_llm", "cost_tts", "cost_token2wav", "n_tokens", "n_tts_tokens")})
        print(i, "listen" if result["is_listen"] else f"speak> {result['text']}", flush=True)
    return log, np.concatenate(audio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--silence", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir)
    assets = args.model_dir / "assets"
    for name, text in QUESTIONS.items():
        wav = args.out / f"{name}.wav"
        if not wav.exists():
            synthesize(model, text, assets / "HT_ref_audio.wav", wav)
    ref = assets / "system_ref_audio.wav"
    for name in QUESTIONS:
        log, audio = duplex(model, args.out / f"{name}.wav", ref, args.silence, args.seed)
        sf.write(args.out / f"{name}_reply.wav", audio, 24000)
        (args.out / f"{name}_reply.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(name, "".join(r["text"] for r in log), flush=True)


if __name__ == "__main__":
    main()
