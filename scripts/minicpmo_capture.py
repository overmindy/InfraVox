# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""BF16 MiniCPM-o 4.5 duplex sessions on spoken questions, recording what every phone graph consumes.

Each question is synthesized once with the model's own zero-shot TTS in a voice other than the
assistant's, then played through the official duplex loop (1 s units, sampling). Per session
OUT/<name>.npz holds:
  llm_embeds [T, 4096]   every embedding the LLM was fed, in order (system prompt included)
  llm_logits_at [K]      indices into llm_embeds after which logits were read
  mel_chunks [C, 80, F]  Whisper input per unit (F = 104, the first unit left-padded, see mel_first)
  tts_inputs [M, 768]    every TTS input embedding in feed order; tts_tokens: the sampled codes
  speech_tokens [S]      codes handed to token2wav; reply.wav the official 24 kHz output
OUT/manifest.json lists sessions with their split (the first `--calibration` are calibration).
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
QUESTIONS = [
    "你好，请问周末适合去爬山吗？给我一点建议吧。",
    "Hi there! Can you tell me a fun fact about the moon?",
    "我最近总是失眠，有什么好办法吗？",
    "What's a good way to start learning to play the guitar?",
    "你能给我讲一个关于小猫的简短故事吗？",
    "How do airplanes stay up in the sky?",
    "今天晚饭吃什么比较好？我不太想做饭。",
    "Could you recommend a book for a long train ride?",
    "为什么天空是蓝色的？",
    "I have a job interview tomorrow and I'm nervous. Any tips?",
    "帮我想一个周五晚上和朋友一起玩的活动。",
    "What is the difference between weather and climate?",
    "你觉得学习编程应该从哪种语言开始？",
    "Tell me something interesting about octopuses.",
    "下雨天在家里可以做些什么有意思的事情？",
    "How can I make my coffee taste better at home?",
    "请用两三句话介绍一下长城。",
    "What should I pack for a weekend camping trip?",
    "我想养一盆植物，什么植物比较好养？",
    "Why do cats like boxes so much?",
]
VOICES = ["HT_ref_audio.wav", "bajie.wav", "haimianbaobao.wav", "nezha.wav", "sunwukong.wav", "system_ref_audio_2.wav"]


def load_model(model_dir: Path):
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True,
    )
    model.eval().cuda()
    model.init_tts()
    return model


def synthesize(model, text: str, voice: Path, out: Path) -> None:
    ref, _ = librosa.load(str(voice), sr=16000, mono=True)
    system = {"role": "system", "content": ["模仿音频样本的音色并生成新的内容。", ref, "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"]}
    user = {"role": "user", "content": ["请朗读以下内容。 " + text]}
    model.chat(msgs=[system, user], do_sample=True, max_new_tokens=512, use_tts_template=True,
               generate_audio=True, temperature=0.1, output_audio_path=str(out))


class Recorder:
    """Wraps the duplex session's LLM feed, Whisper, TTS and token2wav calls."""

    def __init__(self, session) -> None:
        self.clear()
        decoder, model = session.decoder, session.model
        feed, apm, tts_model, stream = decoder.feed, model.apm.forward, model.tts.model.forward, model.tts.audio_tokenizer.stream

        def feed_hook(embeds, return_logits=False):
            self.llm.append(embeds.detach().float().cpu().numpy().reshape(-1, embeds.shape[-1]))
            if return_logits:
                self.logits_at.append(sum(len(e) for e in self.llm) - 1)
            return feed(embeds, return_logits=return_logits)

        def apm_hook(input_features, *args, **kwargs):
            if kwargs.get("use_cache"):
                self.mel.append((input_features.detach().float().cpu().numpy()[0], kwargs.get("prefix_extra_frames", 0)))
            return apm(input_features, *args, **kwargs)

        def tts_hook(*args, inputs_embeds=None, **kwargs):
            self.tts_in.append(inputs_embeds.detach().float().cpu().numpy().reshape(-1, inputs_embeds.shape[-1]))
            return tts_model(*args, inputs_embeds=inputs_embeds, **kwargs)

        def stream_hook(tokens, *args, **kwargs):
            self.speech.append(np.asarray(tokens, np.int64))
            return stream(tokens, *args, **kwargs)

        decoder.feed, model.apm.forward, model.tts.model.forward = feed_hook, apm_hook, tts_hook
        model.tts.audio_tokenizer.stream = stream_hook

    def clear(self) -> None:
        self.llm, self.logits_at, self.mel, self.tts_in, self.speech = [], [], [], [], []


def run_session(session, recorder: Recorder, user_wav: Path, ref_wav: Path, silence_s: int) -> dict:
    ref, _ = librosa.load(str(ref_wav), sr=16000, mono=True)
    user, _ = librosa.load(str(user_wav), sr=16000, mono=True)
    user = np.concatenate([np.zeros(16000, np.float32), user, np.zeros(16000 * silence_s, np.float32)])
    recorder.clear()
    session.prepare(prefix_system_prompt=SYSTEM_PROMPT, ref_audio=ref, prompt_wav_path=str(ref_wav))
    log, audio = [], []
    for i in range(len(user) // 16000):
        session.streaming_prefill(audio_waveform=user[i * 16000 : (i + 1) * 16000])
        result = session.streaming_generate(prompt_wav_path=str(ref_wav), max_new_speak_tokens_per_chunk=20)
        wave = result["audio_waveform"]
        audio.append(np.zeros(24000, np.float32) if wave is None else wave)
        log.append({"is_listen": result["is_listen"], "text": result["text"], "end_of_turn": result["end_of_turn"]})
    width = max(m.shape[1] for m, _ in recorder.mel)
    mel = np.zeros((len(recorder.mel), 80, width), np.float32)
    for i, (m, prefix) in enumerate(recorder.mel):
        mel[i, :, width - m.shape[1] :] = m  # first unit has no 2-frame prefix: left-pad it
    return {
        "llm_embeds": np.concatenate(recorder.llm).astype(np.float32),
        "llm_logits_at": np.asarray(recorder.logits_at, np.int64),
        "mel_chunks": mel,
        "mel_first": np.asarray([recorder.mel[0][0].shape[1]], np.int64),
        "tts_inputs": np.concatenate(recorder.tts_in).astype(np.float32) if recorder.tts_in else np.zeros((0, 768), np.float32),
        "speech_tokens": np.concatenate(recorder.speech) if recorder.speech else np.zeros(0, np.int64),
        "total_ids": np.asarray(session.total_ids, np.int64),
        "log": log,
        "audio": np.concatenate(audio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref-wav", type=Path, required=True, help="assistant voice (also the token2wav prompt)")
    parser.add_argument("--calibration", type=int, default=16)
    parser.add_argument("--silence", type=int, default=16)
    args = parser.parse_args()
    out = args.output_dir
    (out / "questions").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = load_model(args.model_dir)
    for i, text in enumerate(QUESTIONS):
        wav = out / "questions" / f"s{i:02d}.wav"
        if not wav.exists():
            synthesize(model, text, args.model_dir / "assets" / VOICES[i % len(VOICES)], wav)
    session = model.as_duplex()
    recorder = Recorder(session)
    manifest = []
    for i, text in enumerate(QUESTIONS):
        name = f"s{i:02d}"
        wav = out / "questions" / f"{name}.wav"
        if not (out / f"{name}.npz").exists():
            torch.manual_seed(i)
            data = run_session(session, recorder, wav, args.ref_wav, args.silence)
            sf.write(out / f"{name}_reply.wav", data.pop("audio"), 24000)
            log = data.pop("log")
            np.savez(out / f"{name}.npz", **data)
            (out / f"{name}.json").write_text(json.dumps(log, ensure_ascii=False))
            print(name, "".join(r["text"] for r in log), flush=True)
        manifest.append({"name": name, "question": text, "split": "calibration" if i < args.calibration else "validation"})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
