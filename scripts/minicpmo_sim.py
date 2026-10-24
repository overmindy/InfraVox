# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Python model of the phone runner for MiniCPM-o 4.5 duplex: the host logic the C++ runner ports.

Every graph is called through a `Graphs` object with the phone graph I/O (minicpmo_llm / _audio /
_tts / _flow), here backed by the float torch modules and the ORT CPU graphs. The host side
follows MiniCPMODuplex.streaming_prefill / streaming_generate (modeling_minicpmo.py) and
StreamDecoder.decode (utils.py): 1 s units of <unit> + 10 audio embeddings, up to 20 decoded
tokens per unit, the TTS condition from every token after the first, 25-code TTS chunks, and
token2wav in 28-token windows. Writes OUT/<name>.wav (24 kHz) and OUT/<name>.json (unit log).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor

import minicpmo_audio as audio
import minicpmo_llm as llm
import minicpmo_tts as tts
from minicpmo_flow import DiTStep, Token2wavHost

SAMPLE_RATE, UNIT, FIRST_UNIT = 16000, 16000, 16560
FIRST_EMIT = (FIRST_UNIT // 160) * 160  # the mel processor aligns the first chunk to the hop
REDUNDANCY = 2  # cnn_redundancy_ms = 20
LISTEN, SPEAK, TTS_BOS, CHUNK_EOS, CHUNK_TTS_EOS, TURN_EOS, TTS_PAD = 151705, 151706, 151703, 151718, 151721, 151717, 151722
UNIT_TOKEN, UNIT_END = 151683, 151684
CHUNK_TERMINATORS = (LISTEN, CHUNK_EOS, CHUNK_TTS_EOS)
MAX_SPEAK_TOKENS = 20
TTS_EOS, TTS_AUDIO_BOS_TEXT = 6561, 151687
SILENCE_TOKEN, T2W_CHUNK, T2W_LOOKAHEAD = 4218, 25, 3


class StreamingMel:
    """StreamingMelProcessorExact(chunk 1000 ms, first 1035 ms, redundancy 20 ms, 30 s / 10 s slide)
    over MiniCPMAAudioProcessor's log-mel (fixed -10 floor below 5 s of buffer, else max - 8)."""

    def __init__(self, filters: np.ndarray) -> None:
        self.filters = torch.from_numpy(filters.astype(np.float32))  # [201, 80]
        self.buffer = np.zeros(0, np.float32)
        self.emitted, self.base = 0, 0

    def features(self) -> torch.Tensor:
        wave = torch.from_numpy(self.buffer)
        spectrum = torch.stft(wave, 400, 160, window=torch.hann_window(400), return_complex=True)
        power = spectrum[..., :-1].abs() ** 2
        log = torch.clamp(self.filters.T @ power, min=1e-10).log10()
        floor = log.max() - 8.0 if len(self.buffer) >= 5 * SAMPLE_RATE else torch.tensor(-10.0)
        return ((torch.maximum(log, floor) + 4.0) / 4.0).numpy()  # [80, frames]

    def process(self, chunk: np.ndarray) -> np.ndarray:
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32)])
        if len(self.buffer) >= 30 * SAMPLE_RATE:
            drop = min(10 * SAMPLE_RATE, len(self.buffer) - min(SAMPLE_RATE, len(self.buffer) - 10 * SAMPLE_RATE))
            drop = (drop // 160) * 160
            self.buffer, self.base = self.buffer[drop:], self.base + drop // 160
        mel = self.features()
        start = max(0, self.emitted - REDUNDANCY - self.base)
        end = self.emitted + 100 + REDUNDANCY - self.base
        self.emitted += 100
        return mel[:, start:end]


class Graphs:
    """Float graphs with phone I/O; the LLM parts run in bf16 when asked to save memory."""

    def __init__(self, model_dir: Path, work: Path, device: torch.device, llm_dtype: torch.dtype) -> None:
        w = llm.Weights(model_dir)
        self.device, self.dtype = device, llm_dtype
        self.parts = [llm.LLMPart(w, p).to(device, llm_dtype).eval() for p in range(llm.PARTS)]
        self.head = llm.LLMHead(w).to(device).eval()
        self.encoder = audio.AudioEncoder(w).to(device).eval()
        self.tts = tts.TTSModel(w).to(device).eval()
        self.embed = w("llm.model.embed_tokens.weight").to(torch.bfloat16)
        self.emb_text = w("tts.emb_text.weight")
        self.emb_code = w("tts.emb_code.0.weight")
        self.positions = w("apm.embed_positions.weight")
        t2w = work / "token2wav"
        cpu = ["CPUExecutionProvider"]
        self.flow_encoder = onnxruntime.InferenceSession(str(t2w / "flow_encoder.onnx"), providers=cpu)
        self.f0 = onnxruntime.InferenceSession(str(t2w / "hift_f0" / "onnx" / "model.onnx"), providers=cpu)
        self.decode = onnxruntime.InferenceSession(str(t2w / "hift_decode" / "onnx" / "model.onnx"), providers=cpu)
        self.dit = DiTStep(torch.load(t2w / "flow_state.pt")).to(device).eval()
        self.constants = dict(np.load(t2w / "constants.npz"))
        self.mel_filters = WhisperFeatureExtractor(feature_size=80).mel_filters  # [201, 80]

    @torch.no_grad()
    def run(self, module: torch.nn.Module, inputs, dtype=torch.float32) -> list[np.ndarray]:
        tensors = [torch.as_tensor(np.ascontiguousarray(v), dtype=dtype, device=self.device) for v in inputs]
        return [r.float().cpu().numpy() for r in module(*tensors)]

    def token2wav(self) -> Token2wavHost:
        return Token2wavHost(
            self.constants,
            lambda tokens, pad, cnn, att: self.flow_encoder.run(None, {"tokens": tokens[None], "pad": np.zeros((1, pad, 512), np.float32), "cnn_cache": cnn, "att_cache": att}),
            lambda inputs: self.run(self.dit, inputs),
            lambda mel: self.f0.run(None, {"mel": mel})[0],
            lambda mel, s: self.decode.run(None, {"mel": mel, "s_stft": s})[0],
        )


class LLMState:
    """Host KV rings of the four parts. New tokens go to slot position % PAST."""

    def __init__(self, g: Graphs) -> None:
        self.g = g
        self.rings = [llm.HostRing(g.device) for _ in range(llm.PARTS)]
        self.position = 0
        self.pending: list[np.ndarray] = []  # embeddings fed without needing logits yet

    def feed(self, embeds: np.ndarray, need_logits: bool = True):
        """Feeds pending + embeds [n, 4096]; returns (logits [vocab], semantic [768]) of the last row."""
        rows = np.concatenate(self.pending + [embeds]) if self.pending else embeds
        self.pending = []
        if not need_logits:
            self.pending = [rows]
            return None
        hidden = None
        for start in range(0, len(rows), llm.AR_N):
            chunk = rows[start : start + llm.AR_N]
            n = 1 if len(chunk) == 1 else llm.AR_N
            x = np.zeros((1, n, llm.DIM), np.float32)
            x[0, : len(chunk)] = chunk
            for p, ring in enumerate(self.rings):
                mask, cos, sin, past = ring.inputs(self.position, len(chunk), n)
                mask = self.window_mask(mask, len(chunk), n)
                out = self.g.run(self.g.parts[p], [x, mask.cpu(), cos.cpu(), sin.cpu(), *[t.cpu() for t in past]], self.g.dtype)
                ring.write(self.position, len(chunk), [torch.from_numpy(o).to(self.g.device) for o in out[1:]])
                x = out[0]
            hidden = x[0, len(chunk) - 1]
            self.position += len(chunk)
        logits, semantic = self.g.run(self.g.head, [hidden.reshape(1, 1, -1)])
        return logits[0], semantic[0]

    def window_mask(self, mask: torch.Tensor, count: int, n: int) -> torch.Tensor:
        # Past the ring capacity every slot holds one of the last PAST positions.
        if self.position >= llm.PAST:
            mask[..., : llm.PAST] = 0
        return mask


class Sampler:
    """StreamDecoder.decode (sampling) with the duplex defaults."""

    def __init__(self, seed: int, temperature=0.7, top_k=100, top_p=0.8, penalty=1.05, window=512) -> None:
        self.rng = np.random.default_rng(seed)
        self.temperature, self.top_k, self.top_p, self.penalty, self.window = temperature, top_k, top_p, penalty, window
        self.generated: list[int] = []

    def choice(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(len(probs), p=probs / probs.sum()))

    def __call__(self, logits: np.ndarray, greedy: bool = False) -> int:
        logits = logits.astype(np.float64).copy()
        probs = np.exp(logits - logits.max())
        first = int(np.argmax(logits)) if greedy else self.choice(probs)
        if first == CHUNK_EOS:
            return CHUNK_EOS
        logits[[TTS_PAD, CHUNK_EOS]] = -np.inf
        for token in set(self.generated[-self.window :]):
            logits[token] /= self.penalty
        if greedy:
            token = int(np.argmax(logits))
        else:
            logits = logits / self.temperature
            keep = np.argsort(-logits)[: self.top_k]
            filtered = np.full_like(logits, -np.inf)
            filtered[keep] = logits[keep]
            order = np.argsort(-filtered)
            sorted_probs = np.exp(filtered[order] - filtered[order[0]])
            sorted_probs /= sorted_probs.sum()
            remove = np.cumsum(sorted_probs) > self.top_p
            remove[1:] = remove[:-1].copy()
            remove[0] = False
            filtered[order[remove]] = -np.inf
            probs = np.exp(filtered - filtered.max())
            token = self.choice(probs)
        self.generated.append(token)
        return token


class TTSState:
    def __init__(self, g: Graphs, seed: int) -> None:
        self.g, self.rng = g, np.random.default_rng(seed + 1)
        self.reset()

    def reset(self) -> None:
        self.ring = tts.HostRing(self.g.device)
        self.position = 0

    def step(self, embeds: np.ndarray) -> np.ndarray:
        logits = None
        for start in range(0, len(embeds), tts.AR_N):
            chunk = embeds[start : start + tts.AR_N]
            n = 1 if len(chunk) == 1 else tts.AR_N
            x = np.zeros((1, n, tts.DIM), np.float32)
            x[0, : len(chunk)] = chunk
            mask, cos, sin, past = self.ring.inputs(self.position, len(chunk), n)
            if self.position >= tts.PAST:
                mask[..., : tts.PAST] = 0
            out = self.g.run(self.g.tts, [x, mask.cpu(), cos.cpu(), sin.cpu(), *[t.cpu() for t in past]])
            self.ring.write(self.position, len(chunk), [torch.from_numpy(o).to(self.g.device) for o in out[1:]])
            self.position += len(chunk)
            logits = out[0][0, len(chunk) - 1]
        return logits

    def generate(self, condition: np.ndarray, min_tokens: int, max_tokens: int = 26) -> list[int]:
        """MiniCPMTTS.generate_chunk: temperature 0.8, repetition penalty 1.05 over this chunk's last 16."""
        tokens: list[int] = []
        logits = self.step(condition)
        for t in range(max_tokens):
            if t > 0:
                logits = self.step(self.g.emb_code[tokens[-1]].numpy()[None])
            scores = logits.astype(np.float64) / 0.8
            if t > 0:
                for code in set(tokens[-16:]):
                    count = tokens[-16:].count(code)
                    scores[code] = scores[code] * 1.05**count if scores[code] < 0 else scores[code] / 1.05**count
            if t < min_tokens:
                scores[TTS_EOS] = -np.inf
            probs = np.exp(scores - scores.max())
            code = int(self.rng.choice(len(probs), p=probs / probs.sum()))
            if code == TTS_EOS:
                break
            if t == max_tokens - 1:
                break
            tokens.append(code)
        return tokens


class Duplex:
    def __init__(self, g: Graphs, system_embeds: np.ndarray, seed: int, greedy: bool) -> None:
        self.g, self.greedy = g, greedy
        self.mel = StreamingMel(g.mel_filters)
        self.llm = LLMState(g)
        self.sampler = Sampler(seed)
        self.tts = TTSState(g, seed)
        self.t2w = g.token2wav()
        self.audio_ring = audio.AudioRing(g.device)
        self.unit = 0
        self.turn_ended = True
        self.tts_turn_start = True
        self.t2w_buffer = [SILENCE_TOKEN] * T2W_LOOKAHEAD
        self.audio_buffer = np.zeros(0, np.float32)
        self.mels: list[np.ndarray] = []
        self.llm.feed(system_embeds, need_logits=False)

    def embed(self, token: int) -> np.ndarray:
        return self.g.embed[token].float().numpy()[None]

    def encode_audio(self, wave: np.ndarray) -> np.ndarray | None:
        self.audio_buffer = np.concatenate([self.audio_buffer, wave])
        if self.unit == 0 and len(self.audio_buffer) < FIRST_UNIT:
            self.audio_buffer = np.concatenate([np.zeros(FIRST_UNIT - len(self.audio_buffer), np.float32), self.audio_buffer])
        need = FIRST_EMIT if self.unit == 0 else UNIT
        if len(self.audio_buffer) < need:
            return None
        mel = self.mel.process(self.audio_buffer[:need])
        self.audio_buffer = self.audio_buffer[need:]
        padded = np.zeros((80, audio.MEL_IN), np.float32)
        padded[:, audio.MEL_IN - mel.shape[1] :] = mel
        self.mels.append(padded)
        frame_mask, positions, mask, past = self.audio_ring.inputs(self.unit == 0, self.g.positions)
        out = self.g.run(self.g.encoder, [padded[None], frame_mask.cpu(), positions.cpu(), mask.cpu(), *[t.cpu() for t in past]])
        self.audio_ring.write([torch.from_numpy(o).to(self.g.device) for o in out[1:]])
        return out[0][0]

    def step(self, wave: np.ndarray) -> dict:
        """One streaming_prefill + streaming_generate for a 1 s input unit."""
        embeds = self.encode_audio(wave)
        self.unit += 1
        if embeds is None:
            return {"is_listen": True, "text": [], "audio": np.zeros(24000, np.float32)}
        logits, semantic = self.llm.feed(np.concatenate([self.embed(UNIT_TOKEN), embeds]))
        condition_rows, text, is_listen, end_of_turn = [], [], False, False
        for j in range(MAX_SPEAK_TOKENS):
            if j == MAX_SPEAK_TOKENS - 1:
                self.llm.feed(self.embed(CHUNK_EOS), need_logits=False)
                break
            token = self.sampler(logits, self.greedy)
            if token == LISTEN and not self.turn_ended:
                token = TTS_BOS
            is_listen = token == LISTEN
            if token in CHUNK_TERMINATORS:
                self.llm.feed(self.embed(token), need_logits=False)
                break
            self.turn_ended = False
            logits, semantic = self.llm.feed(self.embed(token))
            end_of_turn = token == TURN_EOS
            if end_of_turn:
                self.turn_ended = True
            if j != 0:
                condition_rows.append(self.g.emb_text[token].numpy() + semantic)
                text.append(token)
        self.llm.feed(self.embed(UNIT_END), need_logits=False)
        if is_listen:
            return {"is_listen": True, "text": text, "audio": np.zeros(24000, np.float32)}
        condition = np.stack(condition_rows + [self.g.emb_text[TTS_AUDIO_BOS_TEXT].numpy()])
        turn_start = self.tts_turn_start
        min_tokens = 0 if end_of_turn or turn_start else 26
        codes = self.tts.generate(condition, min_tokens)
        self.tts_turn_start = end_of_turn
        if end_of_turn:
            self.tts.reset()
        wave = self.vocode(codes, end_of_turn, force_flush=turn_start)
        if end_of_turn:
            self.t2w.reset()
            self.t2w_buffer = [SILENCE_TOKEN] * T2W_LOOKAHEAD
        return {"is_listen": False, "text": text, "end_of_turn": end_of_turn, "audio": wave}

    def vocode(self, codes: list[int], last: bool, force_flush: bool) -> np.ndarray:
        self.t2w_buffer += codes
        pieces = []
        if force_flush:
            while len(self.t2w_buffer) >= T2W_LOOKAHEAD + 5:
                take = min(T2W_CHUNK + T2W_LOOKAHEAD, len(self.t2w_buffer))
                pieces.append(self.t2w.stream(self.t2w_buffer[:take]))
                self.t2w_buffer = self.t2w_buffer[min(T2W_CHUNK, take - T2W_LOOKAHEAD) :]
        else:
            while len(self.t2w_buffer) >= T2W_CHUNK + T2W_LOOKAHEAD:
                pieces.append(self.t2w.stream(self.t2w_buffer[: T2W_CHUNK + T2W_LOOKAHEAD]))
                self.t2w_buffer = self.t2w_buffer[T2W_CHUNK:]
        if last and self.t2w_buffer:
            pieces.append(self.t2w.stream(self.t2w_buffer, last_chunk=True))
            self.t2w_buffer = []
        wave = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        wave = np.round(np.clip(wave, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768.0
        if not last and len(wave) < 24000:
            wave = np.pad(wave, (24000 - len(wave), 0))
        return wave


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=["s16", "s17"])
    parser.add_argument("--silence", type=int, default=16)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = Graphs(args.model_dir, args.work_dir, torch.device(args.device), torch.bfloat16 if args.bf16 else torch.float32)
    # System prompt (text + reference-voice audio embeddings): every captured row before the first <unit>.
    unit = g.embed[UNIT_TOKEN].float().numpy()
    reference = np.load(args.capture_dir / f"{args.sessions[0]}.npz")["llm_embeds"]
    prompt = reference[: audio.unit_rows(reference, unit)[0] - 1]
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    for i, name in enumerate(args.sessions):
        rate, user = wavfile.read(args.capture_dir / "questions" / f"{name}.wav")
        scale = np.iinfo(user.dtype).max + 1 if np.issubdtype(user.dtype, np.integer) else 1
        user = (user.reshape(len(user), -1).mean(axis=1) / scale).astype(np.float32)
        if rate != SAMPLE_RATE:
            user = resample_poly(user, SAMPLE_RATE, rate).astype(np.float32)
        user = np.concatenate([np.zeros(16000, np.float32), user, np.zeros(16000 * args.silence, np.float32)])
        duplex = Duplex(g, prompt, seed=i, greedy=args.greedy)
        log, waves = [], []
        for u in range(len(user) // 16000):
            result = duplex.step(user[u * 16000 : (u + 1) * 16000])
            text = tokenizer.decode(result["text"], skip_special_tokens=True)
            waves.append(result["audio"])
            log.append({"is_listen": result["is_listen"], "text": text})
            print(name, u, "listen" if result["is_listen"] else f"speak> {text}", flush=True)
        captured = np.load(args.capture_dir / f"{name}.npz")["mel_chunks"]
        count = min(len(captured), len(duplex.mels))
        error = np.abs(np.stack(duplex.mels[:count]) - captured[:count]).max()
        print(f"{name}: streaming mel max abs error vs capture {error:.2e} over {count} units", flush=True)
        wavfile.write(args.out / f"{name}.wav", 24000, np.concatenate(waves).astype(np.float32))
        (args.out / f"{name}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
