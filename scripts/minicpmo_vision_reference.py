# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Official MiniCPM-o 4.5 vision path (minicpmo env): the reference for scripts/minicpmo_vision.py.

  download  the COCO val2017 photos below into --images/coco (any other photos in --images are used too)
            and the Q&A photos into --qa-dir
  embeds  every photo in --images through the official processor (duplex: max_slice_nums=1, so one
          448x448-area source image, 64 tokens) and vpm + resampler in float32 and BF16
          -> OUT/embeds/<name>.npz (pixel_values [3,14,L*14], tgt_size [2], embeds/embeds_bf16 [64,4096])
  hosttest  host-test vectors of the phone's image preparation for the --qa-images photos
  qa      BF16 duplex sessions: the photo goes in with the first unit (OMNI mode: <unit><image>[64]</image>
          [audio]), then a spoken question -> OUT/qa/<image>_<question>.json with the answer text
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoModel

from minicpmo_capture import SYSTEM_PROMPT, synthesize

# COCO val2017 photos: calibration and held-out set (download), and the Q&A test photos kept out of it.
COCO = [
    139, 632, 724, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818,
    1993, 2006, 2149, 2153, 2157, 2261, 2299, 2431, 2473, 2532, 2587, 2592, 2685, 2923, 3156, 3255, 3501, 3553,
    3661, 3845, 3934, 4134, 4395, 4495, 4765, 4795, 5001, 5037, 5060, 5193, 5477, 5503, 5529, 5586, 5600, 5992,
    6012, 6040, 6213, 6460, 6471, 6614, 6723, 6763, 6771, 6818, 6894, 6954, 7088, 7108, 7278, 7281, 7386, 7511,
    7574, 7784, 7795, 7816, 7818, 7888, 7977, 7991,
]
COCO_QA = [39769, 285, 776]  # two cats on a sofa with remotes, a brown bear, teddy bears
COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
QUESTIONS = {
    "en": "What do you see in this picture? Please describe it.",
    "zh": "这张照片里有什么？请给我描述一下。",
}


def load_model(model_dir: Path):
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        init_vision=True, init_audio=True, init_tts=True,
    )
    model.eval().cuda()
    model.init_tts()
    return model


def download(images_dir: Path, qa_dir: Path) -> None:
    for directory, ids in ((images_dir / "coco", COCO), (qa_dir, COCO_QA)):
        directory.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            target = directory / f"{image_id:012d}.jpg"
            if not target.exists():
                urllib.request.urlretrieve(COCO_URL.format(image_id), target.with_suffix(".part"))
                target.with_suffix(".part").rename(target)


def images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


@torch.no_grad()
def embeds(model, processor, args: argparse.Namespace) -> None:
    out = args.out / "embeds"
    out.mkdir(parents=True, exist_ok=True)
    vpm32, resampler32 = model.vpm, model.resampler
    for path in images(args.images):
        target = out / f"{path.parent.name}_{path.stem}.npz"
        if target.exists():
            continue
        frame = Image.open(path).convert("RGB")
        data = processor.process_image([frame], max_slice_nums=1)
        pixels = data["pixel_values"][0][0]
        tgt = data["tgt_sizes"][0]
        tgt = (tgt if isinstance(tgt, torch.Tensor) else torch.as_tensor(np.asarray(tgt))).reshape(-1, 2).int()
        result = {}
        for dtype, key in ((torch.float32, "embeds"), (torch.bfloat16, "embeds_bf16")):
            vpm32.to(dtype)
            resampler32.to(dtype)
            x = pixels.to("cuda", dtype)[None]
            mask = torch.ones(1, 1, int(tgt[0, 0] * tgt[0, 1]), dtype=torch.bool, device="cuda")
            hidden = vpm32(x, patch_attention_mask=mask, tgt_sizes=tgt.cuda()).last_hidden_state
            result[key] = resampler32(hidden, tgt.cuda())[0].float().cpu().numpy()
        np.savez(target, pixel_values=pixels.float().numpy(), tgt_size=tgt[0].numpy(), size=np.asarray(frame.size), **result)
        print(target.name, tuple(tgt[0].tolist()), flush=True)
    vpm32.to(torch.bfloat16)
    resampler32.to(torch.bfloat16)
    (args.out / "embeds.done").touch()


@torch.no_grad()
def hosttest(model, processor, args: argparse.Namespace) -> None:
    """Vectors for android/minicpmo_host_test.cpp: decoded RGB in, official pixel_values / NaViT position
    ids / resampler key positions out, per image in OUT/hosttest/<i>_*."""
    out = args.out / "hosttest"
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    hook = model.vpm.embeddings.position_embedding.register_forward_hook(lambda module, inputs, output: ids.append(inputs[0]))
    for i, path in enumerate(args.qa_images):
        frame = Image.open(path).convert("RGB")
        data = processor.process_image([frame], max_slice_nums=1)
        pixels = data["pixel_values"][0][0]
        tgt = torch.as_tensor(np.asarray(data["tgt_sizes"][0])).reshape(-1, 2).int()
        h, w = (int(v) for v in tgt[0])
        mask = torch.ones(1, 1, h * w, dtype=torch.bool, device="cuda")
        ids.clear()
        model.vpm(pixels.to("cuda", torch.bfloat16)[None], patch_attention_mask=mask, tgt_sizes=tgt.cuda())
        np.asarray(frame, np.uint8).tofile(out / f"{i}_rgb.u8")
        np.asarray([frame.size[0], frame.size[1], w, h], np.int32).tofile(out / f"{i}_dims.i32")
        pixels.float().numpy().astype(np.float32).tofile(out / f"{i}_pixels.f32")
        ids[0][0].cpu().numpy().astype(np.int32).tofile(out / f"{i}_ids.i32")
        model.resampler.pos_embed[:h, :w].reshape(h * w, -1).float().cpu().numpy().tofile(out / f"{i}_keys.f32")
        print(path.name, frame.size, (h, w), flush=True)
    hook.remove()


def qa(model, args: argparse.Namespace) -> None:
    out = args.out / "qa"
    out.mkdir(parents=True, exist_ok=True)
    for lang, text in QUESTIONS.items():
        wav = out / f"question_{lang}.wav"
        if not wav.exists():
            synthesize(model, text, args.model_dir / "assets" / "HT_ref_audio.wav", wav)
    ref_wav = args.ref_wav
    ref, _ = librosa.load(str(ref_wav), sr=16000, mono=True)
    session = model.as_duplex()
    for image in args.qa_images:
        frame = Image.open(image).convert("RGB")
        for lang in QUESTIONS:
            name = f"{image.stem}_{lang}"
            torch.manual_seed(args.seed)
            user, _ = librosa.load(str(out / f"question_{lang}.wav"), sr=16000, mono=True)
            user = np.concatenate([np.zeros(16000, np.float32), user, np.zeros(16000 * args.silence, np.float32)])
            session.prepare(prefix_system_prompt=SYSTEM_PROMPT, ref_audio=ref, prompt_wav_path=str(ref_wav))
            log, audio = [], []
            for i in range(len(user) // 16000):
                frames = [frame] if i == 0 else None
                session.streaming_prefill(audio_waveform=user[i * 16000 : (i + 1) * 16000], frame_list=frames, max_slice_nums=1)
                result = session.streaming_generate(prompt_wav_path=str(ref_wav), max_new_speak_tokens_per_chunk=20)
                wave = result["audio_waveform"]
                audio.append(np.zeros(24000, np.float32) if wave is None else wave)
                log.append({"is_listen": result["is_listen"], "text": result["text"], "end_of_turn": result["end_of_turn"]})
            sf.write(out / f"{name}.wav", np.concatenate(audio), 24000)
            (out / f"{name}.json").write_text(json.dumps(log, ensure_ascii=False))
            print(name, "".join(r["text"] for r in log), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["download", "embeds", "hosttest", "qa"])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--images", type=Path, help="photo directory (download, embeds)")
    parser.add_argument("--qa-dir", type=Path, help="Q&A photo directory (download)")
    parser.add_argument("--qa-images", type=Path, nargs="*", default=[])
    parser.add_argument("--ref-wav", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--silence", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.step == "download":
        download(args.images, args.qa_dir)
        return
    model = load_model(args.model_dir)
    if args.step == "embeds":
        embeds(model, model.as_duplex().processor, args)
    elif args.step == "hosttest":
        hosttest(model, model.as_duplex().processor, args)
    else:
        qa(model, args)


if __name__ == "__main__":
    main()
