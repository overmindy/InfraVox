"""Separate generated-token changes from same-token codec numerical error.

Keep the direct end-to-end PCM result unchanged. Decoder error must be measured
against the float decoder receiving the exact same generated token sequence.
"""
import argparse
import json
from pathlib import Path
import re
import wave

import numpy as np

from moshi_quantize_mimi_offline import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    full = root/"chain_f38"
    report = json.loads((full/"RESULT.json").read_text())
    short = json.loads((root/"chain_f4/RESULT.json").read_text())
    codec = json.loads((root/"phone/PHONE_RESULT.json").read_text())
    assert short["passed"] and codec["short_numerical_gate_passed"]
    log = (full/"fixed_codes_float.log").read_text()
    assert "DECODED_FRAMES=37" in log and "phase=restore_context" in log
    assert not re.search(r"phase=(open_dlc|compose|finalize)\b", log)

    def audio(path):
        with wave.open(str(path), "rb") as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 71040)
            return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)

    actual = audio(full/"reply.warm.wav")
    reference = audio(full/"fixed_codes_float.wav")
    delta = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-12))
    correlation = float(np.corrcoef(actual, reference)[0, 1])
    codes = np.loadtxt(full/"reply.warm.wav.codes.txt", delimiter=",", dtype=np.int32)
    old_codes = np.loadtxt(report["reference_wav"]+".codes.txt", delimiter=",", dtype=np.int32)
    agreement = float(np.mean(codes[:, 1:] == old_codes[:, 1:]))
    passed = (report["first_warm_wav_identical"] and report["retained_temporal_count"] == 16
              and report["warm_context_restore_count"] == 0 and len(report["context_only_sources"]) == 28
              and report["text_token_agreement"] >= .95 and agreement >= .95
              and np.isfinite(actual).all() and delta <= .1 and correlation >= .99
              and report["clipped_fraction"] == 0)
    result = {"passed": bool(passed), "verification_basis": "same-code decoder error plus generated-token agreement",
              "full_chain_frames": 38, "codec_short_frames": 4,
              "direct_pcm_comparison_passed": report["passed"], "direct_pcm_relative_rms": report["pcm_relative_rms"],
              "generated_audio_token_agreement": agreement, "text_token_agreement": report["text_token_agreement"],
              "same_code_decoder_relative_rms": delta, "same_code_decoder_correlation": correlation,
              "same_code_reference_sha256": sha(full/"fixed_codes_float.wav"),
              "retained_temporal_count": 16, "warm_context_restore_count": 0,
              "independent_listening_verified": False, "long_context_wrap_verified": False,
              "thresholds": {"decoder_relative_rms": .1, "decoder_correlation": .99,
                             "generated_token_agreement": .95, "text_token_agreement": .95}}
    save(root/"COMPONENT_VERIFICATION.json", result)
    print(json.dumps(result))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
