#!/usr/bin/env python3
"""Check PCM files and frame timelines from the phone reply probe.

Amplitude checks do not establish speech quality or semantic correctness.
"""
import argparse
import array
import csv
import hashlib
import json
import math
import sys
import wave
from pathlib import Path


def pcm_summary(path):
    with wave.open(str(path), 'rb') as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 24000):
            raise ValueError('Use mono PCM16 at 24000 Hz')
        samples = array.array('h', reader.readframes(reader.getnframes()))
    if sys.byteorder != 'little':
        samples.byteswap()
    if not samples:
        raise ValueError('The WAV file contains no samples')
    rms = math.sqrt(sum(float(x) * x for x in samples) / len(samples)) / 32768
    first_window = None
    for offset in range(0, len(samples), 480):
        window = samples[offset:offset + 480]
        energy = math.sqrt(sum(float(x) * x for x in window) / len(window)) / 32768
        if energy >= 0.01:  # -40 dBFS, an amplitude threshold, not a voice detector.
            first_window = offset / 24000
            break
    return {
        'seconds': len(samples) / 24000,
        'samples': len(samples),
        'rms_dbfs': 20 * math.log10(rms) if rms > 0 else None,
        'peak_normalized': max(abs(x) for x in samples) / 32768,
        'clipped_fraction': sum(abs(x) >= 32767 for x in samples) / len(samples),
        'nonzero_fraction': sum(x != 0 for x in samples) / len(samples),
        'first_20ms_window_above_minus40_dbfs_seconds': first_window,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', default='reply')
    args = parser.parse_args()
    root = args.directory
    prefix = args.prefix
    latency = json.loads((root / f'{prefix}.latency.json').read_text())
    result = {'latency': latency, 'runs': {}, 'notes': [
        'PCM readiness excludes Java UI, speaker latency, and model file verification.',
        'A PCM block is not proof of an intelligible or relevant spoken reply.',
        'The amplitude threshold is not speech recognition.',
    ]}
    for measured in latency['runs']:
        name = measured['name']
        wav_path = root / f'{prefix}.{name}.wav'
        audio = pcm_summary(wav_path)
        with (root / f'{prefix}.{name}.timeline.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != measured['input_frames']:
            raise ValueError('Frame callback count does not match the report')
        if audio['samples'] != measured['output_frames'] * 1920:
            raise ValueError('PCM sample count does not match the report')
        pcm_rows = [row for row in rows if int(row['pcm_samples']) > 0]
        gaps = [float(b['elapsed_ms']) - float(a['elapsed_ms'])
                for a, b in zip(pcm_rows, pcm_rows[1:])]
        configuration = json.loads((root / f'{prefix}.{name}.wav.runtime.json').read_text())
        with (root / f'{prefix}.{name}.wav.text_tokens.txt').open() as stream:
            text_ids = [int(line.strip().split(',')[1]) for line in stream if line.strip()]
        above_threshold = audio['first_20ms_window_above_minus40_dbfs_seconds']
        first_signal_chunk = None
        if above_threshold is not None:
            emitted_index = round(above_threshold * 24000) // 1920 + 1
            first_signal_chunk = next(float(row['elapsed_ms']) for row in pcm_rows
                                      if int(row['emitted_frames']) == emitted_index)
        first_text = next((index for index, token in enumerate(text_ids)
                           if token not in (0, 3)), None)
        first_text_ready = None if first_text is None else next(
            float(row['elapsed_ms']) for row in pcm_rows
            if int(row['emitted_frames']) == first_text + 1)
        result['runs'][name] = {
            'audio': audio,
            'output_rtf': measured['total_ms'] / (audio['seconds'] * 1000),
            'max_pcm_chunk_gap_ms': max(gaps) if gaps else None,
            'first_chunk_above_minus40_dbfs_ready_ms': first_signal_chunk,
            'first_non_control_text_token_ready_ms': first_text_ready,
            'planned_temporal_count': len(configuration['lm']['planned_temporal']),
            'retained_temporal_count': len(configuration['lm']['retained_temporal']),
            'generated_text_token_ids': text_ids,
        }
    result['first_warm_wav_byte_equal'] = (
        result['runs']['first']['audio']['sha256'] == result['runs']['warm']['audio']['sha256'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
