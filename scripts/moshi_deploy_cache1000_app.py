#!/usr/bin/env python3
"""Promote the already-tested phone bundle without replacing the old App bundle.

Runs no conversion/quantization. Context binaries stay on the same phone.
Checks the recorded four-frame outputs again before creating the App inventory.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adb', default='adb')
    parser.add_argument('--serial', required=True)
    args = parser.parse_args()
    adb = [args.adb, '-s', args.serial]

    def shell(command):
        return subprocess.check_output(adb + ['shell', command], text=True).strip()

    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(block)
        return value.hexdigest()

    shell('test -d /data/local/tmp/moshi-cache1000-full-profile-v1/dlc')
    old_wave = '/data/local/tmp/moshi-cache-profile-baseline-f4.wav.repeat.wav'
    new_wave = '/data/local/tmp/moshi-cache-profile-optimized-f4.wav.repeat.wav'
    # These must be the adjacent control trials, not a fresh unrelated input.
    for suffix in ('', '.codes.txt', '.text_tokens.txt'):
        shell(f'cmp {shlex.quote(old_wave + suffix)} {shlex.quote(new_wave + suffix)}')
    wave_sha = shell(f'sha256sum {new_wave}').split()[0]
    if wave_sha != '2701e5b3b339e135c9a4a55cb776b76f82732329c24099d4a65c486a739b64e1':
        raise RuntimeError('Recorded four-frame WAV no longer matches the verified trial')

    baseline = json.loads((args.baseline / 'dlc_inventory.json').read_text())
    candidate = json.loads((args.candidate / 'temporal_candidate_inventory.json').read_text())
    if not baseline['all_graphs_verified'] or not candidate['all_temporal_converted']:
        raise RuntimeError('Incomplete source inventory')
    graphs = {}
    names = ['frontend'] + [f'temporal_layers_{i}_{i+1}' for i in range(0, 32, 2)]
    names += ['head'] + [f'depformer_codebook_{i}' for i in range(8)]
    root = '/sdcard/Android/data/com.overmindy.mimicodecdemo/files'
    bundle = root + '/models/lm-cache1000-v1'
    cache = root + '/qnn-context-v1/cache1000-v1'
    sources = '/data/local/tmp/moshi-cache1000-full-profile-v1/dlc'
    contexts = '/data/local/tmp/moshi_context_chain_f2_v1.wav.contexts-v1'
    shell(f'mkdir -p {bundle}/dlc {cache}')
    # Android's FUSE assigns app-owned directory permissions; do not chmod
    # those directories from the shell UID. Regular copied files need 0644.
    for name in names:
        temporal = name.startswith('temporal_')
        metadata = dict((candidate if temporal else baseline)['graphs'][name])
        local = (args.candidate if temporal else args.baseline) / metadata['file']
        if digest(local) != metadata['sha256']:
            raise RuntimeError('Local model SHA mismatch: ' + name)
        source, target = f'{sources}/{name}.dlc', f'{bundle}/dlc/{name}.dlc'
        if shell(f'sha256sum {source}').split()[0] != metadata['sha256']:
            raise RuntimeError('Phone source SHA mismatch: ' + name)
        # Resume safely: never overwrite a differing destination.
        shell(f'if test -e {target}; then cmp {source} {target}; else cp -p {source} {target}; fi')
        if shell(f'sha256sum {target}').split()[0] != metadata['sha256']:
            raise RuntimeError('Phone destination SHA mismatch: ' + name)
        size, mtime = shell(f'stat -c "%s %Y" {target}').split()
        context = f'{name}-{size}-{mtime}-2.34.0-5.45.0.bin'
        shell(f'test -s {contexts}/{context}; if test -e {cache}/{context}; then '
              f'cmp {contexts}/{context} {cache}/{context}; else cp -p {contexts}/{context} {cache}/{context}; fi')
        shell(f'chmod 644 {target} {cache}/{context}')
        graphs[name] = metadata
        graphs[name]['device_verified'] = True
        graphs[name]['verification_scope'] = 'same-phone four-frame chained byte parity vs guarded baseline'
        graphs[name]['source_mtime'] = int(mtime)
        graphs[name]['device_context_file'] = context
        print(f'APP MODEL + CONTEXT VERIFIED {name}', flush=True)

    for name in ('mimi_encoder', 'mimi_decoder'):
        source = f'/data/local/tmp/moshi-mimi-codec-v1/{name}.dlc'
        target = f'{bundle}/{name}.dlc'
        shell(f'if test -e {target}; then cmp {source} {target}; else cp -p {source} {target}; fi')
        size, mtime = shell(f'stat -c "%s %Y" {target}').split()
        context = f'{name}-{size}-{mtime}-2.34.0-5.45.0.bin'
        original = f'/data/local/tmp/moshi-codec-context-v1/{context}'
        shell(f'test -s {original}; if test -e {cache}/{context}; then cmp {original} {cache}/{context}; '
              f'else cp -p {original} {cache}/{context}; fi')
        shell(f'chmod 644 {target} {cache}/{context}')
        print(f'APP CODEC + CONTEXT VERIFIED {name}', flush=True)

    inventory = {
        'format': 'moshi-lm-cache1000-app-inventory-v1',
        'runtime_preset': 'cache1000-shared-v1',
        'all_graphs_verified': True,
        'verification_scope': {
            'artifact_sha256': True, 'device_serial': args.serial,
            'native_chained_frames': 4, 'byte_parity_vs_guarded_baseline': True,
            'wav_sha256': wave_sha, 'long_context_wrap_verified': False,
            'long_audio_quality_verified': False, 'installed_app_verified': False,
        },
        'graphs': graphs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + '\n')
    subprocess.run(adb + ['push', str(args.output), bundle + '/dlc_inventory.json'], check=True)
    print('App bundle:', bundle)
    print('Same-phone caches:', cache)
    print('Select both Mimi and LM from this folder; old bundle is unchanged.')


if __name__ == '__main__':
    main()
