# Moshi LM + existing Mimi Android demo

This is an **offline, greedy, unconditioned, batch-one experimental path**.
It preserves the original Mimi codec loopback as a separate switchable mode.
APK compilation, scheduler tests and model transport checks are NOT phone QNN
execution or full-conversation numerical/quality proof.

## Follow the code

1. `demo/src/com/overmindy/mimicodecdemo/MainActivity.java`: imports the two
   existing Mimi DLCs and a complete LM directory via Android SAF, checks every
   LM DLC SHA256 while copying, switches modes, invokes JNI on a worker thread.
2. `demo/mimi_codec_jni.cpp`: supplies the optional LM directory to the native
   prepared session. Empty directory means the unchanged codec path.
3. `mimi_qnn_runner.cpp`, `PreparedMimiCodec::run`: selects `runCodecFrames` or
   `runMoshiFrames`. Mimi encoder/decoder remain loaded with their own states.
4. `moshi_lm_scheduler.h`, `MoshiDelayRing::begin/finish`: reproduces the pinned
   Kyutai `LMGen._step` delay-ring for 17 token streams. It is not sufficient
   to concatenate last-frame generated tokens with current Encoder tokens.
5. `PagedMoshiLm::step`: frontend -> 16 Temporal shards -> head -> 8 DepFormer
   graphs. Temporal K/V states persist across WAV frames. Depth K/V states are
   zeroed each WAV frame and transferred between codebooks. Text uses greedy
   argmax, audio uses the exported greedy token output; nonfinite logits fail.

Pinned upstream: `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`,
`moshi/models/lm.py`, `LMGen._step` and `moshi/models/loaders.py` defaults.
Delays: `[0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]`; text initial ID 32000,
audio initial ID 2048. These constants are specific to this exported bundle.

## Bundle layout and validation

Remote verified-module delivery:
`/data3/yejialei/moshi-lm-guarded-dlc-set-v2`.
Local download directory:
`/Users/e1ainay/d_personal/moshi-models/lm-guarded-v2` (outside Git).

```text
lm-guarded-v2/
  dlc_inventory.json
  quantization_manifest.json
  SHA256SUMS
  DOWNLOAD_VERIFIED.json     # appears only after all local hash checks pass
  dlc/
    frontend.dlc
    temporal_layers_0_1.dlc ... temporal_layers_30_31.dlc
    head.dlc
    depformer_codebook_0.dlc ... depformer_codebook_7.dlc
```

The bundle is about 12 GB; it is deliberately not packed into the APK. Check
download completion before copying to the phone:

```sh
python3 scripts/moshi_verify_downloaded_lm_bundle.py \
  --bundle-dir /Users/e1ainay/d_personal/moshi-models/lm-guarded-v2
```

If a transfer was interrupted, resume using `rsync -a --partial` from the same
remote bundle directory; do not re-export, re-quantize or re-download ONNX.

The phone directory picker must select the folder that contains both `dlc/`
and `dlc_inventory.json`, not the `dlc/` subfolder alone. Android may forbid
selecting the Downloads root: use a named subfolder inside Downloads instead.
The importer needs at least 13 GiB free app-storage space, **in addition to**
the source copy on shared storage. It validates exactly 26 expected filenames
and hashes. A replaced model directory is retained as `lm-previous-*`; failed
imports are retained as `lm-import-*` rather than destroying the old models.

## Memory, delay and limitations

64 full FP32 Temporal K/V caches alone are ~2.93 GiB. All model weights plus
all graph I/O buffers resident at once would exceed many phone memory budgets.
The initial implementation loads, executes and frees one LM graph/context at
a time. It trades very large model-load latency for lower peak memory; it
still needs several GiB available native memory and is **not real-time**.
It does not promise compatibility with low-memory devices or arbitrary HTPs.
Use QAIRT 2.45-compatible HTP libraries and the previously tested phone family.

The app defaults to processing only two frames (160 ms input) in Moshi mode,
even for a longer imported WAV. Increase the LM frame-count field explicitly
when the smoke test passes; zero means all complete frames. Codec mode still
processes the whole clip. A one-frame warm-up produces no Decoder
frame, so N input frames produce N-1 output frames, matching the upstream
streaming loop without artificial end-of-clip flush. Max input is 3000 frames
for this initial fixed-context implementation. Every new clip resets Mimi,
Temporal and depth state. Models are loaded lazily during inference: the
Prepare button does not claim that all 26 LM graphs have been finalized.

The audio token CSV contains **generated, delay-aligned** tokens in Moshi mode,
not Encoder tokens. Text token IDs are displayed/exported separately. No
SentencePiece decoder has been added, so those IDs are not a text transcript.
Greedy decoding and a short numerical check do not establish speech quality.

## Build and use

Use the environment documented in `README.md`, then preserve the existing
signing key if upgrading an installed demo (never uninstall just to bypass a
signature mismatch; that would erase its imported models):

```sh
OUT_DIR="$PWD/android/demo/build/moshi" \
KEYSTORE_PATH="$PWD/android/demo/build/demo.keystore" \
bash android/demo/build.sh
```

Connect the phone, then install the APK with `adb install -r`. Copy the verified
local bundle to a named phone folder (check ~12 GB storage beforehand), e.g.
`adb push /Users/e1ainay/d_personal/moshi-models/lm-guarded-v2 /sdcard/Download/`.
Open **Moshi QNN Demo**, keep/import the existing Mimi encoder and decoder,
import the complete LM folder, enable Moshi, prepare models, and run a short
WAV. Keep codec mode as a control for encoder/decoder-only regressions.

The standalone native runner also accepts an optional last `LM_BUNDLE_DIR`
argument after `FRAMES`. Its codec CLI is unchanged otherwise.

## Checks before claiming success

```sh
clang++ -std=c++17 -Wall -Wextra -Werror android/test_moshi_scheduler.cpp \
  -o /tmp/moshi-scheduler-test
/tmp/moshi-scheduler-test
python3 -m unittest discover -s src/qai_hub_models/test/test_scripts \
  -p test_moshi_verify_downloaded_lm_bundle.py
```

Then separately check APK build/signature, local 26-file hashes, phone graph
load/execute, two-frame tensors/tokens against references, and actual audio.
If a phone failure occurs, identify the first graph name in native logs; do
not replace the whole bundle. The ongoing remote full LM chain is a separate
reference test and does not include Mimi encoder/decoder or host delay-ring.
