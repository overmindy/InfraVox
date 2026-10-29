---
name: add-duplex-model
description: Port a new full-duplex (listen-while-speaking) speech model to InfraVox, the on-device Android app that runs such models on the Snapdragon Hexagon NPU through QNN. Covers studying the upstream streaming loop, splitting it into fixed-shape graphs, AIMET quantization, QAIRT compilation to SoC-specific context binaries, a C++ runner on qnn_runtime.h, the Java engine, the model catalog and publishing. Use when asked to "add / adapt / port model X to InfraVox", or to add a new variant of an existing InfraVox model.
---

# Add a duplex model to InfraVox

You are porting one speech model so that the InfraVox app can hold a live, interruptible voice conversation
with it entirely on the phone. The two existing ports are the reference implementations; **read the matching
files of the closer one before writing anything**, and copy its structure:

| Existing port | Shape of the model | Files |
| --- | --- | --- |
| MiniCPM-o 4.5 | audio encoder → 8B LLM decides listen/speak once per 1 s unit → TTS → token2wav vocoder | `scripts/minicpmo_*`, `android/minicpmo_runner.cpp`, `android/demo/minicpmo_jni.cpp`, `engine/MiniCpmo*.java` |
| Moshi | codec encoder → 7B temporal transformer every 80 ms frame → depth transformer → codec decoder | `scripts/moshi_*`, `android/mimi_qnn_runner.cpp`, `android/demo/moshi_jni.cpp`, `engine/Moshi*.java` |

`android/README.md` documents both ports (data flow, graph contracts, quantization, pitfalls). Read it fully once.

## Fill in before you start

Nothing in this skill names a machine or a path. Find each value yourself (environment variables, existing
`scripts/*_env.sh` files, `which`, asking the user once) and write them into a new `scripts/<model>_env.sh`
in the same style as the existing env files, so every later script sources it. Every value must be
overridable from the environment.

| Placeholder | Meaning | How to find it |
| --- | --- | --- |
| `<model>` | short id used in file names, the catalog `engine` field and Java class names | pick from the upstream name, lowercase, no dashes |
| `<UPSTREAM>` | upstream weights (HF repo id or URL) and the official inference code | model card |
| `<BUILD_HOST>` | the Linux machine with NVIDIA GPUs that runs export/calibration/compilation | ask the user if no env file says |
| `<WORK_ROOT>` | big scratch dir on `<BUILD_HOST>` (captures, encodings, DLCs, bundles; 50–150 GB) | check free space with `df -h` first |
| `<GPUS>` | GPU ids you may use on `<BUILD_HOST>` | `nvidia-smi`; never take GPUs other people are using |
| `<QAIRT_SDK_ROOT>` | Qualcomm AI Runtime SDK, the same version the app ships (see `android/models/catalog.json` `qairt`) | existing env files |
| `<UPSTREAM_PY>` | Python env that runs the official upstream code (float, GPU) | create one if needed |
| `<AIMET_PY>` | Python env with aimet-onnx + onnxruntime-gpu + this repo (editable) | existing env files |
| `<ANDROID_SDK>`, `<NDK>`, `<QNN_INCLUDE>`, `<QNN_ANDROID_LIBS>` | Android build inputs on the dev machine | `scripts/*_deploy.sh` defaults |
| `<PHONE>` | a test phone over `adb`; note its SoC (`adb shell getprop ro.soc.model`) | `adb devices -l` |
| `<HF_REPO>` | Hugging Face model repo the app downloads from | `HubConfig.DEFAULT_REPO` |

If a value cannot be determined, stop and ask the user for exactly that value. Do not guess paths.

## Tools you will use

- **Upstream code + PyTorch** on `<BUILD_HOST>`: the ground truth. Everything is compared against it.
- **ONNX export** (`torch.onnx.export`, fixed shapes, opset ≥ 17) — one ONNX per on-device graph.
- **AIMET (aimet-onnx)**: calibration on captured real inputs → `<graph>.encodings`.
- **QAIRT** in `<QAIRT_SDK_ROOT>/bin/x86_64-linux-clang`:
  `qairt-converter` (ONNX → float DLC, `--quantization_overrides <encodings>`),
  `qairt-quantizer` (→ quantized DLC), `qnn-context-binary-generator` (DLC → SoC context `.bin`),
  `qnn-context-binary-utility` (dump graph names / tensors as JSON). Wrap them like `scripts/minicpmo_qairt_build.sh`.
- **NDK clang++** for the runner, host tests and phone CLIs; **adb** to push and run on `<PHONE>`.
- **`android/qnn_runtime.h`**: the shared QNN layer. `Runtime`, `Context`, `Graph`, `SharedBuffer` (rpcmem),
  `setReal/getReal` (fixed-point I/O).
- **`scripts/hf_publish.py`** (model files + catalog) and **`scripts/app_publish.py`** (APK + update manifest).

## Workflow

Do the phases in order. Each phase has an exit check; do not start the next phase until it passes.
Keep a running log (`<WORK_ROOT>/<model>/NOTES.md`) of every number you measure.

### 1. Reproduce upstream in float

1. Install the upstream code in `<UPSTREAM_PY>` and run its **streaming / duplex** demo, not an offline one.
2. Write down the loop exactly: input sample rate, frame or unit length, what runs per step, state carried
   between steps (KV caches, codec states, conv caches), sampling parameters, how it decides to listen or speak,
   how speech is turned back into audio, and output sample rate.
3. Write `scripts/<model>_capture.py`: run the official loop on 10–20 realistic multi-turn inputs (both languages
   if the model is bilingual) and save, per step, the inputs and outputs of every block you plan to put on the
   NPU. These files are the calibration data **and** the golden reference.

Exit check: the capture replays deterministically (fixed seed) and the saved outputs match a second run.

### 2. Decide the graph split

For each block, decide NPU graph vs CPU:

- NPU graphs must have **static shapes**. Anything whose shape depends on history length (relative position
  encodings over a growing cache, variable-length windows) goes on the CPU (C++ or ONNX Runtime), as the
  MiniCPM-o flow conformer does.
- Transformers with a KV cache use the **AR-1 + AR-N pair** layout (`scripts/minicpmo_llm.py`): same weights,
  one graph for 1 token, one for N (16) tokens; the KV cache is a **host-owned ring** passed in as inputs, each
  graph outputs only the new K/V rows. Split big LLMs into parts of ~8–9 layers (a context must stay < 2 GB).
- Sampling, tokenization, mel/STFT front-ends and small glue run on the CPU.
- Prefer few large graphs over many small ones: each execute costs ~0.3–1 ms of overhead.

Write the contract of every graph (tensor names, shapes, dtypes, what the host fills) into `android/README.md`
under a new section, in the same table style as the existing ones.

### 3. Host simulator

Write `scripts/<model>_sim.py`: the phone's host loop in Python, calling the float graphs (PyTorch modules with
exactly the ONNX graph signatures) and doing on the CPU exactly what the C++ runner will do (ring slots, masks,
RoPE tables, sampling).

Exit check: on the captured inputs the simulator reproduces upstream's text/tokens (greedy: identical; sampled:
same distribution) and audio (listen to it). This simulator is the specification for the C++ port.

### 4. Export and quantize

Per graph, in `scripts/<model>_<block>.py` (pattern: `minicpmo_llm.py`, `minicpmo_audio.py`):

1. Export ONNX with fixed shapes; check ONNX Runtime output equals PyTorch.
2. Run AIMET calibration on the captured inputs. Starting points that worked:
   - LLM / large transformer: **LPBQ W4** weights (block 64), **A16** activations, **8-bit KV**; AR-1 reuses the
     AR-N encodings so both graphs read the same ring.
   - Audio encoders, small transformers, vocoders: **W8A16**.
   - Heads whose argmax matters: keep W8 if W4 drops top-1 agreement noticeably.
3. Measure each graph quantized vs float on held-out captured steps (teacher-forced): token top-1 agreement,
   cosine / relative error of hidden states, audio SNR. Record them.

Exit check: per-graph numbers are written down and the end-to-end simulator with quantized graphs (AIMET
QuantSim) still answers the test questions sensibly.

### 5. Compile with QAIRT

Use one wrapper script modelled on `scripts/minicpmo_qairt_build.sh`:

1. `qairt-converter` → float DLC (with `--quantization_overrides`), then `qairt-quantizer` → quantized DLC.
2. `qnn-context-binary-generator` with a backend config of `O: 3`, VTCM size, the target `soc_model` /
   `dsp_arch`, and `weight_sharing_enabled: true` when several graphs (AR-1 + AR-N) go into one context.
3. Dump the context with `qnn-context-binary-utility` and check every graph name equals the DLC name.

Then a one-click `scripts/<model>_build_all.sh` that skips finished steps (so a failed run resumes) and ends
with a bundle directory plus `SHA256SUMS`.

Context binaries are bound to the target SoC (InfraVox targets SM8975 / HTP v85 with QAIRT 2.51; see
`MCPO_TARGET` in `scripts/minicpmo_env.sh` for the soc_model / dsp_arch table) and the QAIRT version the
app bundles. Do not try to prepare DLCs on the phone instead: an 8B LLM part with AR-1 + AR-N weight sharing
finalizes on the device but cannot be serialized there (QNN status 5007).

### 6. C++ runner

`android/<model>_runner.cpp` on top of `qnn_runtime.h`, exporting `create / begin / step / end / destroy`
functions like `minicpmoCreate` etc. Rules:

- All graph I/O goes through `setReal/getReal` (fixed-point scale/offset come from the context).
- KV rings live in `SharedBuffer` (rpcmem) and are bound to every graph that reads them. Empty slots hold the
  quantized **zero** (`quantizeTo(affine, 0)`), not byte 0.
- Long conversations: when the ring is full, keep (pin) the first positions (system prompt, attention sink)
  and cycle the rest; keys keep absolute RoPE positions.
- Contexts executed one after another share a spill-fill buffer (`Runtime::groupSpillBytes`) and one execute lock.
- Overlap CPU work (vocoder, TTS) with the next step's NPU work on a worker thread when the step budget is tight.
- Write `android/<model>_host_test.cpp` for the CPU pieces (mel, STFT, sampling) against vectors dumped by
  the Python simulator, and a phone CLI (pattern: `android/moshi_cli.cpp` + `scripts/moshi_phone_cli.sh`)
  that runs a WAV through the runner as fast as possible and writes text, audio and per-graph timings.

Exit check: the phone CLI's text and audio match the quantized simulator on the same input, and the
per-step time is below the step length (real-time factor < 1) for a 3-minute input.

### 7. App integration

1. `android/demo/<model>_jni.cpp` — copy `minicpmo_jni.cpp`, rename the symbols.
2. `engine/<Model>Native.java` (native method declarations) and `engine/<Model>Engine.java` implementing
   `DuplexEngine` (input rate, frame size, output rate, playback reserve).
3. Register the engine id in `engine/Engines.java` (`supports` and `create`).
4. Compile the runner + JNI into its own `.so` in `android/demo/build.sh` and load it in the Native class.
5. Add the model and its variants to `android/models/catalog.json` (`engine` = `<model>`; `socs` empty or the
   list of supported SoCs; `htp`, `qairt`, honest `notes` with measured speed) and **increase `revision`**.
6. A deploy script `scripts/<model>_deploy.sh` (pattern: `minicpmo_deploy.sh`) that pushes a bundle to the
   phone over adb, then builds and installs the APK with the existing keystore (never generate a new one:
   reinstalling loses user data and breaks in-app updates).

Exit check on `<PHONE>`: download or push the bundle, load it from the models page, hold a 5-minute
conversation with interruptions. Record load time, step time, RTF, playback underruns, and whether every
question got an answer.

### 8. Publish

1. `python scripts/hf_publish.py --repo <HF_REPO> --variant <variant id> --bundle-dir <bundle>` per variant;
   commit the updated `android/models/catalog.json`.
2. Bump `versionCode` / `versionName` in `android/demo/AndroidManifest.xml`, build the APK, then
   `python scripts/app_publish.py --apk <apk> --notes "<what changed>" --github` so the in-app update check sees it.
3. Update both READMEs (`README.md`, `README.en.md`): supported models table with the measured numbers.

## Pitfalls already paid for

| Symptom | Cause | Fix |
| --- | --- | --- |
| Context generator segfaults on LPBQ | LPBQ on FullyConnected | express projections as 1×1 Conv |
| Hand-written LayerNorm/RMSNorm falls back to float or is numerically wrong | QAIRT fuses `gamma * rsqrt(...)` patterns | use the `layer_norm` op, or run the converter through `scripts/moshi_guard_qairt_converter.py` |
| Converter rejects weights computed at runtime | weight_norm parametrization exported as ops | fold weight_norm into the weights before export |
| Tensors inside AIMET's fused Conv+activation groups have no encodings | QAIRT does not fuse the same way | give `qairt-quantizer` a calibration `--input_list` |
| O3/VTCM options silently ignored | graph name ≠ name in the backend config | graph name must equal the DLC/context name; verify with the utility JSON |
| Model stops talking after a few minutes | KV ring wrapped over the system prompt / attention sink | pin the first positions in the ring |
| Garbage attention after reset | ring filled with byte 0 instead of the quantized zero | fill with `quantizeTo(affine, 0)` |
| 16-bit activation range blown | `-inf` attention masks | use a large finite negative or slice the cache instead |
| Model always stays silent with greedy decoding | quantization shifts the listen/speak logit | sample (temperature ~0.7, top-k / min-p) instead of argmax |
| `contextCreateFromBinary` fails on another phone | contexts are bound to SoC + QAIRT version | only the target SoC is supported; other SoCs need their own context build |
| `deviceCreate` 14001 after `remote work start failed: 3` (Android 17) | the NPU scheduler only serves apps granted by `NpuManager` | call `engine/NpuGate.java` before loading; targetSdk 37 + `uses-feature android.hardware.npu` |

## Report back

End with: what runs where (NPU/CPU per block), quantization per graph with its accuracy numbers, bundle sizes,
measured load time / step time / RTF on `<PHONE>` (name its SoC), the long-conversation result, files added or
changed, and anything left unverified.
