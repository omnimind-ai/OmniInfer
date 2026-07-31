# Android ZO-LoRA CLI

`llama-zo` is a standalone native trainer for running SST-2 zeroth-order
LoRA experiments on Android. It uses the vendored llama.cpp tree directly and
does not depend on OmniInfer's JNI, AAR, Gradle, or Rust control plane.

For the algorithm, execution modes, native reference build, and experiment
protocol, see the [ZO-LoRA example README](../../framework/llama.cpp/examples/zo-lora/README.md).

## Host Requirements

The Android package script is intended for a Linux x86_64 build host. The
validated toolchain is:

- Android NDK r28c
- Qualcomm Hexagon SDK 6.4.0.2
- Hexagon Tools 19.0.04
- CMake 3.22.2 or newer
- Ninja
- ADB for deployment

The Android NDK and Hexagon SDK paths are required. `HEXAGON_TOOLS_ROOT`
defaults to the Tools 19.0.04 installation inside the SDK, and `CMAKE_BIN`
defaults to `cmake` from `PATH`.

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk-r28c
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk-6.4.0.2
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"

./scripts/platforms/android/build-llama-zo.sh --dry-run
./scripts/platforms/android/build-llama-zo.sh --clean
```

The llama.cpp sources are vendored in `framework/llama.cpp`; no submodule
initialization is required. The build targets Android API 26 and `arm64-v8a`.
It produces `llama-zo`, dynamic CPU and Hexagon host libraries, and skels for
v68, v69, v73, v75, v79, and v81 under:

```text
.local/runtime/android/llama-zo/
```

The build script only cross-compiles and packages files. It does not invoke
ADB.

## Prepare SST-2

Download the public GLUE SST-2 archive on the host:

```bash
mkdir -p data
curl -L https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -o data/SST-2.zip
unzip -q data/SST-2.zip -d data
```

The CLI expects the strict two-column `sentence<TAB>label` format used by
`SST-2/train.tsv` and `SST-2/dev.tsv`. A compatible GGUF model must be supplied
separately; model files are not distributed by this repository.

## Deploy

Use a dedicated public example directory instead of overwriting another phone
installation:

```bash
adb shell mkdir -p /data/local/tmp/omniinfer-zo/{models,data,adapters,logs}
adb push .local/runtime/android/llama-zo /data/local/tmp/omniinfer-zo/
adb push /path/to/model-Q4_0.gguf /data/local/tmp/omniinfer-zo/models/model.gguf
adb push data/SST-2/train.tsv /data/local/tmp/omniinfer-zo/data/train.tsv
adb push data/SST-2/dev.tsv /data/local/tmp/omniinfer-zo/data/dev.tsv
```

Run a one-step CPU reference smoke test. Evaluation is disabled, so
`--eval-data` is intentionally omitted:

```bash
adb shell 'cd /data/local/tmp/omniinfer-zo && ./llama-zo/run-llama-zo.sh \
  --model models/model.gguf --train-data data/train.tsv \
  --mode cpu --lora-exec runtime --pipeline false --antithetic false \
  --warmup-steps 2 --steps 1 --eval-step -1 \
  --batch-size 4 --seq-len 128 --rank 8 --alpha 16 \
  --epsilon 1e-2 --lr 5e-5 --seed 1337 --threads 8 \
  --lora-out adapters/cpu-smoke.gguf'
```

Run the paired Hexagon HTP path with HMX enabled for eligible operations:

```bash
adb shell 'cd /data/local/tmp/omniinfer-zo && \
  GGML_HEXAGON_NDEV=1 GGML_HEXAGON_USE_HMX=1 \
  GGML_HEXAGON_VERBOSE=0 GGML_HEXAGON_PROFILE=0 \
  ./llama-zo/run-llama-zo.sh \
  --model models/model.gguf --train-data data/train.tsv \
  --mode coop --lora-exec fused-htp --device HTP0 --n-gpu-layers 99 \
  --hexagon-arch auto --pipeline true --antithetic true \
  --warmup-steps 5 --steps 20 --eval-step -1 \
  --batch-size 4 --seq-len 128 --max-train 67349 --max-eval 872 \
  --rank 8 --alpha 16 --epsilon 1e-2 --lr 5e-5 --seed 1337 --threads 8 \
  --lora-out adapters/htp-paired.gguf'
```

Set `--antithetic false` and `--pipeline false` for the HTP serial reference.
The serial and paired HTP paths use the same seeded batches and perturbations.

## Runtime Rules

- `--train-data` is always required.
- `--eval-data` is required when `--eval-step` is positive and may be omitted
  only when `--eval-step -1` disables evaluation.
- HTP accepts at most 1024 padded tokens per decode. Paired mode therefore
  requires `2 * batch-size * seq-len <= 1024`; serial HTP requires
  `batch-size * seq-len <= 1024`.
- HTP supports Q4_0, Q8_0, and F16 target weights with Adapter rank 8, 16, 24,
  or 32. Unsupported placement, shape, or layout is rejected rather than
  moving the critical ZO-LoRA chain to CPU.
- HTP right-pads every sequence in a batch to the longest sequence. Paired
  mode is side-major: all plus sequences are followed by all minus sequences.
  CPU execution retains its unpadded behavior.
- HMX is available in the packaged v73, v75, v79, and v81 skels. Eligible base
  matrix multiplications and Flash Attention may use HMX; other HTP work uses
  HVX. Enabling HMX does not make the complete graph HMX-only.

The default `--hexagon-arch auto` selects the skel that matches the device.
An explicit override must match the detected architecture.

## Outputs And Timing

Omit `--lora` to generate an Adapter with fixed F16 A tensors and zero-initialized
B tensors. Pass `--lora /path/to/adapter.gguf` to continue from an existing
Adapter. The in-process B master weights are F32; saved standard Adapter GGUF
files contain F16 A and B tensors.

Use a unique `--lora-out` path for reproducible runs. Successful completion,
SIGINT, and SIGTERM perform an atomic save plus a fresh-load check, then print:

```text
adapter_path=/absolute/path/to/adapter.gguf
```

Each measured update emits `timing kind=train`; the final summary reports avg,
p50, and p95. `step_wall_us` is the end-to-end step latency. The reported
`real_tokens_per_s` and `backend_tokens_per_s` use decode time, while
`tokens_real`, `tokens_padding`, and `tokens_backend` expose padding overhead.
Do not add overlapping work fields to `step_wall_us`.
