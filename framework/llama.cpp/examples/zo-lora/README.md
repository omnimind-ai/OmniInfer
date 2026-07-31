# On-device ZO-LoRA

`llama-zo` is a standalone llama.cpp example for training a LoRA Adapter with
two-sided zeroth-order (ZO) updates. The current task implementation uses the
GLUE SST-2 dataset and can run as a native CPU reference or on a supported
Android Hexagon HTP device.

The Android packaging and ADB quick start are also available in the
[OmniInfer Android guide](../../../../docs/android/zo-lora-cli.md).

## Algorithm

For a target linear layer, the Adapter output is:

```text
y = W x + (alpha / rank) B (A x)
```

`A` is fixed for the complete run. Only the F32 master copy of `B` is updated.
At training step `t`, the trainer generates a deterministic standard-normal
noise tensor `z_t` for each LoRA target and evaluates:

```text
L_plus  = L(B_t + epsilon * z_t)
L_minus = L(B_t - epsilon * z_t)
g_t     = (L_plus - L_minus) / (2 * epsilon)
B_t+1   = B_t - learning_rate * g_t * z_t
```

The data batch and each tensor's noise stream are derived independently from
the requested seed, step, and tensor name. CPU, HTP serial, HTP paired, and
pipeline execution therefore use the same `NoisePlan` for the same seed and
step.

When no input Adapter is provided, `A` is sampled once from
`U[-1/sqrt(K), 1/sqrt(K)]` and `B` starts at zero. The generated target list
contains the standard attention Q/K/V/output and feed-forward gate/up/down
linear weights.

The SST-2 prompt is:

```text
<sentence> It was
```

The negative and positive verbalizers are `terrible` and `great`. Both must
append exactly one token with the selected model tokenizer. Accuracy compares
those two token logits. Training and evaluation loss use full-vocabulary
cross-entropy for the target verbalizer token, not a two-logit approximation.

## Execution Paths

| Path | Required options | Plus/minus execution | Padding |
| --- | --- | --- | --- |
| CPU reference | `--mode cpu --lora-exec runtime` | Two standard llama.cpp LoRA decodes | Existing CPU behavior; no HTP padding |
| HTP serial | `--mode coop --lora-exec fused-htp --antithetic false` | Two decodes selecting side 0 then side 1 | Right-padded rectangular batches |
| HTP paired | `--mode coop --lora-exec fused-htp --antithetic true` | One side-major decode containing both sides | Right-padded rectangular batches |

`--pipeline true` overlaps generation of the next host `NoisePlan` with the
current HTP forward pass. It does not change the numerical plan, model graph,
or update rule.

The public CLI name `fused-htp` selects the specialized HTP graph path. Each
LoRA target still contains three explicit HTP nodes:

1. `base = MUL_MAT(W, X)` using the native base-weight matmul path.
2. `tmp = MUL_MAT(A, X)` using the native F16 matmul path.
3. `base += scale * B_side * tmp` using an in-place LoRA accumulate op.

The critical chain is pinned to the HTP that owns the target weight. An
unsupported placement or layout is fatal instead of silently moving these
nodes to CPU. Host-side sampling, noise preparation, loss calculation, and B
updates still execute on CPU.

HTP stores both perturbation sides in one F16 tensor with logical shape:

```text
[round_up(output_dim, 64), rank, 2, 1]
```

Serial HTP uploads this complete pair once and switches the selected side
between decodes. Paired HTP assigns contiguous side values to a side-major
batch: all plus sequences first, then all minus sequences.

## Padding And Token Accounting

HTP preserves a rectangular sequence matrix for HMX-friendly work:

- The longest sequence in the current batch defines `padded_length`.
- Every sequence receives right-side pad tokens up to that length.
- Sample order is unchanged.
- Paired mode duplicates the sequence set in side-major order.
- Only each real final-token batch index requests logits. Padding positions do
  not request output rows.

For input lengths `[5, 2, 5, 3]`, a paired HTP batch has 30 real tokens, 10
padding tokens, and 40 backend tokens. CPU execution retains 30 real/backend
tokens and does not add padding.

Every timing record includes `tokens_real`, `tokens_padding`, and
`tokens_backend`. This separates useful-token throughput from the rectangular
matrix submitted to HTP.

## Supported Configurations

The specialized HTP path accepts:

- Base target weights in Q4_0, Q8_0, or F16.
- F16 LoRA A and B tensors.
- A uniform Adapter rank of 8, 16, 24, or 32.
- Plain two-dimensional token batches with at most 1024 padded tokens per
  decode.
- Target output dimensions aligned to 32 for the in-place accumulation path.

For HTP paired mode:

```text
2 * batch_size * seq_len <= 1024
```

For HTP serial mode:

```text
batch_size * seq_len <= 1024
```

The package includes skels for v68, v69, v73, v75, v79, and v81. HMX kernels
are built for v73, v75, v79, and v81; v68 and v69 use the generic HVX path.
`--hexagon-arch auto` is recommended because an explicit override must match
the architecture detected on the device.

`GGML_HEXAGON_USE_HMX=1` enables HMX for eligible HTP operations. Quantized
base matmul HMX eligibility includes an input dimension aligned to 256, an
output dimension aligned to 32, and at least 32 activation rows. HMX Flash
Attention additionally requires a supported shape, no attention sinks, enough
tokens, and 128-byte-aligned addresses and strides. Other HTP work uses HVX.
For example, the tested rank-8 LoRA-A projection and the LoRA accumulate op use
HVX. The complete graph is not HMX-only.

The trainer validates the Adapter, graph placement, and HTP Flash Attention
allocation before useful training work. Unsupported configurations should be
treated as configuration errors, not as CPU fallback candidates.

## Prepare Data And Model

Run the following commands from the OmniInfer repository root.

Download the public GLUE SST-2 archive:

```bash
mkdir -p data
curl -L https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -o data/SST-2.zip
unzip -q data/SST-2.zip -d data
```

The files used by the published SST-2 runs have these SHA-256 values:

```text
4c5e12ec5fabed1f7aa8b4fde0f0257412d76ab9c4314bb1eec46ba80c7a4438  data/SST-2/train.tsv
db8f4a5951968681dadc4c6ddaf53e8ba31a1436574072c3fa6273a5d333b35a  data/SST-2/dev.tsv
```

The standard splits contain 67,349 training rows and 872 validation rows.
`--max-train` controls the prefix available to deterministic sampling;
`--max-eval` controls the validation prefix.

Obtain a compatible GGUF model separately. The reference experiments use
TinyLlama 1.1B Chat v1.0 in F16 and Q4_0 formats. Model weights are not
redistributed by OmniInfer, and users remain responsible for the model's
license and terms. The evaluated artifacts are identified by:

```text
TinyLlama-1.1B-Chat-v1.0-F16.gguf
sha256 d62855cc687ed9ff7acc9509a88137ff1a3d6505e9f31b60f05b040a571b6630

TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf
sha256 d8a00e32fddab63f9986963397ad2230ea6863b8fae2aa0be8a9b3be7af75192
```

Different conversion or quantization tool versions may produce a different
GGUF hash. Always report the exact model hash with reproduced results.

## Native CPU Build

Configure a native build with the example and focused tests enabled:

```bash
cmake -S framework/llama.cpp -B build/llama-zo-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=OFF

cmake --build build/llama-zo-native \
  --target llama-zo test-zo-lora-loss test-zo-lora-padding -j

ctest --test-dir build/llama-zo-native \
  -R 'test-zo-lora-(loss|padding)' --output-on-failure
```

Run a one-step CPU reference without evaluation:

```bash
./build/llama-zo-native/bin/llama-zo \
  --model /path/to/model.gguf \
  --train-data data/SST-2/train.tsv \
  --mode cpu --lora-exec runtime \
  --pipeline false --antithetic false \
  --warmup-steps 2 --steps 1 --eval-step -1 \
  --batch-size 4 --seq-len 128 \
  --rank 8 --alpha 16 --epsilon 1e-2 --lr 5e-5 \
  --seed 1337 --threads 8 \
  --lora-out cpu-reference.gguf
```

`--train-data` is always required. `--eval-data` is required when
`--eval-step` is positive; it may be omitted only with `--eval-step -1`.

## Android Build And Deployment

The validated build host uses Android NDK r28c, Hexagon SDK 6.4.0.2, Hexagon
Tools 19.0.04, CMake 3.22.2 or newer, and Ninja. Export public installation
paths and run the package script:

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk-r28c
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk-6.4.0.2
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"

./scripts/platforms/android/build-llama-zo.sh --dry-run
./scripts/platforms/android/build-llama-zo.sh --clean
```

`CMAKE_BIN` defaults to `cmake` in `PATH`. The package is written to
`.local/runtime/android/llama-zo` and includes the CLI, dynamic host
libraries, and all six HTP skels.

Deploy into a dedicated directory:

```bash
adb shell mkdir -p /data/local/tmp/omniinfer-zo/{models,data,adapters,logs}
adb push .local/runtime/android/llama-zo /data/local/tmp/omniinfer-zo/
adb push /path/to/model.gguf /data/local/tmp/omniinfer-zo/models/model.gguf
adb push data/SST-2/train.tsv /data/local/tmp/omniinfer-zo/data/train.tsv
adb push data/SST-2/dev.tsv /data/local/tmp/omniinfer-zo/data/dev.tsv
```

The launcher sets `LD_LIBRARY_PATH`, `ADSP_LIBRARY_PATH`, and
`DSP_LIBRARY_PATH` relative to its package. It does not select a model or
dataset automatically.

## Android Run Matrix

The following commands run inside an interactive `adb shell`:

```sh
cd /data/local/tmp/omniinfer-zo
```

CPU reference:

```sh
./llama-zo/run-llama-zo.sh \
  --model models/model.gguf --train-data data/train.tsv \
  --mode cpu --lora-exec runtime \
  --pipeline false --antithetic false \
  --warmup-steps 5 --steps 20 --eval-step -1 \
  --batch-size 4 --seq-len 128 --max-train 67349 --max-eval 872 \
  --rank 8 --alpha 16 --epsilon 1e-2 --lr 5e-5 \
  --seed 1337 --threads 8 --lora-out adapters/cpu-20.gguf
```

HTP serial:

```sh
GGML_HEXAGON_NDEV=1 GGML_HEXAGON_USE_HMX=1 \
GGML_HEXAGON_VERBOSE=0 GGML_HEXAGON_PROFILE=0 \
./llama-zo/run-llama-zo.sh \
  --model models/model.gguf --train-data data/train.tsv \
  --mode coop --lora-exec fused-htp --device HTP0 --n-gpu-layers 99 \
  --hexagon-arch auto --pipeline false --antithetic false \
  --warmup-steps 5 --steps 20 --eval-step -1 \
  --batch-size 4 --seq-len 128 --max-train 67349 --max-eval 872 \
  --rank 8 --alpha 16 --epsilon 1e-2 --lr 5e-5 \
  --seed 1337 --threads 8 --lora-out adapters/htp-serial-20.gguf
```

HTP paired with host NoisePlan pipelining:

```sh
GGML_HEXAGON_NDEV=1 GGML_HEXAGON_USE_HMX=1 \
GGML_HEXAGON_VERBOSE=0 GGML_HEXAGON_PROFILE=0 \
./llama-zo/run-llama-zo.sh \
  --model models/model.gguf --train-data data/train.tsv \
  --mode coop --lora-exec fused-htp --device HTP0 --n-gpu-layers 99 \
  --hexagon-arch auto --pipeline true --antithetic true \
  --warmup-steps 5 --steps 20 --eval-step -1 \
  --batch-size 4 --seq-len 128 --max-train 67349 --max-eval 872 \
  --rank 8 --alpha 16 --epsilon 1e-2 --lr 5e-5 \
  --seed 1337 --threads 8 --lora-out adapters/htp-paired-20.gguf
```

Use a fresh B=0 Adapter, or omit `--lora`, for each path. Reusing an output
from a previous path changes the starting point and invalidates a performance
or parity comparison. Keep the model, data, seed, step count, and tensor
configuration identical across all three runs.

For a device profile, run one additional paired step with:

```sh
GGML_HEXAGON_NDEV=1 GGML_HEXAGON_USE_HMX=1 \
GGML_HEXAGON_VERBOSE=2 GGML_HEXAGON_PROFILE=1 \
LLAMA_ARG_LOG_VERBOSITY=5 \
./llama-zo/run-llama-zo.sh \
  --model models/model.gguf --train-data data/train.tsv \
  --mode coop --lora-exec fused-htp --device HTP0 --n-gpu-layers 99 \
  --pipeline true --antithetic true --warmup-steps 0 \
  --steps 1 --eval-step -1 --batch-size 4 --seq-len 128 \
  --rank 8 --alpha 16 --epsilon 1e-2 --lr 5e-5 \
  --seed 1337 --threads 8 --lora-out adapters/htp-profile.gguf
```

## Reproduce The Learning Result

The historical accuracy run used TinyLlama F16, the first 1,000 training rows,
all 872 validation rows, and independent 500-step and 5,000-step runs from the
same seed-1337 B=0 initialization. To reproduce that protocol, use the HTP
paired command above with:

```text
--eval-data data/dev.tsv
--max-train 1000
--max-eval 872
--batch-size 4
--seq-len 128
--rank 8
--alpha 16
--epsilon 1e-2
--lr 5e-5
--seed 1337
--warmup-steps 5
```

Run once with `--steps 500 --eval-step 500`, then start again from B=0 with
`--steps 5000 --eval-step 5000`. Each run reports a step-0 validation result
before training and a final result after its last update.

## Adapter And Timing Output

Omit `--lora` to generate a compatible Adapter. Pass
`--lora /absolute/path/adapter.gguf` to continue from an existing one. The
input and output paths must differ.

Without `--lora-out`, the output name is based on the current timestamp. For
reproducible automation, always provide a unique path. Normal completion,
SIGINT, and SIGTERM save through an atomic publish and fresh-load verification.
The final stdout line is:

```text
adapter_path=/absolute/path/to/output.gguf
```

The B master remains F32 during training. Standard Adapter GGUF output stores
A and B as F16, so continuing a saved Adapter starts from the F16 round-trip
values.

Warmups do not update B and do not consume the measured data or noise streams.
Each measured step emits a machine-readable `timing kind=train` record, and
the final `timing_summary kind=train` records include avg, p50, and p95.

Use `step_wall_us` as end-to-end update latency. The per-step
`real_tokens_per_s` and `backend_tokens_per_s` fields use decode time. Report
both rates together with `tokens_real`, `tokens_padding`, and `tokens_backend`
when evaluating padding tradeoffs. Work fields such as `loss_work_us` and
`pipeline_noise_work_us` can overlap the critical path and must not be added to
`step_wall_us`.

## Published Results

### Historical F16 learning result

The following result was measured on 2026-07-29 on a Redmi K60 Pro with
Snapdragon 8 Gen 2 / SM8550 (Hexagon v73). It used TinyLlama F16, HTP paired
execution with NoisePlan pipelining, rank 8, alpha 16, epsilon `1e-2`, learning
rate `5e-5`, seed 1337, batch size 4, sequence limit 128, the first 1,000 train
rows, and the full 872-row dev set.

| Independent run | Evaluation loss | Accuracy |
| --- | ---: | ---: |
| B=0, step 0 | 6.298321 | 484/872 = 55.504587% |
| 500 updates | 5.542850 | 500/872 = 57.339450% |
| 5,000 updates | 0.710105 | 707/872 = 81.077982% |

The 500-step and 5,000-step rows come from separate runs initialized from the
same B=0 Adapter. These are learning and correctness results, not current Q4
performance measurements.

### Cache-fixed Q4 one-step correctness

After fixing cross-format reuse of the HTP activation cache, a TinyLlama Q4_0
one-step check with batch size 4, sequence limit 128, the first 1,000 training
rows, epsilon `1e-2`, and seed 1337 produced:

| Backend | Loss plus | Loss minus | Mean |
| --- | ---: | ---: | ---: |
| CPU | 6.460639 | 6.619683 | 6.540161 |
| HTP | 6.478738 | 6.613937 | 6.546338 |

The HTP mean differs from CPU by `+0.006177`. Focused backend checks report
final-output NMSE around `1.0e-5` to `1.6e-5`. This establishes one-step
numerical parity; it is not a Q4 training-accuracy or speed claim.

### Final-source Q4_0 performance

The acceptance run was measured on 2026-07-31 on the same Redmi K60 Pro. It
used the Q4_0 model identified above, rank 8, alpha 16, epsilon `1e-2`, learning
rate `5e-5`, seed 1337, batch size 4, sequence limit 128, all 67,349 training
rows, 5 warmups, and 20 measured steps. CPU, HTP serial, and HTP paired were
run in that order. Each case started below 30 C with no stale `llama-zo`
process.

| Path | Step avg | Step p50 | Step p95 | Real token/s | Backend token/s | Real / padding / backend tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU reference | 1,109,720 us | 1,072,087 us | 1,585,413 us | 127.770 | 127.770 | 2,640 / 0 / 2,640 |
| HTP serial | 832,431 us | 761,579 us | 1,205,786 us | 176.717 | 305.774 | 2,640 / 1,928 / 4,568 |
| HTP paired + pipeline | 639,886 us | 566,687 us | 1,022,787 us | 216.318 | 374.295 | 2,640 / 1,928 / 4,568 |

The token rates are aggregate rates computed as total tokens divided by total
decode time, rather than the mean of per-step rates. Device temperatures were
26.6 to 27.0 C for CPU, 27.8 to 27.9 C for HTP serial, and 28.1 to 28.0 C for
HTP paired.

By step p50, paired HTP was 1.344x faster than serial HTP and 1.892x faster
than CPU. All three paths consumed the same measured real-token sequence.
Serial and paired HTP produced identical per-step losses and gradient scalars,
and their final Adapter files were byte-identical.

The first measured paired batch contained 206 real tokens, 90 padding tokens,
and 296 backend tokens. Three fresh profiles of that exact shape reported:

| Profile | Base matmul | LoRA-A matmul | LoRA accumulate | Flash Attention | Decode | (LoRA-A + accumulate) / decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 141,020 us | 39,081 us | 62,104 us | 162,824 us | 524,132 us | 19.305% |
| 2 | 140,911 us | 39,347 us | 62,088 us | 163,330 us | 522,029 us | 19.431% |
| 3 | 141,445 us | 39,395 us | 61,959 us | 162,461 us | 517,566 us | 19.583% |

Each profile contained 154 base matmuls, 154 LoRA-A matmuls, 154 in-place
LoRA accumulations, and 22 Flash Attention operations. HMX was active for
eligible base matmuls and Flash Attention. Graph auditing reported zero
critical-node CPU splits or fallbacks, and all three runs completed without an
HMX error. The median LoRA-A plus accumulation share was 19.431%.

## Limitations

- The example currently implements one SST-2 prompt and two fixed verbalizers.
- A model is compatible only when both verbalizers append as one token and a
  BOS token is available.
- The trainer updates only LoRA B; it is not a general backpropagation trainer.
- Evaluation-only `--steps 0` is not supported. A positive step count is
  required.
- HMX selection is conditional per operation. HVX execution inside an HTP run
  is expected and does not imply CPU fallback.
- Performance depends on model format, batch shape, SoC, firmware, clocks,
  temperature, and toolchain. Publish all of them with benchmark results.
