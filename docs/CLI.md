# OmniInfer CLI Guide

This guide shows how to use the OmniInfer desktop CLI on Linux, macOS, and Windows.
Android and iOS use the embedded modules under `android/` and `ios/`.

## Before You Start

If you are running OmniInfer from a source checkout, prepare at least one local runtime backend before using the CLI.

- Windows: build or install one of `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `stable-diffusion.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip`; managed `vllm-wsl2-cuda` and `vllm-wsl2-rocm` are available for supported WSL2-capable NVIDIA and AMD systems. See [Build Guide: Windows](build.md#windows).
- Linux: build one of `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `stable-diffusion.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, `llama.cpp-linux-openvino`, or `vllm-linux-cuda` first. See [Build Guide: Linux](build.md#linux).
- macOS: build `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac` first. See [Build Guide: macOS](build.md#macos).

If you are using a packaged release that already includes `runtime/`, you can skip this preparation step and jump straight to the CLI commands below.
Packaged releases do not include the `omniinfer build` command; backend builds are source-checkout tooling only.

## Platform Behavior

- Linux, macOS, Windows:
  the CLI talks to the local OmniInfer service and starts it automatically when needed.
- Android and iOS:
  use the embedded modules under `android/` and `ios/` instead of the desktop CLI.

## Launch The CLI

Run the CLI after your local runtime has been prepared.

For source checkouts, use the repository root.
For packaged releases, use the release directory that already contains `runtime/`.

Linux and macOS:

```sh
./omniinfer --help
```

Run `./omniinfer` without arguments in an interactive terminal to open the Rust control-plane TUI. On first use, the TUI lets you pick an installed backend, choose a model found in OmniInfer-managed `.local` model directories or enter a model path manually, load it, and enter a simple chat loop. When a manual directory is scanned, the selected model is linked into `.local/models/<detected-model-dir>/<model-file>` instead of preserving unrelated parent folders. Later TUI launches automatically reload the last selected backend and model when the model path still exists.

The TUI also surfaces the advisor without adding a setup step. Managed model rows show small advisor fit/backend badges when local recommendations are available. Before a newly selected model is loaded, the TUI shows a short advisor preflight with the recommended backend, fit, and memory estimate. Press Enter to apply the recommendation and continue, `A` for details, `B` to choose another backend, `S` to keep the current backend, or `Q` to cancel. Automatic reload of the last model skips this preflight so repeat launches stay fast.

Windows:

```powershell
.\omniinfer.ps1 --help
```

Packaged Windows releases also keep `.\omniinfer.exe` as the real CLI binary and `.\omniinfer.cmd` for `cmd.exe` compatibility. For interactive TUI use from PowerShell, prefer `.\omniinfer.ps1` or `.\omniinfer.exe`; pressing `Ctrl+C` in a batch wrapper can make `cmd.exe` print `Terminate batch job (Y/N)?`.

## Quick Start

### 1. Check available backends

Linux, macOS, Windows:

```sh
./omniinfer backend list
```

Windows:

```powershell
.\omniinfer.ps1 backend list
```

The human-readable output shows the selected backend and runtime availability.
Use `./omniinfer backend list --json` when automation needs full backend metadata such as capabilities and descriptions.
By default, `backend list` shows compatible backends only. Use `--scope installed` or `--scope all` for a narrower or broader view.
In the table output, an empty `Selected` cell means no backend is selected, and the `Runtime` cell is either `installed` or `missing`.

### 2. Install a backend runtime

Install a prebuilt backend runtime through the CLI:

```sh
./omniinfer backend install <backend>
```

The default install mode is prebuilt. It downloads every runtime and companion archive declared by the built-in catalog, verifies each pinned SHA256, stages and validates the complete runtime, then atomically activates it under `.local/runtime/<platform>/<backend>/bin`. The installer writes `prebuilt.json` with every downloaded asset and digest. If an older installation has a launcher but is missing catalog-required files, rerunning `backend install` repairs it instead of reporting it as installed.

For example:

```sh
./omniinfer backend install llama.cpp-linux
```

Windows:

```powershell
.\omniinfer.ps1 backend install llama.cpp-cpu
.\omniinfer.ps1 backend install llama.cpp-cuda
.\omniinfer.ps1 backend install vllm-wsl2-cuda --wsl-distro Ubuntu
.\omniinfer.ps1 backend install vllm-wsl2-rocm --wsl-distro Ubuntu-24.04
```

The Windows CUDA entry installs both the llama.cpp CUDA package and its matching CUDA runtime companion package. A system CUDA Toolkit is not required; the NVIDIA driver is still required.

#### Windows vLLM through WSL2

Upstream vLLM does not support native Windows. `vllm-wsl2-cuda` and `vllm-wsl2-rocm` are therefore OmniInfer-managed WSL2 backends: the Windows CLI owns installation and lifecycle, while pinned official vLLM Linux wheels and their accelerator runtimes live inside a user Linux distribution.

Prerequisites:

- Windows x64 with WSL2 enabled
- A user Ubuntu distribution; `docker-desktop` and `*-data` distributions are intentionally rejected
- For `vllm-wsl2-cuda`: an NVIDIA GPU exposed to WSL2 and Windows NVIDIA driver 576.02 or newer
- For `vllm-wsl2-rocm`: Ubuntu 24.04, AMD Ryzen AI MAX/AI 300 `gfx1151` or `gfx1150`, and AMD Software 26.2.2 or newer with Ryzen WSL support
- Several GB of download and Linux-disk capacity; the exact size changes with the pinned vLLM/PyTorch dependency set

Install and initialize Ubuntu first if needed:

```powershell
wsl --install -d Ubuntu
wsl --distribution Ubuntu
.\omniinfer.exe backend install vllm-wsl2-cuda --wsl-distro Ubuntu

wsl --install -d Ubuntu-24.04
wsl --distribution Ubuntu-24.04
.\omniinfer.exe backend install vllm-wsl2-rocm --wsl-distro Ubuntu-24.04
```

If exactly one eligible user WSL2 distribution exists, `--wsl-distro` may be omitted. When several exist, it is required. Both installers verify pinned `uv` and vLLM wheel SHA256 values, create a managed Python 3.12 environment, run dependency checks, and execute a real accelerator tensor probe before activation. The ROCm path additionally pins every package in the ROCm 7.2.3 compute-library closure and its SHA256. It downloads missing AMD packages through the Windows network stack with bounded parallelism and resumable partial files, verifies them on Windows, streams them into WSL without loading multi-GB packages into memory, verifies them again, and lets APT perform the final dependency-aware install alongside the pinned OpenMPI runtime, Python 3.12 development headers required by Triton JIT, and ROCDXG 1.2.0. It then requires `rocminfo` to expose `gfx1151` or `gfx1150`. OmniInfer does not install WSL, Ubuntu, the Windows display driver, or an unrelated general-purpose SDK. Managed Python activation is transactional, and rerunning the same command is idempotent.

On unified-memory WSL2 ROCm systems, the managed launcher limits the default vLLM KV cache to 20% of WSL memory, capped at 4 GiB, so an APU's large shared-memory report cannot trigger a host-memory overcommit. It also defaults to eager execution with chunked prefill disabled because the upstream compiled/chunked ROCm path can stall during paged-attention warmup under WSL2. Each compatibility default is skipped when the corresponding vLLM flag is supplied explicitly; advanced users can override memory policy with `--kv-cache-memory-bytes` or `--gpu-memory-utilization`.

`--runtime-root` contains the small Windows launcher manifest, installer tool cache, and logs for these backends. The large Python environment is stored in the selected distribution under `~/.local/share/omniinfer/runtimes/<backend>/<runtime-key>/current`. Keep both locations intact. Normal model unload, backend stop, gateway shutdown, and smoke-test cleanup target only the managed vLLM process group; OmniInfer does not terminate the WSL distribution or unrelated Linux workloads.

Managed WSL2 vLLM launches forward only a documented allowlist of relevant Windows host variables through `WSLENV`: `HF_ENDPOINT`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_XET`, `HF_HUB_ENABLE_HF_TRANSFER`, `HF_HUB_ETAG_TIMEOUT`, `HF_HUB_DOWNLOAD_TIMEOUT`, and upper- or lowercase `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`. Set them before starting `omniinfer serve`. Arbitrary host variables and host cache-path variables are not forwarded.

#### Desktop application integration

Desktop applications can isolate OmniInfer from the package directory with the public global root options. Global options may appear before or after the subcommands:

```powershell
.\omniinfer.exe backend install llama.cpp-cuda `
  --state-root "C:\Users\me\AppData\Local\OmniStudio\omniinfer" `
  --runtime-root "C:\Users\me\AppData\Local\OmniStudio\runtimes"
```

- `--state-root <PATH>` owns OmniInfer state, configuration, logs, models, and the default runtime location.
- `--runtime-root <PATH>` overrides only the directory whose direct children are backend runtime directories such as `llama.cpp-cuda`.
- Relative CLI paths are resolved against the process working directory.
- Precedence is CLI option, public environment variable (`OMNIINFER_STATE_ROOT` or `OMNIINFER_RUNTIME_ROOT`), legacy/internal defaults, then the package-local default. `OMNIINFER_RUST_STATE_ROOT` remains accepted for compatibility but is not the preferred public integration API.
- Explicit CLI roots are preserved across `serve`, its gateway child process, model selection, backend launch, and detached service lifetime.

Use `--json` for streaming machine-readable installation progress. stdout is newline-delimited JSON (JSONL), one complete object per line; human progress is suppressed. Each event has `schema_version: 1`, a monotonic `sequence`, `event`, and `backend`. Download events also report `asset_index`, `asset_count`, `bytes_downloaded`, and `bytes_total` when the server supplies a content length.

```powershell
.\omniinfer.exe backend install llama.cpp-cuda --json --state-root <state> --runtime-root <runtimes>
```

The stable event names are `install_started`, `compatibility_selected`, `asset_planned`, `download_started`, `download_progress`, `checksum_verified`, `checksum_failed`, `download_failed`, `repair_started`, `staging_started`, `command_started`, `command_completed`, `validation_passed`, `already_installed`, `dry_run_completed`, `completed`, and `error`. A failed command exits non-zero and ends stdout with an `error` event. Consumers must ignore unknown event names and fields for forward compatibility.

The legacy compatibility command is still accepted:

```sh
./omniinfer build <backend> --prebuilt
```

Use source builds only from a source checkout when you explicitly need to compile the checked-out submodule:

```sh
./omniinfer build <backend> --from-source
```

Packaged releases support `backend install` for prebuilt runtimes. Source builds still require a cloned repository with `scripts/platforms/...` and the relevant submodules.

### 3. Select a backend

Always pick a backend from `backend list` on your current device.

```sh
./omniinfer backend select <backend>
```

Windows:

```powershell
.\omniinfer.ps1 backend select <backend>
```

Examples:

- Linux: `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `stable-diffusion.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, `llama.cpp-linux-openvino`, `vllm-linux-cuda`, `vla.cpp-linux`, or `vla.cpp-linux-cuda`
- macOS: `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac`
- Windows: `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `stable-diffusion.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, `llama.cpp-hip`, or managed `vllm-wsl2-cuda` / `vllm-wsl2-rocm`

When you select a desktop backend, OmniInfer also creates a backend-specific JSON config template under:

- `.local/config/backend_profiles/<backend>.json`

This file is the advanced path for backend-native parameters only.
Keep basic user inputs such as `-m/--model`, `-mm/--mmproj`, chat prompts, and `--image` on the CLI.

Example:

```json
{
  "schema_version": 2,
  "backend": "llama.cpp-vulkan",
  "family": "llama.cpp",
  "load": {
    "extra_args": ["-ngl", "99", "-t", "8", "-np", "5", "--cache-ram", "32768"]
  },
  "infer": {
    "extra_args": ["--top-k", "40", "--top-p", "0.9"]
  }
}
```

For official `llama.cpp-*` backends, profile load arguments extend OmniInfer's
safe RAM-cache defaults instead of replacing them. Explicit values appear last
and therefore override the matching default. See
[Model Load Parameters](model-load.md#backend-specific-notes) for cache sizing
and checkpoint fallback semantics.

### 3.5. Ask the advisor before loading, optional

The advisor is a local preflight layer. It does not start the gateway or load a model. Use it to inspect current hardware, installed runtimes, model format, approximate memory fit, and a suggested load command.

Inspect hardware and runtime availability:

```sh
./omniinfer advisor system
./omniinfer advisor system --json
```

The text output groups host, GPU, and backend readiness. If no compatible runtime is installed, it shows a prebuilt install command only for a backend present in the current platform catalog.

In `advisor system --json`, hardware compatibility and installer availability are separate contracts:

- `hardware_compatible` means the device can support the backend.
- `prebuilt_installable` means the current platform catalog contains an asset that `backend install` can install.
- `install_command` is non-null only when `prebuilt_installable` is true.
- `summary.recommended_backend_to_install` only selects a hardware-compatible, prebuilt-installable backend. Validated official llama.cpp packages rank ahead of experimental ik_llama.cpp builds; on a supported Windows NVIDIA system the install recommendation is `llama.cpp-cuda`, not `ik_llama.cpp-cuda`.

Inspect a model artifact:

```sh
./omniinfer advisor inspect /path/to/model.gguf
./omniinfer advisor inspect /path/to/model-directory --json
```

Estimate fit and get a recommended backend:

```sh
./omniinfer advisor fit /path/to/model.gguf --ctx-size 8192
./omniinfer advisor fit Qwen/Qwen2.5-7B-Instruct --backend vllm-linux-cuda --json
```

Plan hardware requirements for a model:

```sh
./omniinfer advisor plan /path/to/model.gguf --ctx-size 8192
./omniinfer advisor plan /path/to/model.gguf --gpu-vram 24 --ram 64 --cpu-cores 16
./omniinfer advisor plan /path/to/model.gguf --json
```

The plan command reports GPU, CPU-offload, and CPU-only paths with minimum/recommended VRAM, RAM, CPU cores, current feasibility, and upgrade deltas.

Recommend from OmniInfer-managed local model directories:

```sh
./omniinfer advisor recommend --task coding -n 5
```

Advisor memory numbers are estimates based on local file size, context length, and conservative overhead. Backend startup logs and real benchmark results remain authoritative.

### 4. Load a model

Default path:

```sh
./omniinfer load -m /path/to/model-directory
```

For `llama.cpp-*`, OmniInfer accepts either a model file or a model directory. If you pass a directory, OmniInfer auto-discovers:

- the main text GGUF, including the first file of a standard `-00001-of-000NN.gguf` split set
- the optional `mmproj` GGUF

For `mlx-mac`, OmniInfer passes the model directory directly to the embedded backend.
For `vllm-linux-cuda`, `vllm-wsl2-cuda`, and `vllm-wsl2-rocm`, OmniInfer passes HuggingFace model IDs and other non-path references directly to `vllm serve`. On Windows, an absolute drive path such as `D:\models\Qwen` is translated to the selected distribution's automount path such as `/mnt/d/models/Qwen`. UNC paths are rejected; copy or download the model into a local Windows drive or the selected WSL2 filesystem.

For `vla.cpp-*`, OmniInfer starts and supervises the managed `vla-server` process. vla.cpp uses its own ZeroMQ/protobuf action-prediction protocol instead of the OpenAI chat API, so VLA clients should connect to the reported loopback endpoint with vla.cpp's `src/serving/vla.proto` contract. The gateway does not translate or publish that unauthenticated protocol: `/v1/chat/completions` and `/v1/messages` return `422` while a VLA runtime is loaded. The model must be a VLA checkpoint file path, such as a GGUF or safetensors file. Pass `--mmproj` when the selected VLA architecture requires a separate vision tower GGUF. vla.cpp server-native flags such as `--config` and `--timing-detail phase` can be passed after `--`.

```sh
./omniinfer backend select vla.cpp-linux-cuda
./omniinfer load -m /models/smolvla/smolvla-libero.gguf --mmproj /models/smolvla/mmproj.gguf -- --timing-detail phase
```

For `stable-diffusion.cpp-*`, OmniInfer manages the loopback-only `sd-server`
and exposes its authenticated native async API under `/sdcpp/v1/*`. Chat and
Anthropic endpoints return structured `422` while a diffusion runtime is
loaded. MiniMax H3 uses four separate files; pass the denoiser as `-m` and the
text encoder, video VAE, and optional audio VAE as backend launch arguments.
This Q4 example follows the upstream low-VRAM defaults while enabling Vulkan
flash attention:

```sh
./omniinfer backend select stable-diffusion.cpp-linux-vulkan
./omniinfer load -m /models/MiniMax-H3-FL2VA-Q4/minimax_h3_fl2va_pruned-Q4_K.gguf -- \
  --llm /models/MiniMax-H3-FL2VA-Q4/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae /models/MiniMax-H3-FL2VA-Q4/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /models/MiniMax-H3-FL2VA-Q4/vae/minimax_h3_audio_vae_fp32.safetensors \
  --cfg-scale 1.0 --diffusion-fa --backend te=cpu \
  --rng cpu
```

On Windows, use `stable-diffusion.cpp-vulkan` and native Windows paths for all
four files. After loading, query `/sdcpp/v1/capabilities` before submitting a
job because supported modes and defaults depend on the loaded checkpoint.
MiniMax H3 aligns requested frame counts upward to at least 5 frames with
`video_frames % 17 == 5`; use the completed job's `result.frame_count` as the
actual output count.
The example keeps the text encoder's computation and parameters in host memory
while the denoiser and VAEs use Vulkan memory. Do not add
`--offload-to-cpu` on a machine whose host-visible RAM cannot hold all four
weight files; that flag changes the effective parameter assignment to
`*=cpu`. Use explicit `--backend` and `--params-backend` assignments when a
different split is required.

Explicit file path:

```sh
./omniinfer load -m /path/to/model.gguf
```

Advanced path with backend config JSON:

```sh
./omniinfer backend select llama.cpp-vulkan
./omniinfer load -m /path/to/model-directory --config
```

Windows:

```powershell
.\omniinfer.ps1 load -m C:\path\to\model-directory
```

Vision-language model:

```sh
./omniinfer load -m /path/to/model.gguf -mm /path/to/mmproj.gguf
```

For `mlx-mac`, use a vision-capable model directory instead of a `.gguf` file or `mmproj` sidecar:

```sh
./omniinfer backend select mlx-mac
./omniinfer load -m /path/to/mlx-vlm-model-directory
./omniinfer chat \
  --image /path/to/image.jpg \
  "Describe this image in one sentence."
```

The backend config JSON is where advanced users should put backend-native launch parameters such as `-ngl`, `--threads`, and other backend-specific options.
For `vllm-linux-cuda`, `vllm-wsl2-cuda`, and `vllm-wsl2-rocm`, use `--max-model-len` or the stable OmniInfer `ctx_size` option for context length; OmniInfer maps it to vLLM's `--max-model-len`.

You can also skip `--config` entirely and pass backend-native extra args directly after the stable OmniInfer args. OmniInfer parses those extra args according to the currently selected backend.

Example:

```powershell
.\omniinfer.ps1 backend select llama.cpp-vulkan
.\omniinfer.ps1 load -m C:\models\Qwen3 -ngl 99 -t 8
```

vLLM example:

```sh
./omniinfer backend select vllm-linux-cuda
./omniinfer load -m Qwen/Qwen3.5-4B-Instruct -- --max-model-len 8192 --gpu-memory-utilization 0.85
```

Windows WSL2 vLLM example:

```powershell
.\omniinfer.exe backend select vllm-wsl2-cuda
.\omniinfer.exe load -m Qwen/Qwen3.5-4B-Instruct -- --max-model-len 8192 --gpu-memory-utilization 0.85
```

Use `vllm-wsl2-rocm` instead on a supported Ryzen AI AMD system.

### 5. Chat

Text chat:

```sh
./omniinfer chat "Introduce yourself in one sentence."
```

Vision-language chat:

```sh
./omniinfer chat \
  --image /path/to/image.jpg \
  "Describe this image in one sentence."
```

Windows:

```powershell
.\omniinfer.ps1 chat "Introduce yourself in one sentence."
```

Advanced path with backend config JSON:

```sh
./omniinfer load -m /path/to/model-directory --config
./omniinfer chat "Hello"
```

You can also pass backend-native extra args directly:

```powershell
.\omniinfer.ps1 chat "Hello" -- --top-k 40 --top-p 0.9
```

### 6. Record a benchmark

After loading a text model, `bench run` performs three measured requests by
default and writes submission-compatible JSON under
`.local/benchmarks/results/`:

```sh
mkdir -p .local/benchmarks/slots
./omniinfer load -m /models/model.gguf --ctx-size 4096 -- \
  --cache-ram 0 --no-cache-idle-slots --no-cache-prompt \
  --slot-prompt-similarity 0 --slot-save-path .local/benchmarks/slots

./omniinfer bench run \
  --catalog-model-id <catalog-model-id> \
  --format GGUF \
  --quantization Q4_K_M \
  --model-url https://huggingface.co/owner/model/resolve/<40-character-commit>/model.gguf \
  --baseline \
  --submitter-name <name>

./omniinfer bench list
```

Declare each optional method with `--optimization <slug>` instead of using
`--baseline`. The runtime launch command is captured from OmniInfer state when
available. Known devices and managed runtime provenance are auto-detected;
custom hardware and source builds still require the corresponding metadata
options. Each measured run starts only after every runtime slot is erased, and
results with Prefill or Decode CV above 5% are not archived. See
[Benchmark Results](benchmark.md) for the complete contract, security rules,
and machine-readable output behavior.

Add `--ignore-eos` with `--max-tokens <n>` for a fixed-length benchmark. It
requests `ignore_eos: true`; the CLI then requires every measured response to
report `completion_tokens` equal to `<n>` and aborts rather than archiving a
short or mismatched result. This mode is recorded in the existing
`protocol.notes` field and does not change the benchmark schema.

## Common Commands

```sh
./omniinfer
./omniinfer backend list
./omniinfer backend select <backend>
./omniinfer status
./omniinfer model list
./omniinfer load -m /path/to/model-directory
./omniinfer load -m /path/to/model-directory --config
./omniinfer chat "Hello"
./omniinfer chat --think on "Hello"
./omniinfer bench list
./omniinfer serve --default-thinking off
./omniinfer shutdown
./omniinfer completion bash
```

Thinking mode remains available through request-level options such as
`chat --think on|off`, serve defaults such as `serve --default-thinking on|off`,
TUI `/think`, and the local `/omni/thinking` management API. It is not exposed
as a standalone top-level CLI command.

On packaged Windows releases, replace `./omniinfer` with `.\omniinfer.ps1` in PowerShell. Use `.\omniinfer.cmd` only when you specifically need `cmd.exe` compatibility.

## Platform Notes

### Linux, macOS, Windows

- The CLI uses the Rust control-plane entrypoint. Python control-plane fallback
  has been removed; unsupported commands return a clear Rust error instead of
  running a legacy Python entrypoint.
- The desktop CLI auto-starts the local OmniInfer gateway when required.
- CUDA desktop backends default to one GPU. If `CUDA_VISIBLE_DEVICES` is unset, OmniInfer picks the visible GPU with the most free memory and lowest utilization before launching the backend. Set `CUDA_VISIBLE_DEVICES` or `OMNIINFER_CUDA_VISIBLE_DEVICES` to override this.
- If you want to launch the gateway from an interactive terminal, use:

```sh
./omniinfer serve
```

In a terminal, `serve` opens the Rust server launcher. It asks you to choose a backend and then a model every time. The last selected backend and model are preselected and marked `last selected`, so pressing Enter twice reuses the previous choices. After the model is loaded, the launcher starts the gateway and keeps it running until you press `Ctrl+C`.

When `serve` is used from a non-interactive script, or when `OMNIINFER_SERVE_DIRECT=1` is set, it starts the gateway directly without the launcher. Direct `serve` starts on `127.0.0.1` by default; configuration-file `host` values do not change the listener. Use `--lan` to bind `0.0.0.0`, or pass `--host` explicitly for another address. If no `--model` is supplied, OmniInfer reloads the last selected model from `.local/config/state.json` when one is available; otherwise it starts an empty gateway. Use `--no-restore-model` to disable restore for one startup. To disable later restores persistently without stopping the currently loaded runtime, call `POST /omni/model/clear-selection`.

`serve --startup-timeout <seconds>` applies to both gateway readiness and backend/model cold-start readiness and defaults to 420 seconds. On WSL2 ROCm, a readiness timeout during the first 120 seconds triggers targeted runtime cleanup, a 90-second cooldown, and one retry within the same total budget so a cold Triton cache and ROCm device state can recover without a second user command. OmniInfer does not terminate the WSL distribution or unrelated workloads. Early process exits and explicit total budgets below 360 seconds are not retried.

`POST /omni/backend/stop` is a temporary runtime stop: it preserves the selected model for the next direct startup. The active runtime and future restore selection are exposed separately by `GET /omni/state`. Identical client selections after automatic restore are idempotent and return `already_loaded: true`; changed runtime parameters return `409` with `requires_reload: true`.

```sh
curl -sS -X POST http://127.0.0.1:9000/omni/model/clear-selection
```

To expose only the inference API to trusted devices on the same LAN, use:

```sh
./omniinfer serve --lan
```

LAN mode uses the same launcher in an interactive terminal, then binds the gateway to `0.0.0.0` and requires an API key for remote clients. If no key is supplied through `--api-key` or `OMNIINFER_API_KEY`, OmniInfer generates a session key and prints it with the LAN base URLs. Remote clients can call `/v1/chat/completions` or `/v1/messages`; `/omni/*` management endpoints stay local-only by default.

To create a temporary public HTTPS URL without router port forwarding, use Cloudflare Quick Tunnel mode:

```sh
./omniinfer serve --cloudflare
```

If you already know the model to serve, the same command can start the gateway, open the tunnel, select a backend, load the model, and run a short validation request:

```sh
./omniinfer serve \
  --cloudflare \
  --backend llama.cpp-linux-cuda \
  --model /path/to/model.gguf \
  --ctx-size 8192 \
  --api-key auto \
  --detach
```

Use `--smoke-test` for an ephemeral lifecycle check. It performs one
non-streaming inference request and then stops the gateway, backend, and tunnel,
removes the serve-state record, releases their ports, and exits with code `0`.
Startup, inference, or cleanup failures exit non-zero after the same bounded
cleanup. This behavior also applies when `--detach` is present; omit
`--smoke-test` when the service should remain running.

Windows:

```powershell
.\omniinfer.ps1 serve --cloudflare
```

Cloudflare mode uses the same launcher in an interactive terminal when no `--model` is supplied, keeps OmniInfer bound to `127.0.0.1`, resolves or downloads a verified `cloudflared` helper, prints a temporary `https://*.trycloudflare.com` URL, and requires an API key for remote inference requests. Helper resolution completes before the gateway starts, so a dependency failure does not leave a local-only gateway running. Quick Tunnel is intended for testing and short-lived access; use non-streaming requests for the most reliable behavior. See [Remote Access](remote-access.md).

Detached services can be checked or stopped without remembering process IDs:

```sh
./omniinfer serve status --port 9000
./omniinfer serve stop --port 9000
```

Serve startup records each managed process before continuing. Starting the same
port again first removes the verified previous gateway, backend, and tunnel;
if ownership cannot be verified, startup stops with an error and preserves the
existing state for inspection.

The generic `./omniinfer shutdown` command uses the configured port when it has
a matching serve-state record. Otherwise, it targets the only service recorded
under the active `--state-root`. If several non-default services are recorded,
it exits non-zero and requires `serve stop --port <PORT>` instead of choosing an
arbitrary process.

LAN and Cloudflare access can run at the same time:

```sh
./omniinfer serve --lan --cloudflare
```

In this mode, OmniInfer binds to `0.0.0.0` for LAN clients and starts Cloudflare Quick Tunnel against `http://127.0.0.1:<port>`. Both remote entry points require the same API key, and `/omni/*` management endpoints remain local-only.

For a fixed HTTPS hostname behind a trusted reverse proxy such as nginx + frp, keep OmniInfer on loopback and let the proxy publish the public URL. Use a separate admin key when remote clients need model-management endpoints:

```sh
./omniinfer serve \
  --backend llama.cpp-linux-cuda \
  --public-model-root /path/to/public_models \
  --api-key oi_inference_key \
  --allow-remote-management \
  --behind-proxy \
  --no-restore-model \
  --detach
```

For multiple remote admins, prefer `.local/config/admin_keys.json` over command-line admin keys so secrets do not appear in process lists:

```json
{
  "keys": {
    "admin1": "replace-with-secret",
    "admin2": "replace-with-secret"
  }
}
```

`--public-model-root` is the only model tree remote management requests may select from. Each model lives in a directory with an `omni-model.json` manifest:

```text
public_models/
  qwen3.5-4b-q4_k_m/
    omni-model.json
    Qwen3.5-4B-Q4_K_M.gguf
```

Remote clients list selectable models with `GET /omni/public-models` and switch models with `POST /omni/model/select`:

```sh
curl -sS -H 'Authorization: Bearer oi_admin_key' \
  https://omniinfer.example.com/omni/public-models

curl -sS -H 'Authorization: Bearer oi_admin_key' \
  -H 'Content-Type: application/json' \
  https://omniinfer.example.com/omni/model/select \
  -d '{"model":"qwen3.5-4b-q4_k_m"}'
```

On Windows, allow the port through the Private-network firewall profile when needed:

```powershell
New-NetFirewallRule `
  -DisplayName "OmniInfer LAN 9000" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 9000 `
  -Profile Private `
  -RemoteAddress LocalSubnet
```

### Mobile

- Android is implemented by the root `android/` Gradle module. See [Android Integration Guide](android/integration.md).
- iOS is implemented by the root `ios/OmniInferServer` Swift Package. See [OmniStudio API Service](OmniStudio/api-service.md#ios-client).
