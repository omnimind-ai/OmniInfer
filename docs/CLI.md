# OmniInfer CLI Guide

This guide shows how to use the OmniInfer desktop CLI on Linux, macOS, and Windows.
Android and iOS use the embedded modules under `android/` and `ios/`.

## Before You Start

If you are running OmniInfer from a source checkout, prepare at least one local runtime backend before using the CLI.

- Windows: build one of `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip` first. See [Build Guide: Windows](build.md#windows).
- Linux: build one of `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, `llama.cpp-linux-openvino`, or `vllm-linux-cuda` first. See [Build Guide: Linux](build.md#linux).
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
In the table output, empty `Selected` or `Installed` cells mean the state is false.

### 2. Build a backend from source, optional

Source checkouts on Linux, macOS, and Windows can install a compatible backend through the CLI:

```sh
./omniinfer build <backend>
```

Linux backend scripts default to non-build installation. Use `--from-source` when you explicitly want to compile from the checked-out source submodule:

```sh
./omniinfer build <backend> --from-source
```

Windows:

```powershell
.\omniinfer.ps1 build <backend>
.\omniinfer.ps1 build <backend> --prebuilt
```

The command runs the matching platform script under `scripts/platforms/<platform>/<backend>/build.*` and verifies that the backend launcher exists after the install. `--prebuilt` is still accepted for explicit prebuilt installs where supported.

Packaged releases intentionally do not provide this command because they are designed to run from the included `runtime/` directory without requiring a compiler, CUDA toolkit, CMake, or other build tools.

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

- Linux: `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, `llama.cpp-linux-openvino`, or `vllm-linux-cuda`
- macOS: `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac`
- Windows: `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip`

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
    "extra_args": ["-ngl", "99", "-t", "8"]
  },
  "infer": {
    "extra_args": ["--top-k", "40", "--top-p", "0.9"]
  }
}
```

### 3.5. Ask the advisor before loading, optional

The advisor is a local preflight layer. It does not start the gateway or load a model. Use it to inspect current hardware, installed runtimes, model format, approximate memory fit, and a suggested load command.

Inspect hardware and runtime availability:

```sh
./omniinfer advisor system
./omniinfer advisor system --json
```

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

- the main text GGUF
- the optional `mmproj` GGUF

For `mlx-mac`, OmniInfer passes the model directory directly to the embedded backend.
For `vllm-linux-cuda`, OmniInfer passes the model string directly to `vllm serve`, so it can be a HuggingFace model ID, a local snapshot directory, or another reference accepted by vLLM.

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
For `vllm-linux-cuda`, use `--max-model-len` or the stable OmniInfer `ctx_size` option for context length; OmniInfer maps it to vLLM's `--max-model-len`.

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

### 4. Chat

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

## Common Commands

```sh
./omniinfer
./omniinfer backend list
./omniinfer backend select <backend>
./omniinfer status
./omniinfer model list
./omniinfer load -m /path/to/model-directory
./omniinfer load -m /path/to/model-directory --config
./omniinfer thinking show
./omniinfer thinking set on
./omniinfer chat "Hello"
./omniinfer shutdown
./omniinfer completion bash
```

On packaged Windows releases, replace `./omniinfer` with `.\omniinfer.ps1` in PowerShell. Use `.\omniinfer.cmd` only when you specifically need `cmd.exe` compatibility.

## Platform Notes

### Linux, macOS, Windows

- The CLI uses the Rust control-plane entrypoint. Python control-plane fallback
  has been removed; unsupported commands return a clear Rust error instead of
  running [omniinfer.py](../omniinfer.py).
- The desktop CLI auto-starts the local OmniInfer gateway when required.
- CUDA desktop backends default to one GPU. If `CUDA_VISIBLE_DEVICES` is unset, OmniInfer picks the visible GPU with the most free memory and lowest utilization before launching the backend. Set `CUDA_VISIBLE_DEVICES` or `OMNIINFER_CUDA_VISIBLE_DEVICES` to override this.
- If you want to launch the gateway from an interactive terminal, use:

```sh
./omniinfer serve
```

In a terminal, `serve` opens the Rust server launcher. It asks you to choose a backend and then a model every time. The last selected backend and model are preselected and marked `last selected`, so pressing Enter twice reuses the previous choices. After the model is loaded, the launcher starts the gateway and keeps it running until you press `Ctrl+C`.

When `serve` is used from a non-interactive script, or when `OMNIINFER_SERVE_DIRECT=1` is set, it starts the gateway directly without the launcher. Direct `serve` starts on `127.0.0.1` by default; configuration-file `host` values do not change the listener. Use `--lan` to bind `0.0.0.0`, or pass `--host` explicitly for another address. If no `--model` is supplied, OmniInfer reloads the last selected model from `.local/config/state.json` when one is available; otherwise it starts an empty gateway. Use `--no-restore-model` for managed multi-admin servers where every loaded model should have a named admin owner.

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
  --detach \
  --smoke-test
```

Windows:

```powershell
.\omniinfer.ps1 serve --cloudflare
```

Cloudflare mode uses the same launcher in an interactive terminal when no `--model` is supplied, keeps OmniInfer bound to `127.0.0.1`, downloads or updates a managed `cloudflared` binary under `.local/tools/cloudflared`, prints a temporary `https://*.trycloudflare.com` URL, and requires an API key for remote inference requests. Quick Tunnel is intended for testing and short-lived access; use non-streaming requests for the most reliable behavior. See [Remote Access](remote-access.md).

Detached services can be checked or stopped without remembering process IDs:

```sh
./omniinfer serve status --port 9000
./omniinfer serve stop --port 9000
```

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
