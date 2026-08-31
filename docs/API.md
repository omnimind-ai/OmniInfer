# OmniInfer Desktop API

This document describes the local HTTP API exposed by the desktop OmniInfer gateway.

Base URL:

```text
http://127.0.0.1:9000
```

The port defaults to `9000` and can be changed via `config/omniinfer.json` or the CLI `--port` flag.

All request and response bodies are JSON unless an endpoint explicitly uses Server-Sent Events (SSE). The gateway allows CORS for local browser clients:

- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key`

`OPTIONS *` returns `204` for CORS preflight and also accepts `anthropic-version` and `x-api-key` in `Access-Control-Allow-Headers`.

Windows note: direct gateway processes hide their console window by default. To force a visible direct gateway window, use:

```powershell
.\omniinfer.ps1 serve --window-mode visible
```

## Local Network Access

OmniInfer binds to `127.0.0.1` by default, so only the same machine can access the gateway. To expose the OpenAI-compatible inference API to other devices on the same trusted LAN, start the gateway in LAN mode:

```sh
./omniinfer serve --lan
```

LAN mode binds the gateway to `0.0.0.0`, generates a session API key when one is not provided, and prints the local and LAN base URLs. Remote clients must send the key with either header:

```text
Authorization: Bearer <api-key>
x-api-key: <api-key>
```

For a stable key, pass one explicitly or set `OMNIINFER_API_KEY`:

```sh
OMNIINFER_API_KEY=oi_example ./omniinfer serve --lan
```

LAN access can be combined with Cloudflare Quick Tunnel when you need both same-network and temporary public HTTPS access from the same gateway:

```sh
./omniinfer serve --lan --cloudflare
```

In combined mode, OmniInfer binds the gateway to `0.0.0.0` for LAN clients while `cloudflared` still connects to `http://127.0.0.1:<port>` on the same machine. Both LAN and Cloudflare clients must send the API key.

Only inference-facing endpoints are exposed to remote clients by default:

| Method | Path |
|---|---|
| `GET` | `/health` |
| `GET` | `/v1/models` |
| `POST` | `/v1/chat/completions` |
| `POST` | `/v1/messages` |
| `POST` | `/tokenize` |
| `POST` | `/detokenize` |
| `GET`, `POST` | `/sdcpp/v1/*` |

Management endpoints under `/omni/*`, including model loading, backend switching, backend stop, and gateway shutdown, remain local-only unless the gateway is started with `--allow-remote-management` and an API key. Keep OmniStudio and other local controllers pointed at `http://127.0.0.1:<port>`.

If you intentionally need an unauthenticated LAN test server, use `--allow-insecure-lan`. Do not use that mode on shared networks.

On Windows, the OS firewall may still block inbound LAN traffic. Prefer a Private network profile and a LocalSubnet-only rule:

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

## Cloudflare Quick Tunnel

For temporary remote access without router port forwarding or a public IP address, start OmniInfer with Cloudflare Quick Tunnel mode. In an interactive terminal, OmniInfer first asks you to choose a backend and model, then starts the gateway and tunnel:

```sh
./omniinfer serve --cloudflare
```

This keeps the gateway bound to `127.0.0.1`, resolves an existing `cloudflared` helper or downloads a verified managed copy under `.local/tools/cloudflared`, starts `cloudflared tunnel --url http://127.0.0.1:<port>`, prints a temporary `https://*.trycloudflare.com` URL, and requires an OmniInfer API key for requests arriving through Cloudflare. When combined with `--lan`, the gateway binds to `0.0.0.0` for LAN clients and the tunnel still targets `127.0.0.1`. `/omni/*` management endpoints remain local-only.

Quick Tunnel is intended for demos and short-lived testing. For best compatibility, use non-streaming requests. See [Remote Access](remote-access.md) for setup, security notes, and examples.

## Endpoint Summary

| Method | Path | Purpose |
|---|---|---|
| `OPTIONS` | any path | CORS preflight |
| `GET` | `/health` | Gateway health and basic runtime state |
| `GET` | `/health?deep=true` | Health plus backend process health |
| `GET` | `/omni/state` | Full OmniInfer runtime state |
| `GET` | `/omni/backends?scope=installed\|compatible\|all` | Backend list |
| `GET` | `/omni/thinking` | Default thinking setting |
| `GET` | `/omni/backend/props` | Active backend `/props` payload |
| `GET` | `/omni/models` | Deprecated, returns `410` |
| `GET` | `/omni/public-models` | Public model ids exposed by `--public-model-root` |
| `GET` | `/omni/supported-models?system=windows\|mac\|linux` | Bundled supported-model catalog |
| `GET` | `/omni/supported-models/best?system=windows\|mac\|linux` | Best-backend model catalog |
| `GET` | `/v1/models` | OpenAI-compatible loaded-model list |
| `POST` | `/omni/backend/select` | Select a backend |
| `POST` | `/omni/backend/stop` | Stop current backend runtime |
| `POST` | `/omni/cache/clear` | Clear backend KV cache |
| `POST` | `/omni/shutdown` | Stop the gateway |
| `POST` | `/omni/thinking/select` | Update default thinking setting |
| `POST` | `/omni/model/select` | Load a model |
| `POST` | `/omni/model/clear-selection` | Disable model restore without stopping the runtime |
| `POST` | `/tokenize` | llama.cpp-compatible tokenization |
| `POST` | `/detokenize` | llama.cpp-compatible detokenization |
| `POST` | `/omni/tokenize` | Local-only alias for `/tokenize` |
| `POST` | `/omni/detokenize` | Local-only alias for `/detokenize` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/messages` | Anthropic-compatible Messages API adapter |
| `GET` | `/sdcpp/v1/capabilities` | stable-diffusion.cpp capabilities and defaults |
| `POST` | `/sdcpp/v1/img_gen` | Submit an asynchronous image job |
| `POST` | `/sdcpp/v1/vid_gen` | Submit an asynchronous video/audio job |
| `GET` | `/sdcpp/v1/jobs/{id}` | Poll a diffusion job |
| `POST` | `/sdcpp/v1/jobs/{id}/cancel` | Cancel a diffusion job |

The two chat generation endpoints require an OpenAI-compatible loaded backend.
A loaded `vla.cpp-*` or `stable-diffusion.cpp-*` backend uses a different
protocol; both chat endpoints return `422` with
`error.code=backend_protocol_not_supported` and report its local client
endpoint rather than proxying an incompatible request.

Unknown paths return `404`:

```json
{
  "error": {
    "message": "not found: /path"
  }
}
```

General request bodies are limited to 100 MiB. Native diffusion requests use a
stricter 16 MiB limit and return `413` when it is exceeded. Invalid JSON is
treated as an empty object by the current gateway implementation unless the
selected upstream protocol rejects it.

## Health

### `GET /health`

Returns gateway health and basic runtime state.

Example:

```bash
curl -s http://127.0.0.1:9000/health
```

Example response:

```json
{
  "status": "ok",
  "version": "0.2.1",
  "omni": {
    "backend": "llama.cpp-cpu",
    "model": null,
    "mmproj": null,
    "ctx_size": null,
    "request_defaults": {},
    "backend_ready": false,
    "runtime_mode": null,
    "backend_port": null,
    "backend_pid": null,
    "launch_args": [],
    "launch_command": [],
    "backend_log": null,
    "effective_parameters": {
      "ctx_size": null,
      "ngl": null,
      "threads": null,
      "threads_batch": null,
      "batch_size": null,
      "ubatch_size": null
    }
  },
  "thinking": {
    "default_enabled": false
  }
}
```

### `GET /health?deep=true`

Adds `backend_health`. For external backends, OmniInfer probes the backend process. For embedded backends, readiness is based on the loaded runtime.

Possible shapes include:

```json
{
  "status": "ok",
  "omni": {},
  "thinking": {
    "default_enabled": false
  },
  "backend_health": {
    "status": "no_backend"
  }
}
```

## Runtime State

### `GET /omni/state`

Returns the current runtime state plus all platform-declared backends.

The active runtime and the persisted startup selection are reported separately:

- `backend_ready`, `model`, `model_path`, `mmproj`, `ctx_size`, and `launch_args` describe the runtime that is loaded now.
- `restore_selection` describes the model OmniInfer will try to restore on the next direct `serve` startup, or is `null` when no restore is configured.
- `restore_status` is `not_configured`, `pending`, or `loaded`.
- `restore_completed` is true only when a loaded runtime matches the persisted backend, model, `mmproj`, and context size.
- `generation` and `route_state` identify the currently routable runtime generation.
- `resource_ledger` reports capacity, reserved, committed, and available bytes by host, CUDA-device, or unified-memory domain.
- `runtime_placement` reports the effective llama.cpp CUDA policy, CPU/GPU layer placement, startup-log buffer totals, and reconciled budget. It is `null` for runtimes that do not use placement reconciliation.

Example:

```bash
curl -s http://127.0.0.1:9000/omni/state
```

Example response:

```json
{
  "version": "0.2.1",
  "backend": "llama.cpp-cuda",
  "model": "models/Qwen3.5-2B-Q4_K_M.gguf",
  "mmproj": null,
  "ctx_size": 4096,
  "request_defaults": {
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  },
  "backend_ready": true,
  "runtime_mode": "external_server",
  "backend_port": 12894,
  "backend_pid": 45210,
  "launch_args": [],
  "runtime_placement": {
    "source": "llama.cpp_startup_log",
    "policy": "auto",
    "requested_gpu_layers": null,
    "mode": "partial",
    "offloaded_layers": 28,
    "total_layers": 41,
    "reported_buffer_bytes": {"host": 1500000000, "cuda:0": 4939212390},
    "reconciled_budget": {
      "domains_bytes": {"host": 2147483648, "cuda:0": 5368709120},
      "components": [
        {"name": "reported_model_buffers", "domain": "host", "bytes": 1500000000},
        {"name": "runtime_overhead", "domain": "host", "bytes": 402653184},
        {"name": "reconciliation_slack", "domain": "host", "bytes": 244830464},
        {"name": "reported_model_buffers", "domain": "cuda:0", "bytes": 4939212390},
        {"name": "runtime_overhead", "domain": "cuda:0", "bytes": 134217728},
        {"name": "reconciliation_slack", "domain": "cuda:0", "bytes": 295279002}
      ]
    }
  },
  "restore_selection": {
    "backend": "llama.cpp-cuda",
    "model": "models/Qwen3.5-2B-Q4_K_M.gguf",
    "mmproj": null,
    "ctx_size": 4096
  },
  "restore_status": "loaded",
  "restore_completed": true,
  "launch_command": ["llama-server", "-m", "models/Qwen3.5-2B-Q4_K_M.gguf", "--port", "12894"],
  "backend_log": ".local/runtime/linux/llama.cpp-linux-cuda/logs/runtime.log",
  "effective_parameters": {
    "ctx_size": 4096,
    "ngl": null,
    "threads": null,
    "threads_batch": null,
    "batch_size": null,
    "ubatch_size": null
  },
  "available_backends": [
    {
      "id": "llama.cpp-cuda",
      "label": "llama.cpp CUDA",
      "family": "llama.cpp",
      "selected": true,
      "binary_exists": true,
      "models_dir": "E:\\Coding\\repository\\OmniInfer-2\\.local\\models",
      "capabilities": ["chat", "vision", "stream", "gpu", "cuda"],
      "description": "llama.cpp CUDA backend managed by OmniInfer",
      "loaded_model": "models/Qwen3.5-2B-Q4_K_M.gguf",
      "compatibility": "installed",
      "priority": 0
    }
  ],
  "thinking": {
    "default_enabled": false
  }
}
```

## Backends

### `GET /omni/backends?scope=installed|compatible|all`

Returns backends filtered by scope.

| Scope | Meaning |
|---|---|
| `installed` | Runtime binary or embedded runtime is present |
| `compatible` | Installed or compatible with detected hardware |
| `all` | Every backend declared for the current platform |

The default scope is `installed`.

Example:

```bash
curl -s "http://127.0.0.1:9000/omni/backends?scope=compatible"
```

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama.cpp-cuda",
      "label": "llama.cpp CUDA",
      "family": "llama.cpp",
      "selected": true,
      "binary_exists": true,
      "models_dir": "E:\\Coding\\repository\\OmniInfer-2\\.local\\models",
      "capabilities": ["chat", "vision", "stream", "gpu", "cuda"],
      "description": "llama.cpp CUDA backend managed by OmniInfer",
      "loaded_model": null,
      "compatibility": "installed",
      "priority": 0
    }
  ],
  "recommended": "llama.cpp-cuda"
}
```

Invalid `scope` returns `400`.

### `POST /omni/backend/select`

Selects a backend without loading a model. If another runtime is currently running, it is stopped.

Request body:

```json
{
  "backend": "llama.cpp-cuda"
}
```

Example response:

```json
{
  "ok": true,
  "selected_backend": "llama.cpp-cuda",
  "binary_exists": true,
  "models_dir": "E:\\Coding\\repository\\OmniInfer-2\\.local\\models"
}
```

Status codes:

- `200` on success
- `400` if `backend` is missing or unsupported

### `POST /omni/backend/stop`

Stops the currently loaded backend runtime process. The selected backend and persisted model selection are preserved, so a later direct `serve` can restore the model. Use `/omni/model/clear-selection` when the next startup must remain unloaded.

Example response:

```json
{
  "ok": true,
  "stopped": true,
  "selected_backend": "llama.cpp-cuda",
  "selected_model_preserved": true,
  "restore_status": "pending"
}
```

### `POST /omni/model/clear-selection`

Clears the persisted model, `mmproj`, and context-size selection without stopping a currently loaded runtime. The selected backend is preserved. The operation is idempotent.

Example response while a model is still running:

```json
{
  "ok": true,
  "selection_cleared": true,
  "backend_ready": true,
  "current_model": "models/Qwen3.5-2B-Q4_K_M.gguf",
  "restore_selection": null,
  "restore_status": "not_configured",
  "restore_completed": false
}
```

### `GET /omni/backend/props`

Returns the active external backend's `/props` payload when available. llama.cpp-compatible backends usually include context information such as `n_ctx`.

If no external backend is loaded, or the backend does not expose `/props`, OmniInfer returns `{}`.

## Model Catalog

### `GET /omni/supported-models?system=windows|mac|linux`

Returns the backend-grouped supported model catalog for a target system.

The catalog is bundled with OmniInfer under `crates/omniinfer-core/model_catalogs`; this endpoint does not fetch catalog JSON from a remote service at runtime.

Example:

```bash
curl -s "http://127.0.0.1:9000/omni/supported-models?system=windows"
```

Response shape:

```json
{
  "llama.cpp-cpu": {
    "Qwen3.5": {
      "Qwen3.5-2B": {
        "tag": ["..."],
        "quantization": {
          "Q4_K_M": {
            "download": "https://modelscope.cn/...",
            "size": 1.28,
            "required_memory_gib": 1.28,
            "suitable": true
          }
        },
        "README": "readme/Qwen3.5-2B.md"
      }
    }
  }
}
```

Status codes:

- `200` on success
- `400` if `system` is missing or not one of `windows`, `mac`, `linux`

### `GET /omni/supported-models/best?system=windows|mac|linux`

Returns a merged catalog where each quantization chooses the best installed backend for the current machine.

The response is grouped by model family and model name. Each quantization includes:

- `required_memory_gib`
- `available_memory_gib` (or `null` when unavailable)
- `memory_status`: `sufficient`, `insufficient`, or `unknown`
- `suitable` (true only when memory status is `sufficient`)
- `backend`

`backend` identifies the selected installed, hardware-compatible backend even when memory is
insufficient or unavailable. It is empty only when no installed hardware-compatible backend exists.

Example:

```bash
curl -s "http://127.0.0.1:9000/omni/supported-models/best?system=windows"
```

## Model Loading

### `POST /omni/model/select`

Loads a model and optionally an `mmproj` file. This is the management endpoint that starts the selected backend runtime.
For the stable gateway contract, see [Model Load Parameters](model-load.md).

Request body:

```json
{
  "model": "<relative-or-absolute-model-path>",
  "mmproj": "<optional-relative-or-absolute-mmproj-path>",
  "backend": "<optional-backend-id>",
  "ctx_size": 4096,
  "launch_args": [],
  "strict_capabilities": false,
  "request_defaults": {
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  }
}
```

Notes:

- `model` is required.
- `backend` is optional. If omitted, OmniInfer uses the selected backend or auto-selection logic.
- `ctx_size` is optional and may also be sent as `ctx-size`.
- `launch_args` is optional and intended for backend-native launch arguments.
- Official Linux and Windows llama.cpp CUDA backends default to automatic placement by omitting `-ngl`. Use `-ngl 999` only to require strict full GPU offload.
- `request_defaults` is merged into later inference requests after this model is loaded.
- Effective request defaults are exposed in runtime state and retained when the selected model is restored.
- `strict_capabilities` is optional. When true, unsupported load options fail instead of being ignored with warnings.
- Relative model paths resolve under the selected backend's `models_dir`.
- When the service is started with `--public-model-root` and remote management is enabled, remote clients should pass a public model id such as `qwen3.5-4b-q4_k_m` instead of a server filesystem path. Remote path selection is rejected; local loopback clients may still use explicit paths.

Example response:

```json
{
  "ok": true,
  "already_loaded": false,
  "requires_reload": false,
  "selected_backend": "llama.cpp-cuda",
  "selected_model": "models/Qwen3.5-2B-Q4_K_M.gguf",
  "selected_mmproj": null,
  "selected_ctx_size": 4096,
  "runtime_placement": {
    "source": "llama.cpp_startup_log",
    "policy": "auto",
    "requested_gpu_layers": null,
    "mode": "partial",
    "offloaded_layers": 28,
    "total_layers": 41,
    "reported_buffer_bytes": {"host": 1500000000, "cuda:0": 4939212390},
    "reconciled_budget": {
      "domains_bytes": {"host": 2147483648, "cuda:0": 5368709120},
      "components": [
        {"name": "reported_model_buffers", "domain": "host", "bytes": 1500000000},
        {"name": "runtime_overhead", "domain": "host", "bytes": 402653184},
        {"name": "reconciliation_slack", "domain": "host", "bytes": 244830464},
        {"name": "reported_model_buffers", "domain": "cuda:0", "bytes": 4939212390},
        {"name": "runtime_overhead", "domain": "cuda:0", "bytes": 134217728},
        {"name": "reconciliation_slack", "domain": "cuda:0", "bytes": 295279002}
      ]
    }
  },
  "warnings": []
}
```

For automatic or explicit partial llama.cpp CUDA placement, OmniInfer holds a
provisional host/CUDA reservation during startup and atomically replaces it
with the placement reported by llama.cpp after readiness. Missing or unsafe
placement evidence returns `502`; the runtime process, listener, route, and
ledger reservation are rolled back together.

Selecting the same resolved model, backend, `mmproj`, context size, and effective launch arguments again is idempotent. OmniInfer returns `200` with `already_loaded: true`, `requires_reload: false`, and the existing backend PID instead of starting a second runtime. This also applies when a startup-restored path is selected again through its public model id.

If the model is already loaded with different runtime settings, OmniInfer leaves it unchanged and returns `409` with `requires_reload: true`, `error.code: "model_reload_required"`, and `current` plus `requested` configurations. The client can then explicitly unload or stop the runtime before selecting the new configuration.

### `GET /omni/public-models`

Lists model ids exposed through `--public-model-root`. This endpoint is intended for authenticated management clients and uses the admin API key when `--admin-api-key` is configured.

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-4b-q4_k_m",
      "aliases": ["qwen35-4b-q4_k_m"],
      "display_name": "Qwen3.5 4B Q4_K_M",
      "backend": "llama.cpp-linux-cuda",
      "modalities": ["text"],
      "quant": "Q4_K_M",
      "ctx_size": 8192,
      "model": "/path/to/public_models/qwen3.5-4b-q4_k_m/Qwen3.5-4B-Q4_K_M.gguf",
      "mmproj": null
    }
  ]
}
```

Status codes:

- `200` on success
- `400` for invalid input or missing files
- `409` when the selected model is already loaded with different runtime settings
- `502` when backend startup or post-start resource reconciliation fails

### Streaming model load

If `Accept` contains `text/event-stream`, `/omni/model/select` returns SSE progress events.

Example:

```bash
curl -N -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"model":"models/Qwen3.5-2B-Q4_K_M.gguf"}' \
  http://127.0.0.1:9000/omni/model/select
```

SSE events:

```text
data: {"type":"status","message":"Starting backend..."}
data: {"type":"log","message":"..."}
data: {"type":"done","elapsed_s":4.2,"ok":true,"selected_backend":"llama.cpp-cuda","selected_model":"models/Qwen3.5-2B-Q4_K_M.gguf","selected_mmproj":null,"selected_ctx_size":4096}
data: [DONE]
```

Errors are sent as SSE events:

```text
data: {"type":"error","message":"backend did not become ready in time"}
data: [DONE]
```

## Thinking

### `GET /omni/thinking`

Returns the default thinking mode.

Example response:

```json
{
  "default_enabled": false
}
```

### `POST /omni/thinking/select`

Updates the default thinking mode. The request accepts either `enabled` or `think`.

Request body:

```json
{
  "enabled": false
}
```

Example response:

```json
{
  "ok": true,
  "default_enabled": false
}
```

Status codes:

- `200` on success
- `400` if the field is missing or cannot be parsed as a boolean

## Cache

### `POST /omni/cache/clear`

Erases the backend's KV cache without reloading the model. The current implementation targets llama.cpp-compatible external servers by calling slot erase on slot 0.

Example:

```bash
curl -X POST http://127.0.0.1:9000/omni/cache/clear
```

Example response:

```json
{
  "ok": true,
  "message": "KV cache cleared"
}
```

Status codes:

- `200` on success
- `409` if no external backend is running or the backend rejects cache clearing

Multimodal llama.cpp loads may reject KV cache clearing; reload the model instead.

## stable-diffusion.cpp Native API

When a `stable-diffusion.cpp-*` backend is loaded, OmniInfer keeps `sd-server`
bound to loopback and proxies its native asynchronous API through the gateway.
Remote LAN and Cloudflare clients use the normal OmniInfer inference API key;
they do not need remote-management access. The gateway does not forward client
authorization headers to the loopback runtime.

The stable endpoints are:

- `GET /sdcpp/v1/capabilities`
- `POST /sdcpp/v1/img_gen`
- `POST /sdcpp/v1/vid_gen`
- `GET /sdcpp/v1/jobs/{id}`
- `POST /sdcpp/v1/jobs/{id}/cancel`

Query capabilities after each model load. Defaults, output formats, and
supported modes are checkpoint-dependent. Submissions return `202` with an
`id` and `poll_url`; poll until `status` becomes `completed`, `failed`, or
`cancelled`. A completed video job returns the encoded container in
`result.b64_json` together with its MIME type, FPS, and frame count.
MiniMax H3 normalizes every requested frame count to the smallest supported
value at or above it: at least 5 frames and `video_frames % 17 == 5`. For
example, a request for 25 frames produces 39 frames. Treat
`result.frame_count` as the authoritative output count.

MiniMax H3 example load request:

```json
{
  "backend": "stable-diffusion.cpp-vulkan",
  "model": "C:\\models\\MiniMax-H3-FL2VA-Q4\\minimax_h3_fl2va_pruned-Q4_K.gguf",
  "launch_args": [
    "--llm", "C:\\models\\MiniMax-H3-FL2VA-Q4\\qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    "--vae", "C:\\models\\MiniMax-H3-FL2VA-Q4\\vae\\minimax_h3_video_vae_fp16.safetensors",
    "--audio-vae", "C:\\models\\MiniMax-H3-FL2VA-Q4\\vae\\minimax_h3_audio_vae_fp32.safetensors",
    "--cfg-scale", "1.0",
    "--diffusion-fa",
    "--backend", "te=cpu",
    "--rng", "cpu"
  ]
}
```

OmniInfer budgets stable-diffusion.cpp artifacts by module before launch. The
denoiser is assigned to `diffusion`, `--llm` to `te`, and both `--vae` and
`--audio-vae` to `vae`. Parameter bytes follow `--params-backend` (or the
module's runtime backend when no parameter override exists), while workspace
and safety overhead follow `--backend`. `--offload-to-cpu` is treated exactly
as upstream treats it: an effective default `*=cpu` parameter assignment.
When a module's parameter and runtime domains differ, OmniInfer also reserves a
full runtime-side staging copy because stable-diffusion.cpp copies those
parameters to the compute backend while that module runs.

Vulkan reservations use a `vulkan:<index>` memory domain and the selected
driver's `VK_EXT_memory_budget` free-memory report. Unknown device names,
missing component sizes, unavailable memory-budget evidence, dynamic model
directories, weight type overrides, and `--auto-fit` placement fail closed
before the runtime starts. Use explicit `cpu` or `vulkan<index>` assignments
when overriding placement.

The model-load endpoint remains local-only by default. Once loaded, an
authenticated inference client can submit a short validation job:

```bash
curl -sS http://127.0.0.1:9000/sdcpp/v1/capabilities \
  -H 'Authorization: Bearer <api-key>'

curl -sS http://127.0.0.1:9000/sdcpp/v1/vid_gen \
  -H 'Authorization: Bearer <api-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt":"A silver tabby kitten surfing on a sunny wave",
    "width":640,
    "height":384,
    "video_frames":25,
    "fps":24,
    "sample_params":{"sample_steps":4},
    "output_format":"webm"
  }'
```

Native diffusion request bodies are capped at 16 MiB. Upstream status codes,
content type, and safe response headers pass through. Loading a non-diffusion
backend makes `/sdcpp/v1/*` return structured `422`; calling chat or Messages
with a diffusion backend loaded likewise returns structured `422`.

## OpenAI-Compatible API

### `GET /v1/models`

Returns the currently loaded model in OpenAI-compatible list format. If no model is loaded or the backend is not ready, `data` is empty.

Example response when a model is loaded:

```json
{
  "object": "list",
  "data": [
    {
      "id": "models/Qwen3.5-2B-Q4_K_M.gguf",
      "object": "model",
      "created": 0,
      "owned_by": "omniinfer",
      "permission": [],
      "root": "models/Qwen3.5-2B-Q4_K_M.gguf",
      "parent": null
    }
  ]
}
```

Example response when no model is loaded:

```json
{
  "object": "list",
  "data": []
}
```

### `POST /tokenize`

Tokenizes text with the currently loaded external llama.cpp-compatible backend. This endpoint proxies the upstream llama.cpp server API and preserves its response shape.

Request body:

```json
{
  "content": "Hello",
  "add_special": false,
  "parse_special": true,
  "with_pieces": false
}
```

Example response:

```json
{
  "tokens": [123, 456]
}
```

`/omni/tokenize` is a local-only alias for local controllers. Remote LAN and Cloudflare clients should use `/tokenize` with an API key.

Status codes:

- `200` on success
- `409` if no external backend model is loaded
- `501` if the current backend is embedded and does not expose a llama.cpp tokenizer API
- backend status codes may be passed through

### `POST /detokenize`

Converts token IDs back to text with the currently loaded external llama.cpp-compatible backend.

Request body:

```json
{
  "tokens": [123, 456]
}
```

Example response:

```json
{
  "content": "Hello"
}
```

`/omni/detokenize` is a local-only alias for local controllers. Remote LAN and Cloudflare clients should use `/detokenize` with an API key.

Status codes match `/tokenize`.

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

Important behavior:

- This endpoint does not load a model. Load a model first with `/omni/model/select`.
- `model`, `mmproj`, `backend`, `ctx_size`, and `launch_args` in this request are accepted for compatibility but are not used to start or switch runtimes.
- `request_defaults` can be supplied and is merged with the loaded runtime defaults for the current request.
- Merge precedence is loaded runtime defaults, nested `request_defaults`, then explicit top-level request fields. Non-object defaults return HTTP 400.
- If no runtime is loaded, the endpoint returns `400` or `409`.
- If the loaded backend is not OpenAI-compatible, including `vla.cpp-*`, the endpoint returns structured `422` instead of forwarding the request.

Request body:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "model": "optional-display-model",
  "temperature": 0.2,
  "max_tokens": 128,
  "stream": false,
  "think": false,
  "reasoning_effort": "none",
  "reasoning": {
    "effort": "none"
  },
  "request_defaults": {
    "top_p": 0.9
  }
}
```

Supported OmniInfer thinking hints:

- `think: true|false`
- `reasoning_effort`: values such as `none`, `off`, `disabled`, `false`, or `0` disable thinking
- `reasoning.effort`: same semantics as `reasoning_effort`

OmniInfer maps these hints to the local thinking on/off switch. It does not implement effort-specific reasoning budgets.

Multimodal content follows the OpenAI content-list shape:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,..."
          }
        }
      ]
    }
  ]
}
```

Non-stream response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "models/Qwen3.5-2B-Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "Thinking...",
        "content": "Hello!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 12,
    "total_tokens": 25
  }
}
```

Streaming response:

If `stream` is `true`, the endpoint returns SSE. OmniInfer normalizes `<think>...</think>` output into `delta.reasoning_content` and keeps usage/timing payloads as a final usage event where possible.

```text
data: {"choices":[{"delta":{"reasoning_content":"Thinking...","content":null}}]}
data: {"choices":[{"delta":{"content":"Hel"}}]}
data: {"choices":[{"delta":{"content":"lo"}}]}
data: {"choices":[],"usage":{"prompt_tokens":13,"completion_tokens":12,"total_tokens":25}}
data: [DONE]
```

OmniInfer line streaming:

Set `stream` to `true` and add `omni_stream.format = "lines"` to receive stable line-oriented SSE events instead of OpenAI token chunks. This is useful for clients that render by line while still preserving SSE framing, usage, and finish metadata.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write a short markdown list."
    }
  ],
  "stream": true,
  "omni_stream": {
    "format": "lines",
    "max_line_chars": 240,
    "include_reasoning": false
  }
}
```

Line stream response:

```text
event: line
data: {"index":0,"type":"content","role":"assistant","text":"Here are three points:","newline":true}

event: line
data: {"index":1,"type":"content","role":"assistant","text":"- Fast local inference","newline":true}

event: done
data: {"finish_reason":"stop","usage":{"prompt_tokens":13,"completion_tokens":12,"total_tokens":25}}
```

`max_line_chars` defaults to `240`. If a generated paragraph exceeds that size before a newline, OmniInfer emits a partial line with `"newline": false`. Set `include_reasoning` to `true` to receive reasoning lines as `type: "reasoning"` events when the backend provides `reasoning_content`.

Status codes:

- `200` on success
- `400` for invalid request fields or missing model state
- `409` if the selected backend is not ready
- backend status codes may be passed through for proxied non-stream requests

## Anthropic-Compatible API

### `POST /v1/messages`

Anthropic Messages API compatibility adapter. OmniInfer converts the request to the OpenAI chat-completions format internally and then converts the response back to Anthropic shape.

Like `/v1/chat/completions`, this endpoint does not load a model. Load a model first with `/omni/model/select`.

Request body:

```json
{
  "model": "optional-display-model",
  "max_tokens": 128,
  "system": "You are concise.",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "temperature": 0.2,
  "stream": false
}
```

Supported request features include:

- `system` as a string or list of text blocks
- text messages
- image blocks with base64 or URL sources
- `max_tokens`, `temperature`, `top_p`, `top_k`
- `stop_sequences`
- `stream`
- `thinking.type = enabled|disabled`
- tool definitions, tool use, tool results, and `tool_choice`

Non-stream response shape:

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "models/Qwen3.5-2B-Q4_K_M.gguf",
  "content": [
    {
      "type": "text",
      "text": "Hello!"
    }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 13,
    "output_tokens": 12
  }
}
```

Streaming response uses Anthropic-style SSE events converted from the backend stream.

Status codes:

- `200` on success
- `400` for invalid Anthropic request shape or missing model state
- `409` if the selected backend is not ready
- `502` if a proxied backend returns invalid JSON
- backend status codes may be passed through when the proxy request fails

## Deprecated Endpoint

### `GET /omni/models`

This endpoint is deprecated and intentionally not maintained.

Response:

```json
{
  "error": {
    "message": "GET /omni/models has been deprecated and is no longer maintained"
  }
}
```

Status code: `410`.

## Shutdown

### `POST /omni/shutdown`

Stops the local OmniInfer gateway.

Example response:

```json
{
  "ok": true,
  "message": "shutdown requested"
}
```

## Error Format

Most OmniInfer management endpoints return errors as:

```json
{
  "error": {
    "message": "..."
  }
}
```

The Anthropic-compatible endpoint may return Anthropic-style error objects:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "..."
  }
}
```

## Desktop Versus Mobile

This document covers the desktop gateway exposed by the Rust control plane.

Android and iOS clients expose a smaller OpenAI-compatible subset from the mobile app process. See the OmniStudio API service documents for mobile-specific behavior.
