# Model Load Parameters

This document defines the stable gateway contract for loading a model through
`POST /omni/model/select`.

## Request

```json
{
  "model": "<relative-or-absolute-model-path>",
  "backend": "<optional-backend-id>",
  "mmproj": "<optional-mmproj-path>",
  "no_mmproj": false,
  "ctx_size": 4096,
  "resource_budget_bytes": 7516192768,
  "launch_args": [],
  "request_defaults": {
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  },
  "strict_capabilities": false
}
```

## Fields

| Field | Type | Scope | Reloads runtime | Notes |
|---|---:|---|---:|---|
| `model` | string | load | yes | Required. Relative paths resolve under the selected backend model root for file/directory backends. Reference backends such as `vllm-linux-cuda`, `vllm-wsl2-cuda`, and `vllm-wsl2-rocm` pass model references directly to the backend. |
| `backend` | string | load | maybe | Optional. If omitted, OmniInfer uses selected or automatic backend logic. |
| `mmproj` | string | load | yes | Optional multimodal projector override. |
| `no_mmproj` | boolean | load | yes | Defaults to `false`. Conflicts with `mmproj`; when `true`, disables explicit and automatic projector selection. |
| `ctx_size` / `ctx-size` | integer | load | yes | Optional context length override. |
| `resource_budget_bytes` | positive integer | admission | yes | Optional explicit runtime memory budget. It is required when a reference backend cannot resolve the model to a local file or directory, and it cannot be lower than OmniInfer's local estimate when one is available. |
| `launch_args` | string array or shell string | load | yes | Optional backend-native launch arguments for external server backends. |
| `request_defaults` | object | generation defaults | no | Stored with the loaded runtime and merged into later inference requests. |
| `strict_capabilities` | boolean | validation | no | Optional. When true, unsupported load options fail instead of being ignored with warnings. |

The CLI exposes the same admission field as
`model load --resource-budget-bytes <bytes>` and
`serve --resource-budget-bytes <bytes>`. Supply it when a backend receives a
remote model reference, such as a Hugging Face repository ID, and OmniInfer
cannot inspect the artifact size locally.

The public CLI options `omniinfer model load --no-mmproj` and
`omniinfer serve --no-mmproj` disable both explicit and automatic visual
projector loading. `--no-mmproj` conflicts with `--mmproj`; use one or the
other. Without `--no-mmproj`, an ordinary model load retains automatic
discovery of an available projector.
The selection is saved with the model and honored by automatic restore during a
later ordinary `omniinfer serve` startup.

```sh
omniinfer model load -m /path/to/model-directory --no-mmproj
omniinfer serve -m /path/to/model-directory --no-mmproj
```

`request_defaults` is not a model-load setting. It is a convenient way for a
client to attach generation defaults to the loaded runtime. Changing only
`request_defaults` can reuse the current runtime when the load settings match.
The effective defaults are exposed through runtime state and retained with the
selected model so an automatic restore reapplies the same request behavior.

Common generation defaults include:

```json
{
  "temperature": 0.2,
  "max_tokens": 128,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.05,
  "repeat_penalty": 1.1,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "seed": 1234,
  "stop": ["</s>"],
  "think": false,
  "stream": true
}
```

## Response

```json
{
  "ok": true,
  "already_loaded": false,
  "requires_reload": false,
  "selected_backend": "llama.cpp-cuda",
  "selected_model": "models/Qwen3.5-2B-Q4_K_M.gguf",
  "selected_mmproj": null,
  "selected_ctx_size": 4096,
  "generation": 1,
  "route_state": "ready",
  "allocation_id": 1,
  "resource_budget": {
    "domains_bytes": {"host": 2147483648, "cuda:0": 5368709120},
    "components": [
      {"name": "reported_model_buffers", "domain": "host", "bytes": 1500000000},
      {"name": "runtime_overhead", "domain": "host", "bytes": 402653184},
      {"name": "reconciliation_slack", "domain": "host", "bytes": 244830464},
      {"name": "reported_model_buffers", "domain": "cuda:0", "bytes": 4939212390},
      {"name": "runtime_overhead", "domain": "cuda:0", "bytes": 134217728},
      {"name": "reconciliation_slack", "domain": "cuda:0", "bytes": 295279002}
    ]
  },
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

OmniInfer reserves the reported budget before starting the backend, commits the
allocation only after readiness and local-state persistence succeed, and rolls
it back on failure. Resource domains are reported as `host`, `cuda:<index>`,
`vulkan:<index>`, or `unified:<id>`. For an explicit multi-GPU mapping that
cannot be attributed reliably, the full budget is reserved on every candidate
device rather than risk oversubscription.

For stable-diffusion.cpp, denoiser, text-encoder, and VAE weights are budgeted
separately from runtime workspace and safety overhead. Weight domains follow
the effective `--params-backend` assignment; when it is unset they follow the
corresponding `--backend` module. Vulkan capacity comes from
`VK_EXT_memory_budget`. If parameter storage and execution use different
domains, the runtime-side weight staging copy is reserved as well. OmniInfer
rejects unknown or dynamic placement, dynamic model directories, and weight
type overrides before launch rather than guessing a host/GPU split.

### Official llama.cpp CUDA placement

On Linux and Windows, official llama.cpp CUDA backends leave the GPU-layer
argument unset by default. This preserves llama.cpp's automatic fitter, which
may place model tensors in both host memory and CUDA memory when full GPU
offload does not fit. Sending `-ngl auto` or `--gpu-layers=auto` has the same
effect: OmniInfer removes the argument before launch.

Automatic and explicit partial-offload loads first reserve a conservative host
ceiling and the available memory on the selected CUDA device. After the runtime
is ready, OmniInfer reads llama.cpp's layer and buffer-placement lines from this
startup only, maps logical `CUDA0` to the selected physical device, adds runtime
overhead and safety slack, and atomically reconciles the reservation. To make
that safety evidence available consistently, OmniInfer appends the managed
trace setting `-lv 4` to automatic and explicit partial-offload launches;
`--log-disable` is rejected for those policies. The final values appear in
`resource_budget` and `runtime_placement` in the load response, `GET
/omni/state`, and loaded-model payloads.

Placement mode is derived from the reported model buffers, not only the layer
counter. A runtime that reports both CPU and CUDA model buffers is `partial`
even when llama.cpp reports every repeating layer as offloaded, as can happen
with overflowed tensors in MoE models. Host-only compute or output buffers do
not by themselves make an otherwise CUDA-resident model partial.

An explicit full-offload request such as `-ngl 999`, `--gpu-layers=all`, or
`--gpu-layers=max` keeps strict pre-launch CUDA admission and fails fast when it
cannot fit. If automatic placement cannot be parsed or its reconciled host/CUDA
budget exceeds capacity, the load fails and OmniInfer stops the process tree,
closes the listener, withholds the route, and rolls back the reservation.

## Idempotency and Reloads

The gateway compares the resolved model path, backend, `mmproj`, context size,
and effective backend launch arguments. Repeating an identical selection
returns `200` and reuses the current process:

```json
{
  "ok": true,
  "already_loaded": true,
  "requires_reload": false,
  "backend_pid": 45210
}
```

This includes a model restored during direct `serve` startup and then selected
again by a client. A public model id can take over the restored path identity
without starting a second backend process.

When any runtime setting differs, the gateway does not reload implicitly. It
returns `409` with both configurations so the client can perform a controlled
unload or stop first:

```json
{
  "ok": false,
  "already_loaded": true,
  "requires_reload": true,
  "error": {
    "code": "model_reload_required",
    "message": "model is already loaded with different runtime settings"
  },
  "current": {"ctx_size": 4096},
  "requested": {"ctx_size": 8192}
}
```

`POST /omni/backend/stop` only stops the current runtime and preserves the
startup selection. `POST /omni/model/clear-selection` disables future restore
without stopping a runtime that is currently loaded.

When the gateway accepts a request but drops a load option that the selected
backend cannot use, the response includes a warning:

```json
{
  "field": "ctx_size",
  "reason": "unsupported_by_backend",
  "message": "ctx_size is not supported by mlx-mac and was ignored"
}
```

Clients should treat warnings as user-visible diagnostics, not fatal errors.
For configuration screens that must reject unsupported settings, send
`strict_capabilities: true`.

## Backend-Specific Notes

Official `llama.cpp-*` backends use the following cache-safety defaults:

```text
--slot-prompt-similarity 0 --cache-idle-slots --cache-ram 8192
```

Disabling slot-similarity selection makes an available slot an LRU scheduling
decision, after which llama.cpp can search its RAM prompt cache for a better
compatible state. The RAM value is a MiB capacity limit, not an eager
allocation. Backend-specific `launch_args` extend these defaults and appear
after them, so an explicit later value overrides the matching default.

For five concurrent slots on a host with sufficient memory, a typical override
is:

```json
{
  "launch_args": ["-np", "5", "--cache-ram", "32768"]
}
```

Size the RAM cache for the model architecture, context length, and number of
warm sessions. Cache reuse remains best-effort: when no recurrent/KV checkpoint
exists at or before the common-prefix boundary, llama.cpp rejects the unsafe
state and performs a full prefill. This preserves session semantics but does
not guarantee a cache hit for every warm request.

`vllm-linux-cuda` and Windows `vllm-wsl2-cuda` / `vllm-wsl2-rocm` run the official vLLM
OpenAI-compatible server. OmniInfer starts the Linux backend as:

```text
vllm serve <model> --host <loopback-host> --port <backend-port>
```

For this backend, `ctx_size` maps to vLLM's `--max-model-len`, and OmniInfer
adds `--served-model-name local` unless the user supplies a backend-native
`--served-model-name` in `launch_args`. `mmproj` is not supported and is ignored
with a warning unless `strict_capabilities` is true.

On Windows, vLLM runs inside the WSL2 distribution recorded by the managed
launcher manifest. HuggingFace references are unchanged. Absolute drive paths
are translated to the distribution automount root (`D:\models\x` becomes
`/mnt/d/models/x` with the default WSL mount); UNC paths are rejected. Runtime
shutdown invokes the managed WSL stopper for the vLLM process group and does not
terminate the distribution.

## Chat Requests

`POST /v1/chat/completions` does not load or switch models. It accepts
OpenAI-compatible generation parameters for the current request and merges them
over the runtime `request_defaults`.

The precedence is runtime defaults, then the request's nested
`request_defaults`, then explicit top-level request fields. A non-object
`request_defaults` value is rejected with HTTP 400.

The following fields may appear in a chat request for compatibility but do not
start or switch a runtime there: `model`, `backend`, `mmproj`, `ctx_size`, and
`launch_args`.
