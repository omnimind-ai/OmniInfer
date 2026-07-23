# Model Load Parameters

This document defines the stable gateway contract for loading a model through
`POST /omni/model/select`.

## Request

```json
{
  "model": "<relative-or-absolute-model-path>",
  "backend": "<optional-backend-id>",
  "mmproj": "<optional-mmproj-path>",
  "ctx_size": 4096,
  "launch_args": ["-ngl", "999"],
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
| `model` | string | load | yes | Required. Relative paths resolve under the selected backend model root for file/directory backends. Reference backends such as `vllm-linux-cuda` pass this string directly to the backend. |
| `backend` | string | load | maybe | Optional. If omitted, OmniInfer uses selected or automatic backend logic. |
| `mmproj` | string | load | yes | Optional multimodal projector override. |
| `ctx_size` / `ctx-size` | integer | load | yes | Optional context length override. |
| `launch_args` | string array or shell string | load | yes | Optional backend-native launch arguments for external server backends. |
| `request_defaults` | object | generation defaults | no | Stored with the loaded runtime and merged into later inference requests. |
| `strict_capabilities` | boolean | validation | no | Optional. When true, unsupported load options fail instead of being ignored with warnings. |

`request_defaults` is not a model-load setting. It is a convenient way for a
client to attach generation defaults to the loaded runtime. Changing only
`request_defaults` can reuse the current runtime when the load settings match.

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
  "warnings": []
}
```

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

`vllm-linux-cuda` runs the official vLLM OpenAI-compatible server. OmniInfer
starts it as:

```text
vllm serve <model> --host <loopback-host> --port <backend-port>
```

For this backend, `ctx_size` maps to vLLM's `--max-model-len`, and OmniInfer
adds `--served-model-name local` unless the user supplies a backend-native
`--served-model-name` in `launch_args`. `mmproj` is not supported and is ignored
with a warning unless `strict_capabilities` is true.

## Chat Requests

`POST /v1/chat/completions` does not load or switch models. It accepts
OpenAI-compatible generation parameters for the current request and merges them
over the runtime `request_defaults`.

The following fields may appear in a chat request for compatibility but do not
start or switch a runtime there: `model`, `backend`, `mmproj`, `ctx_size`, and
`launch_args`.
