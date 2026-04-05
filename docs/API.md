# OmniInfer API

This document describes the public local HTTP API exposed by OmniInfer.

Base URL:

```text
http://127.0.0.1:9000
```

All requests use JSON unless noted otherwise.

## 1. Health check

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
  "omni": {
    "backend": "llama.cpp-cpu",
    "model": null,
    "mmproj": null,
    "ctx_size": null,
    "request_defaults": {},
    "backend_ready": false
  },
  "thinking": {
    "default_enabled": false
  }
}
```

## 2. Runtime state

### `GET /omni/state`

Returns the current runtime state.

Example response:

```json
{
  "backend": "llama.cpp-cpu",
  "model": "models/example.gguf",
  "mmproj": "models/mmproj-F32.gguf",
  "ctx_size": 4096,
  "request_defaults": {
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true,
    "think": false
  },
  "backend_ready": true,
  "available_backends": [
    {
      "id": "llama.cpp-cpu",
      "label": "llama.cpp cpu",
      "family": "llama.cpp",
      "selected": true,
      "binary_exists": true,
      "models_dir": "models",
      "capabilities": ["chat", "vision", "stream", "cpu"],
      "description": "..."
    }
  ],
  "thinking": {
    "default_enabled": false
  }
}
```

## 3. List backends

### `GET /omni/backends`

Returns all local backends.

Depending on the current host and which runtime folders are present locally, the list may include backends such as:

- Windows: `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, `llama.cpp-hip`
- Linux: `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, `llama.cpp-linux-openvino`
- macOS: `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, `mlx-mac`

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama.cpp-cpu",
      "label": "llama.cpp cpu",
      "selected": true,
      "binary_exists": true,
      "models_dir": "models",
      "capabilities": ["chat", "vision", "stream", "cpu"],
      "description": "...",
      "loaded_model": null
    }
  ]
}
```

## 4. Supported models

### `GET /omni/supported-models?system=<windows|mac|linux>`

Returns the backend-grouped supported model catalog for a target system.

### `GET /omni/supported-models/best?system=<windows|mac|linux>`

Returns the same catalog after OmniInfer chooses the best backend for each quantization.

Example:

```bash
curl -s "http://127.0.0.1:9000/omni/supported-models/best?system=windows"
```

The response is a JSON object grouped by model family and model name. Each quantization entry includes:

- `required_memory_gib`
- `suitable`
- `backend`

## 5. Load a model

### `POST /omni/model/select`

Loads a model and optionally an `mmproj` file.

Request body:

```json
{
  "model": "<relative-or-absolute-model-path>",
  "mmproj": "<optional-relative-or-absolute-mmproj-path>",
  "backend": "<optional-backend-id>",
  "launch_args": ["-ngl", "999"],
  "request_defaults": {
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  },
  "ctx_size": 4096
}
```

Notes:

- `backend` is optional.
- `ctx_size` is optional and maps to the backend context length.
- `launch_args` is optional and is intended for backend-native launch arguments managed by advanced CLI config files.
- `request_defaults` is optional and stores default inference fields for later requests after the model is loaded.

Example:

```bash
curl -X POST http://127.0.0.1:9000/omni/model/select \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-0.8B-Q4_K_M.gguf",
    "launch_args": ["-ngl", "999"],
    "ctx_size": 4096
  }'
```

Example response:

```json
{
  "ok": true,
  "selected_backend": "llama.cpp-cpu",
  "selected_model": "models/Qwen3.5-0.8B-Q4_K_M.gguf",
  "selected_mmproj": "models/mmproj-F32.gguf",
  "selected_ctx_size": 4096
}
```

## 6. Thinking mode

### `GET /omni/thinking`

Returns the default thinking mode.

### `POST /omni/thinking/select`

Updates the default thinking mode.

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

## 7. Stop the current backend

### `POST /omni/backend/stop`

Stops the currently loaded runtime process.

Example response:

```json
{
  "ok": true,
  "stopped": true,
  "selected_backend": "llama.cpp-cpu"
}
```

## 8. Chat completions

### `POST /v1/chat/completions`

OpenAI-compatible chat endpoint.

Request body:

```json
{
  "model": "<optional-model-path>",
  "mmproj": "<optional-mmproj-path>",
  "backend": "<optional-backend-id>",
  "launch_args": ["-ngl", "999"],
  "request_defaults": {
    "temperature": 0.2
  },
  "ctx_size": 4096,
  "think": false,
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "temperature": 0.2,
  "max_tokens": 128,
  "stream": false
}
```

Notes:

- If a model is already loaded, `model` is optional.
- `ctx_size` is optional.
- `launch_args` and `request_defaults` are optional OmniInfer extensions for backend-specific config-driven flows.
- If `stream=true`, the response uses Server-Sent Events (SSE).

### Non-stream response shape

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "Qwen3.5-0.8B-Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
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

### Stream response shape

Each SSE frame starts with `data:`.

Example:

```text
data: {"choices":[{"delta":{"content":"Hel"}}]}
data: {"choices":[{"delta":{"content":"lo"}}]}
data: {"usage":{"prompt_tokens":13,"completion_tokens":12,"total_tokens":25}}
data: [DONE]
```

## 9. Shutdown

### `POST /omni/shutdown`

Stops the local OmniInfer gateway.

Example response:

```json
{
  "ok": true,
  "message": "shutdown requested"
}
```

## 10. Error format

Errors are returned as JSON:

```json
{
  "error": {
    "message": "..."
  }
}
```
