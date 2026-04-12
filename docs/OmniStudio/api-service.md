# OmniStudio API Service Guide

OmniStudio is a cross-platform model inference client powered by OmniInfer. When a model is loaded in the chat interface, OmniStudio automatically starts an OpenAI-compatible API service in the background, allowing external applications to call the local inference engine via HTTP.

## API Overview

The API service runs on `http://<host>:<port>` (default `http://127.0.0.1:9000`).

Key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and runtime state |
| `/omni/state` | GET | Current backend, model, and config |
| `/omni/backends` | GET | List available backends |
| `/omni/model/select` | POST | Load a model |
| `/omni/backend/stop` | POST | Stop current backend |
| `/v1/models` | GET | List currently loaded models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/omni/shutdown` | POST | Shut down the gateway |

---

## PC Client (Windows / macOS)

Windows and macOS clients share the same architecture: OmniStudio bundles a platform-specific OmniInfer release package and manages the inference gateway as a background process.

### Prerequisites

- OmniStudio desktop client installed
- At least one model file downloaded
- Sufficient RAM/VRAM for the target model

### Step 1: Load a Model

Open the OmniStudio chat interface and select a model. Once the model finishes loading, the API service is automatically started.

At this point, the following services are running in the background:

```
OmniStudio (UI)
  └─ OmniInfer Gateway (HTTP, default port 9000)
```

### Step 2: Verify the Service

Open a terminal and run:

```bash
curl http://127.0.0.1:9000/health
```

Expected response:

```json
{
  "status": "ok",
  "omni": {
    "backend": "llama.cpp-cpu",
    "model": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "backend_ready": true
  }
}
```

`backend_ready: true` confirms the API is ready to accept inference requests.

### Step 3: Call the API

**Non-streaming request:**

```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'
```

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "Qwen3.5-0.8B-Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Streaming request:**

```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a haiku about code."}
    ],
    "stream": true
  }'
```

The response is Server-Sent Events (SSE):

```text
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}
data: {"choices":[{"delta":{"content":"Lines"}}]}
data: {"choices":[{"delta":{"content":" of"}}]}
...
data: {"usage":{"prompt_tokens":12,"completion_tokens":18,"total_tokens":30}}
data: [DONE]
```

### Step 4: Integrate with Third-Party Applications

Any application that supports OpenAI-compatible APIs can connect to OmniStudio. Set the base URL to `http://127.0.0.1:9000` and leave the API key empty (or use any placeholder).

**Python (openai SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**JavaScript/TypeScript (fetch):**

```typescript
const response = await fetch("http://127.0.0.1:9000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello" }],
    stream: false,
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### Available Backends

Backends vary by platform. OmniStudio automatically selects the best available backend, but you can switch via the API:

**Windows:**

| Backend | Acceleration | Requirements |
|---------|-------------|--------------|
| `llama.cpp-cpu` | CPU | None |
| `llama.cpp-cuda` | NVIDIA GPU | CUDA runtime |
| `llama.cpp-vulkan` | GPU (vendor-agnostic) | Vulkan driver |

**macOS:**

| Backend | Acceleration | Requirements |
|---------|-------------|--------------|
| `llama.cpp-mac` | Metal (Apple Silicon) | Apple Silicon Mac |
| `mlx-mac` | MLX (Apple Silicon) | Apple Silicon Mac |

### Advanced: Switch Backend via API

```bash
curl -X POST http://127.0.0.1:9000/omni/model/select \
  -H "Content-Type: application/json" \
  -d '{
    "model": "path/to/model.gguf",
    "backend": "llama.cpp-cuda",
    "ctx_size": 4096,
    "launch_args": ["-ngl", "999"]
  }'
```

### Stop the Service

The API service stops automatically when:
- The model is unloaded in OmniStudio
- OmniStudio is closed

To stop programmatically:

```bash
curl -X POST http://127.0.0.1:9000/omni/shutdown
```

---

## Android Client

> **TODO** — Android API service documentation pending.

---

## iOS Client

> **TODO** — iOS API service documentation pending.
