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

For the full API reference, see [API.md](https://github.com/omnimind-ai/OmniInfer/blob/main/docs/API.md).

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

The Android client runs the inference engine entirely on-device. OmniStudio embeds the OmniInfer native library and exposes the same OpenAI-compatible API through a local HTTP server powered by Ktor.

### Architecture

```
OmniStudio Android (UI)
  └─ OmniInferService (Foreground Service)
       └─ Ktor HTTP Server (127.0.0.1:9099)
            └─ JNI → Native C++
```

Unlike the PC client which runs a separate gateway process, the Android client runs the HTTP server as an Android foreground service within the app process. This keeps everything in a single process and complies with Android's background execution limits.

### Prerequisites

- OmniStudio Android installed on an ARM64 device (Snapdragon 8 Gen 1 or newer recommended)
- At least one model downloaded to local storage
- Sufficient free RAM

### Endpoints

The Android API exposes a subset of the full API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |

> **Note:** The `/omni/*` management endpoints are not available on Android. Model loading and backend selection are controlled through the OmniStudio UI or the Kotlin API (`OmniInferServer.loadModel()`).

### Step 1: Load a Model

Open OmniStudio and select a model to chat with. Once the model finishes loading, the API service starts automatically on port **9099**.

### Step 2: Verify the Service

**From a PC (via adb port forwarding):**

```bash
adb forward tcp:9099 tcp:9099
curl http://127.0.0.1:9099/health
```

**From another app on the same device:**

```kotlin
val url = URL("http://127.0.0.1:9099/health")
val result = url.readText()  // {"status":"ok"}
```

Expected response:

```json
{"status": "ok"}
```

### Step 3: Call the API

**Non-streaming request:**

```bash
curl -X POST http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 256,
    "stream": false
  }'
```

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "omniinfer",
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
    "prompt_tokens": 19,
    "completion_tokens": 8,
    "total_tokens": 27,
    "prompt_tokens_details": {
      "text_tokens": 19,
      "cached_tokens": 0,
      "cache_creation_input_tokens": 19,
      "cache_type": "ephemeral"
    },
    "performance": {
      "prefill_time_ms": 150.0,
      "prefill_tokens_per_second": 126.7,
      "decode_time_ms": 280.0,
      "decode_tokens_per_second": 28.6,
      "total_time_ms": 430.0,
      "time_to_first_token_ms": 150.0
    }
  }
}
```

**Streaming request:**

```bash
curl -X POST http://127.0.0.1:9099/v1/chat/completions \
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
data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":null}]}
data: {"choices":[{"delta":{"content":"Lines"},"index":0,"finish_reason":null}]}
data: {"choices":[{"delta":{"content":" of"},"index":0,"finish_reason":null}]}
...
data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}
data: {"choices":[],"usage":{...}}
data: [DONE]
```

### Step 4: Integrate with Third-Party Applications

**From a PC (via adb):**

Any OpenAI-compatible application on your PC can connect to the phone's inference engine. Set up port forwarding first:

```bash
adb forward tcp:9099 tcp:9099
```

Then point your application to `http://127.0.0.1:9099` as the base URL.

**Python (openai SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9099/v1",  # after adb forward
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

**From another Android app on the same device:**

Any Android app can call `http://127.0.0.1:9099/v1/chat/completions` via OkHttp, Retrofit, Ktor client, or any HTTP library. No special permissions are needed for localhost connections.

### Available Backends

| Backend | Model Format | Acceleration |
|---------|-------------|-------------|
| `llama.cpp` | GGUF (`.gguf`) | CPU (ARM NEON, i8mm, dotprod) |
| `mnn` | MNN (`config.json` + `.mnn` weights) | CPU (ARM82) + optional OpenCL |

Both backends support:
- Multi-turn conversation
- KV cache prefix reuse (automatic, no config needed)
- Multimodal / vision (model must include vision encoder files)
- Thinking / reasoning mode (`enable_thinking` or `reasoning_effort` parameter)
- Tool calling (llama.cpp: all models with tool templates; MNN: Qwen3.5, Qwen3, Hunyuan families)

### Stop the Service

The API service stops automatically when:
- The model is unloaded in OmniStudio
- OmniStudio is closed or the foreground service is dismissed

There is no HTTP shutdown endpoint on Android. Use the OmniStudio UI to manage the service lifecycle.

---

## iOS Client

The iOS client runs the inference engine entirely on-device, similar to the Android client. OmniStudio embeds the OmniInfer Swift Package and exposes the same OpenAI-compatible API through an in-process HTTP server powered by Hummingbird (swift-nio).

### Architecture

```
OmniStudio iOS (SwiftUI)
  └─ OmniInferServer (in-process)
       └─ Hummingbird HTTP Server (127.0.0.1:9099)
```

Unlike Android which uses a foreground service, the iOS client runs the HTTP server as an in-process NIO event loop within the app. iOS does not support background services, so the API is only available while the app is in the foreground.

### Prerequisites

- OmniStudio iOS installed (requires iOS 17+)
- At least one model downloaded to local storage

### Endpoints

The iOS API exposes the same subset as Android:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |

### Step 1: Load a Model

Open OmniStudio and select a model to chat with. Once the model finishes loading, the API service starts automatically on port **9099**.

### Step 2: Verify the Service

**From a Mac on the same network:**

The iOS API binds to `127.0.0.1` (loopback only), so it is only accessible from within the app process or via USB debugging tools. To access from a Mac:

1. Connect the iPhone via USB
2. Use Xcode's Network Link Conditioner or a proxy tool to forward traffic

**From within the app (programmatic access):**

```swift
import OmniInferServer

// Load a model
await OmniInferServer.shared.loadModel(
    modelPath: "/path/to/model.gguf",
    backend: "llama.cpp"  // or "mlx"
)

// The server is now running at http://127.0.0.1:9099
let url = URL(string: "http://127.0.0.1:9099/health")!
let (data, _) = try await URLSession.shared.data(from: url)
// {"status":"ok"}
```

### Step 3: Call the API

**Non-streaming request (from within the app):**

```swift
let url = URL(string: "http://127.0.0.1:9099/v1/chat/completions")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONSerialization.data(withJSONObject: [
    "messages": [["role": "user", "content": "What is 2+2?"]],
    "stream": false
])

let (data, _) = try await URLSession.shared.data(for: request)
let json = try JSONSerialization.jsonObject(with: data)
```

Response:

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4."
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prefill_tokens_per_second": 45.2,
    "decode_tokens_per_second": 21.8
  }
}
```

**Streaming request:**

```swift
let (bytes, _) = try await URLSession.shared.bytes(for: request)
for try await line in bytes.lines {
    guard line.hasPrefix("data: ") else { continue }
    let payload = String(line.dropFirst(6))
    if payload == "[DONE]" { break }
    // Parse SSE chunk...
}
```

### Step 4: Integrate with Other iOS Apps

Third-party iOS apps on the same device can call `http://127.0.0.1:9099/v1/chat/completions` via URLSession or any HTTP library. No special entitlements are needed for localhost connections on iOS.

**Integration via Swift Package:**

For direct integration without HTTP, add the OmniInferServer Swift Package as a dependency:

```swift
// Package.swift
dependencies: [
    .package(path: "Vendor/OmniInfer/ios/OmniInferServer")
]
```

Then use the OmniInferServer API directly:

```swift
import OmniInferServer

// Load model
await OmniInferServer.shared.loadModel(
    modelPath: modelDir,
    backend: "llama.cpp",  // or "mlx"
    port: 9099,
    nCtx: 4096
)

// Server is ready — use HTTP or direct engine access
```

### Stop the Service

The API service stops automatically when:
- The model is unloaded in OmniStudio
- OmniStudio is closed or goes to background

To stop programmatically:

```swift
await OmniInferServer.shared.stop()
```

There is no HTTP shutdown endpoint on iOS. Use the OmniInferServer Swift API to manage the service lifecycle.
