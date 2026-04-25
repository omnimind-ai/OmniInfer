# Android Integration Guide

This guide explains how to integrate the OmniInfer server library into your Android app as a Git submodule.

## Overview

The `android/omniinfer-server` module is a standalone Android library that provides:
- On-device LLM/VLM inference via llama.cpp, MNN, and ExecuTorch QNN (NPU) backends
- An OpenAI-compatible HTTP API (Ktor server) running locally
- JNI bridge to native C++ inference engines

Your app includes it as a Gradle module. The native backends (llama.cpp, MNN) are compiled from source via CMake during the Gradle build. The ExecuTorch QNN backend uses pre-built binaries for Qualcomm NPU acceleration.

## Step 1: Add OmniInfer as a Submodule

```bash
# From your app's repository root:
git submodule add https://github.com/omnimind-ai/OmniInfer.git third_party/omniinfer
```

**Important:** Do NOT run `git submodule update --init --recursive` on the OmniInfer submodule. The `framework/` directory contains many large submodules (mlx, executorch, etc.) that are only needed for non-Android platforms. Instead, initialize only the backends you need:

```bash
# llama.cpp backend (required if OMNIINFER_BACKEND_LLAMA_CPP=ON, which is default)
git submodule update --init third_party/omniinfer/framework/llama.cpp

# MNN backend (required if OMNIINFER_BACKEND_MNN=ON)
git submodule update --init third_party/omniinfer/framework/mnn
```

Typical download sizes:
| Submodule | Size | Required for |
|-----------|------|-------------|
| `framework/llama.cpp` | ~80 MB | llama.cpp backend (GGUF models) |
| `framework/mnn` | ~200 MB | MNN backend (MNN models) |
| All other `framework/*` | ~2 GB+ | Desktop/Linux platforms only — **not needed for Android** |

## Step 2: Configure Gradle

### `settings.gradle.kts`

Include the OmniInfer server module:

```kotlin
include(":omniinfer-server")
project(":omniinfer-server").projectDir =
    file("third_party/omniinfer/android/omniinfer-server")
```

### `app/build.gradle.kts`

Add the module dependency and required Ktor libraries:

```kotlin
val ktorVersion = "3.1.3"  // tested version — other 3.x versions may work

dependencies {
    implementation(project(":omniinfer-server"))

    // Ktor (HTTP server) — omniinfer-server declares these as compileOnly,
    // so the app must provide them. The versions below are tested and recommended.
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-cio:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

> **Why compileOnly?** This avoids transitive dependency conflicts when your app already uses Ktor or kotlinx-serialization at a different version. You control the exact versions in your app's dependency block.

### Backend Selection (CMake Arguments)

Both backends are enabled by default. To use only one, override CMake arguments in `app/build.gradle.kts`:

```kotlin
android {
    defaultConfig {
        externalNativeBuild {
            cmake {
                // Both backends (default):
                arguments += "-DOMNIINFER_BACKEND_LLAMA_CPP=ON"
                arguments += "-DOMNIINFER_BACKEND_MNN=ON"

                // llama.cpp only (smaller binary):
                // arguments += "-DOMNIINFER_BACKEND_MNN=OFF"

                // MNN only:
                // arguments += "-DOMNIINFER_BACKEND_LLAMA_CPP=OFF"
                // arguments += "-DOMNIINFER_BACKEND_MNN=ON"
            }
        }
    }
}
```

### Windows Long Path Workaround

Native CMake builds generate deep directory trees. On Windows, add this to avoid `MAX_PATH` (260 char) failures:

```kotlin
android {
    if (org.gradle.internal.os.OperatingSystem.current().isWindows) {
        externalNativeBuild.cmake.buildStagingDirectory =
            file("${System.getenv("TEMP") ?: "C:/tmp"}/.cxx/omniinfer")
    }
}
```

## Step 3: Allow Cleartext HTTP to Localhost

OmniInfer's HTTP server runs on `http://127.0.0.1`. Android 9+ blocks cleartext HTTP by default. You must add a network security config to allow localhost connections, or the app will fail to reach the server.

**`app/src/main/res/xml/network_security_config.xml`:**

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="false">127.0.0.1</domain>
    </domain-config>
</network-security-config>
```

**`app/src/main/AndroidManifest.xml`:**

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ... >
```

Without this, all HTTP requests to the local inference server will be rejected with `java.net.UnknownServiceException: CLEARTEXT communication to 127.0.0.1 not permitted`.

## Step 4: Use the API

### Lifecycle

```
init() → loadModel() → [requests...] → unloadModel() / stop()
```

- `init()` — call once at app startup. Only passes application context.
- `loadModel()` — loads model weights into memory, starts a foreground service running the Ktor HTTP server. Blocks until the model is ready.
- `unloadModel()` — frees model memory but keeps the HTTP server alive (for loading another model).
- `stop()` — unloads model AND stops the HTTP server.

### Initialize

```kotlin
import com.omniinfer.server.OmniInferServer

// Call once, e.g. in Application.onCreate() or Activity.onCreate()
OmniInferServer.init(context = applicationContext)
```

`init()` only stores the application context. No native resources are allocated.

### Customize the Notification (Optional)

The foreground service notification is customizable. Call `configureNotification()` after `init()` but before `loadModel()`:

```kotlin
OmniInferServer.configureNotification(
    title = "My AI Assistant",
    channelName = "AI Engine",
    smallIcon = R.drawable.ic_my_icon,
    textFormat = { port -> "AI engine active" }
)
```

All parameters are optional and have sensible defaults.

### Load a Model

```kotlin
// Must be called on a background thread (blocks during model loading).
val success: Boolean = OmniInferServer.loadModel(
    modelPath = "/path/to/model",   // required — see "Model Paths" below
    backend   = "llama.cpp",        // "llama.cpp" (default) or "mnn"
    port      = 9099,               // HTTP server port (default 9099)
    nThreads  = 0,                  // CPU threads, 0 = auto (cores - 1)
    nCtx      = 16384               // context window size in tokens (default 16384)
)
```

`loadModel()` does three things internally:
1. Calls `OmniInferBridge.init()` (JNI) to load the model into memory.
2. Starts `OmniInferService` as a foreground service (notification required by Android).
3. The Ktor HTTP server binds to `127.0.0.1:<port>`.

**Single-model behavior:** Only one model can be loaded at a time. Calling `loadModel()` with a different model/backend automatically unloads the previous one first. Calling with the same model is a no-op and returns immediately.

**Model Paths:**

| Backend | `modelPath` points to | Example |
|---------|----------------------|---------|
| `llama.cpp` | The `.gguf` model file | `/data/.../Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_K_M.gguf` |
| `mnn` | The `config.json` in the MNN model directory | `/data/.../Qwen3.5-2B-MNN/config.json` |
| `executorch-qnn` | The `.pte` model file | `/data/.../Qwen3-0.6B/hybrid_llama_qnn.pte` |

### Multimodal (Vision) Models

Both backends auto-detect multimodal support — **no extra API call needed**.

**llama.cpp:** The backend scans the model file's parent directory for a file matching `mmproj*.gguf`. If found, the vision encoder is loaded and image inputs are enabled. **The mmproj file MUST be in the same directory as the model GGUF.**

```
/sdcard/models/Qwen3.5-2B-gguf/
├── Qwen3.5-2B-Q4_K_M.gguf    ← modelPath points here
└── mmproj-F16.gguf             ← auto-discovered, enables vision
```

If the mmproj file is missing or in a different directory, the model loads as text-only and silently ignores image inputs.

**MNN:** The model directory should contain `visual.mnn` and `visual.mnn.weight`. MNN discovers them via `config.json` references.

```
/sdcard/models/Qwen3.5-2B-MNN/
├── config.json                 ← modelPath points here
├── llm.mnn / llm.mnn.weight
└── visual.mnn / visual.mnn.weight  ← enables vision
```

### Send Requests

Once the model is loaded, the OpenAI-compatible API is available at `http://127.0.0.1:<port>`.

**Endpoints:**
- `GET /health` — returns `{"status":"ok"}`
- `GET /v1/models` — lists loaded models
- `POST /v1/chat/completions` — chat inference (streaming and non-streaming)

**Sampling parameters** (all optional, per-request):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | backend default | Sampling temperature (0 = greedy, higher = more random) |
| `top_p` | float | backend default | Nucleus sampling threshold |
| `top_k` | int | backend default | Top-k sampling |
| `repetition_penalty` | float | 1.0 | Repetition penalty (1.0 = disabled) |
| `frequency_penalty` | float | 0.0 | Frequency penalty |
| `presence_penalty` | float | 0.0 | Presence penalty |

If omitted, the backend uses its own defaults (llama.cpp: temp=0.8, top_p=0.95, top_k=40; MNN: model config defaults).

```kotlin
// Text request
val json = """
{
  "model": "any",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
""".trimIndent()

// Multimodal request (image as base64 data URI)
val imageB64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
val json = """
{
  "model": "any",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$imageB64"}}
    ]
  }],
  "stream": true,
  "max_tokens": 200
}
""".trimIndent()

val request = Request.Builder()
    .url("http://127.0.0.1:9099/v1/chat/completions")
    .post(json.toRequestBody("application/json".toMediaType()))
    .build()
```

### Cancel / Stop Generation

Two mechanisms to interrupt an in-progress generation:

**Graceful stop** — Send `POST /v1/cancel` while a streaming request is active. The backend stops generating immediately, the current streaming response finishes normally (finish chunk + usage + `[DONE]`), and the KV cache is preserved. The next request can reuse the cached prefix, skipping re-prefill of prompt + partial response tokens.

```bash
# While a streaming request is in progress:
curl -s -X POST http://127.0.0.1:9099/v1/cancel
# Returns: {"status":"ok"}
```

**Hard cancel (client disconnect)** — If the client closes the HTTP connection (e.g. user navigates away, process killed), the server detects the broken pipe, cancels generation, and cleans up safely. The KV cache is invalidated and the next request will do a full prefill.

| Scenario | Streaming response | KV cache | Next request |
|----------|-------------------|----------|-------------|
| Graceful stop (`/v1/cancel`) | Finishes normally (finish + usage + `[DONE]`) | Preserved | Prefix reuse (`cached_tokens > 0`) |
| Hard cancel (disconnect) | Interrupted | Cleared | Full prefill (`cached_tokens = 0`) |

### Streaming Usage and Performance Metrics

The final SSE chunk of a streaming response contains detailed `usage` and `performance` fields:

```json
{
  "usage": {
    "prompt_tokens": 460,
    "completion_tokens": 18,
    "total_tokens": 478,
    "completion_tokens_details": {
      "reasoning_tokens": 120,
      "text_tokens": 29
    },
    "prompt_tokens_details": {
      "image_tokens": 253,
      "text_tokens": 207,
      "cached_tokens": 438,
      "cache_creation_input_tokens": 22,
      "cache_type": "ephemeral"
    },
    "performance": {
      "prefill_time_ms": 230.6,
      "prefill_tokens_per_second": 1995.1,
      "decode_time_ms": 1021.8,
      "decode_tokens_per_second": 17.6,
      "total_time_ms": 1252.4,
      "time_to_first_token_ms": 230.6
    }
  }
}
```

| Field | Presence | Description |
|-------|----------|-------------|
| `completion_tokens_details` | Only when `reasoning_tokens > 0` | Thinking mode token breakdown |
| `prompt_tokens_details.image_tokens` | Only when > 0 | Tokens consumed by image inputs |
| `cached_tokens` | Always | KV cache prefix reused from previous request (0 on first request) |
| `cache_creation_input_tokens` | Always | Tokens actually prefilled this request (= prompt - cached) |
| `performance` | Always | Timing and throughput metrics |

### Utility Methods

```kotlin
OmniInferServer.isReady()         // true if model loaded and server running
OmniInferServer.getPort()         // current server port (default 9099)
OmniInferServer.getLoadedModels() // list of loaded model filenames
OmniInferServer.getDiagnostics()  // last inference metrics (tokens, timing, etc.)
```

### Cleanup

```kotlin
// Unload model only (server stays alive for loading another model)
OmniInferServer.unloadModel()

// Full shutdown (unload + stop HTTP server)
OmniInferServer.stop()
```

## Updating the Submodule

When OmniInfer releases updates:

```bash
# Step 1: Pull the latest OmniInfer commit
cd third_party/omniinfer
git fetch origin
git checkout origin/main  # or a specific tag

# Step 2: Update the framework submodules you use.
# This is required — git pull/checkout does NOT update submodules automatically.
# Without this step, framework/llama.cpp and framework/mnn stay at their old versions
# even though OmniInfer now points to newer ones.
git submodule update --init framework/llama.cpp
git submodule update --init framework/mnn

# Step 3: Go back to your app root and commit the submodule pointer
cd ../..
git add third_party/omniinfer
git commit -m "Update OmniInfer to latest"
```

**Common mistake:** Running only `git pull` or `git checkout` inside `third_party/omniinfer` without step 2. This updates OmniInfer's own code but leaves the native backends (llama.cpp, MNN) at their old versions, which can cause build failures or missing features.

**Never use `--recursive`** when updating — it would pull all framework submodules including ones you don't need.

## Directory Structure Reference

After setup, your repository should look like:

```
your-app/
├── app/
│   ├── build.gradle.kts        # depends on :omniinfer-server
│   └── src/
├── settings.gradle.kts          # includes :omniinfer-server
└── third_party/
    └── omniinfer/               # Git submodule
        ├── android/
        │   └── omniinfer-server/  # ← the Gradle module
        └── framework/
            ├── llama.cpp/         # ← init this submodule
            └── mnn/               # ← init this submodule (if using MNN)
```

## Supported Models

| Backend | Model Format | Text-only example | Multimodal example |
|---------|-------------|-------------------|-------------------|
| llama.cpp | `.gguf` + optional `mmproj*.gguf` | `Qwen3.5-2B-Q4_K_M.gguf` | Same `.gguf` + `mmproj-F16.gguf` in same dir |
| MNN | Directory with `config.json` | `Qwen3.5-2B-MNN/config.json` | Same dir + `visual.mnn` / `visual.mnn.weight` |
| ExecuTorch QNN | `.pte` + `tokenizer.json` | `hybrid_llama_qnn.pte` | Not yet supported |

**Important for llama.cpp multimodal:** The mmproj and model GGUF must be in the **same directory**. The backend scans the directory automatically. If you place the GGUF in one location and the mmproj elsewhere, vision will silently not work and the model will respond with "I cannot view images."

See `docs/API.md` for the full HTTP API reference.

## Native Library Output

The module produces a single shared library: `libomniinfer-jni.so` (arm64-v8a). It statically links llama.cpp, MNN, and all other native dependencies, so there are no additional `.so` files to manage (unless the ExecuTorch QNN backend is enabled, which bundles separate QNN runtime libraries in `jniLibs/`).

## ExecuTorch QNN Backend (NPU Acceleration)

The ExecuTorch QNN backend runs LLM inference on the Qualcomm Hexagon NPU, delivering significantly higher throughput than CPU-only backends. It requires Snapdragon 8 Gen 1 or newer.

### How It Works

Unlike llama.cpp and MNN which run inside the JNI process, the ET QNN backend spawns a **subprocess** that communicates via stdin/stdout. This is required because Android's linker namespace restrictions prevent QNN's FastRPC from initializing within a JNI-loaded process. The subprocess uses Qualcomm's Unsigned Protection Domain to access the NPU without root.

```
App process (JNI) → fork+exec → libetqnn_runner.so (subprocess)
                  ← stdin/stdout JSON protocol →
                                    ↓
                              QNN SDK → FastRPC HAL → Hexagon NPU
```

### Prerequisites

The ET QNN backend uses **pre-built binaries only** — no compilation is needed from the integrator. You need:

1. **Pre-built binary package** — download from OmniInfer releases
2. **A `.pte` model file** — exported via ExecuTorch's QNN export pipeline
3. **A Snapdragon device** — 8 Gen 1 (SM8450) or newer

### Step 1: Download the Pre-built Package

Download the ET QNN binary package from [OmniInfer Releases](https://github.com/omnimind-ai/OmniInfer/releases). Each package is built for a specific QNN SDK version and contains all required binaries.

The package contains these files, all go into `omniinfer-server/src/main/jniLibs/arm64-v8a/`:

| File | Size | Description |
|------|------|-------------|
| `libetqnn_runner.so` | ~91 MB | Subprocess executable (ET runner) |
| `libqnn_executorch_backend.so` | ~0.6 MB | ET QNN delegate (auto-registration) |
| `libQnnHtp.so` | ~2.5 MB | QNN HTP runtime |
| `libQnnHtpPrepare.so` | ~66-82 MB | QNN HTP ops library |
| `libQnnSystem.so` | ~2.5 MB | QNN system library |
| `libQnnHtpV75Skel.so` | ~9-11 MB | Hexagon DSP skel (select per chip, see below) |
| `libQnnHtpV75Stub.so` | ~0.7 MB | FastRPC stub (select per chip) |

**Skel/Stub file selection by chip:**

| SoC | Chip | Hexagon Version | Skel File | Stub File |
|-----|------|----------------|-----------|-----------|
| SM8650 | 8 Gen 3 | V75 | `libQnnHtpV75Skel.so` | `libQnnHtpV75Stub.so` |
| SM8750 | 8 Elite | V79 | `libQnnHtpV79Skel.so` | `libQnnHtpV79Stub.so` |
| SM8850 | 8 Elite Gen 2 | V81 | `libQnnHtpV81Skel.so` | `libQnnHtpV81Stub.so` |

The pre-built package includes all three skel/stub pairs. Bundle them all — QNN runtime auto-selects the matching version at runtime. Total overhead is ~24 MB for full chip coverage.

### Step 2: Enable in Gradle

```properties
# gradle.properties
omniinfer.backend.executorch_qnn=true
```

No CMake arguments needed — the ET QNN backend does not compile native code. It only needs the pre-built files in `jniLibs/`.

### Step 3: Obtain a Model

The `.pte` model must be exported using ExecuTorch's QNN export pipeline with a **matching QNN SDK version** (same version as the pre-built package).

**Option A: Download pre-exported models** (recommended)

Check [OmniInfer Releases](https://github.com/omnimind-ai/OmniInfer/releases) for pre-exported `.pte` models. Each model listing specifies the required QNN SDK version and supported chips.

**Option B: Export your own model**

Requires a Linux server with the matching QNN SDK installed. See the [ExecuTorch documentation](https://pytorch.org/executorch/stable/llm/getting-started.html) for the export pipeline.

### Step 4: Load and Run

```kotlin
// Place tokenizer.json in the same directory as the .pte file
val success = OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-0.6B/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf("decoder_model_version" to "qwen3")
)
// Once loaded, use the same HTTP API as other backends
```

The `extraConfig` parameter accepts:

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `decoder_model_version` | No | `qwen3` | Chat template to use: `qwen3`, `qwen2_5`, `llama3`, `gemma3` |
| `tokenizer_path` | No | auto-discovered | Path to `tokenizer.json` (auto-discovered from model directory) |

### Performance Reference

Tested on Snapdragon 8 Gen 3 (SM8650):

| Model | Decode (tok/s) | TTFT | Load Time | RAM |
|-------|---------------|------|-----------|-----|
| Qwen3-0.6B (QNN 2.37) | 20.96 | 173ms | 1.0s | 708 MiB |
| Qwen3-1.7B (QNN 2.44) | 22.85 | 67ms | 1.4s | 1715 MiB |

### Limitations

- **Text-only** — multimodal (vision) and tool calling are not yet supported on the ET QNN backend
- **Single-turn only** — KV cache reuse across turns is not yet implemented in the subprocess protocol
- **Qualcomm only** — requires Snapdragon SoC with Hexagon NPU
- **No sampling control** — temperature and other sampling parameters are not yet passed to the subprocess runner

## Troubleshooting

**Native library symbol conflicts:** If your app already includes native libraries that use ggml, llama.cpp, or MNN, you may encounter duplicate symbol errors at link time. The `omniinfer-server` module statically links these libraries into `libomniinfer-jni.so`. To resolve conflicts, either exclude your app's copy of the conflicting library or disable the overlapping OmniInfer backend via CMake arguments.
