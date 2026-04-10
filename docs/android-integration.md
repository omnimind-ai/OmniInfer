# Android Integration Guide

This guide explains how to integrate the OmniInfer server library into your Android app as a Git submodule.

## Overview

The `android/omniinfer-server` module is a standalone Android library that provides:
- On-device LLM/VLM inference via llama.cpp and MNN backends
- An OpenAI-compatible HTTP API (Ktor server) running locally
- JNI bridge to native C++ inference engines

Your app includes it as a Gradle module. The native backends (llama.cpp, MNN) are compiled from source via CMake during the Gradle build.

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
val ktorVersion = "3.1.3"

dependencies {
    implementation(project(":omniinfer-server"))

    // Ktor (HTTP server) — omniinfer-server declares these as compileOnly,
    // so the app must provide them.
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-cio:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

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

## Step 3: Use the API

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

If you call `loadModel()` again with a different model/backend, the previous model is automatically unloaded first. If called with the same model, it returns immediately.

**Model Paths:**

| Backend | `modelPath` points to | Example |
|---------|----------------------|---------|
| `llama.cpp` | The `.gguf` model file | `/data/.../Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_K_M.gguf` |
| `mnn` | The `config.json` in the MNN model directory | `/data/.../Qwen3.5-2B-MNN/config.json` |

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

```kotlin
// Text request
val json = """
{
  "model": "any",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "max_tokens": 100
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
# Pull the latest OmniInfer commit
cd third_party/omniinfer
git fetch origin
git checkout origin/main  # or a specific tag

# Update only the framework submodules you use
git submodule update --init framework/llama.cpp
git submodule update --init framework/mnn

# Go back to your app root and commit the submodule pointer
cd ../..
git add third_party/omniinfer
git commit -m "Update OmniInfer to latest"
```

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

**Important for llama.cpp multimodal:** The mmproj and model GGUF must be in the **same directory**. The backend scans the directory automatically. If you place the GGUF in one location and the mmproj elsewhere, vision will silently not work and the model will respond with "I cannot view images."

See `docs/API.md` for the full HTTP API reference.
