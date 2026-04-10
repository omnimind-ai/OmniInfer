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

### Initialize and Load a Model

```kotlin
import com.omniinfer.server.OmniInferServer

// Initialize (once, at app startup)
OmniInferServer.init(
    context = applicationContext,
    backend = "mnn",  // or "llama.cpp"
    nThreads = 4,
    nCtx = 2048
)

// Load a model
OmniInferServer.loadModel(
    modelPath = "/data/local/tmp/Qwen3.5-0.8B-MNN/config.json",
    port = 9099  // starts HTTP server on this port
)
```

### Send Requests

Once the model is loaded, send requests to `http://127.0.0.1:9099/v1/chat/completions` using any OpenAI-compatible client library or plain HTTP.

```kotlin
// Example with OkHttp
val json = """
{
  "model": "any",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "max_tokens": 100
}
""".trimIndent()

val request = Request.Builder()
    .url("http://127.0.0.1:9099/v1/chat/completions")
    .post(json.toRequestBody("application/json".toMediaType()))
    .build()
```

### Cleanup

```kotlin
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

| Backend | Model Format | Example |
|---------|-------------|---------|
| llama.cpp | `.gguf` files | `gemma-4-E2B-it-Q4_K_M.gguf` |
| MNN | MNN model directory with `config.json` | `Qwen3.5-0.8B-MNN/config.json` |

Both backends support text-only and multimodal (vision) models. See `docs/API.md` for the full API reference.
