# Android Integration Guide

This guide is the handoff document for integrating OmniInfer into a third-party Android app. If you follow the steps below, your app will embed the OmniInfer server in-process, load one local model, and call an OpenAI-compatible local HTTP endpoint at `127.0.0.1`.

## Overview

The `android/omniinfer-server` module is a standalone Android library that provides:
- On-device LLM/VLM inference via llama.cpp, MNN, LiteRT-LM, and ExecuTorch QNN (NPU) backends
- An OpenAI-compatible HTTP API (Ktor server) running locally
- A stable Kotlin facade: `OmniInferServer`
- JNI bridge to native C++ inference engines, plus a Kotlin LiteRT-LM wrapper

Your app includes it as a Gradle module. Native backends are compiled or packaged by the module:

| Backend | Model format | Build/runtime source | Typical use |
|---|---|---|---|
| `llama.cpp` | `.gguf` + optional `mmproj*.gguf` | CMake builds `framework/llama.cpp` | GGUF text/VLM models |
| `mnn` | MNN directory with `config.json` | CMake builds `framework/mnn` | MNN text/VLM models |
| `litert` / `litert-lm` | `.litertlm` | Official AAR `com.google.ai.edge.litertlm:litertlm-android` | Google AI Edge LiteRT-LM models |
| `executorch-qnn` | `.pte` + tokenizer | Prebuilt QNN runner binaries | Qualcomm NPU text models |

The public integration flow is the same for all backends:

```kotlin
OmniInferServer.init(applicationContext)
val ok = OmniInferServer.loadModel(
    modelPath = "/absolute/device/path/to/model",
    backend = "litert", // or "llama.cpp", "mnn", "executorch-qnn"
    nThreads = 4,
    nCtx = 8192,
)
// POST http://127.0.0.1:9099/v1/chat/completions
```

## Compatibility Requirements

Use these versions unless you have a reason to change them:

| Item | Required / tested |
|---|---|
| minSdk | 26+ |
| ABI | `arm64-v8a` |
| JDK | 17 for normal Android builds |
| Android Gradle Plugin | 8.7.3 tested |
| Kotlin Gradle plugin | **2.3.0 or newer** if using current OmniInfer LiteRT-LM integration |
| Ktor | 3.1.3 tested |
| Device model path | Absolute path readable by your app process, commonly `/data/local/tmp/...` for testing or app external files for production |

LiteRT-LM `0.10.2` is compiled with Kotlin metadata `2.3.0`. If your host project uses Kotlin `2.0.x`, `:omniinfer-server:compileKotlin` fails with an incompatible metadata error. Upgrade the host Kotlin plugin to `2.3.0+`.

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
| All other `framework/*` | ~2 GB+ | Desktop/Linux platforms only - **not needed for Android** |

## Step 2: Configure Gradle

### `settings.gradle.kts`

Include the OmniInfer server module:

```kotlin
include(":omniinfer-server")
project(":omniinfer-server").projectDir =
    file("third_party/omniinfer/android/omniinfer-server")
```

### `app/build.gradle.kts`

Add the module dependency and required Ktor libraries. The host app also needs Java/Kotlin 17 settings when using Kotlin 2.3.0:

```kotlin
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

val ktorVersion = "3.1.3"  // tested version - other 3.x versions may work

dependencies {
    implementation(project(":omniinfer-server"))

    // Ktor (HTTP server) - omniinfer-server declares these as compileOnly,
    // so the app must provide them. The versions below are tested and recommended.
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-cio:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Only needed if you use the OkHttp client snippets in this guide.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
```

> **Why compileOnly?** This avoids transitive dependency conflicts when your app already uses Ktor or kotlinx-serialization at a different version. You control the exact versions in your app's dependency block.

The LiteRT-LM AAR is an implementation dependency of `:omniinfer-server`; the host app does not need to add `com.google.ai.edge.litertlm:litertlm-android` directly.

### Root `build.gradle.kts`

If your app uses Gradle plugin versions in the root build file, use Kotlin `2.3.0+`:

```kotlin
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("com.android.library") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.3.0" apply false
}
```

### Backend Selection

The `omniinfer-server` module enables llama.cpp and MNN by default in its Gradle configuration. LiteRT-LM is a Kotlin/AAR backend and does not need CMake. ExecuTorch QNN is disabled unless `omniinfer.backend.executorch_qnn=true`.

If you vendor/fork the module and want to tune native backend size, edit the module's CMake arguments:

```kotlin
android {
    defaultConfig {
        externalNativeBuild {
            cmake {
                // llama.cpp + MNN (default native backends):
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

For normal third-party integration, you usually do not need to add these arguments in the host app. The library module owns its native build configuration.

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

## Step 3.5: Manifest Requirements

The library manifest contributes the foreground service declarations automatically:

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />

<service
    android:name="com.omniinfer.server.OmniInferService"
    android:exported="false"
    android:foregroundServiceType="specialUse" />
```

The module also sets `android:extractNativeLibs="true"` in its manifest because ExecuTorch QNN needs to `fork+exec` a runner `.so` from disk. If your app manifest explicitly sets `android:extractNativeLibs="false"`, it can override the library and break QNN. Remove your override or set it to `true`.

For GPU/OpenCL backends, the library declares optional native libraries:

```xml
<uses-native-library android:name="libOpenCL.so" android:required="false" />
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
```

You normally do not need to copy these declarations into the host manifest unless your manifest merge rules remove library entries.

## Step 4: Use the API

### Lifecycle

```
init() -> loadModel() -> [requests...] -> unloadModel() / stop()
```

- `init()` - call once at app startup. Only passes application context.
- `loadModel()` - loads model weights into memory, starts a foreground service running the Ktor HTTP server. Blocks until the model is ready.
- `unloadModel()` - frees model memory but keeps the HTTP server alive (for loading another model).
- `stop()` - unloads model AND stops the HTTP server.

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
    modelPath = "/path/to/model",   // required - see "Model Paths" below
    backend   = "llama.cpp",        // "llama.cpp", "mnn", "litert", or "executorch-qnn"
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
| `litert`, `litert-lm`, `litertlm` | The `.litertlm` model file | `/data/.../gemma-4-E2B-it.litertlm` |
| `executorch-qnn` | The `.pte` model file | `/data/.../Qwen3-0.6B/hybrid_llama_qnn.pte` |

### Backend-Specific Load Examples

```kotlin
// llama.cpp / GGUF
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-e2b.gguf",
    backend = "llama.cpp",
    nThreads = 6,
    nCtx = 8192,
)

// MNN
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3.5-2B-MNN/config.json",
    backend = "mnn",
    nThreads = 6,
    nCtx = 8192,
)

// LiteRT-LM CPU
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nThreads = 4,
    nCtx = 8192,
    extraConfig = mapOf("backend_type" to "cpu"),
)

// LiteRT-LM GPU
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 8192,
    extraConfig = mapOf("backend_type" to "gpu"),
)

// ExecuTorch QNN
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-1.7B/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf("decoder_model_version" to "qwen3"),
)
```

### LiteRT-LM Notes

LiteRT-LM uses the official Google AI Edge LiteRT-LM AAR. The backend lives in Kotlin, but the public OmniInfer API is identical to the native backends.

Important details:

| Setting | Effect |
|---|---|
| `backend = "litert"` / `"litert-lm"` / `"litertlm"` | Selects LiteRT-LM |
| `extraConfig["backend_type"] = "cpu"` | Uses `Backend.CPU(numOfThreads = nThreads)` |
| `extraConfig["backend_type"] = "gpu"` | Uses `Backend.GPU()` |
| `extraConfig["backend_type"] = "npu"` | Uses `Backend.NPU(nativeLibraryDir)` |
| `nCtx` | Passed to `EngineConfig.maxNumTokens`; set this explicitly for long context |
| `max_tokens` per request | Used as an app-side generation limit; OmniInfer cancels LiteRT-LM once the response budget is reached |

Some `.litertlm` files do not store a max-context metadata field. In that case LiteRT-LM defaults can be smaller than the model card advertises. Always pass an explicit `nCtx` when loading long-context models.

## Production Integration Checklist

Before handing the integration to app feature code, verify this checklist:

| Area | Check |
|---|---|
| Gradle | Host app uses Kotlin Gradle plugin `2.3.0+`, AGP 8.x, and JDK 17 |
| Repositories | Host build can resolve Maven Central and Google's Maven repository |
| Dependencies | Host app provides Ktor server dependencies because `omniinfer-server` declares them as `compileOnly` |
| ABI | App packaging includes `arm64-v8a`; OmniInfer Android does not build x86/armeabi variants |
| Model path | `modelPath` is an absolute path readable by the app process |
| Lifecycle | Call `OmniInferServer.init(applicationContext)` once before loading any model |
| Threading | Call `loadModel()` from a background thread; model load can block for seconds |
| Network | Add `network_security_config.xml` so app code can call `http://127.0.0.1:<port>` |
| Foreground service | Keep the library manifest entries during manifest merge; Android requires the notification while the server runs |
| Shutdown | Call `unloadModel()` to switch models, or `stop()` when the feature/session ends |
| LiteRT long context | Pass explicit `nCtx`; do not rely on `.litertlm` metadata defaults |
| ExecuTorch QNN | Keep `android:extractNativeLibs="true"` if QNN is enabled |

### Multimodal (Vision) Models

llama.cpp and MNN auto-detect multimodal support - **no extra API call needed**.

**llama.cpp:** The backend scans the model file's parent directory for a file matching `mmproj*.gguf`. If found, the vision encoder is loaded and image inputs are enabled. **The mmproj file MUST be in the same directory as the model GGUF.**

```
/sdcard/models/Qwen3.5-2B-gguf/
- Qwen3.5-2B-Q4_K_M.gguf    # modelPath points here
- mmproj-F16.gguf             # auto-discovered, enables vision
```

If the mmproj file is missing or in a different directory, the model loads as text-only and silently ignores image inputs.

**MNN:** The model directory should contain `visual.mnn` and `visual.mnn.weight`. MNN discovers them via `config.json` references.

```
/sdcard/models/Qwen3.5-2B-MNN/
- config.json                 # modelPath points here
- llm.mnn / llm.mnn.weight
- visual.mnn / visual.mnn.weight  # enables vision
```

### Send Requests

Once the model is loaded, the OpenAI-compatible API is available at `http://127.0.0.1:<port>`.

**Endpoints:**
- `GET /health` - returns `{"status":"ok"}`
- `GET /v1/models` - lists loaded models
- `POST /v1/chat/completions` - chat inference (streaming and non-streaming)

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

### Minimal OkHttp Client

Use one `OkHttpClient` and keep read timeout comfortably longer than your largest expected generation. Mobile model loading and long prefill can take seconds.

```kotlin
import com.omniinfer.server.OmniInferServer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

private val client = OkHttpClient.Builder()
    .connectTimeout(10, TimeUnit.SECONDS)
    .readTimeout(300, TimeUnit.SECONDS)
    .build()

suspend fun chatOnce(prompt: String): String = withContext(Dispatchers.IO) {
    val body = """
    {
      "model": "local",
      "reasoning_effort": "none",
      "messages": [{"role": "user", "content": ${JSONObject.quote(prompt)}}],
      "stream": false,
      "max_tokens": 128
    }
    """.trimIndent()

    val request = Request.Builder()
        .url("http://127.0.0.1:${OmniInferServer.getPort()}/v1/chat/completions")
        .post(body.toRequestBody("application/json".toMediaType()))
        .build()

    client.newCall(request).execute().use { response ->
        check(response.isSuccessful) { response.body?.string() ?: response.message }
        val json = JSONObject(response.body!!.string())
        json.getJSONArray("choices")
            .getJSONObject(0)
            .getJSONObject("message")
            .getString("content")
    }
}
```

For tests over `adb forward`, prefer `curl --data-binary @request.json` instead of shell-specific HTTP wrappers that may send an empty request body under timeout/cancellation:

```bash
adb forward tcp:9099 tcp:9099
curl -sS -H "Content-Type: application/json" \
  --data-binary @request.json \
  http://127.0.0.1:9099/v1/chat/completions
```

### Cancel / Stop Generation

Two mechanisms to interrupt an in-progress generation:

**Graceful stop** - Send `POST /v1/cancel` while a streaming request is active. The backend stops generating immediately, the current streaming response finishes normally (finish chunk + usage + `[DONE]`), and the KV cache is preserved. The next request can reuse the cached prefix, skipping re-prefill of prompt + partial response tokens.

```bash
# While a streaming request is in progress:
curl -s -X POST http://127.0.0.1:9099/v1/cancel
# Returns: {"status":"ok"}
```

**Hard cancel (client disconnect)** - If the client closes the HTTP connection (e.g. user navigates away, process killed), the server detects the broken pipe, cancels generation, and cleans up safely. The KV cache is invalidated and the next request will do a full prefill.

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
# This is required - git pull/checkout does NOT update submodules automatically.
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

**Never use `--recursive`** when updating - it would pull all framework submodules including ones you don't need.

## Directory Structure Reference

After setup, your repository should look like:

```
your-app/
  app/
    build.gradle.kts        # depends on :omniinfer-server
    src/
  settings.gradle.kts       # includes :omniinfer-server
  third_party/
    omniinfer/              # Git submodule
      android/
        omniinfer-server/   # the Gradle module
      framework/
        llama.cpp/          # init this submodule
        mnn/                # init this submodule if using MNN
```

## Supported Models

| Backend | Model Format | Text-only example | Multimodal example |
|---------|-------------|-------------------|-------------------|
| llama.cpp | `.gguf` + optional `mmproj*.gguf` | `Qwen3.5-2B-Q4_K_M.gguf` | Same `.gguf` + `mmproj-F16.gguf` in same dir |
| MNN | Directory with `config.json` | `Qwen3.5-2B-MNN/config.json` | Same dir + `visual.mnn` / `visual.mnn.weight` |
| LiteRT-LM | `.litertlm` | `gemma-4-E2B-it.litertlm` | Model-dependent; OmniInfer text path is validated |
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
App process (JNI)
  -> fork+exec libetqnn_runner.so subprocess
  -> stdin/stdout JSON protocol
  -> QNN SDK -> FastRPC HAL -> Hexagon NPU
```

### Prerequisites

The ET QNN backend uses **pre-built binaries only** - no compilation is needed from the integrator. You need:

1. **Pre-built binary package** - download from OmniInfer releases
2. **A `.pte` model file** - exported via ExecuTorch's QNN export pipeline
3. **A Snapdragon device** - 8 Gen 1 (SM8450) or newer

### Step 1: Enable in Gradle

```properties
# gradle.properties
omniinfer.backend.executorch_qnn=true
```

That's it. The first build will **automatically download** all required QNN pre-built binaries (~213 MB) into the omniinfer-server module's `jniLibs/`. Subsequent builds skip the download if files are already present.

No manual downloads, no CMake arguments needed.

<details>
<summary>What gets downloaded (click to expand)</summary>

All files are downloaded from `https://omnimind-model.oss-cn-beijing.aliyuncs.com/omniinfer-android/arm64-v8a/`:

**Universal (all chips):**
- `libetqnn_runner.so` (87 MB) - subprocess runner
- `libqnn_executorch_backend.so` (0.6 MB) - ET QNN delegate
- `libQnnHtp.so` (2.7 MB), `libQnnHtpPrepare.so` (82 MB), `libQnnSystem.so` (2.9 MB), `libQnnHtpNetRunExtensions.so` (0.9 MB) - QNN runtime

**Chip-specific skel/stub (all bundled for broad device support):**
- V75: SM8650 (8 Gen 3)
- V79: SM8750 (8 Elite)
- V81: SM8850 (8 Elite Gen 5)

</details>

### Step 2: Download a Model

Download pre-exported `.pte` models from ModelScope. Each model is exported for a specific SoC - pick the one matching your target device.

**Model repository:** [BiReRa/omniinfer-01001](https://modelscope.cn/models/BiReRa/omniinfer-01001)

**Available models (QNN SDK 2.44, Qwen3 family):**

Two variants are available: **baseline** (short context, max 2048 tokens) and **sink32k** (long context, up to 32K tokens via attention sink).

**Baseline models (seq2048):**

| SoC | Model | Size | Download |
|-----|-------|------|----------|
| SM8650 (8 Gen 3) | Qwen3-0.6B | 680 MB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-0_6b/hybrid_llama_qnn.pte) |
| SM8650 (8 Gen 3) | Qwen3-1.7B | 1.7 GB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-1_7b/hybrid_llama_qnn.pte) |
| SM8650 (8 Gen 3) | Qwen3-4B | 3.1 GB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-4b/hybrid_llama_qnn.pte) |
| SM8750 (8 Elite) | Qwen3-0.6B | 679 MB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8750_qwen3-0_6b/hybrid_llama_qnn.pte) |
| SM8750 (8 Elite) | Qwen3-1.7B | 1.7 GB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8750_qwen3-1_7b/hybrid_llama_qnn.pte) |
| SM8750 (8 Elite) | Qwen3-4B | 3.1 GB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8750_qwen3-4b/hybrid_llama_qnn.pte) |
| SM8850 (8 Elite Gen 5) | Qwen3-0.6B | 682 MB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8850_qwen3-0_6b/hybrid_llama_qnn.pte) |
| SM8850 (8 Elite Gen 5) | Qwen3-1.7B | 1.7 GB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8850_qwen3-1_7b/hybrid_llama_qnn.pte) |
| SM8850 (8 Elite Gen 5) | Qwen3-4B | 3.1 GB | [.pte](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8850_qwen3-4b/hybrid_llama_qnn.pte) |

**Long-context models with Attention Sink (sink32k):**

These models support generation up to 32K tokens total (input + output). Decode speed stays constant regardless of sequence length - no degradation as context grows. Each directory contains two `.pte` files: the main model and an attention sink evictor.

| SoC | Model | .pte + evictor | Download |
|-----|-------|---------------|----------|
| SM8650 (8 Gen 3) | Qwen3-0.6B | 680 MB + 4 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-0_6b_sink32k/hybrid_llama_qnn.pte) |
| SM8650 (8 Gen 3) | Qwen3-1.7B | 1.7 GB + 2 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-1_7b_sink32k/hybrid_llama_qnn.pte) |
| SM8650 (8 Gen 3) | Qwen3-4B | 3.1 GB + 3 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-4b_sink32k/hybrid_llama_qnn.pte) |
| SM8750 (8 Elite) | Qwen3-0.6B | 679 MB + 4 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8750_qwen3-0_6b_sink32k/hybrid_llama_qnn.pte) |
| SM8750 (8 Elite) | Qwen3-1.7B | 1.7 GB + 2 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8750_qwen3-1_7b_sink32k/hybrid_llama_qnn.pte) |
| SM8750 (8 Elite) | Qwen3-4B | 3.1 GB + 3 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8750_qwen3-4b_sink32k/hybrid_llama_qnn.pte) |
| SM8850 (8 Elite Gen 5) | Qwen3-0.6B | 682 MB + 5 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8850_qwen3-0_6b_sink32k/hybrid_llama_qnn.pte) |
| SM8850 (8 Elite Gen 5) | Qwen3-1.7B | 1.7 GB + 3 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8850_qwen3-1_7b_sink32k/hybrid_llama_qnn.pte) |
| SM8850 (8 Elite Gen 5) | Qwen3-4B | 3.1 GB + 3 MB | [sink32k](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8850_qwen3-4b_sink32k/hybrid_llama_qnn.pte) |

**Tokenizer** (same for all models): [tokenizer.json](https://modelscope.cn/models/BiReRa/omniinfer-01001/resolve/master/SM8650_qwen3-0_6b/tokenizer.json) (11 MB)

Or download via CLI:
```bash
# Baseline model
modelscope download --model BiReRa/omniinfer-01001 --include "SM8650_qwen3-1_7b/*" --local_dir ./models

# Long-context (sink32k) model - downloads both .pte files + tokenizer
modelscope download --model BiReRa/omniinfer-01001 --include "SM8650_qwen3-1_7b_sink32k/*" --local_dir ./models
```

Place all files in the same directory on the device. For sink32k models, the directory must contain both `hybrid_llama_qnn.pte` and `attention_sink_evictor.pte` - the evictor is auto-discovered.

**Export your own model** (advanced): Requires a Linux server with the matching QNN SDK (2.44.0) installed. See the [ExecuTorch documentation](https://pytorch.org/executorch/stable/llm/getting-started.html) for the export pipeline.

### Step 3: Load and Run

```kotlin
// Baseline model (short context)
val success = OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-0.6B/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf("decoder_model_version" to "qwen3")
)

// Long-context (sink32k) model - just point to the same directory structure.
// The evictor .pte is auto-discovered and attention sink is enabled automatically.
val success = OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-1.7B-sink32k/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf(
        "decoder_model_version" to "qwen3",
        "seq_len" to "32768"  // max total tokens (input + output)
    )
)
// Once loaded, use the same HTTP API as other backends
```

The `extraConfig` parameter accepts:

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `decoder_model_version` | No | `qwen3` | Chat template to use: `qwen3`, `qwen2_5`, `llama3`, `gemma3` |
| `tokenizer_path` | No | auto-discovered | Path to `tokenizer.json` (auto-discovered from model directory) |
| `seq_len` | No | `32768` if sink model, else `2048` | Max total tokens (input + output). For sink32k models: use `32768` (1.7B/4B) or `4096` (0.6B - see note below) |
| `attention_sink_evictor_path` | No | auto-discovered | Path to `attention_sink_evictor.pte` (auto-discovered from model directory) |

> **Qwen3-0.6B sink32k note:** The 0.6B model is too small to reliably generate EOS with `seq_len=32768`, causing infinite generation. Set `seq_len` to `4096` for 0.6B sink models. The 1.7B and 4B models work correctly with `seq_len=32768`.

### Performance Reference

Tested on Snapdragon 8 Gen 3 (SM8650):

**Baseline models:**

| Model | Decode (tok/s) | TTFT | Load Time | RAM |
|-------|---------------|------|-----------|-----|
| Qwen3-0.6B | 20.96 | 173ms | 1.0s | 708 MiB |
| Qwen3-1.7B | 22.85 | 67ms | 1.4s | 1715 MiB |

**Sink32k models (SM8650, Qwen3-0.6B with seq_len=4096):**

| Input tokens | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------------|----------------|----------------|-------|
| 506 | 439 | 20.0 | Normal output quality |
| 978 | 420 | 19.8 | Normal output quality |
| 1568 | 415 | 20.0 | Normal output quality |
| 2984 | 374 | 19.6 | Degraded output (exceeds ~1916 effective context window) |

Key characteristics:
- **Decode speed is constant** regardless of sequence length - no degradation as context grows
- Effective understanding window: ~1916 tokens (KV cache 2048 - 4 sink tokens - 128 prefill batch)
- Inputs beyond the effective window are evicted; the model retains the first 4 tokens (anchors) + most recent ~1916 tokens
- Long **output** generation works well - tested 1708 tokens at stable 19.8 tok/s

**Sink32k models (SM8750, from export benchmarks):**

| Model | Prefill (tok/s) | Decode (tok/s) | RAM |
|-------|----------------|----------------|-----|
| Qwen3-0.6B | 532 | 49.1 | 714 MiB |
| Qwen3-1.7B | 2018 | 36.9 | 1714 MiB |
| Qwen3-4B | 1118 | 17.8 | 1777 MiB |

### Limitations

- **Text-only** - multimodal (vision) and tool calling are not yet supported on the ET QNN backend
- **Single-turn only** - KV cache reuse across turns is not yet implemented in the subprocess protocol
- **Qualcomm only** - requires Snapdragon SoC with Hexagon NPU
- **No sampling control** - temperature and other sampling parameters are not yet passed to the subprocess runner
- **Sink context window** - attention sink models can generate up to 32K tokens, but the effective understanding window is ~1916 tokens. Content outside this window is evicted.
- **`extractNativeLibs=true` required** - the omniinfer-server manifest sets this via manifest merge. However, if your app's `AndroidManifest.xml` explicitly sets `android:extractNativeLibs="false"`, it will override the library setting. You must either remove the override or set it to `true`. The runner .so must exist as a regular file on disk for fork+exec.

## Troubleshooting

**Kotlin metadata mismatch with LiteRT-LM:** If the build fails with `litertlm-android ... Module was compiled with an incompatible version of Kotlin`, upgrade the host Kotlin Gradle plugin to `2.3.0+`.

**LiteRT-LM model loads but long chat fails near 4k:** Pass `nCtx` explicitly in `loadModel()`. Some `.litertlm` artifacts do not set max-context metadata, so LiteRT-LM may default to `4096`.

**HTTP request times out and logcat says JSON input is empty:** Verify the client is actually sending a body. For command-line testing through `adb forward`, use `curl.exe --data-binary @request.json`.

**`127.0.0.1` request fails with cleartext error:** Add `network_security_config.xml` and reference it from the app manifest as shown above.

**Foreground service fails on Android 14+:** Ensure manifest merge keeps `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_SPECIAL_USE`, and the `OmniInferService` declaration from the library.

**Native library symbol conflicts:** If your app already includes native libraries that use ggml, llama.cpp, or MNN, you may encounter duplicate symbol errors at link time. The `omniinfer-server` module statically links these libraries into `libomniinfer-jni.so`. To resolve conflicts, either exclude your app's copy of the conflicting library or disable the overlapping OmniInfer backend via CMake arguments.
