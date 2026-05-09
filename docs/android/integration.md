# Android Integration Guide

This is the short path for embedding OmniInfer into a third-party Android app.
It keeps the app-side contract simple:

1. Include `android/omniinfer-server` as a Gradle module.
2. Call `OmniInferServer.init(applicationContext)`.
3. Load one local model with `OmniInferServer.loadModel(...)`.
4. Send OpenAI-compatible requests to `http://127.0.0.1:<port>`.

For backend internals and advanced topics, see:

| Topic | Document |
|---|---|
| Backend-specific options | [backends.md](./backends.md) |
| Multimodal model layout | [multimodal.md](./multimodal.md) |
| HTTP request and streaming examples | [api-examples.md](./api-examples.md) |
| ExecuTorch QNN / NPU | [et-qnn.md](./et-qnn.md) |
| Full HTTP API contract | [API.md](../API.md) |

## Requirements

Use these versions unless you have a specific compatibility reason to change them:

| Item | Required / tested |
|---|---|
| minSdk | 26+ |
| ABI | `arm64-v8a` |
| JDK | 17 for normal Android builds |
| Android Gradle Plugin | 8.7.3 tested |
| Kotlin Gradle plugin | 2.3.0+ |
| Ktor | 3.1.3 tested |

Kotlin `2.3.0+` matters because the LiteRT-LM AAR used by OmniInfer is compiled with Kotlin metadata 2.3.0. Host apps using Kotlin 2.0.x will fail during `:omniinfer-server:compileKotlin`.

## Step 1: Add OmniInfer

From your app repository root:

```bash
git submodule add https://github.com/omnimind-ai/OmniInfer.git third_party/omniinfer
```

Do not initialize all submodules recursively. The `framework/` directory contains large desktop/server dependencies that Android does not need.

Initialize only the Android native backends you keep enabled:

```bash
git submodule update --init third_party/omniinfer/framework/llama.cpp
git submodule update --init third_party/omniinfer/framework/mnn
```

`framework/llama.cpp` is needed only when `omniinfer.backend.llama_cpp=true`. `framework/mnn` is needed only when `omniinfer.backend.mnn=true`. LiteRT-LM does not require OmniInfer native submodules.

## Step 2: Configure Gradle

In `settings.gradle.kts`:

```kotlin
include(":omniinfer-server")
project(":omniinfer-server").projectDir =
    file("third_party/omniinfer/android/omniinfer-server")
```

In `app/build.gradle.kts`:

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

val ktorVersion = "3.1.3"

dependencies {
    implementation(project(":omniinfer-server"))

    // omniinfer-server declares Ktor as compileOnly, so the app provides it.
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-cio:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Only needed if you use the OkHttp snippets in this guide.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
```

In the root Gradle plugins block, use Kotlin `2.3.0+`:

```kotlin
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("com.android.library") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.3.0" apply false
}
```

Optional backend switches can live in the host app or root `gradle.properties`. All Android backends default to `true`:

```properties
omniinfer.backend.llama_cpp=true
omniinfer.backend.mnn=true
omniinfer.backend.executorch_qnn=true
omniinfer.backend.litert_lm=true
```

Set unused backends to `false` to reduce native build work, dependency downloads, and APK size. For example, a LiteRT-only app can disable the native backends:

```properties
omniinfer.backend.llama_cpp=false
omniinfer.backend.mnn=false
omniinfer.backend.executorch_qnn=false
```

When `omniinfer.backend.litert_lm=false`, `:omniinfer-server` does not add `com.google.ai.edge.litertlm:litertlm-android` and does not compile the LiteRT-LM source set. See [backends.md](./backends.md) for the full switch table.

### Command-Line LiteRT Test App

For a quick source-build smoke test, create a small host app outside the OmniInfer repository and point its Gradle settings at your local checkout:

```kotlin
// settings.gradle.kts
include(":app")
include(":omniinfer-server")
project(":omniinfer-server").projectDir =
    file("/absolute/path/to/OmniInfer/android/omniinfer-server")
```

Use the Gradle snippets above for the app module, then disable unused native backends in `gradle.properties`:

```properties
omniinfer.backend.llama_cpp=false
omniinfer.backend.mnn=false
omniinfer.backend.executorch_qnn=false
omniinfer.backend.litert_lm=true
```

Build from the host app root:

```bash
./gradlew :app:assembleDebug
```

If you do not use a checked-in Gradle wrapper, use a user-local Gradle install with `JAVA_HOME` and `ANDROID_HOME` set. A no-sudo Linux setup can install JDK 21 under `~/.local/jdks/` and Android SDK command line tools under `~/Android/Sdk`; install at least `platform-tools`, `platforms;android-35`, `build-tools;35.0.0`, `ndk;28.2.13676358`, and `cmake;3.22.1`. The LiteRT-only app still configures `:omniinfer-server`'s CMake project, so NDK and SDK CMake/Ninja must be present even when llama.cpp, MNN, and ExecuTorch QNN are disabled.

## Step 3: Allow Localhost HTTP

OmniInfer serves local HTTP on `127.0.0.1`. Android 9+ blocks cleartext HTTP unless you opt in.

Create `app/src/main/res/xml/network_security_config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="false">127.0.0.1</domain>
    </domain-config>
</network-security-config>
```

Reference it from `AndroidManifest.xml`:

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ... >
```

The library manifest contributes the foreground service declarations. Keep them during manifest merge:

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />

<service
    android:name="com.omniinfer.server.OmniInferService"
    android:exported="false"
    android:foregroundServiceType="specialUse" />
```

If you enable ExecuTorch QNN, keep `android:extractNativeLibs="true"` because the QNN runner must exist as a regular file on disk.

## Step 4: Load a Model

Call `init()` once, then load a model on a background thread:

```kotlin
import com.omniinfer.server.OmniInferServer

OmniInferServer.init(applicationContext)

val ok = OmniInferServer.loadModel(
    modelPath = "/absolute/device/path/to/model",
    backend = "litert", // "llama.cpp", "mnn", "litert", or "executorch-qnn"
    port = 9099,
    nThreads = 4,
    nCtx = 8192,
    extraConfig = mapOf("backend_type" to "cpu"),
)
```

Only one model is loaded at a time. Loading a different model/backend unloads the previous one first.

`modelPath` depends on the backend:

| Backend | `modelPath` points to | Example |
|---|---|---|
| `llama.cpp` | `.gguf` model file | `/sdcard/models/gemma-4-e2b.gguf` |
| `mnn` | MNN `config.json` | `/sdcard/models/Qwen3.5-2B-MNN/config.json` |
| `litert`, `litert-lm`, `litertlm` | `.litertlm` model file | `/sdcard/models/gemma-4-E2B-it.litertlm` |
| `executorch-qnn` | `.pte` model file | `/sdcard/models/Qwen3-1.7B/hybrid_llama_qnn.pte` |

Common backend options:

```kotlin
// LiteRT-LM CPU
extraConfig = mapOf("backend_type" to "cpu")

// LiteRT-LM GPU
extraConfig = mapOf("backend_type" to "gpu")

// LiteRT-LM GPU + multimodal
extraConfig = mapOf(
    "backend_type" to "gpu",
    "vision_backend" to "gpu",
    "max_images" to "1",
)

// LiteRT-LM GPU + speculative decoding
extraConfig = mapOf(
    "backend_type" to "gpu",
    "enable_speculative_decoding" to "true",
)

// ExecuTorch QNN
extraConfig = mapOf("decoder_model_version" to "qwen3")
```

For LiteRT-LM, `backend_type`, `vision_backend`, `max_images`, and
`enable_speculative_decoding` are load-time options. They are consumed when
`Engine.initialize()` runs and cannot be changed by adding fields to an
OpenAI-compatible HTTP request. If the same `modelPath` and `backend` are
already loaded, call `OmniInferServer.unloadModel()` before loading again with
different LiteRT-LM options.

See [backends.md](./backends.md) for the full backend table.

## Step 5: Send Requests

After `loadModel()` returns `true`, call the local OpenAI-compatible endpoint:

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

For command-line tests through adb:

```bash
adb forward tcp:9099 tcp:9099
curl -sS -H "Content-Type: application/json" \
  --data-binary @request.json \
  http://127.0.0.1:9099/v1/chat/completions
```

Prefer `--data-binary @request.json`; some shell-specific wrappers can accidentally send an empty body.

## Lifecycle Helpers

```kotlin
OmniInferServer.isReady()
OmniInferServer.getPort()
OmniInferServer.getLoadedModels()
OmniInferServer.getDiagnostics()

OmniInferServer.unloadModel() // unload model, keep server available
OmniInferServer.stop()        // unload model and stop HTTP server
```

Optional notification customization:

```kotlin
OmniInferServer.configureNotification(
    title = "My AI Assistant",
    channelName = "AI Engine",
    smallIcon = R.drawable.ic_my_icon,
    textFormat = { "AI engine active" },
)
```

Call `configureNotification()` after `init()` and before `loadModel()`.

## Production Checklist

Before shipping the integration, check:

| Area | Check |
|---|---|
| Gradle | Kotlin Gradle plugin `2.3.0+`, AGP 8.x, JDK 17 |
| Repositories | Maven Central and Google's Maven repository are available |
| Dependencies | Host app provides Ktor server dependencies |
| ABI | App packages only `arm64-v8a` unless other ABIs are intentional |
| Backend switches | Disable unused OmniInfer backends in `gradle.properties` if APK size matters |
| Model path | `modelPath` is absolute and readable by the app process |
| Threading | `loadModel()` runs off the UI thread |
| Network | `network_security_config.xml` allows `127.0.0.1` cleartext HTTP |
| Service | Manifest merge keeps `OmniInferService` and foreground service permissions |
| Shutdown | Call `unloadModel()` or `stop()` from your feature lifecycle |
| LiteRT long context | Pass explicit `nCtx`; do not rely on `.litertlm` metadata defaults |
| LiteRT multimodal | Load with `extraConfig["vision_backend"]`; verify logcat does not say `visionBackend=none` |
| LiteRT speculative decoding | Load with `extraConfig["enable_speculative_decoding"] = "true"` and a `.litertlm` package that supports MTP/SD |
| QNN | Keep `android:extractNativeLibs="true"` if ExecuTorch QNN is enabled |

## Updating OmniInfer Later

When updating the OmniInfer submodule:

```bash
cd third_party/omniinfer
git fetch origin
git checkout origin/main  # or a specific tag
git submodule update --init framework/llama.cpp  # if llama.cpp is enabled
git submodule update --init framework/mnn        # if MNN is enabled

cd ../..
git add third_party/omniinfer
git commit -m "Update OmniInfer"
```

Do not use `--recursive`; it downloads Android-irrelevant framework submodules.

## Troubleshooting

**Kotlin metadata mismatch with LiteRT-LM:** upgrade the host Kotlin Gradle plugin to `2.3.0+`.

**`127.0.0.1` request fails with cleartext error:** add `network_security_config.xml` and reference it from the app manifest.

**HTTP request times out and logcat says JSON input is empty:** verify the client sends a body; for adb tests use `curl --data-binary @request.json`.

**LiteRT-LM fails around 4k context:** pass `nCtx` explicitly in `loadModel()`; some `.litertlm` files do not store max-context metadata.

**LiteRT-LM image request returns a `vision_backend` error:** unload the model
and load it again with `extraConfig["vision_backend"] = "cpu"` or `"gpu"`.
`vision_backend` is not a per-request HTTP field.

**LiteRT-LM SD flag appears ignored:** unload and reload with
`extraConfig["enable_speculative_decoding"] = "true"`, then check logcat for
`speculativeDecoding=true` or native `enable_speculative_decoding: true`. The
model package itself must support MTP/speculative decoding; no separate draft
model path is passed by the app.

**Foreground service fails on Android 14+:** ensure manifest merge keeps `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_SPECIAL_USE`, and `OmniInferService`.

**Native library symbol conflicts:** if your app already bundles ggml, llama.cpp, or MNN, exclude the duplicate copy or disable the overlapping OmniInfer backend in the library module.

**CMake configure fails with missing Ninja:** install an Android SDK CMake package such as `cmake;3.22.1`; it includes the Ninja binary used by AGP's external native build.
