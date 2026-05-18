# Android Integration Guide

This is the short path for embedding OmniInfer into a third-party Android app.

1. Add `android/omniinfer-server` as a Gradle module.
2. Call `OmniInferServer.init(applicationContext)`.
3. Load one local model with `OmniInferServer.loadModel(...)`.
4. Send OpenAI-compatible requests to `http://127.0.0.1:<port>`.

For details that are not needed on the first pass:

| Topic | Document |
|---|---|
| Backend options and model paths | [backends.md](./backends.md) |
| HTTP examples, streaming, tools, cancel | [api-examples.md](./api-examples.md) |
| Multimodal model layout and image requests | [multimodal.md](./multimodal.md) |
| Build and device smoke tests | [smoke-tests.md](./smoke-tests.md) |
| Common failures | [troubleshooting.md](./troubleshooting.md) |
| ExecuTorch QNN / NPU | [et-qnn.md](./et-qnn.md) |
| Full HTTP API contract | [API.md](../API.md) |

## Requirements

Use these versions unless your app has a specific compatibility constraint:

| Item | Required / tested |
|---|---|
| minSdk | 26+ |
| ABI | `arm64-v8a` |
| JDK | 17 for normal Android builds |
| Android Gradle Plugin | 8.7.3 tested |
| Kotlin Gradle plugin | 2.3.0+ |
| Ktor | 3.1.3 tested |

Kotlin `2.3.0+` is required because the LiteRT-LM AAR used by OmniInfer is compiled with Kotlin metadata 2.3.0.

## Add OmniInfer

From your app repository root:

```bash
git submodule add https://github.com/omnimind-ai/OmniInfer.git third_party/omniinfer
```

Do not initialize all submodules recursively. The `framework/` directory contains large desktop/server dependencies that Android does not need.

Initialize only the native Android backends you keep enabled:

```bash
git submodule update --init third_party/omniinfer/framework/llama.cpp
git submodule update --init third_party/omniinfer/framework/mnn
```

`framework/llama.cpp` is needed only when `omniinfer.backend.llama_cpp=true`. `framework/mnn` is needed only when `omniinfer.backend.mnn=true`. LiteRT-LM does not require OmniInfer native submodules.

## Configure Gradle

In `settings.gradle.kts`:

```kotlin
include(":omniinfer-server")
project(":omniinfer-server").projectDir =
    file("third_party/omniinfer/android/omniinfer-server")
```

In the root Gradle plugins block, use Kotlin `2.3.0+`:

```kotlin
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("com.android.library") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.3.0" apply false
}
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

    // Only needed if you use the OkHttp examples.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
```

All Android backends default to enabled. Disable unused backends in `gradle.properties` to reduce build work and APK size:

```properties
omniinfer.backend.llama_cpp=false
omniinfer.backend.mnn=false
omniinfer.backend.executorch_qnn=false
omniinfer.backend.litert_lm=true
```

See [backends.md](./backends.md#backend-gradle-switches) for the full switch table.

## Allow Localhost HTTP

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

The library manifest contributes `OmniInferService` and foreground-service declarations. If your manifest merge rules are strict, keep these entries:

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />

<service
    android:name="com.omniinfer.server.OmniInferService"
    android:exported="false"
    android:foregroundServiceType="specialUse" />
```

If you enable ExecuTorch QNN, keep `android:extractNativeLibs="true"` because the QNN runner must exist as a regular file on disk.

## Load A Model

Call `init()` once, then load a model off the UI thread:

```kotlin
import com.omniinfer.server.OmniInferServer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun loadLocalModel(modelPath: String): Boolean = withContext(Dispatchers.IO) {
    OmniInferServer.init(applicationContext)
    OmniInferServer.loadModel(
        modelPath = modelPath,
        backend = "litert", // "llama.cpp", "mnn", "litert", or "executorch-qnn"
        port = 9099,
        nThreads = 4,
        nCtx = 4096,
        extraConfig = mapOf("backend_type" to "gpu"),
    )
}
```

Only one model is loaded at a time. Loading a different model/backend unloads the previous one first.

`modelPath` depends on the backend:

| Backend | `modelPath` points to | Example |
|---|---|---|
| `llama.cpp` | `.gguf` model file | `/sdcard/models/gemma-4-e2b.gguf` |
| `mnn` | MNN `config.json` | `/sdcard/models/Qwen3.5-2B-MNN/config.json` |
| `litert`, `litert-lm`, `litertlm` | `.litertlm` model file | `/sdcard/models/gemma-4-E2B-it.litertlm` |
| `executorch-qnn` | `.pte` model file | `/sdcard/models/Qwen3-1.7B/hybrid_llama_qnn.pte` |

Backend-specific options belong in `extraConfig`. LiteRT-LM GPU, for example:

```kotlin
extraConfig = mapOf("backend_type" to "gpu")
```

LiteRT-LM options such as `backend_type`, `vision_backend`, `max_images`, and `enable_speculative_decoding` are load-time options. Call `OmniInferServer.unloadModel()` before reloading the same model with different values. See [backends.md](./backends.md) for the full backend table.

## Send Requests

After `loadModel()` returns `true`, call the local OpenAI-compatible endpoint:

```kotlin
import com.omniinfer.server.OmniInferServer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody

suspend fun chatOnce(jsonBody: String): String = withContext(Dispatchers.IO) {
    val request = okhttp3.Request.Builder()
        .url("http://127.0.0.1:${OmniInferServer.getPort()}/v1/chat/completions")
        .post(jsonBody.toRequestBody("application/json".toMediaType()))
        .build()

    okhttp3.OkHttpClient().newCall(request).execute().use { response ->
        check(response.isSuccessful) { response.body?.string() ?: response.message }
        response.body!!.string()
    }
}
```

Minimal non-streaming request body:

```json
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "Say READY only."}
  ],
  "stream": false,
  "max_tokens": 16
}
```

For complete Kotlin, curl, streaming, tool-calling, and cancel examples, see [api-examples.md](./api-examples.md).

## Lifecycle

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

## Ship Checklist

| Area | Check |
|---|---|
| Gradle | Kotlin Gradle plugin `2.3.0+`, AGP 8.x, JDK 17 |
| Dependencies | Host app provides Ktor server dependencies |
| ABI | App packages `arm64-v8a` unless other ABIs are intentional |
| Backends | Unused OmniInfer backends are disabled in `gradle.properties` |
| Model path | `modelPath` is absolute and readable by the app process |
| Threading | `loadModel()` runs off the UI thread |
| Network | Cleartext HTTP to `127.0.0.1` is allowed |
| Service | Manifest merge keeps `OmniInferService` and foreground-service permissions |
| Shutdown | App lifecycle calls `unloadModel()` or `stop()` |

If the app builds but fails at runtime, start with [troubleshooting.md](./troubleshooting.md). For adb/curl validation, use [smoke-tests.md](./smoke-tests.md).
