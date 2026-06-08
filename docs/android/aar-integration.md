# Android AAR Integration

This guide is for Android apps that consume OmniInfer as a Maven AAR.

OmniInfer starts an on-device OpenAI-compatible HTTP server inside your app
process. Your app provides a local model file, loads it through the SDK, then
sends requests to `127.0.0.1`.

## Requirements

| Item | Value |
|---|---|
| minSdk | 26+ |
| ABI | `arm64-v8a` |
| compileSdk | 35 recommended |
| Kotlin Gradle plugin | 2.3.0+ |
| AndroidX | enabled |

OmniInfer is distributed with Maven metadata. Use the Maven dependency instead
of copying a flat `.aar` into `libs/`; Maven lets Gradle resolve OmniInfer's
runtime dependencies and consumer ProGuard rules correctly.

## Gradle Setup

In `settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
```

In the root `gradle.properties`:

```properties
android.useAndroidX=true
```

In the app module:

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

    packaging {
        jniLibs {
            useLegacyPackaging = true
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

dependencies {
    implementation("io.github.omnimind-ai:omniinfer:0.2.3")
}
```

`useLegacyPackaging = true` is required because several native runtimes need
real `.so` files on disk.

## Manifest

Add these permissions:

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
```

OmniInfer's AAR contributes its service declaration through manifest merge. The
host app does not need to declare `OmniInferService` manually.

`POST_NOTIFICATIONS` is not required by model inference itself. It is needed on
Android 13+ when the host app targets API 33+ and wants OmniInfer's foreground
service notification to be visible. If the user denies this runtime permission,
the model can still run, but the app should handle notification permission UX
according to its own foreground-service policy.

OmniInfer serves HTTP on localhost. If your app targets Android 9+ and uses the
HTTP endpoint directly, allow cleartext traffic for `127.0.0.1`.

`res/xml/network_security_config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="false">127.0.0.1</domain>
    </domain-config>
</network-security-config>
```

`AndroidManifest.xml`:

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ... >
```

## Model Files

Model weights are not packaged in the AAR. Store downloaded models in a path
your app can read, such as:

```kotlin
val modelDir = File(context.filesDir, "models")
```

OmniInfer includes a model catalog with names, backend settings, download URLs,
file size, and SHA256. Use it to drive your model picker and downloader:

```kotlin
OmniInferServer.init(applicationContext)

val models = OmniInferServer.listCatalogModels()
val selected = models.first { it.id == "qwen35-2b-q4-0-gguf-llamacpp-htp" }
val source = selected.sources.first()
val target = File(modelDir, source.fileName)
```

Download `source.url` into `target`, verify `source.sha256`, then call
`loadModel()`. This AAR package contains llama.cpp CPU, llama.cpp HTP, and
LiteRT-LM GPU. `modelPath` is the absolute path to the model entry file:

| Format | `modelPath` value |
|---|---|
| GGUF / llama.cpp | Path to the `.gguf` file |
| LiteRT-LM | Path to the `.litertlm` file |

Do not pass the containing directory for GGUF or LiteRT-LM models. Do not rely
on `/data/local/tmp` in production apps; normal Android app UIDs often cannot
list or read arbitrary files there.

## Load A Model

The shortest load path is:

```kotlin
val ok = OmniInferServer.loadModel(
    modelPath = target.absolutePath,
)
```

OmniInfer first matches `modelPath` against the bundled catalog file names. If
it finds a match, it applies the catalog's backend, thread, context, and load
defaults automatically. If there is no catalog match, it falls back by file
extension for the backends included in this AAR: `.gguf` -> `llama.cpp/cpu`,
`.litertlm` -> `litert/gpu`.

For explicit selection, use these public backend names:

| Runtime | `backend` selector | Notes |
|---|---|---|
| llama.cpp CPU | `OmniInferBackend.LLAMA_CPP_CPU` / `"llama.cpp/cpu"` | No accelerator options |
| llama.cpp HTP | `OmniInferBackend.LLAMA_CPP_HTP` / `"llama.cpp/htp"` | Adds `HTP0`, offload, and HTP batch defaults |
| LiteRT-LM GPU | `OmniInferBackend.LITERT_GPU` / `"litert/gpu"` | Adds LiteRT GPU backend defaults |

Useful overrides stay small:

```kotlin
val ok = OmniInferServer.loadModel(
    modelPath = target.absolutePath,
    backend = OmniInferBackend.LLAMA_CPP_HTP,
    port = 9099,
    nThreads = 6,
    nCtx = 8192,
)
```

```kotlin
val ok = OmniInferServer.loadModel(
    modelPath = target.absolutePath,
    backend = OmniInferBackend.LITERT_GPU,
    port = 9099,
    nCtx = 8192,
)
```

Use `OmniInferLoadOptions` when you prefer grouping load settings:

```kotlin
val ok = OmniInferServer.loadModel(
    modelPath = target.absolutePath,
    options = OmniInferLoadOptions(
        backend = OmniInferBackend.LLAMA_CPP_HTP,
        port = 9099,
        nThreads = 6,
        nCtx = 8192,
    ),
)
```

Backend-specific `extraConfig` is still available for advanced overrides, but
normal third-party apps should prefer catalog defaults or the backend selectors
above.

Always handle load failure:

```kotlin
if (!ok) {
    val reason = OmniInferServer.getLastError()
}
```

## Send Requests

After `loadModel()` succeeds, send OpenAI-compatible requests to:

```text
http://127.0.0.1:9099/v1/chat/completions
```

Minimal non-streaming body:

```json
{
  "model": "local-model",
  "messages": [
    {"role": "user", "content": "Say READY only."}
  ],
  "stream": false,
  "temperature": 0,
  "top_k": 1,
  "top_p": 1,
  "max_tokens": 8,
  "reasoning_effort": "none"
}
```

The response follows the OpenAI chat completions shape and includes OmniInfer
usage metrics such as prompt tokens, completion tokens, prefill speed, decode
speed, and time to first token.

## Stop Inference

```kotlin
OmniInferServer.unloadModel()
OmniInferServer.stop()
```

Call `unloadModel()` before loading a different model. Call `stop()` when your
app no longer needs the local server.

## Common Integration Checks

- Use Maven dependency resolution, not a flat AAR, so transitive dependencies are present.
- Enable AndroidX with `android.useAndroidX=true`.
- Restrict native packaging to `arm64-v8a` and keep `useLegacyPackaging=true`.
- Store models in app-readable storage and verify SHA256 before loading.
- Allow localhost cleartext HTTP if the app calls `127.0.0.1`.
- For HTP, confirm logcat contains `HTP0 (Hexagon)` or `selected device=HTP0`.
