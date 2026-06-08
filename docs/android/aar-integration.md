# Android AAR Integration

This guide shows the minimal path for a third-party Android app to consume
OmniInfer Android as an AAR.

The example app is available at:

```text
tmp/test_apps/ThirdPartyAarDemo
```

It intentionally does not include the OmniInfer source module.

## 1. Add The Maven Repository

Use Maven Central after the artifact is published:

```kotlin
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
```

For local validation before Central sync, add the generated local Maven repo:

```kotlin
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven {
            name = "omniInferLocal"
            url = uri("/path/to/OmniInfer/android/omniinfer-server/build/repo")
        }
    }
}
```

## 2. Add The Dependency

In the host app module:

```kotlin
dependencies {
    implementation("ai.omnimind:omniinfer-android:0.1.0-alpha01")

    // Only needed if the app sends HTTP requests with OkHttp.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
```

The Maven AAR brings OmniInfer's transitive runtime dependencies, including
Ktor, kotlinx serialization, coroutines, and AndroidX Core. Prefer Maven
consumption over a flat `libs/*.aar` file; flat AAR consumption does not carry
POM dependencies.

## 3. Configure AndroidX And Native Packaging

In the root `gradle.properties`:

```properties
android.useAndroidX=true
```

In the host app module:

```kotlin
android {
    compileSdk = 35

    defaultConfig {
        minSdk = 26
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}
```

`useLegacyPackaging = true` extracts native libraries as regular files. This is
required by accelerator runtimes that need `.so` paths on disk.

## 4. Add Permissions

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
```

`POST_NOTIFICATIONS` is needed on Android 13+ if the host targets API 33+ and
wants the foreground service notification to be visible.

OmniInfer's AAR manifest contributes the foreground service declaration and
optional native library declarations. The host app does not need to declare
`OmniInferService` manually when using the Maven AAR.

## 5. Initialize And Load A Model

```kotlin
import com.omniinfer.server.OmniInferServer

OmniInferServer.init(applicationContext)

val ok = OmniInferServer.loadModel(
    modelPath = "/data/local/tmp/gguf/gemma-4-e2b-it-edited-q4_0.gguf",
    backend = "llama.cpp",
    port = 9099,
    nThreads = 6,
    nCtx = 8192,
    extraConfig = mapOf(
        "accelerator" to "htp",
        "backend_type" to "npu",
    ),
)

if (!ok) {
    val reason = OmniInferServer.getLastError()
}
```

Common backend values:

| Backend | `backend` | Typical `extraConfig` |
|---|---|---|
| llama.cpp CPU | `llama.cpp` | empty |
| llama.cpp Snapdragon HTP | `llama.cpp` | `accelerator=htp`, `backend_type=npu` |
| MNN CPU | `mnn` | empty |
| MNN OpenCL | `mnn` | `backend_type=opencl`, `gpu_mode=68` |
| LiteRT-LM CPU | `litert` | `backend_type=cpu` |
| LiteRT-LM GPU | `litert` | `backend_type=gpu` |

Models are not packaged in the AAR. The host app should download model files at
runtime or manage them in app-specific storage.

## 6. Send An OpenAI-Compatible Request

After `loadModel()` succeeds, OmniInfer starts a local HTTP server:

```text
http://127.0.0.1:9099
```

Minimal non-streaming request:

```kotlin
val json = """
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
""".trimIndent()
```

POST it to:

```text
/v1/chat/completions
```

The response includes OpenAI-compatible `choices` plus OmniInfer `usage`
metrics such as prompt tokens, completion tokens, prefill speed, decode speed,
and time to first token.

## 7. Stop Or Replace The Model

```kotlin
OmniInferServer.unloadModel()
OmniInferServer.stop()
```

Call `stop()` when the host app no longer needs local inference.

## Validation Example

`ThirdPartyAarDemo` was built with only the AAR dependency and installed on
device `6014`. Gemma4 E2B Q4_0 loaded with llama.cpp HTP:

```text
GGML device [1] after load_all: HTP0 (Hexagon)
llama.cpp selected device=HTP0 n_gpu_layers=99 mmap=false
llama.cpp context configured: ctx=8192 batch=1024 ubatch=1024 threads=6 backend_type=npu accelerator=htp device=HTP0
```

The local `/v1/chat/completions` endpoint returned:

```text
READY
```

with `prefill_tokens_per_second=79.8` and `decode_tokens_per_second=19.1` for
the short smoke request.
