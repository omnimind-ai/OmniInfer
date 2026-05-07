# Android Backend Reference

This document collects backend-specific Android integration details. For the short setup path, start with [integration.md](./integration.md).

## Backend Matrix

| Backend | Model format | Runtime | `backend` value | `modelPath` |
|---|---|---|---|---|
| llama.cpp | `.gguf`, optional `mmproj*.gguf` | Native JNI, CMake-built | `llama.cpp` | GGUF model file |
| MNN | Directory with `config.json` | Native JNI, CMake-built | `mnn` | `config.json` |
| LiteRT-LM | `.litertlm` | Google AI Edge LiteRT-LM AAR | `litert`, `litert-lm`, `litertlm` | `.litertlm` file |
| ExecuTorch QNN | `.pte` + tokenizer | QNN subprocess runner | `executorch-qnn` | `.pte` file |

The public API is intentionally the same for all backends:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/absolute/device/path",
    backend = "litert",
    nThreads = 4,
    nCtx = 8192,
    extraConfig = mapOf("backend_type" to "cpu"),
)
```

Only one model is loaded at a time. Loading a different model or backend unloads the previous one.

## llama.cpp

Use llama.cpp for GGUF models and GGUF multimodal models.

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-e2b.gguf",
    backend = "llama.cpp",
    nThreads = 6,
    nCtx = 8192,
)
```

Notes:

- `modelPath` points directly to the `.gguf` model.
- `nThreads = 0` lets OmniInfer choose an automatic CPU thread count.
- Multimodal support is auto-detected when `mmproj*.gguf` is in the same directory.
- The Android library statically links llama.cpp into `libomniinfer-jni.so`.

## MNN

Use MNN for MNN-packaged models.

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3.5-2B-MNN/config.json",
    backend = "mnn",
    nThreads = 6,
    nCtx = 8192,
)
```

Notes:

- `modelPath` points to the model directory's `config.json`.
- Text weights usually live beside it as `llm.mnn` and `llm.mnn.weight`.
- Vision weights, when present, are referenced by `config.json`.
- The Android library statically links MNN into `libomniinfer-jni.so`.

## LiteRT-LM

Use LiteRT-LM for Google AI Edge `.litertlm` artifacts.

```kotlin
// CPU
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nThreads = 4,
    nCtx = 8192,
    extraConfig = mapOf("backend_type" to "cpu"),
)

// GPU
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 8192,
    extraConfig = mapOf("backend_type" to "gpu"),
)
```

LiteRT-LM settings:

| Setting | Effect |
|---|---|
| `backend = "litert"` / `"litert-lm"` / `"litertlm"` | Selects LiteRT-LM |
| `extraConfig["backend_type"] = "cpu"` | Uses `Backend.CPU(numOfThreads = nThreads)` |
| `extraConfig["backend_type"] = "gpu"` | Uses `Backend.GPU()` |
| `extraConfig["backend_type"] = "npu"` | Uses `Backend.NPU(nativeLibraryDir)` |
| `extraConfig["litert_backend"]` | Alias for `backend_type` |
| `extraConfig["gpu_mode"]` | Alias for `backend_type` |
| `nCtx` | Passed to `EngineConfig.maxNumTokens` |
| request `max_tokens` | OmniInfer cancels LiteRT-LM after the response budget is reached |

Important LiteRT-LM details:

- The host app must use Kotlin Gradle plugin `2.3.0+`.
- The host app does not need to declare `com.google.ai.edge.litertlm:litertlm-android`; `:omniinfer-server` owns that dependency when `omniinfer.backend.litert_lm=true`.
- Some `.litertlm` files do not store max-context metadata. Always pass explicit `nCtx` for long-context use.
- Current OmniInfer LiteRT-LM path is text-only.

## ExecuTorch QNN

Use ExecuTorch QNN for Qualcomm NPU `.pte` models.

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-1.7B/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf("decoder_model_version" to "qwen3"),
)
```

Common `extraConfig` keys:

| Key | Default | Description |
|---|---|---|
| `decoder_model_version` | `qwen3` | Chat template family: `qwen3`, `qwen2_5`, `llama3`, `gemma3` |
| `tokenizer_path` | Auto-discovered | Explicit path to `tokenizer.json` |
| `seq_len` | Model-dependent | Max total tokens for the QNN runner |
| `attention_sink_evictor_path` | Auto-discovered | Explicit path to `attention_sink_evictor.pte` |

QNN is a larger topic because it uses a subprocess runner and bundles Qualcomm runtime libraries. See [et-qnn.md](./et-qnn.md).

## Backend Gradle Switches

OmniInfer enables every Android backend by default so the library is ready for all supported runtimes:

```properties
omniinfer.backend.llama_cpp=true
omniinfer.backend.mnn=true
omniinfer.backend.executorch_qnn=true
omniinfer.backend.litert_lm=true
```

Set a backend to `false` in the host or root `gradle.properties` when the app does not need it:

```properties
omniinfer.backend.litert_lm=false
omniinfer.backend.executorch_qnn=false
```

Switch behavior:

| Property | Default | When set to `false` |
|---|---:|---|
| `omniinfer.backend.llama_cpp` | `true` | Skips llama.cpp JNI build path; `llama.cpp` load requests return a disabled-backend error |
| `omniinfer.backend.mnn` | `true` | Skips MNN JNI build path; `mnn` load requests return a disabled-backend error |
| `omniinfer.backend.executorch_qnn` | `true` | Skips QNN native build path and prebuilt runtime download; `executorch-qnn` load requests return a disabled-backend error |
| `omniinfer.backend.litert_lm` | `true` | Skips LiteRT-LM source set and `litertlm-android` dependency; `litert`/`litert-lm` load requests return a disabled-backend error |

For normal third-party integration, prefer these Gradle properties over direct CMake customization. The `:omniinfer-server` module maps the native backend switches to CMake internally.

Host apps should also restrict packaged ABIs unless they intentionally ship emulator or desktop ABIs:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += "arm64-v8a"
        }
    }
}
```

This avoids packaging unused native libraries such as x86_64 copies from transitive AARs. OmniInfer Android targets real `arm64-v8a` devices.

## Native Libraries

The main Android native output is:

```text
libomniinfer-jni.so
```

It statically links the enabled native backends and their dependencies. ExecuTorch QNN is different: when enabled, it also packages QNN and runner libraries under `jniLibs/arm64-v8a/`.

For GPU/OpenCL backends, the library manifest declares optional device libraries:

```xml
<uses-native-library android:name="libOpenCL.so" android:required="false" />
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
```

You normally do not need to copy these declarations into the host manifest unless your manifest merge rules remove library entries.
