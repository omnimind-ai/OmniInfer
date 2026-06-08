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

### llama.cpp Hexagon HTP

On supported Snapdragon devices, llama.cpp can use the Hexagon HTP backend for
GGUF prompt processing:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3.5-2B-Q4_0.gguf",
    backend = "llama.cpp",
    nThreads = 6,
    nCtx = 8192,
    extraConfig = mapOf(
        "accelerator" to "htp",
        "llama_device" to "HTP0",
        "n_gpu_layers" to "99",
        "batch_size" to "1024",
        "ubatch_size" to "1024",
        "cpu_mask" to "0xfc",
        "cpu_strict" to "true",
    ),
)
```

HTP settings:

| Setting | Effect |
|---|---|
| `extraConfig["accelerator"] = "htp"` | Selects the Qualcomm Hexagon path without overloading the framework `backend` name |
| `extraConfig["llama_device"] = "HTP0"` | Selects llama.cpp device `HTP0` |
| `extraConfig["n_gpu_layers"] = "99"` | Offloads all supported layers to the selected llama.cpp device |
| `extraConfig["batch_size"]`, `extraConfig["ubatch_size"]` | Defaults to `1024` for HTP |
| `extraConfig["cpu_mask"]`, `extraConfig["cpu_strict"]` | Optional CPU thread-pool placement for scheduler/helper work |
| `extraConfig["backend_type"] = "npu"` | Legacy alias; kept for compatibility, not recommended for new catalog entries |

Use `OmniInferServer.listCatalogModels()` to read the bundled
`android-llamacpp-htp` catalog. Its `runtime.load_options` map is designed to be
passed directly as `extraConfig`.

For HTP performance, use repackable GGUF quantizations such as Q4_0, Q4_1,
Q8_0, IQ4_NL, or MXFP4. K-quants such as Q4_K_M can run much slower on this
backend and are not the default Android HTP catalog path.

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
| `extraConfig["vision_backend"] = "cpu"` / `"gpu"` / `"npu"` | Enables LiteRT-LM image input and selects the vision encoder backend |
| `extraConfig["enable_speculative_decoding"] = "true"` | Enables LiteRT-LM speculative decoding during normal chat engine initialization |
| `extraConfig["max_images"] = "1"` | Sets `EngineConfig.maxNumImages` when `vision_backend` is enabled |
| `nCtx` | Passed to `EngineConfig.maxNumTokens` |
| request `max_tokens` | OmniInfer cancels LiteRT-LM after the response budget is reached |

### LiteRT-LM Load-Time Options

`backend_type`, `vision_backend`, `max_images`, and `enable_speculative_decoding`
are **model-load options**, not per-request HTTP fields. Pass them in
`OmniInferServer.loadModel(..., extraConfig = ...)` before the LiteRT-LM
`Engine` is initialized.

If the same `modelPath` and `backend` are already loaded, `loadModel()` returns
the existing session and does not rebuild the LiteRT-LM engine just because
`extraConfig` changed. To switch text-only to multimodal, SD-off to SD-on, CPU to
GPU, or one vision backend to another, unload first:

```kotlin
OmniInferServer.unloadModel()

OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 4096,
    extraConfig = mapOf(
        "backend_type" to "gpu",
        "vision_backend" to "gpu",
        "max_images" to "1",
        "enable_speculative_decoding" to "true",
    ),
)
```

LiteRT-LM logs the effective configuration after a successful load. Check
logcat for a line like:

```text
created backend=litert-lm/GPU visionBackend=GPU nCtx=4096 ... speculativeDecoding=true
```

If the line says `visionBackend=none`, image requests will fail even if the
`.litertlm` file contains vision assets. If it says `speculativeDecoding=false`,
SD was not enabled for that engine instance.

Important LiteRT-LM details:

- The host app must use Kotlin Gradle plugin `2.3.0+`.
- The host app does not need to declare `com.google.ai.edge.litertlm:litertlm-android`; `:omniinfer-server` owns that dependency when `omniinfer.backend.litert_lm=true`.
- Some `.litertlm` files do not store max-context metadata. Always pass explicit `nCtx` for long-context use.
- OmniInfer uses LiteRT-LM `0.11.0+` for Gemma 4 multimodal support. Older LiteRT-LM `0.10.2` cannot initialize Gemma 4 E2B's three-signature vision encoder.
- OpenAI-compatible HTTP tool calling is supported through LiteRT-LM's official `ConversationConfig.tools` / `OpenApiTool` path. OmniInfer returns structured `choices[0].message.tool_calls`; it does not execute app tools on the server side.
- For multimodal, OmniInfer passes images as `Content.ImageBytes(...)` and creates the app cache directory before `Engine.initialize()`.
- Speculative decoding is a load-time engine setting. Use `extraConfig["enable_speculative_decoding"] = "true"` before model load; changing it per request requires unloading/reloading the LiteRT-LM engine.
- SD does not require the app to pass a separate draft-model path. The `.litertlm` package itself must support Multi Token Prediction / speculative decoding, for example by including LiteRT-LM MTP sections. If the package does not support it, turning on the flag cannot create an acceleration path.
- Do not use LiteRT-LM's public `benchmark()` helper to validate SD-on behavior in `0.11.0`; that helper uses `nativeCreateBenchmark(...)`, whose public Kotlin/JNI signature does not pass the SD flag. Normal chat generation uses `Engine.initialize()` and `Conversation.getBenchmarkInfo()`, which does report the SD-on path.

LiteRT-LM tool calling notes:

- `tool_choice = "none"` disables tools for that request.
- `tool_choice = "auto"` passes all provided tools to LiteRT-LM.
- `tool_choice = "required"` passes all tools and adds a lightweight instruction that the model must call a tool.
- OpenAI object form, for example `{"type":"function","function":{"name":"product"}}`, is accepted. OmniInfer passes only that function to LiteRT-LM and adds a lightweight instruction that the model must call it.
- Tool-result follow-up requests should include the previous assistant `tool_calls` message plus a `role = "tool"` message. OmniInfer converts the tool result into LiteRT-LM-readable context before continuing generation.
- Required/forced behavior still depends on the model obeying the tool instruction. For hard guarantees, callers should validate that `finish_reason == "tool_calls"` and retry or handle fallback when needed.

Speculative decoding GPU load example:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 4096,
    extraConfig = mapOf(
        "backend_type" to "gpu",
        "enable_speculative_decoding" to "true",
    ),
)
```

Multimodal GPU load example:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 4096,
    extraConfig = mapOf(
        "backend_type" to "gpu",
        "vision_backend" to "gpu",
        "max_images" to "1",
    ),
)
```

Multimodal plus SD example:

```kotlin
OmniInferServer.unloadModel()

OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 4096,
    extraConfig = mapOf(
        "backend_type" to "gpu",
        "vision_backend" to "gpu",
        "max_images" to "1",
        "enable_speculative_decoding" to "true",
    ),
)
```

### LiteRT-LM Smoke Test

For a quick GPU load and HTTP check, see [smoke-tests.md](./smoke-tests.md#litert-lm-gpu-smoke). For common LiteRT-LM load mistakes, see [troubleshooting.md](./troubleshooting.md#litert-lm).

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
| `omniinfer.backend.llama_cpp_htp` | Auto-enabled when a local HTP prebuilt dir exists | Skips packaging llama.cpp Snapdragon HTP runtime libraries |
| `omniinfer.backend.mnn` | `true` | Skips MNN JNI build path; `mnn` load requests return a disabled-backend error |
| `omniinfer.backend.executorch_qnn` | `true` | Skips QNN native build path and prebuilt runtime download; `executorch-qnn` load requests return a disabled-backend error |
| `omniinfer.backend.litert_lm` | `true` | Skips LiteRT-LM source set and `litertlm-android` dependency; `litert`/`litert-lm` load requests return a disabled-backend error |

For normal third-party integration, prefer these Gradle properties over direct CMake customization. The `:omniinfer-server` module maps the native backend switches to CMake internally.

For llama.cpp HTP AAR publishing, provide a Snapdragon llama.cpp package lib
directory when auto-detection is not available:

```bash
-Pomniinfer.backend.llama_cpp_htp=true \
-Pomniinfer.llama_cpp.htp_prebuilt_dir=/path/to/llama.cpp/lib
```

Even a LiteRT-only host app still configures the `:omniinfer-server` external native build for `libomniinfer-jni.so`. Keep Android SDK NDK `28.2.13676358` available, and install SDK CMake/Ninja if AGP reports `[CXX1416] Could not find Ninja on PATH or in SDK CMake bin folders`.

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
