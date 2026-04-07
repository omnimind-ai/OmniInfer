# OmniInfer Android JNI Bridge Integration Guide

Integrate OmniInfer's local LLM inference into your Android project.

OmniInfer JNI Bridge compiles llama.cpp and MNN into a **single** native library (`.so`), allowing you to switch backends at runtime via a Kotlin API. No precompiled binaries needed — Gradle builds everything automatically.

## Supported Backends

| Backend | Model Format | Notes |
|---------|-------------|-------|
| `llama.cpp` | GGUF (`.gguf`) | Wide community model support |
| `mnn` | MNN (`.mnn` + `config.json`) | Better mobile performance, Alibaba MNN engine |

## Prerequisites

- Android Studio (with Android SDK and NDK)
- Git
- Bash (built-in on macOS/Linux; use Git Bash or WSL on Windows)

## Integration Steps

### 1. Clone OmniInfer and initialize submodules

```bash
git clone https://github.com/omnimind-ai/OmniInfer.git
cd OmniInfer
git checkout main_dev

# Both backends
git submodule update --init framework/llama.cpp framework/mnn

# llama.cpp only
git submodule update --init framework/llama.cpp
```

### 2. Generate JNI Bridge code

Run the generator script to inject JNI Bridge code into your Android project:

```bash
bash scripts/platforms/android/jni-bridge/generate-v2.sh \
  --app-dir /path/to/YourAndroidApp \
  --package com.yourcompany.yourapp.omniinfer
```

`--package` specifies the Kotlin package for the generated bridge class. Use any package within your project.

The script generates the following files in your project:

```
app/src/main/
  ├── java/com/yourcompany/yourapp/omniinfer/
  │   └── OmniInferBridge.kt           # Kotlin API
  └── cpp/omniinfer-jni/
      ├── CMakeLists.txt                # CMake build config
      ├── omniinfer_jni.cpp             # JNI entry point
      ├── inference_backend.h           # Backend abstraction
      ├── backend_llama_cpp.h           # llama.cpp implementation
      └── backend_mnn.h                # MNN implementation
```

Optional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--module <name>` | `app` | Android module name |
| `--class <name>` | `OmniInferBridge` | Generated Kotlin class name |
| `--lib-name <name>` | `omniinfer-jni` | Native library name |
| `--llama-cpp-dir <dir>` | auto-detect | Path to llama.cpp source tree |
| `--mnn-dir <dir>` | auto-detect | Path to MNN source tree (omit to disable MNN) |

### 3. Configure build.gradle.kts

Add the following to the `android {}` block in `app/build.gradle.kts`:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_LLAMAFILE=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=ON"

                // Recommended: enable ARM instruction set optimizations
                arguments += "-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod+i8mm"

                // Optional: enable MNN backend (requires framework/mnn submodule)
                // arguments += "-DOMNIINFER_BACKEND_MNN=ON"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/omniinfer-jni/CMakeLists.txt")
        }
    }
}
```

### 4. Build

```bash
./gradlew assembleDebug
```

First build compiles the inference engine sources (~1-2 minutes). Subsequent incremental builds take only a few seconds.

## Kotlin API Reference

### Check availability

```kotlin
if (!OmniInferBridge.isRuntimeAvailable()) {
    // Native library failed to load
    return
}
```

### Initialize a session

```kotlin
// llama.cpp backend with GGUF model
val handle = OmniInferBridge.init(
    modelPath = "/sdcard/models/qwen3.5-0.8b-q4km.gguf",
    backend = "llama.cpp",
    nThreads = 0,       // 0 = auto-detect thread count
    nCtx = 2048         // context window size
)

// MNN backend with MNN model
val handle = OmniInferBridge.init(
    modelPath = "/sdcard/models/Qwen3.5-0.8B-MNN/config.json",
    backend = "mnn"
)

// handle == 0L indicates failure
```

### Generate text

```kotlin
val response = OmniInferBridge.generate(
    handle = handle,
    systemPrompt = "You are a helpful assistant.",
    prompt = "What is the capital of France?"
)
```

### Streaming generation

```kotlin
val response = OmniInferBridge.generate(
    handle = handle,
    systemPrompt = "You are a helpful assistant.",
    prompt = "Write a short poem.",
    callback = object {
        fun onToken(token: String) {
            // Called for each generated token fragment
            runOnUiThread { textView.append(token) }
        }
        fun onMetrics(metrics: String) {
            // Called after generation completes
            // Format: "prefill_tps=73.0, decode_tps=17.0"
        }
    }
)
```

### Multi-turn conversation

The backend automatically maintains conversation history across `generate()` calls:

```kotlin
OmniInferBridge.generate(handle, "You are a translator.", "Hello")
OmniInferBridge.generate(handle, null, "Translate that to French")
```

To manually load conversation history:

```kotlin
OmniInferBridge.loadHistory(
    handle = handle,
    roles = arrayOf("system", "user", "assistant", "user"),
    contents = arrayOf("You are a translator.", "Hello", "Bonjour!", "Now translate to Japanese")
)
```

### Session control

```kotlin
OmniInferBridge.reset(handle)                        // Clear conversation history
OmniInferBridge.cancel(handle)                       // Cancel in-progress generation
OmniInferBridge.setThinkMode(handle, enabled = true) // Enable thinking mode
val diag = OmniInferBridge.collectDiagnostics(handle) // Performance diagnostics
OmniInferBridge.free(handle)                          // Release resources (must call)
```

## Model Preparation

### GGUF models (llama.cpp backend)

```bash
# HuggingFace
curl -L -o model.gguf \
  https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf

# ModelScope (recommended in China)
pip install modelscope
modelscope download --model unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf
```

### MNN models (MNN backend)

```bash
pip install modelscope
modelscope download --model MNN/Qwen3.5-0.8B-MNN --local_dir ./Qwen3.5-0.8B-MNN
```

MNN models are directories. Pass the absolute path to `config.json` when calling `init()`.

### Push to device

For development and testing:

```bash
adb push model.gguf /data/local/tmp/
adb push Qwen3.5-0.8B-MNN/ /data/local/tmp/Qwen3.5-0.8B-MNN/
```

In production, download models to `context.filesDir` or `context.getExternalFilesDir()`.

## Performance Reference

Tested on Qualcomm Snapdragon 8 Gen 3, Qwen3.5-0.8B:

| Backend | Model Format | Prefill | Decode |
|---------|-------------|---------|--------|
| llama.cpp | Q4_K_M GGUF | - | ~1.3 tps |
| mnn | MNN | ~73 tps | ~17 tps |

MNN significantly outperforms llama.cpp on mobile devices. Use MNN for best inference speed.

## Thread Safety

- Different handles can be used concurrently from different threads.
- Do not call `generate()` concurrently on the same handle.
- `cancel()` is safe to call from any thread.
- Run inference methods on `Dispatchers.IO` to avoid blocking the UI thread.

## Architecture

The JNI Bridge uses `RegisterNatives` for dynamic JNI method registration, decoupling C++ code from Java package names. You can regenerate with a different `--package` and `--class` without modifying app code.

```
Kotlin (OmniInferBridge)
    ↓ RegisterNatives
JNI C++ (omniinfer_jni.cpp)
    ↓ InferenceBackend interface
    ├── LlamaCppBackend  →  llama.cpp C API (statically linked)
    └── MnnBackend       →  MNN Llm C++ API (statically linked)
```

Both backends are **statically linked** into a single `libomniinfer-jni.so`. The final APK contains only this one native library with zero external dependencies.

## Troubleshooting

**`isRuntimeAvailable()` returns false**

Missing `externalNativeBuild` config in `build.gradle.kts`, or NDK not installed.

**`init()` returns 0**

Check model path (must be absolute) and file permissions. Inspect logcat:

```bash
adb logcat -s OmniInferJni
```

**MNN build fails**

Ensure submodule is initialized (`git submodule update --init framework/mnn`) and `-DOMNIINFER_BACKEND_MNN=ON` is set in `build.gradle.kts`.

**Slow first build**

Normal. Compiling llama.cpp and MNN C++ sources takes ~1-2 minutes. Subsequent Kotlin-only changes do not trigger native recompilation.
