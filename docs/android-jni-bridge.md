# Android JNI Bridge

OmniInfer can generate a decoupled Android JNI bridge for Android Apps that want to embed the Android runtime directly.

The entry script is:

```sh
bash ./scripts/platforms/android/jni-bridge/generate.sh \
  --app-dir /path/to/YourAndroidApp \
  --module app \
  --package com.example.yourapp.omniinfer \
  --qnn-bundle-dir /path/to/qnn-bundle
```

## What it generates

The generator writes into the target Android module:

- `src/main/java/<package>/OmniInferNativeBridge.kt`
- `src/main/cpp/omniinfer-native-jni/omniinfer_native_jni.cpp`
- `src/main/jniLibs/arm64-v8a/libomniinfer-native-jni.so`
- `src/main/assets/omniinfer-native/runtime/bin/omniinfer-android`
- `src/main/assets/omniinfer-native/runtime/support/common.sh`
- `src/main/assets/omniinfer-native/runtime/backends/llama_cpp/backend.sh`
- `src/main/assets/omniinfer-native/runtime/backends/omniinfer_native/backend.sh`
- `src/main/assets/omniinfer-native/runtime/qnn/*`

`qnn_llama_runner`, `qnn_multimodal_runner`, and the QNN shared libraries are copied only when `--qnn-bundle-dir` is provided.

## Why this shape

The generated bridge is intentionally split into:

- Kotlin bridge layer
- small native JNI layer
- packaged Android runtime assets

This avoids hard-coding app-specific JNI symbol names in upstream source files while keeping the App-side integration stable and easy to regenerate.

The native JNI layer uses `RegisterNatives`, so the app can choose its own package name and bridge class at generation time.

## Runtime behavior

On first use, the generated Kotlin bridge:

1. loads `libomniinfer-native-jni.so`
2. extracts `assets/omniinfer-native/runtime/` into the app private writable directory
3. marks runtime shell files and QNN runners executable
4. creates a per-session writable state root under the extracted runtime
5. dispatches inference requests through the same Android runtime shell entrypoint used by the Android CLI

This means the App path and the CLI path share the same backend semantics.

## Kotlin usage

The generated `OmniInferNativeBridge` is a Kotlin `object` (singleton). A typical lifecycle:

### 1. Check availability

```kotlin
if (!OmniInferNativeBridge.isRuntimeAvailable()) {
    // libomniinfer-native-jni.so failed to load
    return
}
```

### 2. Initialize a session

`init` extracts the bundled runtime assets on first call, selects the `omniinfer-native` backend, and loads the model. It returns a `Long` handle (0 on failure).

```kotlin
val handle = OmniInferNativeBridge.init(
    modelPath = "/data/local/tmp/my-model",   // model directory on device
    tokenizerPath = null,                       // optional, if model dir contains tokenizer.json
    decoderModelVersion = null,                 // optional, for multi-version ExecuTorch models
    nThreads = 4,
    nCtx = 2048
)
if (handle == 0L) {
    // initialization failed
}
```

### 3. Run inference

`generate` sends a prompt and returns the model response. The optional `callback` object receives streaming events via reflection — implement `onToken(String)` and `onMetrics(String)` methods:

```kotlin
class InferenceCallback {
    fun onToken(token: String) {
        // called with the full response text when generation completes
    }
    fun onMetrics(metrics: String) {
        // called with a summary like "prefill_tps=42.5, decode_tps=18.3"
    }
}

val response = OmniInferNativeBridge.generate(
    handle = handle,
    systemPrompt = "You are a helpful assistant.",
    prompt = "What is 2 + 2?",
    imageData = null,           // pass a ByteArray for multimodal models
    nThreads = 4,
    thinkEnabled = false,
    callback = InferenceCallback()
)
```

### 4. Multimodal (image + text)

```kotlin
val imageBytes: ByteArray = loadImageBytes()  // PNG, JPEG, or WebP

val response = OmniInferNativeBridge.generate(
    handle = handle,
    systemPrompt = null,
    prompt = "Describe this image.",
    imageData = imageBytes,
    nThreads = 4,
    thinkEnabled = false,
    callback = null
)
```

You can also pre-warm the image encoder before the first prompt:

```kotlin
OmniInferNativeBridge.prewarmImage(handle, imageBytes, nThreads = 4)
```

### 5. Multi-turn conversation

The bridge tracks conversation history automatically. To restore a prior conversation:

```kotlin
OmniInferNativeBridge.loadHistory(
    handle = handle,
    roles = arrayOf("system", "user", "assistant", "user"),
    contents = arrayOf(
        "You are a helpful assistant.",
        "Hi!",
        "Hello! How can I help?",
        "Tell me a joke."
    )
)
```

### 6. Other controls

```kotlin
OmniInferNativeBridge.setThinkMode(handle, enabled = true)   // toggle extended thinking
OmniInferNativeBridge.cancel(handle)                          // cancel in-progress generation
OmniInferNativeBridge.reset(handle)                           // clear history and system prompt
val diagnostics = OmniInferNativeBridge.collectDiagnostics(handle)  // debug info as Map<String, String>
```

### 7. Release resources

```kotlin
OmniInferNativeBridge.free(handle)
```

## Recommended App integration

For an app with existing backend abstractions:

- keep llama.cpp JNI backends as-is
- add one unified bridge facade that can dispatch to:
  - `llama-jni`
  - `mtmd-jni`
  - `omniinfer-native-jni`
- keep `libomniinfer-native-jni.so` and QNN runtime files out of Git if they are regenerated locally

## Validation notes

The generated runtime bundle has been validated on-device by:

1. pushing the generated `runtime/` tree to Android
2. selecting `omniinfer-native`
3. loading a `hybrid_llama_qnn.pte` package
4. running text inference successfully through the generated runtime assets

Multimodal ExecuTorch/QNN packaging is supported by the generator and runtime layout, but validation still depends on having an actual `qnn_multimodal_runner` bundle and matching multimodal `.pte` artifacts available.
