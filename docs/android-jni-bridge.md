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
