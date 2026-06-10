# Android Troubleshooting

This document collects common Android integration failures. For the short setup path, start with [integration.md](./integration.md).

## Build And Gradle

### Kotlin metadata mismatch with LiteRT-LM

Symptom:

```text
:omniinfer-server:compileKotlin
Module was compiled with an incompatible version of Kotlin
```

Fix: upgrade the host Kotlin Gradle plugin to `2.3.0+`. LiteRT-LM Android is compiled with Kotlin metadata 2.3.0.

### Missing Ktor classes

Symptom: the app compiles OmniInfer but fails at runtime or compile time with missing Ktor server classes.

Fix: prefer the Maven-published OmniInfer Android artifact so Gradle can resolve
transitive dependencies from the generated POM. If you use a flat local AAR file,
Gradle cannot read Maven dependency metadata; add Ktor and kotlinx dependencies
manually:

```kotlin
implementation("io.ktor:ktor-server-core:3.1.3")
implementation("io.ktor:ktor-server-cio:3.1.3")
implementation("io.ktor:ktor-server-content-negotiation:3.1.3")
implementation("io.ktor:ktor-serialization-kotlinx-json:3.1.3")
implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
```

The same rule applies to LiteRT-LM: host apps should not declare
`com.google.ai.edge.litertlm:litertlm-android` manually. Use the OmniInfer Maven
coordinate so the tested LiteRT-LM version is resolved transitively.

### Unused x86_64 LiteRT libraries in APK

Symptom: the final APK contains `lib/x86_64/libLiteRt.so`,
`lib/x86_64/libLiteRtClGlAccelerator.so`, or
`lib/x86_64/liblitertlm_jni.so`.

Fix: add an ABI filter in the host app if the app only targets arm64 phones.
These libraries come from the LiteRT-LM transitive dependency, not from
OmniInfer's own AAR native payload.

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += "arm64-v8a"
        }
    }
}
```

### CMake configure fails with missing Ninja

Symptom:

```text
[CXX1416] Could not find Ninja on PATH or in SDK CMake bin folders
```

Fix: install an Android SDK CMake package such as `cmake;3.22.1`; it includes the Ninja binary used by AGP's external native build. A LiteRT-only host still configures `:omniinfer-server`'s CMake project for `libomniinfer-jni.so`.

### Native submodules are missing

Symptom: CMake fails because `framework/llama.cpp` or `framework/mnn` is empty.

Fix: either initialize the backend submodule you enabled, or disable that backend in `gradle.properties`.

```bash
git submodule update --init framework/llama.cpp
git submodule update --init framework/mnn
```

```properties
omniinfer.backend.llama_cpp=false
omniinfer.backend.mnn=false
```

## App And HTTP

### `127.0.0.1` request fails with a cleartext error

Fix: add `network_security_config.xml` and reference it from the app manifest. See [integration.md](./integration.md#allow-localhost-http).

### HTTP request times out and logcat says JSON input is empty

Fix: verify the client sends a request body. For adb tests, prefer:

```bash
curl -sS -H "Content-Type: application/json" \
  --data-binary @request.json \
  http://127.0.0.1:9099/v1/chat/completions
```

### Foreground service fails on Android 14+

Fix: ensure manifest merge keeps `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_SPECIAL_USE`, and `OmniInferService`.

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />
```

### Native library symbol conflicts

If your app already bundles ggml, llama.cpp, or MNN, exclude the duplicate copy or disable the overlapping OmniInfer backend in the library module.

## llama.cpp HTP

### HTP is not actually selected

Fix: load with `extraConfig["accelerator"] = "htp"` and
`extraConfig["llama_device"] = "HTP0"`, then check logcat for:

```text
GGML backend ... OpenCL
GGML backend ... HTP
llama.cpp selected device=HTP0
HTP0 new session
HTP0-REPACK model buffer
```

If those rows are missing, the app is not using the Hexagon backend. Confirm the
AAR packages `libggml-opencl.so`, `libggml-hexagon.so`, and the
`libggml-htp-v*.so` files for the target Snapdragon generation.

### HTP prefill is much slower than expected

Fix: check the model quantization first. Q4_K_M and other K-quants are not the
recommended Android HTP path. Use the bundled model catalog and start with Q4_0
GGUF files before comparing against llama.cpp Snapdragon benchmark numbers.

### Qwen3.5 4B Q4_0 loads on HTP but outputs incorrect text

Qwen3.5 4B Q4_0 can load on the llama.cpp Hexagon/HTP backend but produce
incorrect text when `SSM_CONV` is fully offloaded on tested Snapdragon devices.
OmniInfer configures `GGML_HEXAGON_OPFILTER=SSM_CONV` before loading the HTP
backend for `llama.cpp/htp`, which keeps HTP selected while letting `SSM_CONV`
fall back to CPU. If a host process initializes the Hexagon backend before this
environment variable is set, the filter may not take effect until the process is
restarted.

## LiteRT-LM

### LiteRT-LM fails around 4k context

Fix: pass `nCtx` explicitly in `loadModel()`. Some `.litertlm` files do not store max-context metadata, so relying on model defaults can give a smaller context than expected.

### Image request returns a `vision_backend` error

Symptom:

```text
LiteRT-LM image input requires loading the model with extraConfig vision_backend=cpu|gpu|npu
```

Fix: unload the model and load it again with `extraConfig["vision_backend"] = "cpu"` or `"gpu"`. `vision_backend` is a load-time option, not a per-request HTTP field.

### Speculative decoding flag appears ignored

Fix:

1. Unload and reload with `extraConfig["enable_speculative_decoding"] = "true"`.
2. Check logcat for `speculativeDecoding=true` or native `enable_speculative_decoding: true`.
3. Verify the `.litertlm` package supports MTP/speculative decoding.
4. Use a decode-heavy prompt when measuring speed; very short outputs can hide or reverse the benefit.

No separate draft model path is passed by the app.

### Changing LiteRT-LM load options has no effect

LiteRT-LM consumes `backend_type`, `vision_backend`, `max_images`, and `enable_speculative_decoding` during `Engine.initialize()`. If the same model is already loaded, call `OmniInferServer.unloadModel()` before loading again with different values.

### GPU backend is not actually selected

Check logcat for:

```text
created backend=litert-lm/GPU
MainExecutorSettings: backend: GPU
```

If those rows are missing, confirm the app passed `extraConfig["backend_type"] = "gpu"` to `loadModel()` and that the model was reloaded after changing options.

## MNN

### OpenCL silently falls back to CPU

Android linker namespaces can block app-process access to `/vendor/lib64/libOpenCL.so`. The manifest must declare:

```xml
<uses-native-library android:name="libOpenCL.so" android:required="false" />
```

Without this, MNN may only log an OpenCL init fallback line.

### Gemma4 E2B MNN load asks for audio files

Some MNN multimodal packages require `audio.mnn` and `audio.mnn.weight` even when you only run text or vision. Download all model files referenced by the package, not only `llm.mnn` and `visual.mnn`.

### Gemma4 E2B MNN stops after one token

This has reproduced in upstream MNN `llm_demo` for specific benchmark prompts. Treat it as a model/package/template stop behavior unless a simple prompt also fails.

## ExecuTorch QNN

For QNN-specific subprocess, model file, and Qualcomm runtime issues, see [et-qnn.md](./et-qnn.md#troubleshooting).

One common host-app requirement remains: keep `android:extractNativeLibs="true"` when ExecuTorch QNN is enabled, because the QNN runner must exist as a regular file on disk.

## Device And Install

### OEM install confirmation blocks `adb install`

On some OEM ROMs, `adb install -r` and `pm install -r` can appear to hang because the device opened an unknown-source confirmation screen. Check the device UI or `dumpsys activity`; after confirming install, verify:

```bash
adb shell pm list packages <package-name>
```

### Remote TCP ADB becomes unauthorized

Avoid `adb kill-server` for remote devices unless someone can accept the RSA prompt. Prefer `adb disconnect` / `adb connect` first.
