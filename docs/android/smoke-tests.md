# Android Smoke Tests

Use this document when you need to validate an Android build or a device setup. For app integration, start with [integration.md](./integration.md).

## Source-Build Host App

Create a small host app outside the OmniInfer repository and point its Gradle settings at your local checkout:

```kotlin
// settings.gradle.kts
include(":app")
include(":omniinfer-server")
project(":omniinfer-server").projectDir =
    file("/absolute/path/to/OmniInfer/android/omniinfer-server")
```

Use the Gradle snippets from [integration.md](./integration.md#configure-gradle), then disable unused native backends in `gradle.properties` for a LiteRT-only smoke build:

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

If you do not use a checked-in Gradle wrapper, use a user-local Gradle install with `JAVA_HOME` and `ANDROID_HOME` set. A no-sudo Linux setup can install JDK 21 under `~/.local/jdks/` and Android SDK command line tools under `~/Android/Sdk`; install at least:

```text
platform-tools
platforms;android-35
build-tools;35.0.0
ndk;28.2.13676358
cmake;3.22.1
```

The LiteRT-only app still configures `:omniinfer-server`'s CMake project, so NDK and SDK CMake/Ninja must be present even when llama.cpp, MNN, and ExecuTorch QNN are disabled.

## adb / curl HTTP Smoke

After the host app loads a model and starts the local server:

```bash
cat > request.json <<'JSON'
{
  "model": "local",
  "messages": [
    { "role": "user", "content": "Say READY only." }
  ],
  "stream": false,
  "reasoning_effort": "none",
  "temperature": 0.0,
  "max_tokens": 16
}
JSON

adb forward tcp:9099 tcp:9099
curl -sS -H "Content-Type: application/json" \
  --data-binary @request.json \
  http://127.0.0.1:9099/v1/chat/completions
```

Prefer `--data-binary @request.json`; some shell wrappers can accidentally send an empty body when JSON is quoted inline.

## LiteRT-LM GPU Smoke

Load the model with an explicit GPU backend:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 4096,
    extraConfig = mapOf("backend_type" to "gpu"),
)
```

Then send the HTTP smoke request above. A successful response should include usage metrics and a `performance` object.

Use logcat to confirm the backend:

```bash
adb logcat -d | grep -E "LiteRtLmBackend|MainExecutorSettings|created backend|enable_speculative"
```

Useful LiteRT-LM rows include:

```text
MainExecutorSettings: backend: GPU
created backend=litert-lm/GPU
```

For speculative decoding, also verify:

```text
speculativeDecoding=true
enable_speculative_decoding: true
```

The model package must include MTP/speculative-decoding support. No separate draft-model path is passed by the app.

## Multimodal Smoke

For LiteRT-LM image requests, load with a vision backend:

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

Then send an OpenAI-compatible image request as shown in [multimodal.md](./multimodal.md#request-shape). If the response says `vision_backend` is missing, unload and reload the model; `vision_backend` is not a per-request field.

## Install Guard Checks

On some OEM ROMs, `adb install` or `pm install` opens an interactive unknown-source confirmation page and appears to hang. Check the device screen, `dumpsys activity`, or a screenshot. After confirming the install, verify the package:

```bash
adb shell pm list packages com.example.yourapp
```

## When To Escalate

Move from smoke tests to focused debugging when:

- `/health` is not reachable after `loadModel()` returned success.
- The response has HTTP 500 and logcat shows a native backend error.
- LiteRT-LM logcat does not show the expected backend or load-time option.
- The model loads but output metrics are missing or implausible.

See [troubleshooting.md](./troubleshooting.md) for common failure signatures.
