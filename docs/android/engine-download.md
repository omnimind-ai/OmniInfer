# Android Runtime Engine Download

Use this mode when the app should ship the OmniInfer SDK without native
inference libraries. The app downloads a versioned engine package, verifies it,
installs it in private storage, and loads it with `System.load`.

For a self-contained APK, use the standard
[AAR integration](./aar-integration.md). The two modes have separate Maven
artifacts at the same release version:

| Mode | Dependency |
|---|---|
| Bundled runtime | `io.github.omnimind-ai:omniinfer:0.2.7` |
| Downloaded runtime | `io.github.omnimind-ai:omniinfer-lite:0.2.7` |

Choose one dependency; do not add both.

## Requirements

- arm64 Android 8.0+ (`minSdk 26`)
- an HTTPS endpoint containing a matching engine zip and `.sha256`
- an app-readable model file with a trusted size and SHA-256
- a private, enterprise, or non-Play distribution channel; Google Play apps
  must not download executable code from outside Play

## Quickstart

### 1. Add the Lite SDK

Keep `google()` and `mavenCentral()` in `settings.gradle.kts`, then add:

```kotlin
dependencies {
    implementation("io.github.omnimind-ai:omniinfer-lite:0.2.7")
}
```

The resulting APK contains the SDK and its Maven dependencies but no native
inference `.so` files.

Add `INTERNET` and the foreground-service permissions described in
[AAR integration](./aar-integration.md#manifest). Allow cleartext only for
`127.0.0.1`; engine and model downloads should use HTTPS.

### 2. Download and install the engine

1. Download `<engine>.zip.sha256` and parse the first whitespace-delimited
   value as a 64-character lowercase SHA-256.
2. Stream the zip to a temporary file while hashing it, then compare hashes.
3. Extract into a staging directory under app-private storage. Reject entries
   whose canonical path escapes that directory.
4. Atomically rename the verified staging directory into place.
5. Verify the manifest and every library before loading:

```kotlin
val manifest = OmniInferEngineLoader.install(
    engineDir,
    verifyHashes = true,
)
```

`install` validates the manifest format, SDK/native interface version, ABI,
minimum SDK, library sizes and SHA-256 values, and rejects unlisted `.so`
files. Catch `OmniInferEngineException`, retain the last verified engine, and
offer a clean re-download.

### 3. Load a model

Store the verified model in app-private storage, initialize OmniInfer after the
engine is installed, and select a backend listed in `manifest.backends`:

```kotlin
OmniInferServer.init(applicationContext)
val ok = OmniInferServer.loadModel(
    modelPath = modelPath,
    options = OmniInferLoadOptions(
        backend = "llama.cpp-cpu",
        port = 9099,
        nCtx = 4096,
    ),
)
check(ok) { OmniInferServer.getLastError() }
```

Use `llama.cpp-htp` only when the installed engine lists it and supports the
device. Normal apps must not depend on `/data/local/tmp`.

### 4. Send a request

```text
POST http://127.0.0.1:9099/v1/chat/completions
Content-Type: application/json

{"model":"local","messages":[{"role":"user","content":"Reply READY only."}],"stream":false,"max_tokens":8}
```

The response uses the OpenAI chat-completions shape and includes usage and
performance fields.

## Runtime Contract

```text
Lite APK
  → verified engine zip in app-private storage
  → OmniInferEngineLoader.install(engineDir)
  → manifest and per-library verification
  → core libraries loaded with System.load
  → ggml backends discovered from the same engine directory
  → OmniInferServer on 127.0.0.1
```

An engine zip contains one ABI:

```text
manifest.json
lib/arm64-v8a/*.so
```

| Manifest field | Meaning |
|---|---|
| `formatVersion` | Package format, currently `1` |
| `engineVersion` | OmniInfer version used to build the package |
| `interfaceVersion` | SDK/native contract, currently `1` |
| `abi` / `minSdk` | Device compatibility gates |
| `backends` | Included selectors such as `llama.cpp-cpu` |
| `coreLibs` | `System.load` order ending in `libomniinfer-jni.so` |
| `libs` | Expected name, size and SHA-256 of every `.so` |

Only one engine can be loaded per process. Switching versions requires an app
restart because registered native backends cannot be safely unloaded.

For production downloads, use WorkManager, check free space with `StatFs`, and
use the Android 14+ `dataSync` foreground-service type where applicable. Keep
SDK and engine versions in lockstep. Verify 16 KB page-size compatibility for
every included prebuilt library before supporting 16 KB devices.

## Building an Engine Package

This section is for engine distributors, not normal app developers. Clone the
repository with submodules and run from `android/` with JDK 17 or 21, Gradle
8.10.2, Android SDK 35, NDK `28.2.13676358`, and SDK CMake/Ninja:

```bash
git clone --recurse-submodules https://github.com/omnimind-ai/OmniInfer.git
cd OmniInfer/android
export ANDROID_HOME=/absolute/path/to/Android/Sdk

gradle :omniinfer-server:bundleEnginePackage \
  -Pomniinfer.backend.llama_cpp=true \
  -Pomniinfer.backend.mnn=false \
  -Pomniinfer.backend.executorch_qnn=false \
  -Pomniinfer.backend.litert_lm=false \
  -Pomniinfer.backend.llama_cpp_htp=false \
  -Pomniinfer.maven.version=0.2.7
```

The zip and checksum are written under
`omniinfer-server/build/distributions/engine/`. The task reopens the finished
zip and verifies every entry against its manifest.

For HTP, enable `omniinfer.backend.llama_cpp_htp` and pass
`omniinfer.llama_cpp.htp_prebuilt_dir` pointing to one complete,
version-matched Snapdragon runtime set. Never combine host and DSP libraries
from different llama.cpp commits.
