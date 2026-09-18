# Android Runtime Engine Download

This guide is for Android apps that do not bundle the OmniInfer native runtime
in the APK. The host app ships only the Kotlin SDK layer; the native inference
libraries (`libllama.so`, ggml backends, JNI bridge, ...) are downloaded at
runtime as a versioned engine package, verified, extracted into app-private
storage, and loaded with `System.load`.

Apps that prefer a self-contained install should use the standard
[AAR integration](./aar-integration.md) instead; both modes share the same
Kotlin API, server, and model catalog.

## How It Works

```
host APK (Kotlin SDK only, no .so)
  → download engine package zip (zip + .sha256 from your distribution point)
  → verify zip SHA-256, extract into app-private storage (atomic rename)
  → OmniInferEngineLoader.install(engineDir)
      manifest validation → per-lib SHA-256 → core libs System.load
  → OmniInferServer loads models with the engine dir as its native lib dir
  → OpenAI-compatible server on 127.0.0.1 as usual
```

Key platform facts that make this safe:

- Android 10+ (targetSdk 29+) forbids `exec()` of app-data files, but `dlopen`
  through `System.load` of a `.so` in app-private storage is allowed; logcat
  shows `avc: granted { execute }` for the loaded library.
- ggml discovers runtime backends (CPU variants, OpenCL, Hexagon/HTP) with
  `ggml_backend_load_all_from_path`, which accepts the engine lib dir. The JNI
  layer also sets `ADSP_LIBRARY_PATH` to the same dir for Hexagon sessions.

## Engine Package Layout

A package is a zip containing `manifest.json` and the native libraries of one
ABI:

```
manifest.json
lib/arm64-v8a/*.so
```

`manifest.json` fields:

| Field | Meaning |
|---|---|
| `formatVersion` | Manifest format version, currently `1` |
| `engineVersion` | OmniInfer release the package was built from |
| `interfaceVersion` | Kotlin SDK ↔ native interface contract, currently `1` |
| `abi` | `arm64-v8a` |
| `minSdk` | Minimum Android SDK the package supports |
| `backends` | Backend selectors included, e.g. `llama.cpp-cpu`, `llama.cpp-htp` |
| `coreLibs` | The `DT_NEEDED` chain in `System.load` order (must end with `libomniinfer-jni.so`) |
| `libs` | Every packaged `.so` with `name`, `sha256`, `sizeBytes` |

Build both artifacts from the OmniInfer repo with Gradle:

```bash
# 1. Lite AAR: Kotlin SDK only, no native libs
gradle :omniinfer-server:publishReleasePublicationToOmniInferLocalRepository \
  -Pomniinfer.packaging.native_bundled=false \
  -Pomniinfer.backend.llama_cpp=true -Pomniinfer.backend.mnn=false \
  -Pomniinfer.backend.executorch_qnn=false -Pomniinfer.backend.litert_lm=false \
  -Pomniinfer.publication.require_litert_lm=false \
  -Pomniinfer.maven.version=<version> \
  -Pomniinfer.maven.repo=<local repo dir>

# 2. Engine package: zip + manifest + .sha256 (output: build/distributions/engine)
gradle :omniinfer-server:bundleEnginePackage \
  -Pomniinfer.backend.llama_cpp=true -Pomniinfer.backend.mnn=false \
  -Pomniinfer.backend.executorch_qnn=false -Pomniinfer.backend.litert_lm=false \
  -Pomniinfer.backend.llama_cpp_htp=true \
  -Pomniinfer.maven.version=<version>
```

The bundle task re-opens the finished zip and re-verifies every entry against
the manifest before publishing it, and emits `<name>.zip.sha256` for the
download step. The per-lib `coreLibs` order is not optional: `libomp.so` must
precede `libggml-base.so` and `libllama.so` must precede `libllama-common.so`
(the order was verified against the `DT_NEEDED` entries of the packaged libs).

## Host App Integration

### Gradle Setup

Consume the lite AAR as a normal Maven coordinate (repositories per
[aar-integration](./aar-integration.md#gradle-setup)):

```kotlin
dependencies {
    implementation("io.github.omnimind-ai:omniinfer:<version>")
}
```

The APK contains the SDK dex and its Maven dependencies but zero native
inference libraries.

### Download And Install

1. Fetch `<name>.zip.sha256` from your distribution point (HTTPS), then stream
   the zip while hashing it; compare before extracting.
2. Extract into a staging directory under app-private storage, then rename it
   into place atomically. Reject any zip entry whose canonical path escapes the
   target directory. Never extract to external storage: it is `noexec` and
   world-visible.
3. Verify and load:

```kotlin
try {
    val manifest = OmniInferEngineLoader.install(engineDir, verifyHashes = true)
    // engine libs are now loaded; OmniInferServer routes through them
} catch (error: OmniInferEngineException) {
    // message tells the user which step failed; delete engineDir and re-download
}
```

`install` validates `formatVersion`, `interfaceVersion`, `abi` against
`Build.SUPPORTED_ABIS`, and `minSdk` against the device, then checks every lib
size and SHA-256 and rejects unlisted `.so` files, before loading the
`coreLibs` chain.

### Load A Model

No new API: after `install`, use
[`OmniInferServer`](./aar-integration.md#load-a-model) exactly as in the
bundled AAR flow. `OmniInferServer` automatically passes the engine lib dir to
the JNI backend; before `install` it falls back to `applicationInfo.nativeLibraryDir`.

```kotlin
OmniInferServer.init(applicationContext)
OmniInferServer.loadModel(
    modelPath = modelPath,
    options = OmniInferLoadOptions(backend = "llama.cpp-htp", port = 9099, nCtx = 4096),
)
```

## Rules And Gotchas

- **One engine per process.** Loaded native libs cannot be unloaded while ggml
  backends may have registered. Installing a different engine version requires
  an app restart; `install` throws `OmniInferEngineException` if another
  package is already active.
- **Google Play policy.** Apps distributed on Google Play must not download
  executable code from outside Play. This flow is for private, enterprise, or
  non-Play channels; on Play use dynamic feature delivery instead.
- **Store the engine in app-private storage** (`context.getDir(...,
  MODE_PRIVATE)`), never `/sdcard` or any world-writable path.
- **Verify before load**: zip-level SHA-256 from the distribution point,
  plus the per-lib SHA-256 the loader performs. Keep the `.sha256` files next
  to the zip on the server.
- **Page size**: llama.cpp libs built with NDK r27+ are 16 KB-page aligned;
  the Snapdragon HTP/OpenCL prebuilt libs in the current package are 4 KB
  aligned and therefore limited to 4 KB-page devices. Devices with 16 KB
  kernels need a rebuilt prebuilt set (tracked as a follow-up).
- **Download UX**: use `WorkManager` for resumable downloads, check free space
  (`StatFs`) before starting, and declare the `dataSync` foreground service
  type for long background downloads on Android 14+.
- **Keep SDK and engine versions in lockstep.** `interfaceVersion` changes are
  breaking: the loader refuses a package whose interface version differs from
  the SDK's.
