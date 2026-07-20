# Building OmniInfer

This guide explains how OmniInfer builds or installs local runtime backends from a source checkout.

## Build Model

Desktop build scripts follow one consistent layout:

- `scripts/platforms/<platform>/build-*.{sh,ps1}`
  Stable user-facing entrypoints committed to Git.
- `scripts/platforms/<platform>/<backend>/build.{sh,ps1}`
  Backend-specific implementation scripts.
- `.local/runtime/<platform>/<backend>/`
  Local runtime output directories for binaries, logs, models, and intermediate build files.

Portable releases are separate from source builds:

- `runtime/`
  Runtime tree inside a packaged release.
- `release/portable/`
  Release output area used by platform packaging scripts.

Mobile builds do not use `scripts/platforms/`. Android is implemented by the
Gradle module under `android/`, and iOS is implemented by the Swift package and
native bridge under `ios/`.

## Common Prerequisites

For all desktop builds:

- Git
- Python 3
- `cmake`
- `framework/llama.cpp` available for every `llama.cpp-*` backend

Additional framework notes:

- `framework/llama-cpp-turboquant` is required for `turboquant-mac`
- `mlx-mac` is embedded and uses Python packages instead of building `framework/mlx`
- `framework/vllm` pins the upstream vLLM source release used for provenance and future source-build work
- `vllm-linux-cuda` installs vLLM Python wheels into an OmniInfer-managed local venv by default, which matches vLLM's normal binary distribution path

Submodule behavior:

- Linux `llama.cpp-*` scripts can bootstrap `framework/llama.cpp` automatically unless you pass `--no-bootstrap`
- macOS `llama.cpp-*` scripts can bootstrap `framework/llama.cpp` automatically unless you pass `--no-bootstrap`
- macOS `turboquant-mac` can bootstrap `framework/llama-cpp-turboquant` automatically unless you pass `--no-bootstrap`
- Windows `llama.cpp-*` scripts do not bootstrap submodules automatically; initialize `framework/llama.cpp` first if it is missing
- `vllm-linux-cuda` does not bootstrap `framework/vllm` during normal wheel installation

Example:

```bash
git submodule update --init --recursive framework/llama.cpp
```

## Prebuilt Runtime Installs

The supported user-facing prebuilt runtime install command is implemented in Rust:

```bash
./omniinfer backend install <backend>
```

For example:

```bash
./omniinfer backend install llama.cpp-linux
```

`./omniinfer build <backend>` and `./omniinfer build <backend> --prebuilt` are retained as compatibility aliases for the same prebuilt installer. Source builds are explicit and require a source checkout:

```bash
./omniinfer build <backend> --from-source
```

The Rust installer owns multi-asset download, pinned SHA256 verification, staged extraction, required-file validation, atomic activation, and manifest writing. Source build scripts still own compilation from checked-out submodules. Shared llama.cpp release URLs live in `scripts/prebuilt_backends.json`, but a backend is only offered as prebuilt when that catalog contains a matching entry for the current platform.

Prebuilt versioning is explicit:

- `scripts/prebuilt_backends.json` schema 3 keeps each upstream release tag and expected submodule commit once under `sources`, while platform entries record primary archives, companion assets, pinned SHA256 values, required runtime files, and launcher names.
- A prebuilt llama.cpp runtime is an upstream release artifact. It is only source-aligned when the catalog tag and `framework/llama.cpp` submodule are pinned to the same upstream release tag or commit.
- If a source checkout has a different `framework/llama.cpp` commit than the catalog entry, the Rust installer prints a version note and records the catalog metadata in `prebuilt.json`.
- If no official asset exists, leave the catalog entry absent. For example, llama.cpp `b9500` publishes Linux CPU, ROCm, Vulkan, OpenVINO, macOS, and Windows CUDA assets, but not a Linux CUDA archive.
- Each prebuilt install writes `.local/runtime/<platform>/<backend>/prebuilt.json` with the source tag and all downloaded URLs and digests.
- Windows `llama.cpp-cuda` requires the matching llama.cpp CUDA runtime companion asset. The three required CUDA DLLs are validated before activation, and an incomplete older install is repaired on the next `backend install` invocation.
- Existing `prebuilt.json` archive digests are compared with newly pinned catalog digests before an installed runtime is accepted. A mismatched or malformed managed manifest triggers a transactional reinstall; an unmanaged/source-built runtime without `prebuilt.json` is not overwritten merely because it exists.

Validate catalog structure, URL/tag consistency, and complete SHA256 coverage before committing:

```bash
python scripts/update_prebuilt_catalog.py check
```

When a future llama.cpp submodule update should move the prebuilt release at the same time, update the gitlink first and run the catalog updater. It fetches the named official GitHub Release, updates every primary and companion URL/digest for that source, and records the gitlink commit once:

```bash
git submodule update --init framework/llama.cpp
# Move framework/llama.cpp to the reviewed upstream release commit and stage the gitlink.
python scripts/update_prebuilt_catalog.py update \
  --source ggml-org/llama.cpp \
  --tag bNNNN \
  --submodule-commit current
python scripts/update_prebuilt_catalog.py check --require-gitlink-match
```

Do not use the update command for a submodule-only development commit that has no matching official prebuilt Release. In that case, retain the existing catalog source metadata so the installer continues to report the intentional source/prebuilt mismatch.

## Runtime Output Layout

Current desktop runtime directories:

- Windows x64 CPU: `.local/runtime/windows/llama.cpp-cpu`
- Windows x64 CUDA: `.local/runtime/windows/llama.cpp-cuda`
- Windows x64 Vulkan: `.local/runtime/windows/llama.cpp-vulkan`
- Windows arm64 CPU: `.local/runtime/windows/llama.cpp-windows-arm64`
- Windows x64 SYCL: `.local/runtime/windows/llama.cpp-sycl`
- Windows x64 HIP: `.local/runtime/windows/llama.cpp-hip`
- Linux x64 CPU: `.local/runtime/linux/llama.cpp-linux`
- Linux x64 ROCm: `.local/runtime/linux/llama.cpp-linux-rocm`
- Linux x64 Vulkan: `.local/runtime/linux/llama.cpp-linux-vulkan`
- Linux s390x CPU: `.local/runtime/linux/llama.cpp-linux-s390x`
- Linux x64 OpenVINO: `.local/runtime/linux/llama.cpp-linux-openvino`
- Linux x64 vLLM CUDA: `.local/runtime/linux/vllm-linux-cuda`
- macOS Apple Silicon Metal: `.local/runtime/macos/llama.cpp-mac`
- macOS Intel x64 CPU: `.local/runtime/macos/llama.cpp-mac-intel`
- macOS TurboQuant: `.local/runtime/macos/turboquant-mac`
- macOS MLX embedded runtime: `.local/runtime/macos/mlx-mac`

Typical subfolders:

- `bin/`
- `logs/`
- `models/`
- `build/`

## Windows

### Available Scripts

- `scripts/platforms/windows/build-llama-cpu.ps1`
- `scripts/platforms/windows/build-llama-cuda.ps1`
- `scripts/platforms/windows/build-llama-vulkan.ps1`
- `scripts/platforms/windows/build-llama-arm64.ps1`
- `scripts/platforms/windows/build-llama-sycl.ps1`
- `scripts/platforms/windows/build-llama-hip.ps1`
- `scripts/platforms/windows/build-release.ps1`

### Backend Notes

`llama.cpp-cpu`:

- Target: Windows x64 CPU
- Toolchains: Visual Studio 2022 Build Tools, MSYS2 UCRT64, or MinGW POSIX

`llama.cpp-cuda`:

- Target: Windows x64 CUDA
- Requires: NVIDIA CUDA Toolkit, `nvcc`, and MSVC `cl.exe`

`llama.cpp-vulkan`:

- Target: Windows x64 Vulkan
- Requires: Vulkan SDK or equivalent MSYS2 Vulkan toolchain

`llama.cpp-windows-arm64`:

- Target: Windows arm64 CPU
- Uses llama.cpp's `cmake/arm64-windows-llvm.cmake`
- Best run from a Visual Studio 2022 Developer PowerShell with arm64-capable LLVM/MSVC tooling available

`llama.cpp-sycl`:

- Target: Windows x64 SYCL
- Requires: Intel oneAPI compiler/runtime environment
- Best run from the Intel oneAPI command prompt after `setvars.bat`

`llama.cpp-hip`:

- Target: Windows x64 HIP
- Requires: AMD HIP SDK / ROCm for Windows
- Optional: pass `-GpuTargets` to tune for a specific AMD GPU architecture

### Build Commands

Windows x64 CPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1
```

Windows x64 CUDA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1
```

Windows x64 Vulkan:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-vulkan.ps1
```

Windows arm64 CPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-arm64.ps1
```

Windows x64 SYCL:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-sycl.ps1
```

Windows x64 HIP:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-hip.ps1
```

Optional HIP GPU target override:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-hip.ps1 -GpuTargets gfx1151
```

### Dry-Run Checks

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-arm64.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-sycl.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-hip.ps1 -DryRun
```

### Windows Packaging Wrapper

`scripts/platforms/windows/build-release.ps1` builds the Windows portable package.

- It can prebuild any of the Windows runtimes above before packaging
- It packages only the requested backends when `-Backends` is provided; otherwise it packages every built backend under `.local/runtime/windows`
- It builds the Rust control-plane CLI with `cargo build --release -p omniinfer-cli` and installs it as `omniinfer.exe`
- It packages the Rust control-plane launcher only; Python control-plane fallback is not packaged
- It writes PowerShell and cmd.exe launchers; use `omniinfer.ps1` from PowerShell and keep `omniinfer.cmd` for cmd.exe compatibility

## Linux

### Available Scripts

- `scripts/platforms/linux/build-llama-cpu.sh`
- `scripts/platforms/linux/build-llama-rocm.sh`
- `scripts/platforms/linux/build-llama-vulkan.sh`
- `scripts/platforms/linux/build-llama-s390x.sh`
- `scripts/platforms/linux/build-llama-openvino.sh`
- `scripts/platforms/linux/vllm-linux-cuda/build.sh`
- `scripts/platforms/linux/build-release.sh`

### Backend Notes

Linux backend script behavior:

| Backend | Default action | Source build action |
|---|---|---|
| `llama.cpp-linux` | Downloads official `b9500` Linux CPU archive | `--from-source` builds `framework/llama.cpp` with CPU settings |
| `llama.cpp-linux-rocm` | Downloads official `b9500` ROCm archive | `--from-source` builds `framework/llama.cpp` with ROCm settings |
| `llama.cpp-linux-vulkan` | Downloads official `b9500` Vulkan archive | `--from-source` builds `framework/llama.cpp` with Vulkan settings |
| `llama.cpp-linux-s390x` | Downloads official `b9500` s390x archive | `--from-source` builds `framework/llama.cpp` for s390x |
| `llama.cpp-linux-openvino` | Downloads official `b9500` OpenVINO archive | `--from-source` builds `framework/llama.cpp` with OpenVINO settings |
| `llama.cpp-linux-cuda` | Fails with a clear "no prebuilt configured" message because upstream `b9500` has no Linux CUDA archive | `--from-source` builds `framework/llama.cpp` with CUDA settings |
| `vllm-linux-cuda` | Creates an OmniInfer-managed venv and installs vLLM wheels | Not a C++ source build path |
| `mnn-linux` | Creates an OmniInfer-managed venv and installs the official `MNN==3.5.0` wheel | `--from-source` builds PyMNN from `framework/mnn` |
| `ik_llama.cpp-linux` | Fails with a clear "no prebuilt configured" message | `--from-source` builds `framework/ik_llama.cpp` CPU |
| `ik_llama.cpp-linux-cuda` | Fails with a clear "no prebuilt configured" message | `--from-source` builds `framework/ik_llama.cpp` CUDA |
| `omniinfer-native-linux` | Fails with a clear "no prebuilt configured" message | `--from-source` builds `framework/omniinfer-native` |

`llama.cpp-linux`:

- Target: Linux x64 CPU

`llama.cpp-linux-rocm`:

- Target: Linux x64 ROCm
- Requires: ROCm userspace plus `hipcc` / `hipconfig`

`llama.cpp-linux-vulkan`:

- Target: Linux x64 Vulkan
- Requires: Vulkan loader, headers, and shader tooling

`llama.cpp-linux-s390x`:

- Target: Linux s390x CPU
- Uses the same `llama-server` flow as other Linux CPU builds but writes to its own runtime directory

`llama.cpp-linux-openvino`:

- Target: Linux x64 OpenVINO
- Requires: OpenVINO runtime and build environment
- The script accepts `--openvino-root` and also honors `OPENVINO_ROOT`
- Runtime selection remains controlled at inference time with environment variables such as `GGML_OPENVINO_DEVICE=CPU` or `GGML_OPENVINO_DEVICE=GPU`

`vllm-linux-cuda`:

- Target: Linux x64 CUDA
- Uses the official vLLM OpenAI-compatible server through `vllm serve`
- Installs into `.local/runtime/linux/vllm-linux-cuda` without `sudo`
- Requires a CUDA-capable NVIDIA GPU and a vLLM-compatible Python/PyTorch wheel stack
- Accepts HuggingFace model IDs, local snapshot directories, or other model references that vLLM can load

`mnn-linux`:

- Target: Linux embedded PyMNN runtime
- Default path installs the official `MNN==3.5.0` Python wheel into `.local/runtime/linux/mnn-linux/venv`
- The MNN GitHub release also publishes a Linux x64 CPU/OpenCL zip, but OmniInfer's embedded driver needs the Python modules `MNN`, `MNN.cv`, and `MNN.llm`, so the wheel is the default prebuilt path
- `--from-source` keeps the previous PyMNN source-build path and is required for `--opencl` / `--cuda`

### Build Commands

Linux x64 CPU:

```bash
bash ./scripts/platforms/linux/build-llama-cpu.sh
```

Linux x64 ROCm:

```bash
bash ./scripts/platforms/linux/build-llama-rocm.sh
```

Linux x64 Vulkan:

```bash
bash ./scripts/platforms/linux/build-llama-vulkan.sh
```

Linux s390x CPU:

```bash
bash ./scripts/platforms/linux/build-llama-s390x.sh
```

Linux x64 OpenVINO:

```bash
bash ./scripts/platforms/linux/build-llama-openvino.sh --openvino-root /opt/intel/openvino
```

Linux x64 vLLM CUDA:

```bash
bash ./scripts/platforms/linux/vllm-linux-cuda/build.sh --smoke-test
```

Pin a specific vLLM wheel when reproducibility matters:

```bash
bash ./scripts/platforms/linux/vllm-linux-cuda/build.sh --package 'vllm==0.9.2'
```

### Linux Portable Packaging

`scripts/platforms/linux/build-release.sh` can now package any Linux runtime directories that already exist locally, including:

- `llama.cpp-linux`
- `llama.cpp-linux-cuda`
- `llama.cpp-linux-rocm`
- `llama.cpp-linux-vulkan`
- `llama.cpp-linux-s390x`
- `llama.cpp-linux-openvino`
- `ik_llama.cpp-linux`
- `ik_llama.cpp-linux-cuda`
- `vllm-linux-cuda`
- `mnn-linux`

Runtime discovery is driven by the Linux backend registry rather than a
hard-coded `llama-server` filename check. External server backends are packaged
when their registered launcher exists under `bin/`. Embedded Python runtimes
such as `mnn-linux` are rejected until they are exposed through an adapter
service or Rust-native driver.
`llama.cpp` and `ik_llama.cpp` backends copy the minimal binary `bin/` payload,
while external-server Python runtime backends such as `vllm-linux-cuda` copy
the runtime environment needed by their launcher.

The portable package exposes a single user-facing `omniinfer` Rust
control-plane binary. Packaging builds the CLI with
`cargo build --release -p omniinfer-cli`. Embedded backend paths that still need
an in-process Python control plane, such as `mnn-linux`, are rejected until they
have an adapter service or Rust-native driver.

Optional prebuild examples:

```bash
bash ./scripts/platforms/linux/build-release.sh --build-cpu-backend --build-vulkan-backend
bash ./scripts/platforms/linux/build-release.sh --build-openvino-backend --openvino-root /opt/intel/openvino
```

## macOS

### Available Scripts

- `scripts/platforms/macos/build-llama-mac.sh`
- `scripts/platforms/macos/build-llama-mac-intel.sh`
- `scripts/platforms/macos/build-turboquant-mac.sh`
- `scripts/platforms/macos/build-mlx-mac.sh`
- `scripts/platforms/macos/build-release.sh`

### Backend Notes

`llama.cpp-mac`:

- Target: macOS Apple Silicon
- Uses Metal acceleration

`llama.cpp-mac-intel`:

- Target: macOS Intel x64
- Aligned with the official llama.cpp macOS x64 release direction
- Uses a CPU-oriented x64 build and disables Metal by default

`turboquant-mac`:

- Target: macOS Apple Silicon
- Built from `framework/llama-cpp-turboquant`

`mlx-mac`:

- Target: macOS Apple Silicon
- Embedded backend
- Requires Python packages rather than a compiled `llama-server`
- Not supported by Rust-only portable packaging until an adapter service or
  Rust-native driver exists
- Use Python `3.10+`
- The MLX backend environment must be able to import `mlx`, `mlx_lm`,
  `mlx_vlm`, `torch`, and `torchvision`
- Runtime dependencies are listed in [`scripts/platforms/macos/mlx-mac/requirements.txt`](../scripts/platforms/macos/mlx-mac/requirements.txt)
- A dedicated `conda` environment such as `mlx` is recommended

### Build Commands

macOS Apple Silicon Metal:

```bash
bash ./scripts/platforms/macos/build-llama-mac.sh
```

macOS Intel x64:

```bash
bash ./scripts/platforms/macos/build-llama-mac-intel.sh
```

TurboQuant:

```bash
bash ./scripts/platforms/macos/build-turboquant-mac.sh
```

MLX:

```bash
bash ./scripts/platforms/macos/build-mlx-mac.sh
```

MLX can still be built as a backend environment, but it is not exposed through
Rust-only release packages until the embedded driver is replaced by an adapter
service or Rust-native driver.

### macOS Portable Packaging

`scripts/platforms/macos/build-release.sh` packages a macOS portable release under:

- `release/portable/macos-arm64/OmniInfer` on Apple Silicon
- `release/portable/macos-x64/OmniInfer` on Intel

By default, the script packages already-built macOS runtimes from `.local/runtime/macos`.
If no macOS runtime exists yet, it builds the host default backend first:

- Apple Silicon: `llama.cpp-mac`
- Intel: `llama.cpp-mac-intel`

You can package only selected backends. Requested backends are reused when already
present, and missing requested backends are built through their normal backend build
scripts before packaging:

```bash
bash ./scripts/platforms/macos/build-release.sh --backend llama.cpp-mac --backend turboquant-mac
```

The default output directory is the stable package root:

- `release/portable/macos-arm64/OmniInfer`
- `release/portable/macos-x64/OmniInfer`

The script does not create a zip archive by default. Pass `--archive` when you
need a distributable zip; the archive name is generated from the package,
platform, and selected backend set, for example
`OmniInfer-macos-arm64-llama-mlx.zip`.

Use strict mode when you want packaging to fail instead of building missing runtimes:

```bash
bash ./scripts/platforms/macos/build-release.sh --backends llama.cpp-mac --no-build-missing
```

`--all-supported` and any explicit `mlx-mac` selection currently fail because
embedded backends are not supported by the Rust-only portable package.

The release exposes a single user-facing `omniinfer` Rust control-plane binary.
Packaging builds the CLI with `cargo build --release -p omniinfer-cli`.
`mlx-mac` is rejected by release packaging until it is served through an adapter
service or Rust-native driver.

## Android

Android is built from the root `android/` module, not from `scripts/platforms`.
Use the Android integration docs for embedding and backend setup:

- [Android App Integration](android/integration.md)
- [Android Backend Reference](android/backends.md)
- [Android Multimodal Guide](android/multimodal.md)

## iOS

iOS is built from the root `ios/` implementation:

- `ios/OmniInferServer/`
  Swift Package facade and in-process HTTP service.
- `ios/native/`
  Native bridge sources for the embedded inference backends.

The legacy iOS script helper has been removed.

## After Building

From a source checkout on Linux, macOS, or Windows, the shortest path for desktop backends is:

```bash
./omniinfer backend install <backend>
```

The CLI installs a configured prebuilt runtime and checks that the backend launcher was produced. Use `./omniinfer build <backend> --from-source` only when you intentionally want to compile the checked-out runtime source with local build tools.

List the local backends:

```bash
./omniinfer backend list
```

Select one backend:

```bash
./omniinfer backend select llama.cpp-linux-vulkan
```

Load a model:

```bash
./omniinfer model load -m /absolute/path/to/model.gguf
```

For more runtime usage examples, see [CLI Guide](CLI.md).
