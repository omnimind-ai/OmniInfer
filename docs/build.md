# Building OmniInfer

This guide explains how OmniInfer builds local runtime backends from a source checkout.

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

Android uses a direct-mode runtime installer instead of the desktop-style backend-per-runtime build layout.

## Common Prerequisites

For all desktop builds:

- Git
- Python 3
- `cmake`
- `framework/llama.cpp` available for every `llama.cpp-*` backend

Additional framework notes:

- `framework/llama-cpp-turboquant` is required for `turboquant-mac`
- `mlx-mac` is embedded and uses Python packages instead of building `framework/mlx`

Submodule behavior:

- Linux `llama.cpp-*` scripts can bootstrap `framework/llama.cpp` automatically unless you pass `--no-bootstrap`
- macOS `llama.cpp-*` scripts can bootstrap `framework/llama.cpp` automatically unless you pass `--no-bootstrap`
- macOS `turboquant-mac` can bootstrap `framework/llama-cpp-turboquant` automatically unless you pass `--no-bootstrap`
- Windows `llama.cpp-*` scripts do not bootstrap submodules automatically; initialize `framework/llama.cpp` first if it is missing

Example:

```bash
git submodule update --init --recursive framework/llama.cpp
```

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

`scripts/platforms/windows/build-release.ps1` is a wrapper around the repository's Windows portable packager.

- It can prebuild any of the Windows runtimes above before packaging
- It still expects a repo-specific `release/build_portable.ps1` packager to exist

## Linux

### Available Scripts

- `scripts/platforms/linux/build-llama-cpu.sh`
- `scripts/platforms/linux/build-llama-rocm.sh`
- `scripts/platforms/linux/build-llama-vulkan.sh`
- `scripts/platforms/linux/build-llama-s390x.sh`
- `scripts/platforms/linux/build-llama-openvino.sh`
- `scripts/platforms/linux/build-release.sh`

### Backend Notes

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

### Linux Portable Packaging

`scripts/platforms/linux/build-release.sh` can now package any Linux runtime directories that already exist locally, including:

- `llama.cpp-linux`
- `llama.cpp-linux-rocm`
- `llama.cpp-linux-vulkan`
- `llama.cpp-linux-s390x`
- `llama.cpp-linux-openvino`

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

## Android

### Available Scripts

- `scripts/platforms/android/build-runtime.sh`
- `scripts/platforms/android/runtime/install.sh`
- `scripts/platforms/android/runtime/omniinfer-android`

### What The Android Script Does

Android does not build or launch the desktop HTTP gateway.

Instead, the Android script prepares a direct-mode runtime tree under:

- `.local/runtime/android/bin/omniinfer-android`
- `.local/runtime/android/lib/arm64-v8a/libllama-cli.so`
- `.local/runtime/android/lib/arm64-v8a/libmtmd-cli.so`

The installer always writes the launcher. It can also copy prebuilt Android binaries from an artifact directory or explicit paths.

For `omniinfer-native`, the Android runtime can also host the Qualcomm ExecuTorch runners and QNN shared libraries under `.local/runtime/android/qnn/`. If your model package uses the official ExecuTorch Qualcomm llama multimodal flow, include `qnn_multimodal_runner` in that bundle as well.

The recommended model-package format for official ExecuTorch Qualcomm llama artifacts is a model directory that contains:

- `omniinfer-native.env`
- `tokenizer.json` or another runtime tokenizer file referenced by the manifest
- one or more `.pte` files such as `hybrid_llama_qnn.pte`, `vision_encoder_qnn.pte`, `tok_embedding_qnn.pte`, or `attention_sink_evictor.pte`

`omniinfer-native.env` is a simple shell-style manifest consumed by the Android launcher. It lets OmniInfer map one model directory to the correct runner and artifact set without hard-coding file-name guesses in the CLI.

### Prepare The Android Runtime

Launcher only:

```bash
bash ./scripts/platforms/android/build-runtime.sh --launcher-only
```

Install launcher plus prebuilt Android binaries from a directory:

```bash
bash ./scripts/platforms/android/build-runtime.sh --artifact-dir /path/to/android/artifacts
```

Install with explicit binary paths:

```bash
bash ./scripts/platforms/android/build-runtime.sh \
  --llama-cli /path/to/libllama-cli.so \
  --mtmd-cli /path/to/libmtmd-cli.so
```

Dry run:

```bash
bash ./scripts/platforms/android/build-runtime.sh --artifact-dir /path/to/android/artifacts --dry-run
```

After the runtime is prepared, the repo-root `./omniinfer` script will detect Android automatically and forward commands into `.local/runtime/android/bin/omniinfer-android`.

## After Building

List the local backends:

```bash
./omniinfer backend list
```

Select one backend:

```bash
./omniinfer select llama.cpp-linux-vulkan
```

Load a model:

```bash
./omniinfer model load -m /absolute/path/to/model.gguf
```

For more runtime usage examples, see [CLI Guide](CLI.md).
