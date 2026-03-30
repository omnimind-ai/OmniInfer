# Building OmniInfer

This guide explains how to build OmniInfer runtime backends and package local releases on Windows, Linux, macOS, and Android.

## Build Model

OmniInfer uses the same high-level structure across desktop platforms:

- `scripts/platforms/<platform>/...`
  Tracked build and packaging scripts committed to Git.
- `.local/runtime/<platform>/...`
  Local runtime output directories. Build artifacts are written here, but the directory is intentionally not tracked in Git.
- `release/portable/...`
  Locally packaged portable releases.

The tracked scripts are split into two layers:

- Platform entry scripts
  Stable entrypoints such as `scripts/platforms/windows/build-llama-cpu.ps1`.
- Backend implementation scripts
  Backend-specific build logic such as `scripts/platforms/windows/llama.cpp-cpu/build.ps1`.

## Common Prerequisites

For all platforms:

- Git
- Python 3
- `framework/llama.cpp` available in the repository

If the `framework/llama.cpp` submodule is missing, the Linux and macOS backend scripts can bootstrap it automatically unless you pass `--no-bootstrap`.

## Output Conventions

Backend builds emit runtime files into local platform folders:

- Windows CPU: `.local/runtime/windows/llama.cpp-cpu`
- Windows CUDA: `.local/runtime/windows/llama.cpp-cuda`
- Windows Vulkan: `.local/runtime/windows/llama.cpp-vulkan`
- Linux CPU: `.local/runtime/linux/llama.cpp-linux`
- Linux ROCm: `.local/runtime/linux/llama.cpp-linux-rocm`
- macOS Metal: `.local/runtime/macos/llama.cpp-mac`
- Android CLI assets: `.local/runtime/android`

OmniInfer still falls back to the legacy `platform/<Platform>/...` layout if it already exists locally, but new builds should use `.local/runtime/<platform>/...`.

Typical runtime subfolders:

- `bin/`
- `logs/`
- `models/`
- `build/`

## Windows

### Scripts

- `scripts/platforms/windows/build-llama-cpu.ps1`
  Build the Windows CPU backend.
- `scripts/platforms/windows/build-llama-cuda.ps1`
  Build the Windows CUDA backend.
- `scripts/platforms/windows/build-llama-vulkan.ps1`
  Build the Windows Vulkan backend.
- `scripts/platforms/windows/build-release.ps1`
  Package a Windows portable release.

### Prerequisites

CPU backend:

- `cmake`
- One supported Windows C/C++ toolchain:
  - Visual Studio 2022 Build Tools with the C++ workload, or
  - MSYS2 UCRT64 with `gcc`, `g++`, and `ninja`, or
  - MinGW POSIX with `gcc`, `g++`, and `mingw32-make`

CUDA backend:

- `cmake`
- NVIDIA CUDA Toolkit with `nvcc`
- MSVC `cl.exe` available in `PATH`

Vulkan backend:

- `cmake`
- One supported Windows C/C++ toolchain:
  - Visual Studio 2022 Build Tools with the C++ workload, or
  - MSYS2 UCRT64 with `gcc`, `g++`, `ninja`, `vulkan-devel`, and `shaderc`, or
  - MinGW POSIX with `gcc`, `g++`, `mingw32-make`, and a Vulkan SDK that exposes `glslc.exe`
- A Windows Vulkan SDK install, or an MSYS2 environment that already provides Vulkan headers/libs and `glslc`

Recommended:

- Run CUDA builds from a Visual Studio 2022 Developer PowerShell.

### Build The CPU Backend

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1
```

Optional:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1 -BuildType Release
```

Expected output:

- `.local/runtime/windows/llama.cpp-cpu/bin/llama-server.exe`
- Required runtime DLLs copied into the same directory when needed

### Build The CUDA Backend

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1
```

Optional architecture override:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1 -CudaArchitectures 86
```

Expected output:

- `.local/runtime/windows/llama.cpp-cuda/bin/llama-server.exe`
- CUDA runtime DLLs copied into the same directory when available

### Build The Vulkan Backend

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-vulkan.ps1
```

Optional:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-vulkan.ps1 -BuildType Release
```

Expected output:

- `.local/runtime/windows/llama.cpp-vulkan/bin/llama-server.exe`
- Vulkan backend DLLs copied from the llama.cpp build output into the same directory

### Build A Windows Portable Release

The Windows release build packages:

- `OmniInfer.exe`
- `omniinfer-cli.exe`
- `omniinfer.cmd`
- `runtime/`
- `config/omniinfer.json`
- `usage.md`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1
```

Optional:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -BuildCpuBackend
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -BuildCpuBackend -BuildCudaBackend
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -BuildVulkanBackend
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -BuildCpuBackend -BuildVulkanBackend
```

Expected output:

- `release/portable/OmniInfer`

### Dry-Run Validation

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-vulkan.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -DryRun
```

## Linux

### Scripts

- `scripts/platforms/linux/build-llama-cpu.sh`
  Build the Linux CPU backend.
- `scripts/platforms/linux/build-llama-rocm.sh`
  Build the Linux ROCm backend.
- `scripts/platforms/linux/build-release.sh`
  Package a Linux portable release.

### Prerequisites

CPU backend:

- `cmake`
- A C/C++ compiler toolchain such as `gcc` and `g++`
- `ninja` recommended

ROCm backend:

- `cmake`
- ROCm with `hipcc` and `hipconfig`
- A compatible AMD GPU and working ROCm userspace installation
- `ninja` recommended

Recommended:

- Ensure your user has the correct render and ROCm device permissions before running the ROCm backend.

### Build The Linux CPU Backend

```bash
bash ./scripts/platforms/linux/build-llama-cpu.sh
```

Useful options:

```bash
bash ./scripts/platforms/linux/build-llama-cpu.sh --build-type Release --smoke-test
bash ./scripts/platforms/linux/build-llama-cpu.sh --clean
```

Expected output:

- `.local/runtime/linux/llama.cpp-linux/bin/llama-server`

### Build The Linux ROCm Backend

```bash
bash ./scripts/platforms/linux/build-llama-rocm.sh
```

Useful options:

```bash
bash ./scripts/platforms/linux/build-llama-rocm.sh --gpu-targets gfx1151
bash ./scripts/platforms/linux/build-llama-rocm.sh --rocm-path /opt/rocm
bash ./scripts/platforms/linux/build-llama-rocm.sh --smoke-test
```

Expected output:

- `.local/runtime/linux/llama.cpp-linux-rocm/bin/llama-server`
- Required ROCm shared libraries copied into the same directory

### Build A Linux Portable Release

```bash
bash ./scripts/platforms/linux/build-release.sh
```

Optional:

```bash
bash ./scripts/platforms/linux/build-release.sh --build-cpu-backend
bash ./scripts/platforms/linux/build-release.sh --build-rocm-backend
bash ./scripts/platforms/linux/build-release.sh --build-cpu-backend --build-rocm-backend
```

Expected output:

- `release/portable/OmniInfer`

### Dry-Run Validation

```bash
bash ./scripts/platforms/linux/build-llama-cpu.sh --dry-run
bash ./scripts/platforms/linux/build-llama-rocm.sh --dry-run
bash ./scripts/platforms/linux/build-release.sh --dry-run
```

## macOS

### Scripts

- `scripts/platforms/macos/build-llama-mac.sh`
  Build the macOS Metal backend.

The macOS build scripts use the same two-layer pattern as the other desktop platforms:

- Platform entry: `scripts/platforms/macos/build-llama-mac.sh`
- Backend implementation: `scripts/platforms/macos/llama.cpp-mac/build.sh`

### Prerequisites

- `cmake`
- Xcode Command Line Tools
- A recent Apple Clang toolchain
- `ninja` recommended

Recommended:

- Run on Apple Silicon for the intended Metal backend target.

### Build The macOS Metal Backend

```bash
bash ./scripts/platforms/macos/build-llama-mac.sh
```

Useful options:

```bash
bash ./scripts/platforms/macos/build-llama-mac.sh --build-type Release --smoke-test
bash ./scripts/platforms/macos/build-llama-mac.sh --clean
```

Expected output:

- `.local/runtime/macos/llama.cpp-mac/bin/llama-server`

### Dry-Run Validation

```bash
bash ./scripts/platforms/macos/build-llama-mac.sh --dry-run
```

### Build A macOS Portable Release

The macOS release build packages:

- `OmniInfer` (gateway executable)
- `runtime/llama.cpp-mac/`
- `config/omniinfer.json`
- `release-metadata.json`
- `OmniInfer-macos-<arch>.tar.gz`

Before packaging, build or prepare the macOS runtime binary first:

```bash
bash ./scripts/platforms/macos/build-llama-mac.sh --build-type Release
```

Then package the portable release:

```bash
bash ./release/mac/build_portable.sh
```

Optional package name:

```bash
bash ./release/mac/build_portable.sh OmniInferDev
```

Expected output:

- `release/mac/portable/OmniInfer`
- `release/mac/portable/OmniInfer-macos-<arch>.tar.gz`

### Validate The macOS Release

Run the release validation script:

```bash
python3 ./release/mac/test_release.py --package-dir ./release/mac/portable/OmniInfer
```

This validation checks:

- gateway startup and health endpoint
- backend list/state and `llama.cpp-mac` availability
- control APIs such as thinking toggle, backend stop, and shutdown
- release metadata consistency (`git_commit`, source fingerprint, and packaged `llama-server` hash)

For strict "latest build" checks, build from a clean working tree so `release-metadata.json` reflects the current committed source exactly.

## Android

### Current Repository Model

Android is different from the desktop platforms:

- OmniInfer runs in direct CLI mode on Android instead of starting the local HTTP gateway.
- The repository currently expects Android runtime assets to be prepared and copied into `.local/runtime/android/`.
- There is no tracked Android build wrapper script in `scripts/platforms/android/` yet.

Expected Android runtime layout:

- `.local/runtime/android/bin/omniinfer-android`
- `.local/runtime/android/lib/arm64-v8a/libllama-cli.so`
- `.local/runtime/android/lib/arm64-v8a/libmtmd-cli.so`

For runtime behavior details, see `docs/android-cli.md`.

### Recommended Toolchain

If you want to rebuild the Android artifacts from source, prepare:

- Android NDK
- CMake
- An Android-capable `clang` toolchain from the NDK
- Target ABI: `arm64-v8a`

Recommended:

- Keep Android-specific native build logic in a dedicated Android project or native packaging pipeline, then copy the final artifacts into this repository's expected `.local/runtime/android/` layout.

### Prepare Android Runtime Assets

After producing the Android-native binaries in your Android build pipeline, copy them into:

```text
.local/runtime/android/bin/omniinfer-android
.local/runtime/android/lib/arm64-v8a/libllama-cli.so
.local/runtime/android/lib/arm64-v8a/libmtmd-cli.so
```

Make sure the launcher script is executable:

```bash
chmod +x .local/runtime/android/bin/omniinfer-android
```

### Validate The Android CLI Layout

On the Android device:

```bash
./omniinfer backend list
./omniinfer select llama.cpp-llama
./omniinfer model load -m /absolute/path/to/model.gguf
./omniinfer chat --message "Hello"
```

## Best Practices

- Treat `scripts/platforms/...` as the source of truth for tracked build logic.
- Treat `.local/runtime/...` as disposable local output that can be rebuilt.
- Keep packaging steps separate from backend compilation steps.
- Validate wrapper scripts with dry-run mode before a full rebuild when changing paths or arguments.
- If gateway or CLI behavior changes, update the usage docs and rebuild the affected platform release package.
