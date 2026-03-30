# Building OmniInfer on Windows

This document explains how to build the Windows llama.cpp backends and how to package a Windows portable release.

## Repository layout

- `scripts/platforms/windows/build-llama-cpu.ps1`
  Builds the Windows CPU backend package in `platform/Windows/llama.cpp-cpu/bin`.
- `scripts/platforms/windows/build-llama-cuda.ps1`
  Builds the Windows CUDA backend package in `platform/Windows/llama.cpp-cuda/bin`.
- `scripts/platforms/windows/build-release.ps1`
  Builds a Windows portable release in `release/portable/OmniInfer`.

## 1. Prerequisites

### CPU backend

You need:

- `cmake`
- One supported Windows C/C++ toolchain:
  - Visual Studio 2022 Build Tools with C++ workload, or
  - MSYS2 UCRT64 with `gcc`, `g++`, and `ninja`, or
  - MinGW POSIX toolchain with `gcc`, `g++`, and `mingw32-make`

The source code is expected at:

- `framework/llama.cpp`

### CUDA backend

You need:

- `cmake`
- NVIDIA CUDA Toolkit with `nvcc`
- MSVC `cl.exe` available in `PATH`

Recommended:

- Run the script from a Visual Studio 2022 Developer PowerShell.

## 2. Build the CPU backend

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1
```

Optional:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1 -BuildType Release
```

Expected output:

- `platform/Windows/llama.cpp-cpu/bin/llama-server.exe`
- Required runtime DLLs copied into the same `bin` directory

## 3. Build the CUDA backend

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1
```

Optional architecture override:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1 -CudaArchitectures 86
```

Expected output:

- `platform/Windows/llama.cpp-cuda/bin/llama-server.exe`
- CUDA runtime DLLs copied into the same `bin` directory when available

## 4. Build a Windows portable release

The release build packages:

- `OmniInfer.exe` gateway
- `omniinfer-cli.exe` and `omniinfer.cmd`
- runtime binaries under `runtime/`
- `config/omniinfer.json`
- `usage.md`

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1
```

Output:

- `release/portable/OmniInfer`

If you want to rebuild the CPU backend first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -BuildCpuBackend
```

If you want to rebuild both backends before packaging:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -BuildCpuBackend -BuildCudaBackend
```

## 5. Dry-run checks

To verify the wrapper scripts and command wiring without starting a real build:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cpu.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-llama-cuda.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\platforms\windows\build-release.ps1 -DryRun
```

## 6. Notes

- The release directory is generated locally and can be rebuilt at any time from source.
- Runtime payloads are still emitted under the local `platform/` directory during builds and packaging, but that directory is intentionally not tracked in Git.
- The release build expects `tmp/usage.md` to exist so it can copy the latest usage document into the package.
- If you change gateway behavior, also update the CLI and rebuild the corresponding release package.
