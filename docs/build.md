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

`scripts/platforms/windows/build-release.ps1` builds the Windows portable package.

- It can prebuild any of the Windows runtimes above before packaging
- It packages only the requested backends when `-Backends` is provided; otherwise it packages every built backend under `.local/runtime/windows`
- It builds `omniinfer.exe` with PyInstaller `--onedir` and excludes optional heavyweight scientific/image stacks such as MKL, NumPy, SciPy, PIL, Torch, pandas, and OpenCV from the CLI bundle
- It writes PowerShell and cmd.exe launchers; use `omniinfer.ps1` from PowerShell and keep `omniinfer.cmd` for cmd.exe compatibility

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

The portable package exposes `omniinfer` as the real CLI binary.

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
- Use Python `3.10+`
- The Python interpreter that launches OmniInfer must be able to import `mlx`, `mlx_lm`, `mlx_vlm`, `torch`, and `torchvision`
- Runtime dependencies are listed in [`scripts/platforms/macos/mlx-mac/requirements.txt`](../scripts/platforms/macos/mlx-mac/requirements.txt)
- A dedicated `conda` environment such as `mlx` is recommended
- If you use a custom interpreter, set `OMNIINFER_PYTHON` so the CLI and auto-started gateway use the same Python

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

Example custom Python environment:

```bash
export OMNIINFER_PYTHON="$HOME/miniconda3/envs/mlx/bin/python"
./omniinfer backend list
```

On macOS source checkouts, `./omniinfer` also auto-prefers `.local/runtime/macos/mlx-mac/venv/bin/python3` when that runtime venv exists.

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
bash ./scripts/platforms/macos/build-release.sh --backends llama.cpp-mac,mlx-mac
bash ./scripts/platforms/macos/build-release.sh --backend llama.cpp-mac --backend turboquant-mac
```

Use strict mode when you want packaging to fail instead of building missing runtimes:

```bash
bash ./scripts/platforms/macos/build-release.sh --backends llama.cpp-mac --no-build-missing
```

To build and package every supported macOS backend:

```bash
bash ./scripts/platforms/macos/build-release.sh --all-supported
```

The release exposes `omniinfer` as the user-facing CLI entrypoint. The packaging
mode depends on the selected backends:

- Releases without `mlx-mac` build `omniinfer` as a PyInstaller binary.
- Releases with `mlx-mac` skip PyInstaller and install `omniinfer` as a launcher
  that runs `omniinfer.py` with `runtime/mlx-mac/venv/bin/python3`, so the embedded
  MLX backend can import its Python runtime packages.

For `mlx-mac` releases, the packaging script creates a fresh venv inside the
release package and installs `scripts/platforms/macos/mlx-mac/requirements.txt`.
The default `--mlx-env-manager auto` mode creates a copied standard-library venv
and uses `uv pip install` when `uv` is available, otherwise it falls back to the
standard library `venv` module and pip. Use `--mlx-env-manager uv`,
`--mlx-env-manager venv`, or
`--mlx-env-manager conda-pack` when you need to compare or pin the packaging
strategy. The `conda-pack` strategy creates a temporary conda environment,
installs the same MLX requirements, packs it with `conda-pack`, and unpacks it
into `runtime/mlx-mac/venv`; the launcher runs `conda-unpack` before starting
OmniInfer so prefix relocation happens after extraction. Use `--mlx-python
<path>` when you need to choose a specific Python 3.10 through 3.13 interpreter
for the release environment version. Use `--python-index-url <url>` when a
regional PyPI mirror is needed for MLX dependency downloads. In `conda-pack`
mode, use repeated `--conda-channel <channel>` arguments and
`--conda-override-channels` when conda itself should resolve Python and base
packages from regional Anaconda mirrors.

By default, `mlx-mac` release environments are slimmed after dependency
installation: Python bytecode caches, test trees, pip/wheel, and build-only
Torch headers are removed from the package. Pass `--no-slim` when you need an
unmodified Python environment for debugging.

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
./omniinfer build <backend>
```

The CLI delegates to the matching script under `scripts/platforms/<platform>/<backend>/build.*`, uses a `Release` CMake build, and checks that the backend launcher was produced. This build command is intentionally omitted from packaged releases; release archives are expected to run directly from their bundled `runtime/` directory without local build tools.

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
