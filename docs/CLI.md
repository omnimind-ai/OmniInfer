# OmniInfer CLI Guide

This guide shows how to use the OmniInfer CLI on Linux, macOS, Windows, and Android.

## Before You Start

If you are running OmniInfer from a source checkout, prepare at least one local runtime backend before using the CLI.

- Windows: build `llama.cpp-cpu`, `llama.cpp-cuda`, or `llama.cpp-vulkan` first. See [Build Guide: Windows](build.md#windows).
- Linux: build `llama.cpp-linux` or `llama.cpp-linux-rocm` first. See [Build Guide: Linux](build.md#linux).
- macOS: build `llama.cpp-mac` first. See [Build Guide: macOS](build.md#macos).
- Android: prepare the Android runtime assets first. See [Build Guide: Android](build.md#android).

If you are using a packaged release that already includes `runtime/`, you can skip this preparation step and jump straight to the CLI commands below.

## Platform Behavior

- Linux, macOS, Windows:
  the CLI talks to the local OmniInfer service and starts it automatically when needed.
- Android:
  the CLI runs in direct mode and calls the Android native backend binaries directly.

## Launch The CLI

Run the CLI after your local runtime has been prepared.

For source checkouts, use the repository root.
For packaged releases, use the release directory that already contains `runtime/`.

Linux and macOS:

```sh
./omniinfer --help
```

Windows:

```powershell
.\omniinfer.cmd --help
```

Android:

```sh
./omniinfer --help
```

## Quick Start

### 1. Check available backends

Linux, macOS, Windows:

```sh
./omniinfer backend list
```

Windows:

```powershell
.\omniinfer.cmd backend list
```

Android:

```sh
./omniinfer backend list
```

### 2. Select a backend

Always pick a backend from `backend list` on your current device.

```sh
./omniinfer select <backend>
```

Windows:

```powershell
.\omniinfer.cmd select <backend>
```

Examples:

- Linux: `llama.cpp-linux` or `llama.cpp-linux-rocm`
- macOS: `llama.cpp-mac`
- Windows: `llama.cpp-cpu`, `llama.cpp-cuda`, or `llama.cpp-vulkan`
- Android: `llama.cpp-llama` or `llama.cpp-mtmd`

### 3. Load a model

Text model:

```sh
./omniinfer model load -m /path/to/model.gguf
```

Windows:

```powershell
.\omniinfer.cmd model load -m C:\path\to\model.gguf
```

Vision-language model:

```sh
./omniinfer model load -m /path/to/model.gguf -mm /path/to/mmproj.gguf
```

### 4. Chat

Text chat:

```sh
./omniinfer chat --message "Introduce yourself in one sentence."
```

Vision-language chat:

```sh
./omniinfer chat \
  --image /path/to/image.jpg \
  --message "Describe this image in one sentence."
```

Windows:

```powershell
.\omniinfer.cmd chat --message "Introduce yourself in one sentence."
```

## Common Commands

```sh
./omniinfer backend list
./omniinfer select <backend>
./omniinfer status
./omniinfer model list
./omniinfer model load -m /path/to/model.gguf
./omniinfer thinking show
./omniinfer thinking set on
./omniinfer chat --message "Hello"
./omniinfer shutdown
./omniinfer completion bash
```

On Windows, replace `./omniinfer` with `.\omniinfer.cmd`.

## Useful Notes

- `select` stores your current backend choice for later runs.
- `model load` stores the current model path, optional `mmproj`, and optional `ctx-size`.
- `chat` streams output by default.
- `status` shows the current backend, model, and thinking state.
- `shutdown` stops the local desktop service. On Android it just confirms that direct mode has no background gateway.

## Platform Notes

### Linux, macOS, Windows

- The CLI uses the Python entrypoint in [omniinfer.py](../omniinfer.py).
- The desktop CLI auto-starts the local OmniInfer gateway when required.
- If you want to run the gateway in the foreground, use:

```sh
./omniinfer serve
```

### Android

- The repo-root [omniinfer](../omniinfer) script detects Android automatically.
- Android direct mode uses the local launcher at `.local/runtime/android/bin/omniinfer-android`.
- Android backend binaries live under `.local/runtime/android/lib/arm64-v8a`.
- For more Android-specific details, see [Android CLI Notes](android-cli.md).
