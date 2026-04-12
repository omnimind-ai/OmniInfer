# OmniInfer CLI Guide

This guide shows how to use the OmniInfer CLI on Linux, macOS, Windows, and Android.

## Before You Start

If you are running OmniInfer from a source checkout, prepare at least one local runtime backend before using the CLI.

- Windows: build one of `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip` first. See [Build Guide: Windows](build.md#windows).
- Linux: build one of `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, or `llama.cpp-linux-openvino` first. See [Build Guide: Linux](build.md#linux).
- macOS: build `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac` first. See [Build Guide: macOS](build.md#macos).
- Android: prepare the Android runtime assets first. See [Build Guide: Android](build.md#android).

If you are using a packaged release that already includes `runtime/`, you can skip this preparation step and jump straight to the CLI commands below.

### macOS `mlx-mac` prerequisites

If you want to use the embedded `mlx-mac` backend from a source checkout:

- Use Python `3.10+`.
- Make sure the Python interpreter that launches OmniInfer can import `mlx`, `mlx_lm`, `mlx_vlm`, `torch`, and `torchvision`.
- The repository includes [`scripts/platforms/macos/mlx-mac/requirements.txt`](../scripts/platforms/macos/mlx-mac/requirements.txt) for that runtime.
- The recommended local setup is your `conda` environment named `mlx`.

Examples:

```sh
export OMNIINFER_PYTHON="$HOME/miniconda3/envs/mlx/bin/python"
./omniinfer backend list
```

or:

```sh
"$HOME/miniconda3/envs/mlx/bin/python" omniinfer.py backend list
```

On macOS source checkouts, `./omniinfer` also auto-prefers `.local/runtime/macos/mlx-mac/venv/bin/python3` when that runtime venv exists.

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

- Linux: `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, or `llama.cpp-linux-openvino`
- macOS: `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac`
- Windows: `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip`
- Android: `llama.cpp-llama`, `llama.cpp-mtmd`, or `omniinfer-native`

When you select a desktop backend, OmniInfer also creates a backend-specific JSON config template under:

- `~/.config/omniinfer/backend_profiles/<backend>.json`

This file is the advanced path for backend-native parameters only.
Keep basic user inputs such as `-m/--model`, `-mm/--mmproj`, `--message`, and `--image` on the CLI.

Example:

```json
{
  "schema_version": 2,
  "backend": "llama.cpp-vulkan",
  "family": "llama.cpp",
  "load": {
    "extra_args": ["-ngl", "99", "-t", "8"]
  },
  "infer": {
    "extra_args": ["--top-k", "40", "--top-p", "0.9"]
  }
}
```

### 3. Load a model

Default path:

```sh
./omniinfer model load -m /path/to/model-directory
```

For `llama.cpp-*`, OmniInfer accepts either a model file or a model directory. If you pass a directory, OmniInfer auto-discovers:

- the main text GGUF
- the optional `mmproj` GGUF

For `mlx-mac`, OmniInfer passes the model directory directly to the embedded backend.

For `omniinfer-native` on Android, OmniInfer accepts either:

- a `.pte` model file
- a model directory that contains `omniinfer-native.env`
- or a compatibility-mode model directory that contains `hybrid_llama_qnn.pte`

In package mode, `omniinfer-native.env` tells OmniInfer which ExecuTorch Qualcomm llama runner to use and which `.pte` artifacts belong to the package.
The recommended way to generate that manifest is:

```sh
bash ./scripts/platforms/android/package-omniinfer-native.sh \
  --artifact-dir /path/to/executorch-artifacts \
  --decoder-model-version qwen3
```

OmniInfer also auto-discovers `tokenizer.json` beside the selected `.pte` file when available.

Explicit file path:

```sh
./omniinfer model load -m /path/to/model.gguf
```

Advanced path with backend config JSON:

```sh
./omniinfer select llama.cpp-vulkan
./omniinfer model load -m /path/to/model-directory --config
```

Windows:

```powershell
.\omniinfer.cmd model load -m C:\path\to\model-directory
```

Vision-language model:

```sh
./omniinfer model load -m /path/to/model.gguf -mm /path/to/mmproj.gguf
```

Android OmniInfer Native QNN:

```sh
./omniinfer select omniinfer-native
./omniinfer model load -m /data/local/tmp/syf/executorch/static_llm
./omniinfer chat --message "你好啊，你是谁？"
```

For `mlx-mac`, use a vision-capable model directory instead of a `.gguf` file or `mmproj` sidecar:

```sh
./omniinfer select mlx-mac
./omniinfer model load -m /path/to/mlx-vlm-model-directory
./omniinfer chat \
  --image /path/to/image.jpg \
  --message "Describe this image in one sentence."
```

The backend config JSON is where advanced users should put backend-native launch parameters such as `-ngl`, `--threads`, and other backend-specific options.

You can also skip `--config` entirely and pass backend-native extra args directly after the stable OmniInfer args. OmniInfer parses those extra args according to the currently selected backend.

Example:

```powershell
.\omniinfer.cmd select llama.cpp-vulkan
.\omniinfer.cmd model load -m C:\models\Qwen3 -ngl 99 -t 8
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

Advanced path with backend config JSON:

```sh
./omniinfer chat --message "Hello" --config
```

You can also pass backend-native extra args directly:

```powershell
.\omniinfer.cmd chat --message "Hello" -- --top-k 40 --top-p 0.9
```

## Common Commands

```sh
./omniinfer backend list
./omniinfer select <backend>
./omniinfer status
./omniinfer model list
./omniinfer model load -m /path/to/model-directory
./omniinfer model load -m /path/to/model-directory --config
./omniinfer thinking show
./omniinfer thinking set on
./omniinfer chat --message "Hello"
./omniinfer chat --message "Hello" --config
./omniinfer shutdown
./omniinfer completion bash
```

On Windows, replace `./omniinfer` with `.\omniinfer.cmd`.

## Useful Notes

- `select` stores your current backend choice for later runs.
- `select` also creates a backend-specific config JSON template for advanced backend-native parameters.
- `--config` without a path means "use the selected backend profile under `~/.config/omniinfer/backend_profiles/`".
- Backend profile JSON files should only hold backend-native extra parameters. Keep model paths, prompts/messages, and images on the CLI.
- `model load` stores the current model path, optional `mmproj`, optional `ctx-size`, and any request defaults loaded from backend-native extra args.
- `llama.cpp-*` backends accept either a model file such as `.gguf` or a model directory. Passing a model directory is the simplest cross-backend habit.
- If a `llama.cpp-*` model directory contains multiple text GGUF files or multiple `mmproj` GGUF files, OmniInfer stops and asks you to make the choice explicit.
- `turboquant-mac` uses the same `llama-server` HTTP protocol family as `llama.cpp-*`, but it remains a separate backend id.
- `mlx-mac` supports both text model directories and vision-language model directories.
- `mlx-mac` does not use `-mm/--mmproj`; multimodal support comes from the selected MLX model directory itself.
- `chat` streams output by default. Backend-native request defaults can come from `--config` or from backend-specific extra args typed directly on the CLI.
- Load-time backend-native extra args are broadly passthrough for `llama.cpp-*` and `turboquant-mac`. Chat-time backend-native extra args support many common official flags plus generic long-form request overrides, but they are still interpreted through the current backend family rather than exposed as a blind global flag bag.
- Do not combine `--auto` with backend-native extra args or `--config`, because those flows need a concrete selected backend to interpret flags correctly.
- `status` shows the current backend, model, and thinking state.
- `shutdown` stops the local desktop service. On Android it just confirms that direct mode has no background gateway.

## Platform Notes

### Linux, macOS, Windows

- The CLI uses the Python entrypoint in [omniinfer.py](../omniinfer.py).
- The desktop CLI auto-starts the local OmniInfer gateway when required.
- If you use `mlx-mac`, keep the same Python interpreter for both the CLI and the auto-started gateway. `OMNIINFER_PYTHON` is the safest way to enforce that.
- If you want to run the gateway in the foreground, use:

```sh
./omniinfer serve
```

### Android

- The repo-root [omniinfer](../omniinfer) script detects Android automatically.
- Android direct mode uses the local launcher at `.local/runtime/android/bin/omniinfer-android`.
- Android backend binaries live under `.local/runtime/android/lib/arm64-v8a`.
- Android OmniInfer Native QNN runtime files can also live under `.local/runtime/android/qnn`.
- For more Android-specific details, see [Android CLI Notes](android-cli.md).
