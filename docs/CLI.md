# OmniInfer CLI Guide

This guide shows how to use the OmniInfer desktop CLI on Linux, macOS, and Windows.
Android and iOS use the embedded modules under `android/` and `ios/`.

## Before You Start

If you are running OmniInfer from a source checkout, prepare at least one local runtime backend before using the CLI.

- Windows: build one of `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip` first. See [Build Guide: Windows](build.md#windows).
- Linux: build one of `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, or `llama.cpp-linux-openvino` first. See [Build Guide: Linux](build.md#linux).
- macOS: build `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac` first. See [Build Guide: macOS](build.md#macos).

If you are using a packaged release that already includes `runtime/`, you can skip this preparation step and jump straight to the CLI commands below.
Packaged releases do not include the `omniinfer build` command; backend builds are source-checkout tooling only.

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
- Android and iOS:
  use the embedded modules under `android/` and `ios/` instead of the desktop CLI.

## Launch The CLI

Run the CLI after your local runtime has been prepared.

For source checkouts, use the repository root.
For packaged releases, use the release directory that already contains `runtime/`.

Linux and macOS:

```sh
./omniinfer --help
```

Run `./omniinfer` without arguments in an interactive terminal to open the basic TUI. On first use, the TUI lets you pick an installed backend, choose a model found in OmniInfer-managed `.local` model directories or enter a model path manually, load it, and enter a simple chat loop. When a manual directory is scanned, the selected model is linked into `.local/models/<detected-model-dir>/<model-file>` instead of preserving unrelated parent folders. Later TUI launches automatically reload the last selected backend and model when the model path still exists.

Windows:

```powershell
.\omniinfer.ps1 --help
```

Packaged Windows releases also keep `.\omniinfer.cmd` for `cmd.exe` compatibility. For interactive TUI use from PowerShell, prefer `.\omniinfer.ps1` or `.\omniinfer-cli.exe`; pressing `Ctrl+C` in a batch wrapper can make `cmd.exe` print `Terminate batch job (Y/N)?`.

## Quick Start

### 1. Check available backends

Linux, macOS, Windows:

```sh
./omniinfer backend list
```

Windows:

```powershell
.\omniinfer.ps1 backend list
```

The human-readable output shows the selected backend and runtime availability.
Use `./omniinfer backend list --json` when automation needs full backend metadata such as capabilities and descriptions.
By default, `backend list` shows compatible backends only. Use `--scope installed` or `--scope all` for a narrower or broader view.
In the table output, empty `Selected` or `Installed` cells mean the state is false.

### 2. Build a backend from source, optional

Source checkouts on Linux, macOS, and Windows can build a compatible backend through the CLI:

```sh
./omniinfer build <backend>
```

Windows:

```powershell
.\omniinfer.ps1 build <backend>
```

The command runs the matching platform script under `scripts/platforms/<platform>/<backend>/build.*`, defaults to a `Release` build, and verifies that the backend launcher exists after the build. Use `--build-type <type>` to choose a different CMake build type, or `--dry-run` to print the build command without running it.

Packaged releases intentionally do not provide this command because they are designed to run from the included `runtime/` directory without requiring a compiler, CUDA toolkit, CMake, or other build tools.

### 3. Select a backend

Always pick a backend from `backend list` on your current device.

```sh
./omniinfer backend select <backend>
```

Windows:

```powershell
.\omniinfer.ps1 backend select <backend>
```

Examples:

- Linux: `llama.cpp-linux`, `llama.cpp-linux-rocm`, `llama.cpp-linux-vulkan`, `llama.cpp-linux-s390x`, or `llama.cpp-linux-openvino`
- macOS: `llama.cpp-mac`, `llama.cpp-mac-intel`, `turboquant-mac`, or `mlx-mac`
- Windows: `llama.cpp-cpu`, `llama.cpp-cuda`, `llama.cpp-vulkan`, `llama.cpp-windows-arm64`, `llama.cpp-sycl`, or `llama.cpp-hip`

When you select a desktop backend, OmniInfer also creates a backend-specific JSON config template under:

- `.local/config/backend_profiles/<backend>.json`

This file is the advanced path for backend-native parameters only.
Keep basic user inputs such as `-m/--model`, `-mm/--mmproj`, chat prompts, and `--image` on the CLI.

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

### 4. Load a model

Default path:

```sh
./omniinfer load -m /path/to/model-directory
```

For `llama.cpp-*`, OmniInfer accepts either a model file or a model directory. If you pass a directory, OmniInfer auto-discovers:

- the main text GGUF
- the optional `mmproj` GGUF

For `mlx-mac`, OmniInfer passes the model directory directly to the embedded backend.

Explicit file path:

```sh
./omniinfer load -m /path/to/model.gguf
```

Advanced path with backend config JSON:

```sh
./omniinfer backend select llama.cpp-vulkan
./omniinfer load -m /path/to/model-directory --config
```

Windows:

```powershell
.\omniinfer.ps1 load -m C:\path\to\model-directory
```

Vision-language model:

```sh
./omniinfer load -m /path/to/model.gguf -mm /path/to/mmproj.gguf
```

For `mlx-mac`, use a vision-capable model directory instead of a `.gguf` file or `mmproj` sidecar:

```sh
./omniinfer backend select mlx-mac
./omniinfer load -m /path/to/mlx-vlm-model-directory
./omniinfer chat \
  --image /path/to/image.jpg \
  "Describe this image in one sentence."
```

The backend config JSON is where advanced users should put backend-native launch parameters such as `-ngl`, `--threads`, and other backend-specific options.

You can also skip `--config` entirely and pass backend-native extra args directly after the stable OmniInfer args. OmniInfer parses those extra args according to the currently selected backend.

Example:

```powershell
.\omniinfer.ps1 backend select llama.cpp-vulkan
.\omniinfer.ps1 load -m C:\models\Qwen3 -ngl 99 -t 8
```

### 4. Chat

Text chat:

```sh
./omniinfer chat "Introduce yourself in one sentence."
```

Vision-language chat:

```sh
./omniinfer chat \
  --image /path/to/image.jpg \
  "Describe this image in one sentence."
```

Windows:

```powershell
.\omniinfer.ps1 chat "Introduce yourself in one sentence."
```

Advanced path with backend config JSON:

```sh
./omniinfer load -m /path/to/model-directory --config
./omniinfer chat "Hello"
```

You can also pass backend-native extra args directly:

```powershell
.\omniinfer.ps1 chat "Hello" -- --top-k 40 --top-p 0.9
```

## Common Commands

```sh
./omniinfer
./omniinfer backend list
./omniinfer backend select <backend>
./omniinfer status
./omniinfer model list
./omniinfer load -m /path/to/model-directory
./omniinfer load -m /path/to/model-directory --config
./omniinfer thinking show
./omniinfer thinking set on
./omniinfer chat "Hello"
./omniinfer shutdown
./omniinfer completion bash
```

On packaged Windows releases, replace `./omniinfer` with `.\omniinfer.ps1` in PowerShell. Use `.\omniinfer.cmd` only when you specifically need `cmd.exe` compatibility.

## Useful Notes

- `backend select` stores your current backend choice for later CLI and TUI runs under `.local/config/state.json`.
- `backend select` also creates a backend-specific config JSON template for advanced backend-native parameters.
- Running `./omniinfer` with no arguments opens the TUI only in an interactive terminal; non-interactive usage prints CLI help instead of blocking for input.
- The TUI auto-discovers models from the shared OmniInfer-managed `.local/models` directory. When you enter a model file path manually, the TUI creates a model subdirectory under `.local/models` and places the symlink inside that directory so the model appears in later TUI runs. When you enter a directory, the TUI scans it recursively for non-`mmproj` GGUF model files and links the selected file under a `.local/models/<directory-name>/` folder.
- After a successful model load, the TUI stores the selected backend, model path, optional `mmproj`, and optional `ctx-size` under `.local/config/state.json`; the next TUI launch reloads them and enters chat directly when the model still exists.
- The TUI chat commands are `/backend`, `/model`, `/think`, `/reasoning`, `/status`, `/clear`, `/help`, and `/exit`. Use `/think` to toggle the gateway default thinking mode, or `/think on` / `/think off` to set it explicitly. The thinking choice is saved under `.local/config/state.json` and reused the next time the gateway starts. Typing `/` in the chat prompt shows inline command hints.
- Use `/reasoning` to toggle whether streamed reasoning is shown in the transcript, or `/reasoning on` / `/reasoning off` to set it explicitly. This only changes TUI display; `/think` controls whether requests ask the backend to think. The reasoning display choice is saved under `.local/config/state.json`.
- The TUI backend, model, and command menus are searchable overlays in interactive terminals. Type to filter, use Up/Down to move, Enter to select, and Esc to return to chat.
- The TUI keeps a fixed bottom input bar during chat and while responses stream so the conversation history stays visible above the prompt. The prompt area also shows a compact status line with the active backend, loaded model, thinking mode, runtime readiness/port, context size or recent context usage, backend device class, launch thread settings when available, and transient notices from backend/model/thinking commands.
- TUI system notices such as backend switches, model load completion, and chat-stream errors are routed through the prompt status line instead of being appended as assistant chat content.
- TUI chat keeps an in-memory multi-turn `messages` list for the current backend/model session. Each turn sends the full accumulated user/assistant history plus the new user message. Switching backend or model clears this in-memory conversation.
- TUI transcript output uses lightweight role blocks for user, assistant, and optional reasoning content. TUI `/status` shows grouped runtime, model, generation, and conversation details, including context usage after a chat response returns a `usage` payload.
- On terminals with readline support, the TUI chat prompt supports Unicode-aware editing and Up/Down input history.
- By default, the TUI suppresses a leading `<think>...</think>` block in streamed model output and shows only the visible answer text; `/reasoning on` shows that reasoning block when the backend provides it.
- `load --config` without a path means "use the selected backend profile under `.local/config/backend_profiles/`".
- Backend profile JSON files should only hold backend-native extra parameters. Keep model paths, prompts/messages, and images on the CLI.
- `load` is the short form of `model load`; both store the current model path, optional `mmproj`, optional `ctx-size`, and any request defaults loaded from backend-native extra args.
- `llama.cpp-*` backends accept either a model file such as `.gguf` or a model directory. Passing a model directory is the simplest cross-backend habit.
- If a `llama.cpp-*` model directory contains multiple text GGUF files or multiple `mmproj` GGUF files, OmniInfer stops and asks you to make the choice explicit.
- `turboquant-mac` uses the same `llama-server` HTTP protocol family as `llama.cpp-*`, but it remains a separate backend id.
- `mlx-mac` supports both text model directories and vision-language model directories.
- `mlx-mac` does not use `-mm/--mmproj`; multimodal support comes from the selected MLX model directory itself.
- `chat` streams output by default. Backend-native request defaults can come from the profile used during `load --config` or from backend-specific extra args typed directly on the CLI.
- Unless overridden by `--max-tokens` or a backend profile request default, chat requests use a default completion budget of 2048 tokens.
- Load-time backend-native extra args are broadly passthrough for `llama.cpp-*` and `turboquant-mac`. Chat-time backend-native extra args support many common official flags plus generic long-form request overrides, but they are still interpreted through the current backend family rather than exposed as a blind global flag bag.
- Do not combine `--auto` with backend-native extra args or load profiles, because those flows need a concrete selected backend to interpret flags correctly.
- `status` shows the current backend, model, and thinking state.
- `shutdown` stops the local desktop service.

## Platform Notes

### Linux, macOS, Windows

- The CLI uses the Python entrypoint in [omniinfer.py](../omniinfer.py).
- The desktop CLI auto-starts the local OmniInfer gateway when required.
- CUDA desktop backends default to one GPU. If `CUDA_VISIBLE_DEVICES` is unset, OmniInfer picks the visible GPU with the most free memory and lowest utilization before launching the backend. Set `CUDA_VISIBLE_DEVICES` or `OMNIINFER_CUDA_VISIBLE_DEVICES` to override this.
- If you use `mlx-mac`, keep the same Python interpreter for both the CLI and the auto-started gateway. `OMNIINFER_PYTHON` is the safest way to enforce that.
- If you want to run the gateway in the foreground, use:

```sh
./omniinfer serve
```

### Mobile

- Android is implemented by the root `android/` Gradle module. See [Android Integration Guide](android/integration.md).
- iOS is implemented by the root `ios/OmniInferServer` Swift Package. See [OmniStudio API Service](OmniStudio/api-service.md#ios-client).
