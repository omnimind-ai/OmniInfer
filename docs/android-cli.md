# OmniInfer Android CLI

OmniInfer supports Android in direct mode.

## Layout

- `.local/runtime/android/bin/omniinfer-android`
  Local Android shell frontend used by the repo-root `./omniinfer` entrypoint.
- `.local/runtime/android/lib/arm64-v8a/libllama-cli.so`
  Local Android llama.cpp backend binary.
- `.local/runtime/android/lib/arm64-v8a/libmtmd-cli.so`
  Local Android multimodal backend binary.

At the moment both backend files are wired through the same validated Android llama.cpp CLI binary,
so text chat and multimodal chat both work out of the box while the dedicated mtmd Android build is
still being stabilized.

## How it works

Android does not run the local HTTP gateway.

Instead, the repo-root `./omniinfer` script detects Android and forwards commands to
`.local/runtime/android/bin/omniinfer-android`, which:

- persists backend/model state
- maps OmniInfer commands to Android-native llama.cpp backends
- runs text and multimodal inference directly in the current shell

These Android runtime assets stay under the local `.local/runtime/android/` directory even though that directory is not tracked in Git.

## State

The Android CLI stores state under one of these directories:

- `$HOME/.config/omniinfer`
- `./.omniinfer/android-cli`

It keeps:

- selected backend
- selected model path
- selected mmproj path
- ctx size
- default thinking mode

## Common commands

```sh
./omniinfer backend list
./omniinfer select llama.cpp-llama
./omniinfer model load -m /data/local/tmp/Qwen2.5-3B-Instruct-Q8_0.gguf
./omniinfer chat --message "Introduce yourself in one sentence."
./omniinfer model load \
  -m /data/local/tmp/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  -mm /data/local/tmp/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
./omniinfer chat \
  --image /data/local/tmp/benchmark_images/test_448x448.jpg \
  --message "Describe this image in one sentence."
```
