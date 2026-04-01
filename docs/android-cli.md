# OmniInfer Android CLI

OmniInfer supports Android in direct mode.

The repository now includes an Android runtime preparation script:

```sh
bash ./scripts/platforms/android/build-runtime.sh \
  --artifact-dir /path/to/android/artifacts \
  --qnn-bundle-dir /path/to/qnn-bundle
```

## Layout

- `.local/runtime/android/bin/omniinfer-android`
  Local Android shell frontend used by the repo-root `./omniinfer` entrypoint.
- `.local/runtime/android/lib/arm64-v8a/libllama-cli.so`
  Local Android llama.cpp backend binary.
- `.local/runtime/android/lib/arm64-v8a/libmtmd-cli.so`
  Local Android multimodal backend binary.
- `.local/runtime/android/qnn/qnn_llama_runner`
  Local OmniInfer Native text backend launcher for ExecuTorch/QNN.
- `.local/runtime/android/qnn/libQnn*.so`
  Local QNN runtime libraries used by `qnn_llama_runner`.

At the moment both backend files are wired through the same validated Android llama.cpp CLI binary,
so text chat and multimodal chat both work out of the box while the dedicated mtmd Android build is
still being stabilized.

## How it works

Android does not run the local HTTP gateway.

Instead, the repo-root `./omniinfer` script detects Android and forwards commands to
`.local/runtime/android/bin/omniinfer-android`, which:

- persists backend/model state
- maps OmniInfer commands to Android-native llama.cpp backends
- maps OmniInfer commands to the OmniInfer Native QNN backend when selected
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
- tokenizer path for OmniInfer Native QNN
- decoder model version for OmniInfer Native QNN

## Common commands

```sh
./omniinfer backend list
./omniinfer select llama.cpp-llama
./omniinfer model load -m /data/local/tmp/Qwen2.5-3B-Instruct-Q8_0.gguf
./omniinfer chat --message "Introduce yourself in one sentence."
./omniinfer select omniinfer-native
./omniinfer model load -m /data/local/tmp/syf/executorch/static_llm
./omniinfer chat --message "你好啊，你是谁？"
./omniinfer model load \
  -m /data/local/tmp/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  -mm /data/local/tmp/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
./omniinfer chat \
  --image /data/local/tmp/benchmark_images/test_448x448.jpg \
  --message "Describe this image in one sentence."
```

## OmniInfer Native QNN Notes

- The Android backend id is `omniinfer-native`.
- It is currently text-only.
- If you pass a directory to `model load`, OmniInfer first looks for `hybrid_llama_qnn.pte`.
- `tokenizer.json` is auto-discovered next to the selected `.pte` model.
- `decoder_model_version` is auto-inferred when possible. The current `hybrid_llama_qnn.pte` convention defaults to `qwen3`.
- You can still override either value explicitly:

```sh
./omniinfer select omniinfer-native
./omniinfer model load \
  -m /data/local/tmp/syf/executorch/static_llm \
  --tokenizer-path /data/local/tmp/syf/executorch/static_llm/tokenizer.json \
  --decoder-model-version qwen3
```
