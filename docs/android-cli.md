# OmniInfer Android CLI

OmniInfer supports Android in direct mode.

The repository now includes two Android helper scripts:

```sh
bash ./scripts/platforms/android/build-runtime.sh \
  --artifact-dir /path/to/android/artifacts \
  --qnn-bundle-dir /path/to/qnn-bundle
```

```sh
bash ./scripts/platforms/android/package-omniinfer-native.sh \
  --artifact-dir /path/to/executorch-artifacts \
  --decoder-model-version qwen3
```

## Layout

- `.local/runtime/android/bin/omniinfer-android`
  Local Android shell frontend used by the repo-root `./omniinfer` entrypoint.
- `.local/runtime/android/support/common.sh`
  Shared Android CLI state and command dispatcher helpers.
- `.local/runtime/android/backends/llama_cpp/backend.sh`
  Android llama.cpp backend adapter.
- `.local/runtime/android/backends/omniinfer_native/backend.sh`
  Android ExecuTorch/QNN backend adapter.
- `.local/runtime/android/lib/arm64-v8a/libllama-cli.so`
  Local Android llama.cpp backend binary.
- `.local/runtime/android/lib/arm64-v8a/libmtmd-cli.so`
  Local Android multimodal backend binary.
- `.local/runtime/android/qnn/qnn_llama_runner`
  Local OmniInfer Native text backend launcher for ExecuTorch/QNN.
- `.local/runtime/android/qnn/qnn_multimodal_runner`
  Local OmniInfer Native multimodal backend launcher for ExecuTorch/QNN.
- `.local/runtime/android/qnn/libQnn*.so`
  Local QNN runtime libraries used by `qnn_llama_runner`.

This layout is intentional: the main `omniinfer-android` entrypoint stays thin, while backend-specific logic lives in separate backend modules.

At the moment both llama.cpp backend ids can still fall back to the same validated Android llama.cpp CLI binary, so text chat and multimodal chat both work out of the box while the dedicated mtmd Android build is still being stabilized.

## How it works

Android does not run the local HTTP gateway.

Instead, the repo-root `./omniinfer` script detects Android and forwards commands to
`.local/runtime/android/bin/omniinfer-android`, which:

- loads shared Android runtime helpers from `support/`
- loads backend-specific adapters from `backends/`
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
- The recommended workflow is to package official ExecuTorch Qualcomm llama artifacts first:

```sh
bash ./scripts/platforms/android/package-omniinfer-native.sh \
  --artifact-dir /path/to/executorch-artifacts \
  --decoder-model-version qwen3
```

- `omniinfer-native` now supports two package styles:
  - compatibility mode: a single `.pte` text model such as `hybrid_llama_qnn.pte`
  - package mode: a model directory that contains `omniinfer-native.env`
- If you pass a directory to `model load` and `omniinfer-native.env` exists, OmniInfer treats that directory as the authoritative ExecuTorch package root.
- If no package manifest exists, OmniInfer falls back to the old text-only discovery path and first looks for `hybrid_llama_qnn.pte`.
- `tokenizer.json` is auto-discovered next to the selected `.pte` model.
- `decoder_model_version` is auto-inferred when possible. The current `hybrid_llama_qnn.pte` convention defaults to `qwen3`.
- Package mode is the recommended path for official `framework/executorch/examples/qualcomm/oss_scripts/llama` artifacts because those models may include multiple `.pte` files such as decoder, vision encoder, token embedding, and attention sink helpers.
- You can still override either value explicitly:

```sh
./omniinfer select omniinfer-native
./omniinfer model load \
  -m /data/local/tmp/syf/executorch/static_llm \
  --tokenizer-path /data/local/tmp/syf/executorch/static_llm/tokenizer.json \
  --decoder-model-version qwen3
```

Recommended package manifest example:

```sh
# /path/to/artifact/omniinfer-native.env
OMNIINFER_NATIVE_FORMAT=1
OMNIINFER_NATIVE_RUNNER=llama
OMNIINFER_NATIVE_DECODER_MODEL_VERSION=qwen3
OMNIINFER_NATIVE_TOKENIZER=tokenizer.json
OMNIINFER_NATIVE_TEXT_DECODER=hybrid_llama_qnn.pte
OMNIINFER_NATIVE_EVAL_MODE=1
```

The package script can auto-detect the common official artifact names:

- `hybrid_llama_qnn.pte`, `kv_llama_qnn.pte`, `lookahead_llama_qnn.pte`
- `tokenizer.json`, `tokenizer.bin`, or `tokenizer.model`
- `vision_encoder_qnn.pte`
- `tok_embedding_qnn.pte`
- `attention_sink_evictor.pte`

For multimodal Qualcomm llama artifacts, use `OMNIINFER_NATIVE_RUNNER=multimodal` and add:

```sh
OMNIINFER_NATIVE_VISION_ENCODER=vision_encoder_qnn.pte
OMNIINFER_NATIVE_TOK_EMBEDDING=tok_embedding_qnn.pte
OMNIINFER_NATIVE_TEXT_DECODER=hybrid_llama_qnn.pte
```
