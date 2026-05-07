# Android Multimodal Models

This document describes how OmniInfer Android discovers vision model files. The app API is the same as text-only chat: send `/v1/chat/completions` with text and image content.

## Supported Backends

| Backend | Android multimodal status | Required layout |
|---|---|---|
| llama.cpp | Supported | GGUF model and `mmproj*.gguf` in the same directory |
| MNN | Supported | `config.json` references text and vision MNN files |
| LiteRT-LM | Text-only in OmniInfer path | Model-dependent; image input is not enabled through OmniInfer yet |
| ExecuTorch QNN | Text-only | Not supported |

## llama.cpp Layout

The backend scans the model file's parent directory for a file matching `mmproj*.gguf`. If found, the vision encoder is loaded automatically.

```text
/sdcard/models/Qwen3.5-2B-gguf/
  Qwen3.5-2B-Q4_K_M.gguf    # modelPath points here
  mmproj-F16.gguf           # auto-discovered, enables vision
```

Load call:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_K_M.gguf",
    backend = "llama.cpp",
    nThreads = 6,
    nCtx = 8192,
)
```

If the mmproj file is missing or placed elsewhere, the model loads as text-only and image inputs are ignored.

## MNN Layout

The MNN model directory should contain `config.json` and the files referenced by it.

```text
/sdcard/models/Qwen3.5-2B-MNN/
  config.json                 # modelPath points here
  llm.mnn
  llm.mnn.weight
  visual.mnn                  # enables vision
  visual.mnn.weight
```

Load call:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3.5-2B-MNN/config.json",
    backend = "mnn",
    nThreads = 6,
    nCtx = 8192,
)
```

MNN discovers vision files through `config.json`; keep the packaged directory structure intact.

## Request Shape

Send image data as an OpenAI-compatible `image_url` item. Data URIs are supported:

```json
{
  "model": "local",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }],
  "stream": true,
  "max_tokens": 200
}
```

Kotlin example:

```kotlin
val imageB64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
val json = """
{
  "model": "local",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$imageB64"}}
    ]
  }],
  "stream": true,
  "max_tokens": 200
}
""".trimIndent()
```

## Troubleshooting

**Model says it cannot see the image:** verify the backend actually loaded vision files. For llama.cpp, the `mmproj*.gguf` must be in the same directory as the model GGUF. For MNN, check `config.json` references `visual.mnn`.

**Image request works on one backend but not another:** check the backend support table above. LiteRT-LM and ExecuTorch QNN are currently text-only through OmniInfer Android.

**Large image request is slow:** image inputs add prompt tokens and prefill work. Check the final response `usage.prompt_tokens_details.image_tokens` and `performance.prefill_tokens_per_second`.
