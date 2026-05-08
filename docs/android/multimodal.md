# Android Multimodal Models

This document describes how OmniInfer Android discovers vision model files. The app API is the same as text-only chat: send `/v1/chat/completions` with text and image content.

## Supported Backends

| Backend | Android multimodal status | Required layout |
|---|---|---|
| llama.cpp | Supported | GGUF model and `mmproj*.gguf` in the same directory |
| MNN | Supported | `config.json` references text and vision MNN files |
| LiteRT-LM | Supported for `.litertlm` models with vision encoder | Enable `extraConfig["vision_backend"]` at model load |
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

## LiteRT-LM Layout

LiteRT-LM multimodal models package the text and vision assets inside a `.litertlm` file.

```text
/sdcard/models/gemma-4-E2B-it-litert-lm/
  gemma-4-E2B-it.litertlm    # modelPath points here
```

Load call:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm",
    backend = "litert",
    nCtx = 4096,
    extraConfig = mapOf(
        "backend_type" to "gpu",
        "vision_backend" to "gpu",
        "max_images" to "1",
    ),
)
```

LiteRT-LM details:

- OmniInfer uses `Content.ImageBytes(...)` for image input.
- The app cache directory is created before `Engine.initialize()` so LiteRT-LM can place GPU/vision cache files.
- Use LiteRT-LM `0.11.0+` for Gemma 4 E2B/E4B vision models. `0.10.2` cannot initialize Gemma 4 E2B's `vision_70` / `vision_140` / `vision_280` encoder signatures.
- `nCtx` is passed to `EngineConfig.maxNumTokens`; image inputs consume prompt tokens, so keep enough context for image + text + output.

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

**Model says it cannot see the image:** verify the backend actually loaded vision files. For llama.cpp, the `mmproj*.gguf` must be in the same directory as the model GGUF. For MNN, check `config.json` references `visual.mnn`. For LiteRT-LM, load with `extraConfig["vision_backend"] = "cpu"` or `"gpu"`.

**Image request works on one backend but not another:** check the backend support table above. ExecuTorch QNN is currently text-only through OmniInfer Android.

**Large image request is slow:** image inputs add prompt tokens and prefill work. Check the final response `usage.prompt_tokens`, `usage.prompt_tokens_details`, and `performance.prefill_tokens_per_second`. LiteRT-LM currently reports total prompt tokens and image count/bytes; it does not expose an official separate image-token count through `BenchmarkInfo`.
