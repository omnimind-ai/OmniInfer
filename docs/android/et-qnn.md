# Android ExecuTorch QNN Backend

This document covers the Android `executorch-qnn` backend. For normal CPU/GPU/LiteRT integration, use [integration.md](./integration.md) and [backends.md](./backends.md).

## Overview

The ExecuTorch QNN backend runs LLM inference on Qualcomm Hexagon NPU. It is useful when you have a Snapdragon target device and a model exported to ExecuTorch/QNN `.pte` format.

The app-facing API is the same as other OmniInfer Android backends:

```kotlin
OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-1.7B/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf("decoder_model_version" to "qwen3"),
)
```

## How It Works

Unlike llama.cpp, MNN, and LiteRT-LM, the QNN backend does not run entirely inside the JNI process. It spawns a subprocess runner and communicates through stdin/stdout JSON.

```text
App process
  -> fork+exec libetqnn_runner.so
  -> stdin/stdout JSON protocol
  -> QNN SDK -> FastRPC HAL -> Hexagon NPU
```

This design avoids Android linker namespace restrictions that prevent QNN FastRPC from initializing reliably inside a JNI-loaded process.

## Requirements

| Requirement | Notes |
|---|---|
| Snapdragon SoC | Snapdragon 8 Gen 1 or newer is the intended target |
| Android ABI | `arm64-v8a` |
| Model | `.pte` exported for the target SoC / QNN SDK |
| Runtime libraries | Bundled by OmniInfer when QNN is enabled |
| Manifest | `android:extractNativeLibs="true"` must be preserved |

`extractNativeLibs=true` is required because the subprocess runner `.so` must exist as a regular file on disk.

## Gradle Switch

The ExecuTorch QNN backend is enabled by default:

```properties
omniinfer.backend.executorch_qnn=true
```

Set it to `false` if the app does not use QNN:

```properties
omniinfer.backend.executorch_qnn=false
```

When enabled, the first build downloads required QNN prebuilt binaries into `android/omniinfer-server/src/main/jniLibs/arm64-v8a/`. Later builds skip the download when the files already exist.

Downloaded files include:

| File | Purpose |
|---|---|
| `libetqnn_runner.so` | Subprocess runner |
| `libqnn_executorch_backend.so` | ExecuTorch QNN delegate |
| `libQnnHtp.so` | QNN HTP runtime |
| `libQnnHtpPrepare.so` | QNN preparation/runtime support |
| `libQnnSystem.so` | QNN system library |
| `libQnnHtpNetRunExtensions.so` | QNN net-run extensions |
| `libQnnHtpV75Skel.so`, `libQnnHtpV75Stub.so` | SM8650 / 8 Gen 3 support |
| `libQnnHtpV79Skel.so`, `libQnnHtpV79Stub.so` | SM8750 / 8 Elite support |
| `libQnnHtpV81Skel.so`, `libQnnHtpV81Stub.so` | SM8850 / 8 Elite Gen 5 support |

## Model Layout

Place all model files in the same directory on the device.

Baseline model:

```text
/sdcard/models/Qwen3-1.7B/
  hybrid_llama_qnn.pte
  tokenizer.json
```

Attention sink model:

```text
/sdcard/models/Qwen3-1.7B-sink32k/
  hybrid_llama_qnn.pte
  attention_sink_evictor.pte
  tokenizer.json
```

The tokenizer and attention sink evictor are auto-discovered from the model directory unless you pass explicit paths in `extraConfig`.

## Load Examples

Baseline:

```kotlin
val success = OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-1.7B/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf("decoder_model_version" to "qwen3"),
)
```

Attention sink:

```kotlin
val success = OmniInferServer.loadModel(
    modelPath = "/sdcard/models/Qwen3-1.7B-sink32k/hybrid_llama_qnn.pte",
    backend = "executorch-qnn",
    extraConfig = mapOf(
        "decoder_model_version" to "qwen3",
        "seq_len" to "32768",
    ),
)
```

Supported `extraConfig` keys:

| Key | Required | Default | Description |
|---|---|---|---|
| `decoder_model_version` | No | `qwen3` | Chat template family: `qwen3`, `qwen2_5`, `llama3`, `gemma3` |
| `tokenizer_path` | No | Auto-discovered | Path to `tokenizer.json` |
| `seq_len` | No | `32768` for sink model, otherwise `2048` | Max total tokens |
| `attention_sink_evictor_path` | No | Auto-discovered | Path to `attention_sink_evictor.pte` |

## Prebuilt Models

Pre-exported models are hosted on ModelScope:

```text
BiReRa/omniinfer-01001
```

Example downloads:

```bash
# Baseline model
modelscope download \
  --model BiReRa/omniinfer-01001 \
  --include "SM8650_qwen3-1_7b/*" \
  --local_dir ./models

# Long-context attention sink model
modelscope download \
  --model BiReRa/omniinfer-01001 \
  --include "SM8650_qwen3-1_7b_sink32k/*" \
  --local_dir ./models
```

Pick the directory matching the target SoC:

| SoC | Directory prefix |
|---|---|
| SM8650 / Snapdragon 8 Gen 3 | `SM8650_...` |
| SM8750 / Snapdragon 8 Elite | `SM8750_...` |
| SM8850 / Snapdragon 8 Elite Gen 5 | `SM8850_...` |

Available Qwen3 sizes include 0.6B, 1.7B, and 4B variants. Both baseline and attention sink directories are available for supported SoCs.

## Performance Reference

Measured on Snapdragon 8 Gen 3 / SM8650:

| Model | Decode tok/s | TTFT | Load time | RAM |
|---|---:|---:|---:|---:|
| Qwen3-0.6B baseline | 20.96 | 173 ms | 1.0 s | 708 MiB |
| Qwen3-1.7B baseline | 22.85 | 67 ms | 1.4 s | 1715 MiB |

Attention sink observations on SM8650, Qwen3-0.6B with `seq_len=4096`:

| Input tokens | Prefill tok/s | Decode tok/s | Notes |
|---:|---:|---:|---|
| 506 | 439 | 20.0 | Normal output |
| 978 | 420 | 19.8 | Normal output |
| 1568 | 415 | 20.0 | Normal output |
| 2984 | 374 | 19.6 | Degraded output beyond effective window |

Reference export benchmarks for SM8750:

| Model | Prefill tok/s | Decode tok/s | RAM |
|---|---:|---:|---:|
| Qwen3-0.6B | 532 | 49.1 | 714 MiB |
| Qwen3-1.7B | 2018 | 36.9 | 1714 MiB |
| Qwen3-4B | 1118 | 17.8 | 1777 MiB |

## Limitations

- Text-only: multimodal and tool calling are not supported on the QNN backend.
- Single-turn subprocess protocol: KV cache reuse across app turns is not implemented there yet.
- Qualcomm only: requires Snapdragon SoC with Hexagon NPU.
- Sampling control is limited: some sampling parameters are not passed to the subprocess runner.
- Attention sink is better for long output than long input. Content outside the effective sink window can be evicted.
- `extractNativeLibs=true` is required for subprocess execution.

## Troubleshooting

**Runner cannot start:** verify `android:extractNativeLibs="true"` survived manifest merge and the QNN libraries are present in `jniLibs/arm64-v8a/`.

**Model hangs or outputs degraded text with very long input:** reduce input length or use a larger model. Attention sink keeps generation length high, but the effective understanding window is still limited.

**Wrong model for device:** QNN `.pte` files are SoC/export specific. Use the ModelScope directory matching the target chipset.

**Tokenizer not found:** keep `tokenizer.json` beside `hybrid_llama_qnn.pte`, or pass `extraConfig["tokenizer_path"]`.
