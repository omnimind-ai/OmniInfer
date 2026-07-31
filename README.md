<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/omniinfer-logo-dark.svg">
    <img src="docs/assets/omniinfer-logo-light.svg" alt="OmniInfer logo" width="520">
  </picture>
</p>

# OmniInfer

Easy, fast, and private LLM & VLM inference for every device

| [Demo](#demo) | [Getting Started](#getting-started) | [About](#about) | [ZO-LoRA](#on-device-zeroth-order-optimization) | [Documentation](#documentation) | [Architecture](#architecture) |

## Demo

OmniInfer includes a terminal UI for selecting backends, loading models, and chatting with local models.

<table width="100%">
  <tr>
    <td width="100%">
      <video src="https://github.com/user-attachments/assets/4ac5329e-8c54-4ea9-8a51-02306c0607e9" controls="controls" style="max-width: 100%;"></video>
    </td>
  </tr>
</table>

## Getting Started

### Quick Install

Linux x64 CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash
```

Install a specific release:

```bash
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash -s -- --version v0.3.5
```

The lightweight installer downloads the CLI-only GitHub Release archive, verifies `checksums.txt`, and installs `omniinfer` into `~/.local/bin` by default. It does not clone this repository, install backend runtimes, download models, or use sudo.

Install a prebuilt runtime after the CLI is available:

```bash
omniinfer backend list
omniinfer backend install llama.cpp-linux
```

Desktop integrations can add `--state-root <path>` and `--runtime-root <path>` to isolate managed files, and `backend install ... --json` emits streaming JSONL progress. See [CLI Usage](docs/CLI.md#2-install-a-backend-runtime) for the stable integration contract.

On Windows, official vLLM is available through the managed `vllm-wsl2-cuda` and `vllm-wsl2-rocm` backends. They run pinned upstream Linux wheels inside a user WSL2 distribution for NVIDIA CUDA or supported AMD Ryzen AI GPUs; vLLM does not support native Windows. See the [Windows vLLM setup](docs/CLI.md#windows-vllm-through-wsl2).

You can also run `omniinfer` with no arguments to open the TUI; when a compatible backend is missing, the TUI can install the prebuilt runtime before model loading.

macOS arm64 and Windows x64 CLI-only archives are available from [GitHub Releases](https://github.com/omnimind-ai/OmniInfer/releases). Homebrew, Scoop, npm, and platform-native one-line installers are planned.

`omniinfer serve --cloudflare` automatically downloads and verifies a pinned
`cloudflared` helper when neither a managed copy nor a system installation is
available. On macOS, the manual fallback is `brew install cloudflared`; retry
the command afterward or pass `--cloudflared-path "$(command -v cloudflared)"`.

### Source And Backend Setup

Use the source installer when you want a repository checkout plus backend runtime setup, source builds, and optional model setup.

Linux and macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install-from-source.sh | bash
```

Windows PowerShell:

```powershell
irm "https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.ps1?$(Get-Random)" | iex
```

The source installer detects your platform and hardware, recommends a backend, and walks you through model setup interactively.
Use `--model /path/to/model.gguf` for explicit model setup or `--no-model` / `-NoModel` to skip model setup without prompting.
Install summaries are written to `.local/install-summary.json`; source builds also save logs under `tmp/test_results/install/`.

### Source Checkout

If you already cloned this repository, build at least one local runtime backend first.

- Windows: see [Build Guide: Windows](docs/build.md#windows)
- Linux: see [Build Guide: Linux](docs/build.md#linux)
- macOS: see [Build Guide: macOS](docs/build.md#macos)
- Android: see [Build Guide: Android](docs/build.md#android)

After the runtime is ready, start with the OmniInfer CLI from the repository root.

Linux and macOS:

```sh
./omniinfer --help
```

Windows:

```powershell
.\omniinfer.ps1 --help
```

Android:

```sh
./omniinfer --help
```

## About

OmniInfer is a high-performance, cross-platform inference engine for running Large Language Models (LLM) and Vision-Language Models (VLM) locally. It abstracts away model compilation, hardware adaptation, and deployment complexity, enabling efficient local inference with minimal configuration.

> OmniInfer powers the inference layer of [Omni Studio](https://omnimind.com.cn/omnistudio), a unified model orchestration platform.

OmniInfer is fast with:

- Optimized token generation speed and minimal memory footprint
- Multiple backend engines, including llama.cpp, ik_llama.cpp, MNN, MLX, TurboQuant, LiteRT-LM, ExecuTorch QNN, and OmniInfer Native where supported
- Hardware-aware adaptation and optimization

OmniInfer is flexible and easy to use with:

- Seamless multi-backend switching for the best available engine on each device
- OpenAI-compatible and Anthropic-compatible local API endpoints
- Support for text and vision-language workloads
- Fine-grained parameter control for context length, GPU offloading, KV cache, and backend-native launch options

OmniInfer runs everywhere:

- Linux, macOS, Windows — desktop and server
- Android and iOS — mobile and edge devices
- One codebase across CLI, HTTP gateway, and mobile modules

## On-device Zeroth-Order Optimization

OmniInfer also includes an experimental `llama-zo` workflow for optimizing a
LoRA Adapter directly on an Android device without backpropagation. The SST-2
reference task keeps LoRA A fixed, updates only an F32 master copy of LoRA B,
and estimates each update from two seeded loss evaluations at `B + epsilon*z`
and `B - epsilon*z`.

The same deterministic batches and perturbations are available through:

- A standard llama.cpp CPU reference path
- Hexagon HTP serial execution with separate plus and minus decodes
- Hexagon HTP paired execution with both sides in one rectangular decode
- Optional host pipelining for preparation of the next perturbation plan

The HTP implementation preserves native Q4_0, Q8_0, and F16 base matmul paths,
then applies an explicit F16 LoRA-A matmul and in-place LoRA accumulation. HMX
accelerates eligible base matmuls and Flash Attention on supported SoCs; other
HTP work uses HVX. In particular, the tested rank-8 LoRA-A projection and LoRA
accumulation are HVX operations, so an HTP run is not an HMX-only graph.
Critical ZO-LoRA graph nodes are pinned to HTP and unsupported placement fails
instead of silently falling back to CPU.

### Result Snapshot

Historical TinyLlama F16 learning results measured on a Redmi K60 Pro
(Snapdragon 8 Gen 2 / SM8550, Hexagon v73) on 2026-07-29 used HTP paired
execution, rank 8, seed 1337, the first 1,000 SST-2 training rows, and the full
872-row development set:

| Independent run | Dev loss | Dev accuracy |
| --- | ---: | ---: |
| B=0, step 0 | 6.298321 | 484/872 = 55.504587% |
| 500 updates | 5.542850 | 500/872 = 57.339450% |
| 5,000 updates | 0.710105 | 707/872 = 81.077982% |

After the HTP activation-cache lifecycle fix, a one-step TinyLlama Q4_0 check
reported mean loss `6.540161` on CPU and `6.546338` on HTP, a difference of
`+0.006177`. This is a numerical correctness result, not a Q4 accuracy or
performance claim.

The final-source Q4_0 performance run on the same Redmi K60 Pro used 5 warmups,
20 measured steps, rank 8, batch size 4, sequence limit 128, and seed 1337:

| Path | Step p50 | Real token/s | Backend token/s | Real / padding / backend tokens |
| --- | ---: | ---: | ---: | ---: |
| CPU reference | 1,072,087 us | 127.770 | 127.770 | 2,640 / 0 / 2,640 |
| HTP serial | 761,579 us | 176.717 | 305.774 | 2,640 / 1,928 / 4,568 |
| HTP paired + pipeline | 566,687 us | 216.318 | 374.295 | 2,640 / 1,928 / 4,568 |

Paired HTP was 1.344x faster than serial HTP and 1.892x faster than CPU by
step p50. A profile of the first measured shape confirmed HMX base matmul and
Flash Attention, zero critical-node CPU fallbacks, and a median LoRA-A plus
LoRA-accumulation share of 19.431% of decode time across three runs.

Start with the [complete ZO-LoRA example](framework/llama.cpp/examples/zo-lora/README.md)
for the algorithm, native build, mode matrix, reproducibility protocol, and
result details. The [Android ZO-LoRA guide](docs/android/zo-lora-cli.md) covers
toolchain setup, packaging, and ADB deployment.

## Documentation

Recommended docs:

- [CLI Guide](docs/CLI.md): end-to-end CLI usage for Linux, macOS, Windows, and Android
- [Android App Integration](docs/android/integration.md): embed OmniInfer in a third-party Android app
- [Android Backend Reference](docs/android/backends.md): Android backend options for llama.cpp, MNN, LiteRT-LM, and ExecuTorch QNN
- [Android ZO-LoRA CLI](docs/android/zo-lora-cli.md): build and run the standalone native SST-2 trainer with CPU or Hexagon HTP
- [Android Smoke Tests](docs/android/smoke-tests.md): adb/curl checks and source-build validation
- [Android Troubleshooting](docs/android/troubleshooting.md): common build, runtime, and backend failures
- [Build Guide](docs/build.md): build and platform packaging notes
- [API Reference](docs/API.md): OpenAI-compatible local API usage

## Architecture

![omni_studio_architecture](./docs/assets/architecture.drawio.svg)

## Citation

If you use OmniInfer in research, please cite this repository.
GitHub can automatically generate citation formats from [CITATION.cff](CITATION.cff).

```bibtex
@software{omniinfer,
  author = {{Omnimind AI}},
  title = {OmniInfer},
  url = {https://github.com/omnimind-ai/OmniInfer}
}
```

## Contributing

We welcome and value any contributions and collaborations. Please check out [Contributing to OmniInfer](CONTRIBUTING.md) for how to get involved.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

`framework/llama.cpp` is vendored and modified from
[ggml-org/llama.cpp at revision `8a091c47abe67e0a03b85bc7c9eee8bdb9b14b05`](https://github.com/ggml-org/llama.cpp/commit/8a091c47abe67e0a03b85bc7c9eee8bdb9b14b05)
and remains available under its MIT license; see
[framework/llama.cpp/LICENSE](framework/llama.cpp/LICENSE).
