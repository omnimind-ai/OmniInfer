<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/omniinfer-logo-dark.svg">
    <img src="docs/assets/omniinfer-logo-light.svg" alt="OmniInfer logo" width="520">
  </picture>
</p>

# OmniInfer

Easy, fast, and private LLM & VLM inference for every device

| [Demo](#demo) | [Getting Started](#getting-started) | [About](#about) | [Documentation](#documentation) | [Architecture](#architecture) |

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
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash -s -- --version v0.3.2
```

The lightweight installer downloads the CLI-only GitHub Release archive, verifies `checksums.txt`, and installs `omniinfer` into `~/.local/bin` by default. It does not clone this repository, install backend runtimes, download models, or use sudo.

macOS arm64 and Windows x64 CLI-only archives are available from [GitHub Releases](https://github.com/omnimind-ai/OmniInfer/releases). Homebrew, Scoop, npm, and platform-native one-line installers are planned.

### Source And Backend Setup

Use the source installer when you want a repository checkout plus backend runtime setup and optional model setup.

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

## Documentation

Recommended docs:

- [CLI Guide](docs/CLI.md): end-to-end CLI usage for Linux, macOS, Windows, and Android
- [Android App Integration](docs/android/integration.md): embed OmniInfer in a third-party Android app
- [Android Backend Reference](docs/android/backends.md): Android backend options for llama.cpp, MNN, LiteRT-LM, and ExecuTorch QNN
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
