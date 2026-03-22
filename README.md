# OmniInfer

<p align="center">
  <img src="assets/OmniInfer_logo.png" alt="OmniInfer Logo" width="400">
</p>

<p align="center">
  <strong>A High-Performance Cross-Platform Inference Engine for LLMs and VLMs</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#supported-backends">Backends</a> •
  <a href="#supported-platforms">Platforms</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**OmniInfer** is the high-performance inference backend engine for [Omni Studio](https://github.com/omnimind-ai), designed to enable efficient local deployment of large language models (LLMs) and vision-language models (VLMs) across multiple platforms and backends.

OmniInfer abstracts away the complexity of model compilation, hardware adaptation, and deployment, allowing users to run LLMs and VLMs locally with minimal configuration. It serves as the core inference engine powering Omni Studio's seamless cross-platform experience.

## Features

- **🚀 High Performance** - Optimized inference with minimal memory footprint and fast token generation
- **🔧 Multi-Backend Support** - Flexible backend architecture supporting various inference engines
- **📱 Cross-Platform** - Run models on Android, iOS, macOS, Windows, and Linux
- **🤖 LLM & VLM Support** - Run both large language models and vision-language models
- **🔌 OpenAI API Compatible** - Easy integration with existing applications and workflows
- **🛡️ Privacy First** - All inference runs locally, keeping your data private

## Supported Backends

OmniInfer supports multiple inference backends, currently organized as separate branches:

| Backend | Branch | Description |
|---------|--------|-------------|
| **llama.cpp** | `main` | GGML-based inference with broad model format support |
| **OmniInfer Native** | `feature/llm-backend` | Our self-developed inference engine with optimized performance |

> **🚧 Work in Progress:** We are actively integrating multiple backends into a unified architecture. A more streamlined and user-friendly multi-backend experience is coming soon!

## Supported Platforms

| Platform | Status |
|----------|--------|
| Linux | ✅ Supported |
| macOS | ✅ Supported |
| Windows | ✅ Supported |
| Android | ✅ Supported |
| iOS | ✅ Supported |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

OmniInfer builds upon the excellent work of the open-source community, including:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGML-based inference
- [GGML](https://github.com/ggerganov/ggml) - Tensor library for machine learning

---

<p align="center">
  Made with ❤️ by the <a href="https://github.com/omnimind-ai">OmniMind AI</a> team
</p>