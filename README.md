<p align="center">
  <picture>
    <img alt="OmniInfer-VLM" src="./assets/OmniInfer-VLM-logo.png" width="60%">
  </picture>
</p>

<h3 align="center">
High-Performance Multimodal Embedding Generation with HTP Acceleration
</h3>

<p align="center">
| 
<a href="#build-instructions"><b>Build</b></a> 
| 
<a href="#usage"><b>Usage</b></a> 
| 
<a href="#performance-benchmarks"><b>Benchmarks</b></a> 
| 
<a href="https://github.com/omnimind-ai/OmniOp-NPU"><b>Backend (OmniOp)</b></a> 
|
</p>

---
**OmniInfer-VLM** is a specialized inference framework designed to generate embedding layer data for Vision-Language Models (VLMs). It is optimized for high-performance deployment, featuring support for Qualcomm's Hexagon Tensor Processor (HTP) via the [OmniOp-NPU](https://github.com/omnimind-ai/OmniOp-NPU) backend, alongside standard CPU execution.

This project streamlines the pipeline for multimodal inference, allowing efficient processing of image and text inputs on both Linux (CPU) and Android (NPU) platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Build Instructions](#build-instructions)
  - [Linux (CPU)](#linux-cpu)
  - [Android (NPU)](#android-npu)
- [Usage](#usage)
  - [Command Arguments](#command-arguments)
  - [Running on Linux](#running-on-linux)
  - [Running on Android](#running-on-android)
- [Important Notes](#important-notes)

## Prerequisites

To utilize the hardware acceleration features of this repository, specifically for the HTP backend, you must have the dependencies from the **OmniOp-NPU** project.

*   **OmniOp-NPU Repository:** [https://github.com/omnimind-ai/OmniOp-NPU](https://github.com/omnimind-ai/OmniOp-NPU)

For Android builds, ensure you have a valid Android NDK environment configured.

## Build Instructions

### Linux (CPU)

Follow these steps to build the CLI tool for a Linux environment using the CPU backend.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/OmniInfer-VLM.git
    cd OmniInfer-VLM
    ```

2.  **Configure and Build**
    Use the following CMake commands to configure the build with HTP support enabled (simulated or strictly linked) and compile the binary.

    ```bash
    cmake -B build \
        -DGGML_OPENML=OFF \
        -DGGML_HTP=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_OPENMP=OFF

    cmake --build build --config Release --target llama-mtmd-cli
    ```

### Android (NPU)

To target Android devices with Qualcomm NPUs, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/OmniInfer-VLM.git
    cd OmniInfer-VLM
    ```

2.  **Configure and Build**
    *Note: Ensure your CMake environment is configured for Android cross-compilation (e.g., using the NDK toolchain).*

    ```bash
    cmake -B build \
        -DGGML_OPENML=OFF \
        -DGGML_HTP=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_OPENMP=OFF

    cmake --build build --config Release --target llama-mtmd-cli
    ```

3.  **Install OmniOp-NPU Dependencies**
    After compiling `llama-mtmd-cli`, you must copy the **two generated dependency files** (shared libraries or artifacts) from the [OmniOp-NPU](https://github.com/omnimind-ai/OmniOp-NPU) build process into the root directory of this repository.

## Usage

The main executable is `llama-mtmd-cli`. It requires a text model, a vision projection (ViT) model, a prompt, and an input image.

### Command Arguments

| Argument | Description |
| :--- | :--- |
| `-m <file>` | Path to the main text model file (GGUF format). |
| `--mmproj <file>` | Path to the Vision Transformer/Projector file (GGUF format). |
| `-p "text"` | The text prompt/instruction (e.g., "Describe this image"). |
| `--image <file>` | Path to the input image file (JPEG/PNG). |

### Running on Linux

Execute the binary directly from the build directory:

```bash
./build/bin/llama-mtmd-cli \
    -m <YOUR_TEXTMODEL_GGUF_FILE> \
    --mmproj <YOUR_VIT_GGUF_FILE> \
    -p "Describe this image in short." \
    --image <YOUR_IMAGE_PATH>
```

### Running on Android

When running on Android, you must ensure the dynamic libraries are correctly located by setting the `LD_LIBRARY_PATH`.

```bash
LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:. ./build/bin/llama-mtmd-cli \
    -m <YOUR_TEXTMODEL_GGUF_FILE> \
    --mmproj <YOUR_VIT_GGUF_FILE> \
    -p "Describe this image in short." \
    --image <YOUR_IMAGE_PATH>
```

## Important Notes

*   **ViT Model Selection (HTP Backend):** When using the HTP backend (especially on NPU), it is **strongly recommended** to use a **cropped or appropriately pruned** version of the `<YOUR_VIT_GGUF_FILE>`. Using a full-sized or incompatible ViT model may result in undefined behavior, execution failures, or unknown errors during inference.

## License and Acknowledgements

This project is licensed under the Apache License, Version 2.0.

This repository contains code derived from the following open-source project:

- <Upstream Project Name> (https://github.com/ggml-org/llama.cpp), licensed under the MIT License

## Community & Support

Join our community to stay updated with the latest VLM optimization techniques and mobile deployment guides.

<table align="center">
  <tr>
    <td align="center">
      <img src="./assets/wechat_qr.png" alt="WeChat Group" width="220"/><br/>
      <b>WeChat Group</b>
    </td>
    <td align="center">
       <a href="https://github.com/omnimind-ai/OmniInfer-VLM/issues">
        <img src="https://img.shields.io/badge/GitHub-Issues-blue?logo=github" alt="GitHub Issues"/>
      </a><br/>
      <b>Technical Support</b>
    </td>
  </tr>
</table>

---
