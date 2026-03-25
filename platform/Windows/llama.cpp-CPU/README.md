# OmniInfer Windows CPU Package

This directory contains the minimum Windows package for the first OmniInfer
backend:

- `llama.cpp-CPU` as the runtime backend
- `llama-server.exe` as the OpenAI-compatible inference engine
- the repository-root `omniinfer_gateway.py` as the OmniInfer control plane and proxy
- an implementation derived from `E:\Coding\repository\llama.cpp\dist\omniserver.py`

## Layout

- `bin/`: runtime binaries copied from the llama.cpp build
- `models/`: local GGUF model directory used by the OmniInfer gateway
- `logs/`: runtime logs
- `scripts/build-llama-cpu.ps1`: build and package llama-server
- `scripts/test-multimodal.ps1`: send a multimodal OpenAI-style request with a local image

## Toolchain Notes

- Preferred: Visual Studio 2022 Build Tools from a Developer PowerShell
- Supported fallback: MinGW-w64 with the `posix` thread model
- Not supported: MinGW-w64 with the `win32` thread model

The build script checks the detected compiler and stops early if the toolchain
is known to be incompatible.

## Control Plane Endpoints

- `GET /omni/state`
- `GET /omni/backends`
- `POST /omni/backend/select`
- `POST /omni/backend/stop`
- `GET /omni/models`
- `POST /omni/model/select`

The gateway starts with `llama.cpp(CPU)` selected by default.

## OpenAI-Compatible Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`

Model loading is handled by the gateway. After a model is loaded, requests are
forwarded to `llama-server` through `POST /v1/chat/completions`.

## Start Command

Run the gateway directly from the repository root:

```powershell
python .\omniinfer_gateway.py
```
