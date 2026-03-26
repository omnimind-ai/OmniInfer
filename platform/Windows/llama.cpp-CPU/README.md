# OmniInfer Windows CPU Backend

This package hosts the `llama.cpp-CPU` runtime used by the unified OmniInfer service.

Key paths:

- `bin/`: put the CPU `llama-server.exe` here
- `models/`: CPU-local model directory
- `logs/`: runtime logs
- `scripts/build-llama-cpu.ps1`: build a CPU `llama-server.exe`
- `scripts/test-multimodal.ps1`: smoke-test a multimodal request through OmniInfer

The OmniInfer service selects this backend by default. You can also select it explicitly:

```http
POST /omni/backend/select
```

with:

```json
{
  "backend": "llama.cpp-CPU"
}
```

For actual inference, use:

- `POST /omni/model/select`
- `POST /v1/chat/completions`
