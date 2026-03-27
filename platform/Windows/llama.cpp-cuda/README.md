# OmniInfer Windows CUDA Backend

This package hosts the `llama.cpp-cuda` runtime used by the unified OmniInfer service.

Key paths:

- `bin/`: put the CUDA-enabled `llama-server.exe` here
- `models/`: optional GPU-local model directory
- `logs/`: runtime logs
- `scripts/build-llama-cuda.ps1`: build a CUDA-enabled `llama-server.exe`

The OmniInfer service selects this backend through:

```http
POST /omni/backend/select
```

with:

```json
{
  "backend": "llama.cpp-cuda"
}
```

For actual inference, use:

- `POST /omni/model/select`
- `POST /v1/chat/completions`
