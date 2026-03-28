# OmniInfer macOS Metal Backend

This package hosts the `llama.cpp-mac` runtime used by OmniInfer on Apple Silicon/macOS.

Key paths:

- `bin/`: put the macOS `llama-server` binary here
- `models/`: optional local model directory for macOS runs
- `logs/`: runtime logs
- `scripts/build-llama-mac.sh`: build a Metal-enabled `llama-server`

The OmniInfer service selects this backend by default on macOS. You can also select it explicitly:

```http
POST /omni/backend/select
```

with:

```json
{
  "backend": "llama.cpp-mac"
}
```

For actual inference, use:

- `POST /omni/model/select`
- `POST /v1/chat/completions`
