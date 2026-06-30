# TODO

## Embedded Backend Strategy

- Define the long-term path for embedded MLX/MNN runtimes after the Rust-only
  control-plane migration.
- Preferred directions to evaluate later:
  - adapter services that expose an OpenAI-compatible local HTTP server;
  - Rust-native runtime drivers where an embedded integration is genuinely
    needed.
- Until that work is designed and validated, release packaging and the Rust
  gateway should reject embedded backends with clear errors.
