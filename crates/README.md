# OmniInfer Rust Control Plane

This workspace is an incremental Rust rewrite of the OmniInfer control plane:
CLI parsing, local state/config handling, gateway orchestration, and the future
TUI. It intentionally does not rewrite inference runtimes such as llama.cpp,
vLLM, MLX, or MNN.

Current scope:

- `omniinfer-core`: shared local paths, config compatibility, state parsing, and
  minimal local HTTP helpers.
- `omniinfer-cli`: Rust control-plane binary with the target command surface,
  gateway orchestration, runtime management, and shell completion generation.

The production entrypoint is `./omniinfer`, which starts the Rust control
plane. Python control-plane fallback has been removed; unsupported commands
return explicit Rust errors.

Use `OMNIINFER_RUST_STATE_ROOT=/tmp/omniinfer-state` when running isolated
integration checks so test state, logs, and backend profiles do not mutate the
real checkout.

## Local Development

```bash
cargo test --workspace
cargo run -p omniinfer-cli -- --help
cargo run -p omniinfer-cli -- status
cargo run -p omniinfer-cli -- completion bash
```

Run `cargo fmt --all -- --check` and `cargo test --workspace` before each
Rust control-plane commit.

## Profiling

Capture Rust command profiles:

```bash
python3 scripts/profile_python_cli.py \
  --runs 7 \
  --binary target/debug/omniinfer-rs \
  --scenario help \
  --scenario status \
  --skip-import-trace \
  --output-dir tmp/test_results/20260622-rust-control-plane-rust-profile
```
