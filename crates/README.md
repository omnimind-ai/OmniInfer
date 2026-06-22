# OmniInfer Rust Control Plane

This workspace is an incremental Rust rewrite of the OmniInfer control plane:
CLI parsing, local state/config handling, gateway orchestration, and the future
TUI. It intentionally does not rewrite inference runtimes such as llama.cpp,
vLLM, MLX, or MNN.

Current scope:

- `omniinfer-core`: shared local paths, config compatibility, state parsing, and
  minimal local HTTP helpers.
- `omniinfer-cli`: experimental `omniinfer-rs` binary with the target command
  surface, read-only `status`, and shell completion generation.

The production entrypoint remains `./omniinfer` until the Rust implementation
covers the required behavior and passes compatibility checks.

See [`docs/rust-control-plane.md`](../docs/rust-control-plane.md) for the
switching checklist, fallback controls, and profiling commands.

## Local Development

```bash
cargo test --workspace
cargo run -p omniinfer-cli -- --help
cargo run -p omniinfer-cli -- status
cargo run -p omniinfer-cli -- completion bash
```

This machine currently lacks the `rustfmt` component. Run `cargo fmt --all`
when rustfmt is available.

## Profiling

Capture the Python baseline:

```bash
python3 scripts/profile_python_cli.py \
  --runs 7 \
  --output-dir tmp/test_results/20260622-rust-control-plane-python-profile
```

Capture the Rust prototype for migrated commands:

```bash
python3 scripts/profile_python_cli.py \
  --runs 7 \
  --binary target/debug/omniinfer-rs \
  --scenario help \
  --scenario status \
  --skip-import-trace \
  --output-dir tmp/test_results/20260622-rust-control-plane-rust-profile
```
