# Python Runtime Audit

Last updated: 2026-06-30.

This audit tracks Python usage that matters to the desktop OmniInfer control
plane. Vendored framework trees under `framework/` are excluded unless an
OmniInfer release script calls them directly.

## Target

The default user-facing CLI and portable packages should run through the Rust
control plane without requiring `omniinfer.py` or `service_core/`.

Python may remain in explicitly declared places:

- Development, packaging, and validation tooling.
- Backend-specific runtimes where the backend itself is distributed as Python
  packages, such as MLX, MNN, or vLLM.
- Legacy source files and service-core tests while they are still useful for
  reference and parity audits. They are not part of the default user runtime.

## User Runtime Surface

| Surface | Classification | Current action |
|---|---|---|
| `omniinfer` / `omniinfer.cmd` / `omniinfer.ps1` repo launchers | Rust entrypoints | Start the Rust control plane directly. They no longer honor legacy Python control-plane environment variables. |
| `crates/omniinfer-cli` default CLI path | Rust runtime | Primary user runtime. Unsupported commands return a clear Rust error instead of delegating to Python. |
| `crates/omniinfer-cli` embedded backend handling | Rust runtime policy | Embedded backends are rejected until they are exposed through an external adapter service or Rust-native driver. |
| `omniinfer.py` and `service_core/` | Legacy Python control plane source | Kept in the repository for reference/tests, but excluded from portable packages and not used as fallback. |

## Backend Runtime Surface

| Backend family | Classification | Notes |
|---|---|---|
| `llama.cpp-*`, `ik_llama.cpp-*`, `turboquant-*` external server backends | Rust-compatible runtime | Suitable for no-Python portable packages when the backend launcher is packaged under `runtime/<backend>/bin`. |
| `mlx-mac` | Backend-specific Python runtime | Embedded backend. Packaging rejects it until an adapter service or Rust-native driver exists. |
| `mnn-linux` | Backend-specific Python runtime | Embedded backend. Linux release packaging rejects it until an adapter service or Rust-native driver exists. |
| `vllm-linux-cuda` | Backend-specific Python runtime | vLLM is distributed as a Python/PyTorch runtime. Treat it as backend-specific Python, not as the default OmniInfer control-plane dependency. |

## Tooling Surface

| Surface | Classification | Notes |
|---|---|---|
| `scripts/platforms/common/package-rust-cli.py` | Packaging tool | May use host Python. It produces no-Python portable roots by default. |
| `scripts/platforms/linux/release_runtime_backends.py` | Packaging tool | Uses Python to discover/copy Linux runtime packages. It is not copied into the release as user runtime. |
| `scripts/platforms/*/build-release.*` | Packaging tool | May require host Python to build or assemble packages. This is acceptable for release builders. |
| `scripts/validate_rust_control_plane.py`, `scripts/capture_cli_contracts.py`, `scripts/profile_python_cli.py` | Validation tooling | Not part of user runtime. `profile_python_cli.py` is still used for process profiling despite its historical name. |
| `tests/*.py` | Test tooling | Kept for legacy service-core coverage and parity reference. |
| `pyproject.toml` | Development package metadata | Tracks legacy Python package/test dependencies only. |

## No-Python Release Rules

1. Default portable packaging must not copy `omniinfer.py` or `service_core/`.
2. Portable packages must contain only Rust-compatible external-server
   backends.
3. Embedded Python backends must fail package assembly until they have an
   external adapter service or Rust-native driver.
4. Validation must check that default portable roots do not contain
   `omniinfer.py`, `service_core/`, or launcher text that auto-selects Python
   control-plane runtimes.
