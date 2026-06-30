# Python Runtime Audit

Last updated: 2026-06-30.

This audit tracks Python usage that matters to the desktop OmniInfer control
plane. Vendored framework trees under `framework/` are excluded unless an
OmniInfer release script calls them directly.

## Target

The default user-facing CLI and portable packages should run through the Rust
control plane without requiring `omniinfer.py` or `service_core/`.

Python may remain in three explicitly declared places:

- Development, packaging, and validation tooling.
- A temporary legacy fallback package mode, enabled explicitly with
  `--include-python-fallback`.
- Backend-specific runtimes where the backend itself is distributed as Python
  packages, such as MLX, MNN, or vLLM.

## User Runtime Surface

| Surface | Classification | Current action |
|---|---|---|
| `omniinfer` / `omniinfer.cmd` / `omniinfer.ps1` repo launchers | Compatibility entrypoints | Rust is the default. `OMNIINFER_FORCE_PYTHON=1` remains available only when `omniinfer.py` exists in the checkout. |
| `crates/omniinfer-cli` default CLI path | Rust runtime | Primary user runtime. Missing Python fallback returns a clear package-mode error. |
| `crates/omniinfer-cli` Python fallback | Legacy fallback | Kept for source checkout rollback and explicit compatibility packages. Not copied into portable packages by default. |
| `crates/omniinfer-cli` Python compatibility upstream for embedded backends | Backend compatibility | Required only when the selected backend has `runtime_mode = embedded`. No-Python packages must use external-server backends only. |
| `omniinfer.py` and `service_core/` | Legacy Python control plane | Excluded from default portable packages. Included only with `--include-python-fallback`. |

## Backend Runtime Surface

| Backend family | Classification | Notes |
|---|---|---|
| `llama.cpp-*`, `ik_llama.cpp-*`, `turboquant-*` external server backends | Rust-compatible runtime | Suitable for no-Python portable packages when the backend launcher is packaged under `runtime/<backend>/bin`. |
| `mlx-mac` | Backend-specific Python runtime | Requires packaged Python environment and legacy compatibility path today. macOS release packaging requires `--include-python-fallback` when `mlx-mac` is selected. |
| `mnn-linux` | Backend-specific Python runtime | Requires embedded Python modules. Linux release packaging rejects no-Python packages that include embedded backends. |
| `vllm-linux-cuda` | Backend-specific Python runtime | vLLM is distributed as a Python/PyTorch runtime. Treat it as backend-specific Python, not as the default OmniInfer control-plane dependency. |

## Tooling Surface

| Surface | Classification | Notes |
|---|---|---|
| `scripts/platforms/common/package-rust-cli.py` | Packaging tool | May use host Python. It produces no-Python portable roots by default. |
| `scripts/platforms/linux/release_runtime_backends.py` | Packaging tool | Uses Python to discover/copy Linux runtime packages. It is not copied into the release as user runtime. |
| `scripts/platforms/*/build-release.*` | Packaging tool | May require host Python to build or assemble packages. This is acceptable for release builders. |
| `scripts/validate_rust_control_plane.py`, `scripts/capture_cli_contracts.py`, `scripts/profile_python_cli.py` | Validation tooling | Historical Rust/Python parity tooling. Not part of user runtime. |
| `tests/*.py` | Test tooling | Kept for legacy behavior and service-core coverage while fallback exists. |
| `pyproject.toml` | Development package metadata | Tracks legacy Python package/test dependencies only. |

## No-Python Release Rules

1. Default portable packaging must not copy `omniinfer.py` or `service_core/`.
2. Compatibility packages must opt in with `--include-python-fallback`.
3. No-Python packages must contain only Rust-compatible external-server
   backends.
4. Embedded Python backends must fail package assembly unless
   `--include-python-fallback` is explicit.
5. Validation must check that default portable roots do not contain
   `omniinfer.py`, `service_core/`, or launcher text that auto-selects Python
   fallback runtimes.
