# Rust Control Plane Migration

OmniInfer is migrating CLI, TUI, gateway orchestration, and local service
management to Rust. Inference runtimes remain external runtime backends such as
llama.cpp, vLLM, MLX, and MNN.

The production entrypoint is still `./omniinfer`. The experimental Rust
entrypoint is:

```bash
cargo build -p omniinfer-cli
target/debug/omniinfer-rs --help
```

## Current Rust Coverage

Implemented directly in Rust:

- `--help`
- `status`
- `ps`, including detached serve pid files and `--json`
- `backend list`
- `backend select`
- `backend stop`
- `model list`
- `model load` / `load`, including `/omni/model/select` JSON responses, SSE
  progress events, backend extra args, and selected backend/model state updates.
- `advisor system` / `advisor system --json`, including host RAM/CPU probe,
  CUDA device probe, usable backend table, full JSON probe, and recommended
  installed backend derived from the current backend registry.
- `advisor inspect`, `advisor fit`, `advisor plan`, and `advisor recommend`,
  including local model artifact inspection, memory breakdown estimates,
  evidence/confidence labels, backend fit ranking, hardware planning
  simulations, and locally managed model recommendations.
- no-argument TUI, including backend/model selection, managed model linking,
  advisor badges and load preflight, streaming chat, conversation slash
  commands, thinking/reasoning toggles, status display, and shutdown on exit.
- interactive `serve` launcher in TTY sessions, reusing the Rust serve
  orchestration path after backend/model selection.
- `thinking show`
- `thinking set on|off`
- `serve status --port <port>`
- `serve stop --port <port>`
- local `serve --detach`, including optional backend selection, model loading,
  local smoke tests, serve pid files, and ready/curl output.
- foreground `serve`, using the same orchestration path as detached serve and
  waiting on the service process after the ready block.
- `serve --lan` in foreground or detached mode, including safe default
  `0.0.0.0` binding, generated/configured API keys unless explicitly insecure,
  LAN URL/curl output, and remote-management key validation.
- `serve --cloudflare --detach`, including loopback gateway binding, managed or
  explicit `cloudflared` resolution, Quick Tunnel URL parsing, generated or
  configured API keys, serve pid files with tunnel pid, stop cleanup, and
  ready/curl output. `--smoke-test` first validates the local OpenAI endpoint,
  then validates the public HTTPS `trycloudflare.com` URL with the same bearer
  key and short retries for fresh Quick Tunnel DNS/edge propagation. Transient
  public DNS/edge failures are reported as warnings after local smoke passes,
  because the tunnel URL can become reachable after the CLI has printed the
  ready block.
- Rust-facing HTTP gateway/proxy used by `serve`, including auth policy, CORS
  preflight responses, OpenAI/Anthropic/public endpoint forwarding, local-only
  management protection, SSE streaming responses, body/query forwarding, and
  graceful shutdown after `/omni/shutdown`.
- `shutdown`
- `completion bash`
- `chat <prompt>`, `chat --no-stream <prompt>`, and local `chat --image <path>`
  requests.

Fallback to the Python implementation:

- all other unported commands

## Fallback Controls

The Rust entrypoint supports explicit fallback controls:

```bash
OMNIINFER_FORCE_PYTHON=1 target/debug/omniinfer-rs status
OMNIINFER_RUST_STRICT=1 target/debug/omniinfer-rs advisor system
OMNIINFER_RUST_STATE_ROOT=/tmp/omniinfer-state target/debug/omniinfer-rs status
```

- `OMNIINFER_FORCE_PYTHON=1` runs `omniinfer.py` for every command.
- `OMNIINFER_RUST_STRICT=1` disables fallback and shows which command is still
  pending in Rust.
- `OMNIINFER_RUST_STATE_ROOT=/path/to/root` keeps `.local/`, `config/`, logs,
  run files, state, and backend profiles under a separate root while
  `OMNIINFER_RUST_REPO_ROOT` continues to identify the source checkout. This is
  useful for isolated integration tests against the real Python gateway. Rust
  serve orchestration passes both path overrides into Python child processes so
  model loading, health checks, logs, and pid files use the same state root.
- `OMNIINFER_PYTHON=/path/to/python` selects the Python executable used by
  fallback.

Migrated Rust commands that need the local gateway automatically start
`python omniinfer.py serve` with `config/omniinfer.json` host, port,
startup-timeout, window-mode, default-thinking, and default-backend settings if
the gateway is not already healthy. `status` and `shutdown` intentionally do
not auto-start the gateway.

Rust core now includes pre-switch helpers for model loading: backend-native load
extra-arg parsing for llama.cpp/turboquant/vLLM/MLX/generic families, and
backend profile loading for `load.extra_args`, legacy `load.launcher_args`, and
infer request defaults. It also includes pure payload construction for
`/omni/model/select`, including selected-backend resolution, model/mmproj path
validation, ctx-size precedence, launch args, and request defaults. The Rust
path can POST JSON model-load responses and persist selected backend/model
state. It also parses model-load SSE progress, done, and error events while
printing progress as event lines arrive.

Rust core also includes a small HTTPS JSON client for public smoke tests. Local
HTTP streaming paths still use the existing hand-written localhost client, while
public JSON checks use a blocking rustls-backed client suitable for CLI control
flow.

Gateway replacement now covers the user-facing HTTP server layer. Rust listens
on the public/local `serve` port and proxies to a loopback Python upstream that
still owns runtime/backend orchestration and request normalization. This keeps
runtime behavior compatible while moving the externally exposed gateway surface,
auth boundary, CORS handling, SSE pass-through, and `/omni/shutdown` lifecycle
into Rust.

## Contract Snapshots

Capture current Python contracts:

```bash
python3 scripts/capture_cli_contracts.py \
  --output-dir tmp/test_results/rust-control-plane-python-contracts
```

Capture Rust wrapper contracts:

```bash
python3 scripts/capture_cli_contracts.py \
  --binary target/debug/omniinfer-rs \
  --output-dir tmp/test_results/rust-control-plane-wrapper-contracts
```

Capture only strict Rust paths, with fallback disabled:

```bash
python3 scripts/capture_cli_contracts.py \
  --binary target/debug/omniinfer-rs \
  --rust-strict \
  --scenario status \
  --scenario serve-status \
  --output-dir tmp/test_results/rust-control-plane-strict-contracts
```

Capture forced Python fallback through the Rust wrapper:

```bash
python3 scripts/capture_cli_contracts.py \
  --binary target/debug/omniinfer-rs \
  --force-python \
  --output-dir tmp/test_results/rust-control-plane-force-python-contracts
```

The contract snapshot records command, exit code, stdout/stderr hashes, preview
text, and whether read-only scenarios modified `.local/config/state.json`.
Use `--state-root tmp/test_results/<run>/state` when validating the Rust
entrypoint so contract runs do not mutate the real checkout state.

## Profiling

Capture Python baseline:

```bash
python3 scripts/profile_python_cli.py \
  --runs 7 \
  --output-dir tmp/test_results/rust-control-plane-python-profile
```

Capture Rust read-only paths:

```bash
python3 scripts/profile_python_cli.py \
  --runs 7 \
  --binary target/debug/omniinfer-rs \
  --scenario status \
  --scenario thinking-show \
  --scenario serve-status \
  --skip-import-trace \
  --output-dir tmp/test_results/rust-control-plane-rust-readonly-profile
```

Use `--state-root` for the same reason as contract snapshots.

## Pre-Switch Validation

Run the consolidated local validation before changing `./omniinfer` to the
Rust entrypoint:

```bash
python3 scripts/validate_rust_control_plane.py \
  --runs 3 \
  --output-dir tmp/test_results/rust-control-plane-pre-switch-validation
```

The script records:

- `cargo fmt --all -- --check`
- `cargo test --workspace`
- Python, strict Rust, and forced-Python CLI contract snapshots
- Python and Rust CLI profiling summaries
- `git diff --check`

Each contract/profile run uses an isolated state root and writes artifacts under
the selected output directory.

Latest local validation artifact:
`tmp/test_results/20260623-rust-control-plane-validation-smoke-2/summary.md`.
That run passed formatting, workspace tests, Python contracts, strict Rust
contracts, forced-Python contracts, Python/Rust profiles, and `git diff
--check`.

## Before Switching `./omniinfer`

Do not switch the default entrypoint until these conditions are satisfied:

- All user-visible commands parse in Rust and either run natively or fallback
  without changing stdout/stderr/exit-code semantics.
- Contract snapshots pass for Python, strict Rust, and forced-Python wrapper
  entrypoints.
- State/config compatibility is covered for `.local/config/state.json`,
  legacy `cli_state.json`, backend profiles, serve pid files, and
  `config/omniinfer.json`.
- Runtime process management is ported or explicitly delegated to fallback.
- Gateway API compatibility is covered for `/health`, OpenAI chat, Anthropic
  messages, management endpoints, CORS, auth, SSE pass-through, and shutdown.
- One-command public serve still supports Cloudflare, `--detach`,
  `--api-key auto`, smoke tests, status, stop, logs, and curl output.
- TUI either has a Rust implementation or intentionally falls back to Python.
- Linux, macOS, and Windows launchers have equivalent fallback behavior.
- `cargo fmt --all -- --check`, `cargo test --workspace`, Python compatibility
  tests, contract snapshots, and profiling comparisons are recorded.

Only after that should `./omniinfer` default to Rust. Keep the Python fallback
available for at least one release cycle.

Remaining pre-switch work after the current Linux validation:

- Run the consolidated validation with more profile samples, for example
  `--runs 3` or `--runs 7`, before making performance claims.
- Run cross-platform launcher validation on Windows and macOS, including
  PowerShell/cmd wrappers and packaged source-checkout behavior.
- Run a real VLM smoke matrix for image chat and mmproj load paths.
- Decide whether the long-term no-Python target requires replacing the Python
  runtime manager/backend registry/request normalization; the current Rust
  gateway intentionally proxies to a loopback Python upstream.
- Switch `./omniinfer` only after the above checks are recorded, then keep
  `OMNIINFER_FORCE_PYTHON=1` available for rollback during at least one release.
