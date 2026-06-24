# Rust Control Plane Migration

OmniInfer is migrating CLI, TUI, gateway orchestration, and local service
management to Rust. Inference runtimes remain external runtime backends such as
llama.cpp, vLLM, MLX, and MNN.

The source-checkout `./omniinfer` entrypoint now defaults to the Rust control
plane on this branch. Build or refresh it with:

```bash
cargo build -p omniinfer-cli
./omniinfer --help
```

Set `OMNIINFER_FORCE_PYTHON=1` to run the legacy Python entrypoint during the
rollback window.

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
  graceful shutdown after `/omni/shutdown`. Rust normalizes
  `/v1/chat/completions` bodies before forwarding, including thinking/reasoning
  aliases, legacy function tools, request defaults, and stream-format options.
- Rust-native gateway runtime management for external-server backends,
  including `/omni/backend/select`, `/omni/backend/stop`,
  `/omni/model/select`, `/health`, `/omni/state`, `/omni/backends`,
  `/v1/models`, and direct `/v1/chat/completions` forwarding to the loaded
  backend after Rust request normalization.
- Embedded backend compatibility in the Rust gateway: embedded
  `/omni/model/select` requests are delegated to the Python upstream while the
  Rust external-server runtime is stopped, preserving MNN/MLX embedded driver
  behavior without making Rust own those in-process runtimes.
- Rust-native compatibility endpoints for loaded external-server backends:
  Anthropic `/v1/messages` request/response conversion, true incremental
  Anthropic SSE streaming converted from OpenAI chat streams, `/tokenize`,
  `/detokenize`, `/omni/tokenize`, `/omni/detokenize`, and
  `/omni/cache/clear` via llama.cpp slot erase. When no Rust external runtime
  is loaded, these endpoints still fall back to the Python upstream for
  embedded and legacy compatibility.
- Rust-native small management endpoints for loaded external-server backends:
  `/omni/thinking`, `/omni/thinking/select`, `/omni/backend/props`, and the
  deprecated `/omni/models` compatibility response.
- Bundled model catalog handling for `model list`,
  `/omni/supported-models`, and `/omni/supported-models/best`, including
  installed-backend filtering, best-backend merge, and local RAM/VRAM fit
  annotations.
- llama.cpp-compatible model artifact discovery, including recursive directory
  scanning for a single text GGUF, mmproj GGUF detection, sibling/project-root
  mmproj auto-discovery, and ambiguity errors for multiple text models or
  projectors.
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

Rust backend registry now mirrors the Python backend templates for Linux,
Windows, macOS, Android, and iOS, including backend priority, runtime paths,
environment/config overrides, installed-runtime probing, capabilities,
model-artifact semantics, and OpenAI-server protocol metadata. Read-only
backend/advisor/TUI flows use this registry directly, so `backend list`,
`advisor system`, `advisor fit`, `advisor plan`, `advisor recommend`, and TUI
advisor preflight no longer need to auto-start the Python gateway just to read
backend metadata.

Rust runtime preparation now includes two no-user-visible core layers:

- `runtime_plan` builds Python-compatible external backend launch commands for
  llama.cpp-compatible servers, ik_llama.cpp, and vLLM OpenAI servers,
  including ctx-size replacement, reserved OmniInfer-managed flag validation,
  slot-save paths, mmproj args, vLLM served-model-name defaults, cwd, log file
  name, and proxy model ref.
- `runtime_process` starts an external runtime command, redirects stdout/stderr
  to the runtime log, waits for `/health`, and stops the child with graceful
  termination followed by kill-on-timeout cleanup. This is covered by unit tests
  and is wired into the Rust gateway path for external-server backends.

Gateway replacement covers the user-facing HTTP server layer and the primary
external-server runtime path. Rust listens on the public/local `serve` port,
handles the core management endpoints listed above, launches external
llama.cpp/ik/vLLM-style runtimes directly, synthesizes Python-compatible
health/state snapshots, and forwards OpenAI chat directly to the loaded backend.
Endpoints that are not yet native still proxy to the loopback Python upstream,
preserving compatibility while the remaining control-plane surface is migrated.

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

Capture Python baseline through the forced fallback path:

```bash
python3 scripts/profile_python_cli.py \
  --force-python \
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

## Post-Switch Validation

Run the consolidated local validation after changing `./omniinfer` to the Rust
entrypoint:

```bash
python3 scripts/validate_rust_control_plane.py \
  --runs 7 \
  --output-dir tmp/test_results/rust-control-plane-post-switch-validation
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
`tmp/test_results/20260624-final-rust-entrypoint-validation-2/summary.md`.
That run passed formatting, workspace tests, Python contracts through
`OMNIINFER_FORCE_PYTHON=1`, strict Rust contracts, forced-Python contracts,
Python/Rust profiles, and `git diff --check`.

Latest Rust-native gateway smoke artifact:
`tmp/test_results/20260623-rust-native-gateway-real-smoke/summary.md`. It
loaded Qwen3.5 4B through the Rust gateway's native `/omni/model/select`,
reported `backend_ready=true`, exposed one model through `/v1/models`, and
returned `rust-native-ok` through direct OpenAI chat forwarding.

Latest compatibility endpoint smoke artifact:
`tmp/test_results/20260624-real-smoke-entrypoint-anthropic-stream/summary.md`.
It started the switched `./omniinfer` entrypoint, loaded Qwen3.5 4B through
Rust serve on an isolated state root, and verified real Anthropic
`/v1/messages` streaming produced `message_start`, `content_block_delta`, and
`message_stop` before the temporary service was stopped.

Latest VLM/mmproj smoke artifact:
`tmp/test_results/20260623-vlm-mmproj-after-rust-native-full/summary.md`. It
validated the Rust `serve` orchestration path with Stepfun 4B plus mmproj and
Qwen3.6 27B with both BF16 and F16 projectors, including real image chat
requests through the OpenAI-compatible endpoint.

## After Switching `./omniinfer`

The default source-checkout entrypoint has been switched to Rust on
`refactor/rust-control-plane`. The following entrypoints are expected to work:

- `./omniinfer ...` uses `target/debug/omniinfer-rs` by default when the binary
  exists.
- `OMNIINFER_FORCE_PYTHON=1 ./omniinfer ...` keeps the Python fallback available
  for rollback.
- `target/debug/omniinfer-rs ...` remains available for direct Rust testing.

The switch is intentionally limited to this source-checkout launcher until
packaging and cross-platform wrapper validation are recorded.

Remaining work after the current Linux switch:

- Run cross-platform launcher validation on Windows and macOS, including
  PowerShell/cmd wrappers and packaged source-checkout behavior.
- Keep `OMNIINFER_FORCE_PYTHON=1` available for at least one release cycle and
  monitor real user paths before removing the fallback.
- Finish packaging/release wiring after the user validates the local source
  checkout behavior.
- Decide whether embedded MLX/MNN in-process drivers should remain delegated to
  Python or become a separate Rust-native runtime-driver project.
