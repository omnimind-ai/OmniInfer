# OmniInfer vla.cpp LIBERO live demo (Linux)

This example demonstrates the complete managed path:

```text
browser dashboard
  -> LIBERO simulator
  -> vla.cpp-compatible request preprocessing and protobuf client
  -> OmniInfer-managed vla-server
```

The dashboard shows the live front/wrist camera views, the 7-DoF action sent to
LIBERO, model/policy/simulator/control-loop latency, task text, episode progress,
and explicit success, failure, stop, or error results. LIBERO also writes the
episode MP4 under `--output-dir`.

The browser can select any of the ten predefined `libero_object` tasks before
starting a rollout. The task selector sends a LIBERO task id, not arbitrary
prompt text, so the language instruction, scene, target object, and success
condition remain consistent. Selection is locked while a rollout is active.
For multi-episode runs, the final status is `success`, `failed`, or `partial`;
`partial` means that the same run contained both successful and failed episodes.
The dashboard exposes the SmolVLA and PI0.5 request formats supported by the
vla.cpp LIBERO client. SmolVLA is the validated end-to-end example path. PI0.5
is an experimental request path: its tokenizer, state-normalization, and action
chunk wiring are covered here, but this example has not published reproducible
real-checkpoint rollout evidence. Do not treat it as a validated success-rate
or parity claim until that evidence is reviewed.

This is an optional Linux developer example. It is not packaged with OmniInfer:
the setup process downloads LIBERO and creates its own Python environment, and
model files remain user-provided.

## Quick start

Run these commands from an OmniInfer checkout on Linux. They install only the
optional demo dependencies; no Python environment, LIBERO source, model, or
cache is added to the OmniInfer release package.

```sh
# 1. System tools (Ubuntu/Debian example).
sudo apt-get update
sudo apt-get install -y git protobuf-compiler

# 2. Install uv if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Fetch the vla.cpp submodule required by the demo client.
git submodule update --init framework/vla.cpp

# 4. Create the small CPU-only demo environment (default).
examples/vla-libero/setup.sh

# 5. Build the VLA runtime, copy the complete backend package into a private
#    runtime root, choose an unused gateway port, then start OmniInfer.
DEMO_ROOT="${XDG_STATE_HOME:-$HOME/.local/state}/omniinfer/vla-libero-demo"
bash scripts/platforms/linux/vla.cpp-linux-cuda/build.sh --from-source
mkdir -p "$DEMO_ROOT/runtimes"
cp -a .local/runtime/linux/vla.cpp-linux-cuda "$DEMO_ROOT/runtimes/"

OMNIINFER_SERVE_DIRECT=1 ./omniinfer serve \
  --host 127.0.0.1 --port <gateway-port> --no-restore-model \
  --state-root "$DEMO_ROOT/state" \
  --runtime-root "$DEMO_ROOT/runtimes"

MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --omniinfer-url http://127.0.0.1:<gateway-port> \
  --backend vla.cpp-linux-cuda \
  --model <path-to-smolvla.gguf> \
  --arch smolvla --task-id 0 --episodes 1
```

The dashboard prints its selected URL at startup. By default it asks the OS for
an unused loopback port, which avoids collisions on multi-user hosts. Choose a
task at that URL and press **Start**. For a remote Linux host, forward the
printed server port with
`ssh -L <local-port>:127.0.0.1:<server-port> <host>` and open the local port.
The two ports may differ. The dashboard never exposes a network listener by
default. Pass `--listen-port <port>` only when a fixed port is required.

To create a Python environment with CUDA PyTorch instead of the CPU default:

```sh
examples/vla-libero/setup.sh --torch-backend cu124
```

Choose the CUDA option only when Python-side GPU preprocessing is required.
`vla-server` performs model inference independently, so the CPU environment is
the recommended default for this example. Select the backend when creating the
environment. Re-running setup with a different backend does not convert an
existing venv. To change backend, use a new venv path or remove the old venv
first; for example:

```sh
examples/vla-libero/setup.sh \
  --torch-backend cu124 \
  --venv "${XDG_CACHE_HOME:-$HOME/.cache}/omniinfer/vla-libero-demo/venv-cu124"
```

## Simulation flow

The dashboard is deliberately a visible, end-to-end rollout rather than a
hidden benchmark runner:

1. Select one predefined `libero_object` task in the browser. Each choice fixes
   the language instruction, scene, object, and LIBERO success condition.
2. Press **Start**. The dashboard validates the selected task and asks
   OmniInfer to load the requested VLA model, or reuses the ready managed
   runtime when no `--model` is supplied.
3. OmniInfer launches or supervises `vla-server` and reports its loopback
   ZeroMQ/protobuf endpoint. The dashboard rejects any non-vla.cpp or
   non-loopback endpoint.
4. LIBERO/MuJoCo resets the selected scene and yields front-camera, wrist-camera
   and robot-state observations.
5. The demo converts those observations to the vla.cpp request format and sends
   them directly to the OmniInfer-managed `vla-server`.
6. `vla-server` returns an action chunk. The demo applies the next 7-DoF action
   to LIBERO, then repeats the observation → action loop until the episode ends
   or the user presses **Stop**.
7. Throughout the rollout, the browser shows the latest sampled camera frame,
   current action, task text, episode/step count, and
   model/policy/simulator/control-loop latency. Each rollout writes an MP4 to
   its own timestamped subdirectory under `--output-dir`.
8. LIBERO reports the final success condition. The page explicitly displays
   `success`, `failed`, `partial`, `stopped`, or `error`; it never treats a
   completed process as success by itself.

The simulator and MP4 recorder still process every environment step. The
browser display samples the latest observation at up to `--fps` and deliberately
drops intermediate display frames when action chunks advance faster than the
page can render. A new rollout clears the previous image immediately, and stale
requests from an older rollout cannot replace the current frame.

The policy defaults to its validated 256x256 LIBERO camera observations. The
dashboard separately samples the same simulator state at 512x512 for browser
display and serves it as quality-92 4:2:2 JPEG at up to 30 FPS. Thus the
multi-view browser frame is 1024x512 before CSS scaling, without changing the
pixels sent to the model. Override the policy, display, or display/video rate
only when needed:

```sh
examples/vla-libero/run.sh -- \\
  --render-size 256 \\
  --display-render-size 512 \\
  --fps 30 [other demo options]
```

`--render-size` controls the raw policy observation and can change rollout
behavior even though each client applies its architecture-specific resize. It
is therefore a benchmark-relevant setting, not a visual-quality control. Keep
it at 256 for the validated SmolVLA/LIBERO path and record any override in a
rollout comparison. `--display-render-size` is browser-only. Some wrist cameras
include an uninformative lower border, so the dashboard crops its display to
the upper 84% by default before resizing it to the front-view height. Pass
`--wrist-display-crop-ratio 1.0` to retain the full wrist frame. This crop never
changes the observation sent to the policy.

## Prerequisites

1. Linux x86_64, Python 3.10, Git, `uv`, and `protoc`.
2. An NVIDIA driver plus EGL-capable rendering for CUDA vla.cpp runtimes.
3. Build `vla.cpp-linux` or `vla.cpp-linux-cuda` from an OmniInfer source
   checkout, then place or copy the complete backend directory into the same
   per-user runtime root used by the gateway. The current prebuilt catalog does
   not publish either VLA backend, so `backend install` is not available for
   this example yet. See `docs/build.md` for source-build dependencies.
4. Initialize `framework/vla.cpp`.
5. Have a SmolVLA or PI0.5 checkpoint compatible with the vla.cpp runtime.

The vla.cpp Python client invokes `protoc` to generate its local protobuf stub.
If `protoc` is installed outside `PATH`, pass `--protoc <path-to-protoc>`; the
demo validates it before initializing the client.

When all required model resources are already cached and the demonstration host
has no Internet access, set `HF_HUB_OFFLINE=1` to avoid Hugging Face Hub
connection retries during startup.

Build the VLA runtime and copy the complete backend package into a private
runtime root. The source checkout currently uses this path rather than assuming
that a matching prebuilt archive is available:

```sh
DEMO_ROOT="${XDG_STATE_HOME:-$HOME/.local/state}/omniinfer/vla-libero-demo"
bash scripts/platforms/linux/vla.cpp-linux-cuda/build.sh --from-source
mkdir -p "$DEMO_ROOT/runtimes"
cp -a .local/runtime/linux/vla.cpp-linux-cuda "$DEMO_ROOT/runtimes/"
```

Then start a loopback OmniInfer gateway. On a multi-user host, both the port and
the state/runtime roots must be private to your session or user; changing only
the port still shares OmniInfer state, PID files, and logs:

```sh
OMNIINFER_SERVE_DIRECT=1 ./omniinfer serve \
  --host 127.0.0.1 \
  --port <gateway-port> \
  --no-restore-model \
  --state-root "$DEMO_ROOT/state" \
  --runtime-root "$DEMO_ROOT/runtimes"
```

For a CPU runtime, use the corresponding `vla.cpp-linux` build script and
backend directory. See `docs/build.md` for build dependencies and options. Do
not point two concurrently running gateway instances at the same state root.

Create the dedicated dashboard environment:

```sh
examples/vla-libero/setup.sh
```

This clones LIBERO at commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01` on first use and creates an
isolated source checkout and environment under
`${XDG_CACHE_HOME:-~/.cache}/omniinfer/`. Pinning the revision makes the demo
setup reproducible instead of following LIBERO's moving default branch. It does
not install or modify vla.cpp's complete `setup_libero.sh` evaluation
environment. The dashboard's default is CPU-only PyTorch: model inference
continues to run in the separately managed `vla-server`, so Python needs torch
only for simulation and request preprocessing. On the validation host this
reduced the environment from about 9.6 GB to about 2.3 GB.

To retain a CUDA PyTorch environment, request one explicitly:

```sh
examples/vla-libero/setup.sh \
  --torch-backend cu124 \
  --venv "${XDG_CACHE_HOME:-$HOME/.cache}/omniinfer/vla-libero-demo/venv-cu124"
```

The CUDA option is for compatibility or local GPU-side preprocessing; it does
not move vla-server inference into Python. `uv` and a system `protoc` are still
required. Choose CPU or CUDA when the venv is first created; changing
`--torch-backend` does not rewrite an existing environment. Use a separate
`--venv` path as above, or remove the existing venv before recreating it.
`setup.sh --help` documents alternate venv, LIBERO source, and uv paths.

Setup does not start a simulator by default, so creating the environment does
not require an EGL-capable device. To validate the installed dependencies,
assets, renderer, and one real LIBERO environment reset immediately after
setup, add `--smoke-test`:

```sh
examples/vla-libero/setup.sh --smoke-test
```

The smoke test defaults to `MUJOCO_GL=egl`. Select another MuJoCo renderer when
needed:

```sh
MUJOCO_GL=osmesa examples/vla-libero/setup.sh --smoke-test
```

This check initializes the simulator only; it does not load a VLA model or run
an episode.

## Disk space planning

The demo does not bundle its Python environment, LIBERO checkout, model files,
runtime build, or download caches into the OmniInfer release. They are created
or supplied on the machine that runs the demo. Plan disk space before running
setup or building a backend, especially on a shared host.

The Python environment is the main setup cost. On the validation host, the
CPU-only uv environment used about 2.2 GB. The complete CUDA environment used
about 9.6 GB, so reserve at least 10 GB when selecting `--torch-backend cu124`.
Exact sizes can vary with PyTorch, CUDA, platform wheels, and dependency
versions.

CPU and CUDA environments should use separate `--venv` paths and therefore
consume space independently when both are retained. The uv and Hugging Face
download caches also use additional, variable disk space; they are shared user
caches rather than part of either venv or the OmniInfer release. Models and
rollout recordings are user-provided/generated and must be budgeted separately.

Inspect the actual paths before and after setup or a build:

```sh
df -h .
du -sh "${XDG_CACHE_HOME:-$HOME/.cache}/omniinfer/vla-libero-demo" 2>/dev/null || true
du -sh "${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}" 2>/dev/null || true
```

## SmolVLA

SmolVLA uses the public
`HuggingFaceTB/SmolVLM2-500M-Instruct` tokenizer by default. The first use
requires Hugging Face connectivity unless the tokenizer is already fully
cached. When the network is unavailable but the cache is complete,
initialization can still succeed, but the Hub's online version check may wait
through connection timeouts and retries on every new rollout. After confirming
that the tokenizer is fully cached, set `HF_HUB_OFFLINE=1` when starting the
dashboard to skip those checks. Do not enable offline mode before the required
files are cached; initialization will fail instead of downloading them.

For subsequent offline runs after the cache has been verified, prefix the
dashboard command with the environment variable:

```sh
HF_HUB_OFFLINE=1 MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --omniinfer-url http://127.0.0.1:<gateway-port> \
  --model-profiles <path-to-model-profiles.json>
```

This setting applies to that dashboard process only. Omit
`HF_HUB_OFFLINE=1` whenever files still need to be downloaded.

```sh
MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --omniinfer-url http://127.0.0.1:<gateway-port> \
  --backend vla.cpp-linux-cuda \
  --model <path-to-smolvla.gguf> \
  --arch smolvla \
  --task libero_object \
  --task-id 0 \
  --episodes 1 \
  --n-action-steps 1
```

## PI0.5

> **Experimental:** the PI0.5 request/configuration path is implemented, but a
> reproducible real-checkpoint rollout has not been published for this example.
> Use SmolVLA for the currently validated end-to-end demonstration.

PI0.5 requires LIBERO state quantiles. If `--stats-json` is omitted, the
vla.cpp client follows its official default and obtains
`lerobot/libero` `meta/stats.json` from Hugging Face. Pass a local file for a
reproducible or offline demonstration. `--tokenizer` is optional; when omitted,
the vla.cpp client uses its PI0.5 PaliGemma tokenizer preset. That upstream
repository is gated, so hosts without an authenticated/cached copy should pass
a compatible local tokenizer directory explicitly.

```sh
MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --omniinfer-url http://127.0.0.1:<gateway-port> \
  --backend vla.cpp-linux-cuda \
  --model <path-to-pi05.gguf> \
  --arch pi05 \
  --tokenizer <path-or-id-to-paligemma-tokenizer> \
  --stats-json <path-to-libero-meta-stats.json> \
  --task libero_object \
  --task-id 0 \
  --episodes 1 \
  --n-action-steps 10
```

The tokenizer flag can also override SmolVLA's tokenizer preset. The stats flag
is PI0.5-specific and is rejected with other architectures. An explicitly
supplied stats path is checked before the dashboard starts, so a typo does not
fail only after the simulator is running.

## Selecting models in the dashboard

The commands above bind the dashboard to one model for the lifetime of the
process. To let users choose between approved models in the page, copy
`model-profiles.example.json`, replace its placeholder paths, and start with:

```sh
MUJOCO_GL=egl examples/vla-libero/run.sh -- \
  --omniinfer-url http://127.0.0.1:<gateway-port> \
  --model-profiles <path-to-model-profiles.json>
```

The first profile in the JSON file is selected by default. The dashboard state
exposes only each profile's `id`, display `label`, and `arch`; checkpoint,
tokenizer, stats, and server arguments are never accepted from a browser
request. Model and task selection are frozen while a rollout is active.

Each profile supports:

- required: `label` and `arch`;
- exactly one of `model` or `use_loaded_runtime: true`;
- optional: `omniinfer_url`, `backend`, `mmproj`, `server_args`, `tokenizer`,
  `stats_json`, and `n_action_steps`;
- `omniinfer_url` may bind each profile to a different loopback gateway, so one
  dashboard can switch between independently managed model runtimes;
- profile gateway URLs must be explicit loopback `http://` endpoints;
- use `use_loaded_runtime: true` only when that profile's gateway already has
  the intended VLA runtime loaded;
- relative `model`, `mmproj`, and `stats_json` paths resolved from the profile
  JSON directory;
- tokenizer values accepted as Hugging Face IDs or local paths; prefix a local
  relative tokenizer with `./` to distinguish it from an ID such as `org/repo`;
- default `n_action_steps`: 1 for SmolVLA and 10 for PI0.5. PI0.5 produces a
  50-step chunk; executing 10 steps balances visible synchronous replanning
  pauses while retaining more closed-loop correction than longer horizons.

Any configured model files are validated when the dashboard starts. A profile
using an already loaded runtime skips file validation. Selecting a different
profile asks its configured OmniInfer gateway to load that model before the
rollout, so a
model switch includes normal runtime startup and weight-loading latency. It is
not an instantaneous in-process policy switch. OmniInfer keeps independently
loaded model paths as separate managed runtimes; selecting another profile does
not unload the previous runtime, so RAM/VRAM use can accumulate. Before loading
a model that will not fit alongside the current one, unload the old model with
`POST /omni/model/unload` using its loaded model id, or stop the isolated demo
gateway after the rollout. Reloading the same model path with different runtime
settings returns `409 model_reload_required` until the existing runtime is
explicitly unloaded.

`--model-profiles` cannot be combined with the single-model `--model`,
`--mmproj`, `--server-arg`, `--tokenizer`, or `--stats-json` options. The
configuration file is trusted local input: keep it outside public artifacts
when it contains private filesystem layout, and do not expose the dashboard
beyond loopback.

If OmniInfer already manages the desired VLA model, omit `--model`; the demo
validates `/omni/state` and uses its reported `client_endpoint`. It rejects a
non-VLA protocol, non-VLA backend, missing endpoint, and non-loopback ZMQ
endpoint instead of silently connecting to the wrong runtime.

If the gateway uses an admin key, place it in
`OMNIINFER_ADMIN_API_KEY`; the demo reads the environment variable and sends a
Bearer header without putting the secret in the process command line. To keep
that credential local, `--omniinfer-url` accepts only an explicit loopback IP
and port over HTTP; the client ignores environment proxies and refuses HTTP
redirects. Run the demo on the same host as the gateway and use the dashboard's
SSH-forwarding instructions for remote browser access.

Open the URL printed by `run.sh`. The dashboard is intentionally loopback-only:
it can start and stop GPU rollouts, so it must not be exposed directly to a
network. When it runs on a remote machine, forward it over SSH:

```sh
ssh -L <port>:127.0.0.1:<port> <remote-host>
```

The page is idle on startup. Choose a predefined task and press **Start**;
arbitrary text is deliberately not accepted, because each LIBERO task binds its
instruction, scene, object, and success condition. Stop requests take effect
after the current policy or simulator step returns.

## Files and cleanup

- `setup.sh` downloads the pinned LIBERO revision and creates a venv under
  `${XDG_CACHE_HOME:-~/.cache}/omniinfer/vla-libero-demo/` by default. These
  per-user defaults avoid writing generated state into a shared source checkout.
- `run.sh` only starts the dashboard; it never installs packages.
- `demo.py` never installs Python packages. Set `HF_HUB_OFFLINE=1` when all
  required model resources are already cached and network access is undesired.
- Rollout videos default to
  `${XDG_STATE_HOME:-~/.local/state}/omniinfer/vla-libero-demo/outputs/`, not
  the current checkout. Override this with `--output-dir` when needed.
- The page receives a per-process CSRF token and all start/stop requests must
  return it in a dedicated header. Loopback binding remains mandatory; CSRF
  protection does not make direct network exposure supported.
- Remove the chosen venv and LIBERO checkout manually when no longer needed.
  Neither belongs in Git or the OmniInfer release archive.

## Metric semantics

- **Model prediction**: preprocessing plus the synchronous vla.cpp
  request/response on steps that request a new action chunk.
- **Policy call**: every action request, including inexpensive action-queue
  replay when `--n-action-steps` is greater than one.
- **Simulator step**: the LIBERO environment step.
- **Control loop**: policy call plus simulator step.

Latency cards show P50 as the primary value and also report mean and P95 over
the most recent 500 steps. This keeps an episode-ending LIBERO environment
reset visible in the aggregate without presenting that terminal reset as the
normal per-step latency.

The dashboard is a rollout demonstration, not a full LIBERO benchmark. Use
vla.cpp's official evaluation runners for benchmark-scale success-rate claims.
