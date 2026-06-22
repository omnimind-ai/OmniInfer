from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import re
import secrets
import signal
import sys
import threading
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from service_core import commands
from service_core.remote_access import parse_trycloudflare_url
from service_core.service import APP_ROOT, REPO_ROOT, load_app_config


_cli_port_override: int | None = None
DEFAULT_IMAGE_PATH = Path(REPO_ROOT) / "tests" / "pictures" / "test1.png"
SYSTEM_CHOICES = ("linux", "mac", "windows")
PUBLIC_SMOKE_PROMPT = "Reply exactly: omniinfer-public-ok"


BACKEND_HELP = """\
Manage inference runtimes.

Examples:
  omniinfer backend list
  omniinfer backend select llama.cpp-linux-cuda
"""


MODEL_HELP = """\
Discover and load models.

Examples:
  omniinfer model list
  omniinfer model load -m /path/to/model.gguf --ctx-size 8192
"""


ADVISOR_HELP = """\
Inspect hardware, estimate model fit, and plan deployments.

Examples:
  omniinfer advisor system
  omniinfer advisor fit /path/to/model.gguf --ctx-size 8192
  omniinfer advisor plan /path/to/model.gguf --gpu-vram 24 --ram 64
"""


THINKING_HELP = """\
Manage the default thinking mode used by chat requests.

Examples:
  omniinfer thinking show
  omniinfer thinking set off
"""


class CommandGroupParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"Error: {message}\n")


HELP_TEXT = """\
OmniInfer CLI

Common commands:
  omniinfer
  omniinfer advisor system
  omniinfer advisor fit /path/to/model.gguf
  omniinfer advisor plan /path/to/model.gguf
  omniinfer backend list
  omniinfer backend select <backend>
  omniinfer status
  omniinfer load -m /path/to/model.gguf
  omniinfer load -m /path/to/model.gguf --config
  omniinfer chat "Introduce yourself in one sentence."
  omniinfer chat "Describe this image." --image photo.png
  omniinfer serve --cloudflare --model /path/to/model.gguf --ctx-size 8192
  omniinfer serve status --port 9000
  omniinfer serve stop --port 9000
  omniinfer shutdown

Design notes:
  1. The CLI automatically starts the service and selects the best backend when needed.
  2. Host and port details are hidden by default; the CLI uses local app configuration automatically.
  3. `omniinfer backend select <backend>` persists the current backend selection.
  4. `omniinfer chat` requires a model to be loaded first via `omniinfer load`.
  5. `omniinfer chat` streams tokens to stdout by default. Use `--no-stream` for batch output.
  6. `omniinfer` without arguments opens the basic TUI in an interactive terminal.

Command map:
  backend list              -> show available backends
  backend select <backend>  -> choose a backend
  advisor system/fit/plan   -> inspect hardware, model fit, and hardware requirements
  status                    -> show service status, selected backend, and loaded model
  load                      -> load a model
  model list                -> show models supported on the current system
  model load                -> load a model, optionally with a backend config JSON
  thinking show/set         -> show or change the default think setting
  chat                      -> run text or multimodal inference on the loaded model
  backend stop              -> stop the currently running backend process
  shutdown                  -> stop the OmniInfer service
  serve                     -> start the gateway; add --cloudflare/--lan for remote inference access
  completion bash           -> print the bash completion script
"""


def _help_text() -> str:
    if not commands.is_backend_build_supported():
        return HELP_TEXT
    return HELP_TEXT.replace(
        "  omniinfer backend select <backend>\n",
        "  omniinfer backend select <backend>\n"
        "  omniinfer build <backend>\n",
    ).replace(
        "  backend select <backend>  -> choose a backend\n",
        "  backend select <backend>  -> choose a backend\n"
        "  build <backend>           -> build a backend from a source checkout\n",
    )


BASH_COMPLETION_SCRIPT = r"""# bash completion for omniinfer
_omniinfer_completion() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local suggestions
    suggestions="$("${COMP_WORDS[0]}" __complete "${COMP_WORDS[@]:1}" 2>/dev/null)"

    if [[ -n "$suggestions" ]]; then
        mapfile -t COMPREPLY < <(compgen -W "$suggestions" -- "$cur")
        return 0
    fi

    return 0
}

complete -o bashdefault -o default -F _omniinfer_completion omniinfer
"""


SERVE_HELP_TEXT = """\
usage: omniinfer serve [status|stop] [options]

Start and manage the OmniInfer gateway.

Common examples:
  omniinfer serve --cloudflare --model /path/to/model.gguf
  omniinfer serve --cloudflare --backend llama.cpp-linux-cuda --model /path/to/model.gguf --ctx-size 8192 --api-key auto --detach --smoke-test
  omniinfer serve status --port 9000
  omniinfer serve stop --port 9000

Controls:
  status                  Show the service status for the selected port
  stop                    Stop the service on the selected port
  -m, --model PATH        Load this model after the gateway is healthy
  -mm, --mmproj PATH      Optional mmproj for GGUF vision models
  --ctx-size N            Context length used when loading the model
  --backend BACKEND       Select this backend before loading the model
  --detach                Keep the service running in the background
  --smoke-test            Run a short chat completion after loading
  --no-smoke-test         Disable smoke testing if enabled by a wrapper

Remote access:
  --cloudflare            Expose inference through Cloudflare Quick Tunnel
  --lan                   Expose inference to the local network
  --api-key KEY|auto      API key for remote clients; auto generates one
  --cloudflare-no-print-key
                          Do not print the key in Cloudflare output
  --allow-insecure-lan    Allow LAN access without an API key
  --allow-remote-management
                          Allow authenticated remote /omni/* access

Gateway:
  --port PORT             Gateway port
  --host HOST             Advanced bind override; not needed for --cloudflare
  --cloudflared-path PATH Override the managed cloudflared binary
  --default-thinking on|off
  --window-mode visible|hidden
  --startup-timeout N
  --log-level DEBUG|INFO|WARNING|ERROR
  --verbose
  --debug-body
"""


def detect_system_name() -> str:
    system = platform.system().lower()
    if system.startswith("linux"):
        return "linux"
    if system.startswith("darwin") or system.startswith("mac"):
        return "mac"
    if system.startswith("win"):
        return "windows"
    raise SystemExit(f"unsupported host system: {platform.system()}")


def get_service_config() -> dict[str, Any]:
    config = load_app_config(Path(APP_ROOT))
    if _cli_port_override is not None:
        config["port"] = _cli_port_override
    return config


def service_base_url() -> str:
    config = get_service_config()
    host = str(config["host"]).strip()
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{int(config['port'])}"


def request_json(
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
    allow_http_error: bool = False,
) -> tuple[int, Any, bytes]:
    url = f"{service_base_url()}{endpoint}"
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(url=url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
        if not allow_http_error:
            parsed = try_parse_json(raw)
            message = parsed if parsed is not None else raw.decode("utf-8", errors="replace")
            raise SystemExit(f"{method} {endpoint} failed with status {status}: {message}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"unable to reach local OmniInfer service: {exc}") from exc
    except OSError as exc:
        raise SystemExit(f"unable to reach local OmniInfer service: {exc}") from exc

    return status, try_parse_json(raw), raw


def try_parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except json.JSONDecodeError:
        return None


def is_service_running() -> bool:
    try:
        status, payload, _ = request_json("GET", "/health", timeout=2.0, allow_http_error=True)
    except SystemExit:
        return False
    return status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"


def local_backend_ids() -> list[str]:
    return commands.local_backend_ids()


def ensure_service_running() -> None:
    commands.ensure_service_running()


def ensure_service_and_selection() -> str | None:
    ensure_service_running()
    _status, payload, _ = request_json("GET", "/omni/state", timeout=10.0)
    if isinstance(payload, dict) and payload.get("backend"):
        return str(payload["backend"])
    return None


def format_bool(value: bool | None) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "unknown"


def print_backend_list(scope: str = "compatible", json_output: bool = False) -> int:
    payload = commands.list_backends(scope=scope)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        if json_output:
            print(json.dumps({"object": "list", "data": [], "recommended": None}))
            return 0
        raise SystemExit("No backends are available on this system.")

    if json_output:
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    title_by_scope = {
        "compatible": "Compatible backends",
        "installed": "Installed backends",
        "all": "Available backends",
    }
    print(title_by_scope.get(scope, "Available backends"))
    backend_width = max(len("Backend"), *(len(str(item.get("id", ""))) for item in rows))
    header = f"{'Backend':<{backend_width}}  Selected  Installed"
    print(header)
    print(f"{'-' * backend_width}  --------  ---------")
    for item in rows:
        backend_id = str(item.get("id", ""))
        selected = "yes" if item.get("selected") else ""
        installed = "yes" if item.get("binary_exists") else ""
        print(f"{backend_id:<{backend_width}}  {selected:<8}  {installed:<9}".rstrip())
    return 0


def _dump_json(payload: Any) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _print_table(rows: list[list[str]], *, indent: str = "") -> None:
    if not rows:
        return
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row_index, row in enumerate(rows):
        line = "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()
        print(f"{indent}{line}")
        if row_index == 0:
            rule = "  ".join("-" * width for width in widths).rstrip()
            print(f"{indent}{rule}")


def print_advisor_system(json_output: bool = False) -> int:
    payload = commands.advisor_system()
    if json_output:
        return _dump_json(payload)

    host = payload.get("host") if isinstance(payload.get("host"), dict) else {}
    cuda = payload.get("cuda") if isinstance(payload.get("cuda"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print("OmniInfer Advisor System")
    print(f"System: {host.get('system')} ({host.get('machine')})")
    print(f"CPU cores: {host.get('cpu_cores') or '-'}")
    print(f"RAM: {host.get('available_ram_gib') or '-'} GiB available / {host.get('total_ram_gib') or '-'} GiB total")
    devices = cuda.get("visible_devices") or cuda.get("devices") or []
    if devices:
        print("CUDA devices:")
        for device in devices:
            print(
                "  "
                f"GPU {device.get('index')}: {device.get('name')} "
                f"free={device.get('free_gib')} GiB total={device.get('total_gib')} GiB util={device.get('utilization_pct')}%"
            )
    else:
        print("CUDA devices: none detected")
    print(f"Recommended installed backend: {summary.get('recommended_installed_backend') or '-'}")
    usable_backends = [
        backend
        for backend in payload.get("backends", [])
        if isinstance(backend, dict) and backend.get("installed") and backend.get("hardware_compatible")
    ]
    print("Usable backends:")
    if usable_backends:
        rows = [["Backend", "Family", "Capabilities"]]
        for backend in usable_backends:
            capabilities = ", ".join(str(item) for item in backend.get("capabilities", [])[:4])
            rows.append([str(backend.get("id") or "-"), str(backend.get("family") or "-"), capabilities or "-"])
        _print_table(rows, indent="  ")
    else:
        print("  none installed and compatible")
    hidden_count = len([backend for backend in payload.get("backends", []) if isinstance(backend, dict)]) - len(usable_backends)
    if hidden_count > 0:
        print(f"Hidden backends: {hidden_count} unavailable or incompatible; use --json for the full probe.")
    return 0


def print_advisor_inspect(model: str, *, mmproj: str | None = None, json_output: bool = False) -> int:
    payload = commands.advisor_inspect(model, mmproj=mmproj)
    if json_output:
        return _dump_json(payload)
    print("OmniInfer Advisor Inspect")
    print(f"Model: {payload.get('model')}")
    print(f"Format: {payload.get('format')}")
    print(f"Size: {payload.get('size_gib') or '-'} GiB")
    print(f"mmproj: {payload.get('mmproj') or '-'}")
    print(f"Quantization: {payload.get('quantization') or '-'}")
    print(f"Params: {payload.get('params_b') or '-'}B")
    print(f"Capabilities: {', '.join(payload.get('capabilities') or []) or '-'}")
    estimate = payload.get("estimate") if isinstance(payload.get("estimate"), dict) else {}
    print(f"Estimated memory: {estimate.get('estimated_gpu_memory_gib') or '-'} GiB ({estimate.get('confidence') or 'unknown'} confidence)")
    _print_memory_breakdown(estimate.get("breakdown") if isinstance(estimate.get("breakdown"), dict) else estimate)
    for warning in payload.get("warnings", []):
        print(f"Warning: {warning}")
    return 0


def print_advisor_fit(
    model: str,
    *,
    mmproj: str | None = None,
    ctx_size: int | None = None,
    backend: str | None = None,
    json_output: bool = False,
) -> int:
    payload = commands.advisor_fit(model, mmproj=mmproj, ctx_size=ctx_size, backend=backend)
    if json_output:
        return _dump_json(payload)
    print("OmniInfer Advisor Fit")
    model_info = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    print(f"Model: {model_info.get('model')}")
    print(f"Context size: {payload.get('context_size')}")
    recommended = payload.get("recommended") if isinstance(payload.get("recommended"), dict) else None
    if recommended:
        print(f"Recommended backend: {recommended.get('backend')}")
        print(f"Fit: {recommended.get('fit')}")
        print(f"Confidence: {recommended.get('recommendation_confidence') or '-'}")
        evidence = recommended.get("evidence") if isinstance(recommended.get("evidence"), dict) else {}
        print(f"Evidence: {evidence.get('level') or '-'}")
        print(f"Installed: {'yes' if recommended.get('installed') else 'no'}")
        print(f"Memory: {recommended.get('memory_required_gib')} GiB required / {recommended.get('memory_available_gib') or '-'} GiB available")
        _print_memory_breakdown(recommended.get("memory_breakdown") if isinstance(recommended.get("memory_breakdown"), dict) else {})
        for reason in recommended.get("why_recommended", [])[:5]:
            print(f"Why: {reason}")
    else:
        print("Recommended backend: -")
    if payload.get("next_command"):
        print(f"Next command: {payload['next_command']}")
    alternatives = payload.get("alternatives") if isinstance(payload.get("alternatives"), list) else []
    if alternatives:
        print("Alternatives:")
        for candidate in alternatives[:5]:
            print(f"  {candidate.get('backend')}: fit={candidate.get('fit')}, installed={'yes' if candidate.get('installed') else 'no'}")
            why_not = candidate.get("why_not") if isinstance(candidate.get("why_not"), list) else []
            if why_not:
                print(f"    why_not={why_not[0]}")
    for warning in payload.get("warnings", []):
        print(f"Warning: {warning}")
    return 0


def print_advisor_plan(
    model: str,
    *,
    mmproj: str | None = None,
    ctx_size: int | None = None,
    gpu_vram_gib: float | None = None,
    ram_gib: float | None = None,
    cpu_cores: int | None = None,
    json_output: bool = False,
) -> int:
    payload = commands.advisor_plan(
        model,
        mmproj=mmproj,
        ctx_size=ctx_size,
        gpu_vram_gib=gpu_vram_gib,
        ram_gib=ram_gib,
        cpu_cores=cpu_cores,
    )
    if json_output:
        return _dump_json(payload)
    model_info = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    planning = payload.get("planning_hardware") if isinstance(payload.get("planning_hardware"), dict) else {}
    print("OmniInfer Advisor Plan")
    print(f"Model: {model_info.get('model')}")
    print(f"Context size: {payload.get('context_size')}")
    print(
        "Planning hardware: "
        f"free_vram={planning.get('gpu_vram_free_gib') or '-'} GiB, "
        f"available_ram={planning.get('available_ram_gib') or '-'} GiB, "
        f"cpu_cores={planning.get('cpu_cores') or '-'}"
    )
    estimate = model_info.get("estimate") if isinstance(model_info.get("estimate"), dict) else {}
    _print_memory_breakdown(estimate.get("breakdown") if isinstance(estimate.get("breakdown"), dict) else {})
    recommended = payload.get("recommended_path") if isinstance(payload.get("recommended_path"), dict) else None
    if recommended:
        print(f"Recommended path: {recommended.get('path')} ({recommended.get('fit')})")
    print("Run paths:")
    for path in payload.get("run_paths", []):
        if not isinstance(path, dict):
            continue
        minimum = path.get("minimum") if isinstance(path.get("minimum"), dict) else {}
        rec = path.get("recommended") if isinstance(path.get("recommended"), dict) else {}
        print(
            "  "
            f"{path.get('path')}: feasible={'yes' if path.get('feasible_now') else 'no'}, "
            f"fit={path.get('fit')}, speed={path.get('estimated_relative_speed')}"
        )
        print(
            "    "
            f"minimum: vram={minimum.get('vram_gib') if minimum.get('vram_gib') is not None else '-'} GiB, "
            f"ram={minimum.get('ram_gib')} GiB, cpu={minimum.get('cpu_cores')}"
        )
        print(
            "    "
            f"recommended: vram={rec.get('vram_gib') if rec.get('vram_gib') is not None else '-'} GiB, "
            f"ram={rec.get('ram_gib')} GiB, cpu={rec.get('cpu_cores')}"
        )
    upgrades = payload.get("upgrade_deltas") if isinstance(payload.get("upgrade_deltas"), list) else []
    if upgrades:
        print("Upgrade deltas:")
        for item in upgrades[:5]:
            print(f"  {item.get('description')}")
    commands_list = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    if commands_list:
        print("Next commands:")
        for command in commands_list:
            print(f"  {command}")
    for warning in payload.get("warnings", []):
        print(f"Warning: {warning}")
    return 0


def _print_memory_breakdown(breakdown: dict[str, Any]) -> None:
    if not breakdown:
        return
    fields = [
        ("weights", breakdown.get("weights_gib")),
        ("mmproj", breakdown.get("mmproj_gib")),
        ("kv", breakdown.get("kv_cache_gib")),
        ("activation", breakdown.get("activation_gib")),
        ("framework", breakdown.get("framework_overhead_gib")),
        ("slack", breakdown.get("allocator_slack_gib")),
    ]
    rendered = [f"{name}={value} GiB" for name, value in fields if value is not None]
    if rendered:
        print("Memory breakdown: " + ", ".join(rendered))


def print_advisor_recommend(
    *,
    task: str | None = None,
    limit: int = 5,
    ctx_size: int | None = None,
    json_output: bool = False,
) -> int:
    payload = commands.advisor_recommend(task=task, limit=limit, ctx_size=ctx_size)
    if json_output:
        return _dump_json(payload)
    print("OmniInfer Advisor Recommend")
    print(f"Task: {payload.get('task')}")
    print(f"Models scanned: {payload.get('models_scanned')}")
    rows = payload.get("recommendations") if isinstance(payload.get("recommendations"), list) else []
    if not rows:
        print("No local model recommendations found.")
        return 0
    for index, row in enumerate(rows, 1):
        model_info = row.get("model") if isinstance(row.get("model"), dict) else {}
        recommended = row.get("recommended") if isinstance(row.get("recommended"), dict) else {}
        print(f"{index}. {model_info.get('model')}")
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        print(
            f"   backend={recommended.get('backend')} fit={recommended.get('fit')} "
            f"score={row.get('score')} confidence={row.get('recommendation_confidence') or '-'} "
            f"evidence={evidence.get('level') or '-'}"
        )
        why = row.get("why_recommended") if isinstance(row.get("why_recommended"), list) else []
        if why:
            print(f"   why={why[0]}")
        if row.get("next_command"):
            print(f"   command={row.get('next_command')}")
    return 0


def select_backend(name: str) -> int:
    result = commands.select_backend(name)
    print(f"Selected backend: {result.backend}")
    if result.models_dir:
        print(f"Models directory: {result.models_dir}")
    print(
        "Backend config: "
        f"{result.profile_path} ({'created' if result.profile_created else 'already exists'})"
    )
    return 0


def build_backend(name: str, *, prebuilt: bool = False, from_source: bool = False) -> int:
    options = commands.BackendBuildOptions(backend=name, prebuilt=prebuilt, from_source=from_source)
    command, _script_path = commands.backend_build_command(options)
    action = "Building backend" if from_source else "Installing backend"
    print(f"{action}: {name}")
    if from_source:
        install_mode = "source"
    elif prebuilt:
        install_mode = "prebuilt"
    else:
        install_mode = "default"
    print(f"Install mode: {install_mode}")
    print("Command: " + " ".join(command))

    result = commands.build_backend(options)
    print(f"Backend {'build' if from_source else 'install'} completed: {result.backend}")
    if result.binary_path:
        print(f"Binary: {result.binary_path}")
    return 0


def print_status() -> int:
    selected_backend = ensure_service_and_selection()
    _status, payload, _ = request_json("GET", "/omni/state", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to read the current runtime state.")

    print("OmniInfer Status")
    print("CLI: ready")
    print("Local service: running")
    print(f"Selected backend: {selected_backend or 'none'}")
    print(f"Backend ready: {'yes' if payload.get('backend_ready') else 'no'}")
    print(f"Loaded model: {payload.get('model') or 'not loaded'}")
    print(f"Loaded mmproj: {payload.get('mmproj') or 'not loaded'}")
    print(f"Loaded ctx-size: {payload.get('ctx_size') or 'not loaded'}")
    request_defaults = payload.get("request_defaults") if isinstance(payload.get("request_defaults"), dict) else {}
    if request_defaults:
        print(f"Request defaults: {json.dumps(request_defaults, ensure_ascii=False)}")
    thinking = payload.get("thinking") if isinstance(payload.get("thinking"), dict) else {}
    print(f"Default thinking: {format_bool(thinking.get('default_enabled'))}")
    return 0


def print_ps(json_output: bool = False) -> int:
    """List all running OmniInfer services on common ports."""
    import socket

    # Common ports to scan
    ports_to_scan = [9000, 9001, 9002, 9003, 9004, 9005, 9010, 9020, 9050, 9100, 8900, 8800, 19000]

    def is_port_listening(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                result = s.connect_ex(("127.0.0.1", port))
                return result == 0
        except Exception:
            return False

    def is_omniinfer_service(port: int) -> bool:
        """Check if the port is running an OmniInfer gateway."""
        try:
            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
                return isinstance(data, dict) and "status" in data
        except Exception:
            return False

    def get_service_state(port: int) -> dict[str, Any] | None:
        """Get the state of an OmniInfer service."""
        try:
            url = f"http://127.0.0.1:{port}/omni/state"
            req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:
            return None

    services: list[dict[str, Any]] = []

    for port in ports_to_scan:
        if not is_port_listening(port):
            continue

        if not is_omniinfer_service(port):
            continue

        state = get_service_state(port)
        if state is None:
            services.append({
                "port": port,
                "status": "running",
                "backend": "unknown",
                "backend_ready": False,
                "model": "not loaded",
                "error": "Failed to get state"
            })
            continue

        services.append({
            "port": port,
            "status": "running",
            "backend": state.get("backend", "none"),
            "backend_ready": state.get("backend_ready", False),
            "model": state.get("model", "not loaded"),
            "mmproj": state.get("mmproj"),
            "ctx_size": state.get("ctx_size"),
        })

    if json_output:
        print(json.dumps(services, ensure_ascii=False, indent=2))
        return 0

    if not services:
        print("No running OmniInfer services found.")
        return 0

    print("Running OmniInfer Services:")
    print()
    for svc in services:
        print(f"  Port {svc['port']}:")
        print(f"    Status: {svc['status']}")
        print(f"    Backend: {svc['backend']}")
        print(f"    Backend Ready: {'yes' if svc['backend_ready'] else 'no'}")
        print(f"    Model: {svc['model']}")
        if svc.get('mmproj'):
            print(f"    MMProj: {svc['mmproj']}")
        if svc.get('ctx_size'):
            print(f"    Context Size: {svc['ctx_size']}")
        if svc.get('error'):
            print(f"    Error: {svc['error']}")
        print()

    return 0


def flatten_best_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, family_models in payload.items():
        if not isinstance(family_models, dict):
            continue
        for model_name, model_info in family_models.items():
            if not isinstance(model_info, dict):
                continue
            quantizations = model_info.get("quantization")
            if not isinstance(quantizations, dict):
                continue
            for quant_name, quant_info in quantizations.items():
                if not isinstance(quant_info, dict):
                    continue
                rows.append(
                    {
                        "family": family,
                        "model": model_name,
                        "quantization": quant_name,
                        "backend": str(quant_info.get("backend", "")),
                        "suitable": bool(quant_info.get("suitable")),
                        "required_memory_gib": quant_info.get("required_memory_gib"),
                    }
                )
    return rows


def print_model_list(system_name: str, best: bool) -> int:
    ensure_service_running()
    query = urllib.parse.urlencode({"system": system_name})
    endpoint = "/omni/supported-models/best" if best else "/omni/supported-models"
    _status, payload, _ = request_json("GET", f"{endpoint}?{query}", timeout=60.0)
    if not isinstance(payload, dict):
        raise SystemExit("The supported models response has an unexpected format.")

    print(f"Supported models ({system_name})")
    if best:
        rows = flatten_best_models(payload)
        if not rows:
            print("No models are available to display.")
            return 0
        current_family = None
        current_model = None
        for row in rows:
            if row["family"] != current_family:
                current_family = row["family"]
                current_model = None
                print(f"\n[{current_family}]")
            if row["model"] != current_model:
                current_model = row["model"]
                print(f"  {current_model}")
            backend = row["backend"] or "-"
            suitable = "yes" if row["suitable"] else "no"
            required = row["required_memory_gib"]
            required_text = f"{required} GiB" if required is not None else "-"
            print(f"    - {row['quantization']}: backend={backend}, suitable={suitable}, memory={required_text}")
        return 0

    for backend_name, backend_payload in payload.items():
        print(f"\n[{backend_name}]")
        if not isinstance(backend_payload, dict):
            continue
        for family, family_models in backend_payload.items():
            if not isinstance(family_models, dict):
                continue
            print(f"  {family}")
            for model_name, model_info in family_models.items():
                if not isinstance(model_info, dict):
                    continue
                print(f"    {model_name}")
                quantizations = model_info.get("quantization")
                if not isinstance(quantizations, dict):
                    continue
                for quant_name, quant_info in quantizations.items():
                    if not isinstance(quant_info, dict):
                        continue
                    suitable = "yes" if quant_info.get("suitable") else "no"
                    required = quant_info.get("required_memory_gib")
                    required_text = f"{required} GiB" if required is not None else "-"
                    print(f"      - {quant_name}: suitable={suitable}, memory={required_text}")
    return 0


def print_model_load(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    model_input = getattr(args, "model", None)
    if not model_input:
        raise SystemExit("Please specify a model path with -m or --model.")

    spinner = _Spinner("Starting service...")
    if not verbose:
        spinner.start()

    def on_progress(event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        msg = event.get("message", "")
        if event_type == "status":
            if verbose and msg:
                print(msg)
            elif msg:
                spinner.update(msg)
        elif event_type == "log":
            if verbose and msg:
                print(f"  {msg}")
        elif event_type == "done":
            spinner.stop()
            elapsed = event.get("elapsed_s")
            if elapsed is not None:
                print(f"Backend ready ({elapsed}s)")
        elif event_type == "error":
            spinner.stop()

    try:
        spinner.update("Preparing model...")
        response, selection = commands.load_model(
            commands.ModelLoadOptions(
                model=str(model_input),
                mmproj=getattr(args, "mmproj", None),
                ctx_size=getattr(args, "ctx_size", None),
                config=getattr(args, "config", None),
                backend_extra_args=list(getattr(args, "backend_extra_args", [])),
                verbose=verbose,
            ),
            progress=on_progress,
        )
    except SystemExit:
        spinner.stop()
        raise

    if selection.auto_selected:
        print(f"No backend selected. Auto-selected: {selection.backend}")
    print("Model loaded")
    print(f"Backend: {response.get('selected_backend') or '-'}")
    print(f"Model: {response.get('selected_model') or '-'}")
    print(f"mmproj: {response.get('selected_mmproj') or '-'}")
    print(f"ctx-size: {response.get('selected_ctx_size') or '-'}")
    return 0


def add_model_load_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", help="Path to the model file or model directory")
    parser.add_argument("-mm", "--mmproj", help="Path to the mmproj file")
    parser.add_argument("--ctx-size", type=int, help="Optional llama.cpp context length override for this load")
    parser.add_argument(
        "--config",
        nargs="?",
        const="",
        help="Use the selected backend config JSON, or pass an explicit config path",
    )
    parser.add_argument("--verbose", action="store_true", help="Show full backend log output during loading")


_SPINNER_FRAMES = "⠋⠙⠸⢰⣠⣄⡆⠇"
_SPINNER_INTERVAL = 0.08  # ~12 fps


class _Spinner:
    """Thread-driven spinner that renders independently of the event stream."""

    def __init__(self, text: str = "Loading model...") -> None:
        self._text = text
        self._start = time.monotonic()
        self._active = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._is_tty:
            return
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, text: str) -> None:
        with self._lock:
            self._text = text

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._active:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._active = False

    def _run(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            with self._lock:
                text = self._text
            elapsed = time.monotonic() - self._start
            frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
            idx += 1
            sys.stdout.write(f"\r{frame} {text} ({elapsed:.1f}s)\033[K")
            sys.stdout.flush()
            self._stop_event.wait(_SPINNER_INTERVAL)


def print_thinking_show() -> int:
    print(f"Default thinking: {format_bool(commands.get_default_thinking())}")
    return 0


def print_thinking_set(value: str) -> int:
    enabled = value == "on"
    print(f"Default thinking set to: {format_bool(commands.set_default_thinking(enabled))}")
    return 0


def split_thinking_blocks(text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", text, flags=re.DOTALL)
    if not match:
        return None, text.strip()
    thinking = match.group(1).strip()
    answer = match.group(2).strip()
    return (thinking or None), answer


def print_performance(payload: dict[str, Any]) -> None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    print("\nPerformance")
    print(f"  Model: {payload.get('model') or '-'}")
    if usage:
        print(
            "  Tokens: "
            f"prompt={usage.get('prompt_tokens', '-')}, "
            f"completion={usage.get('completion_tokens', '-')}, "
            f"total={usage.get('total_tokens', '-')}"
        )
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and "cached_tokens" in details:
            print(f"  Cached prompt tokens: {details.get('cached_tokens')}")
    if timings:
        if "prompt_ms" in timings:
            print(f"  Prompt time: {timings.get('prompt_ms')} ms")
        if "predicted_ms" in timings:
            print(f"  Generation time: {timings.get('predicted_ms')} ms")
        if "predicted_per_second" in timings:
            print(f"  Generation speed: {timings.get('predicted_per_second')} tok/s")


def flush_stream_prefix(prefix_buffer: str) -> tuple[str, bool]:
    if not prefix_buffer:
        return "", True
    if not prefix_buffer.lstrip().startswith("<think>"):
        return prefix_buffer, True
    if "</think>" not in prefix_buffer:
        return "", False
    empty_match = re.match(r"^\s*<think>\s*</think>\s*(.*)$", prefix_buffer, flags=re.DOTALL)
    if empty_match:
        return empty_match.group(1), True
    return prefix_buffer, True


def print_chat_output(payload: dict[str, Any]) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit("Inference response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise SystemExit("Inference response did not include a message.")
    content = message.get("content")
    if isinstance(content, list):
        text = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        text = str(content or "").strip()

    thinking, answer = split_thinking_blocks(text)
    if thinking:
        print("Thinking")
        print(textwrap.indent(thinking, "  "))
        print()
    print("Response")
    print(answer or "(empty)")
    print_performance(payload)


def chat(args: argparse.Namespace) -> int:
    if args.message and args.prompt:
        raise SystemExit("Use either positional prompt or --message, not both.")
    message_text = args.message or args.prompt
    if not message_text:
        raise SystemExit("Please provide a message, for example: omniinfer chat \"Hello\".")

    payload = commands.build_chat_payload(
        commands.ChatOptions(
            message=message_text,
            image=args.image,
            stream=args.stream,
            think=args.think,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    effective_stream = bool(payload.get("stream"))
    if effective_stream:
        payload["stream_options"] = {"include_usage": True}
        print("Response")
        final_payload: dict[str, Any] | None = None
        prefix_buffer = ""
        prefix_flushed = False
        for chunk in commands.iter_chat_stream_payload(payload):
            if chunk.text:
                if prefix_flushed:
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
                else:
                    prefix_buffer += chunk.text
                    out, prefix_flushed = flush_stream_prefix(prefix_buffer)
                    if prefix_flushed and out:
                        sys.stdout.write(out)
                        sys.stdout.flush()
            if chunk.final_payload:
                final_payload = chunk.final_payload
        if not prefix_flushed and prefix_buffer:
            sys.stdout.write(prefix_buffer)
            sys.stdout.flush()
        print()
        if final_payload:
            print_performance(final_payload)
        return 0

    response = commands.request_chat_payload(payload)
    print_chat_output(response)
    return 0


def backend_stop() -> int:
    ensure_service_running()
    request_json("POST", "/omni/backend/stop", timeout=30.0)
    print("Current backend process stopped")
    return 0


def shutdown_service() -> int:
    if not commands.shutdown_service():
        print("OmniInfer service is not running")
        return 0
    print("OmniInfer service stopped")
    return 0


@dataclass(frozen=True)
class _ServePlan:
    action: str | None
    service_args: list[str]
    backend: str | None
    model: str | None
    mmproj: str | None
    ctx_size: int | None
    detach: bool
    smoke_test: bool
    api_key: str | None
    api_key_generated: bool
    print_api_key: bool


def _pop_option(args: list[str], names: set[str]) -> tuple[list[str], str | None]:
    result: list[str] = []
    value: str | None = None
    index = 0
    while index < len(args):
        token = args[index]
        matched_inline = False
        for name in names:
            prefix = f"{name}="
            if token.startswith(prefix):
                value = token.split("=", 1)[1]
                matched_inline = True
                break
        if matched_inline:
            index += 1
            continue
        if token in names:
            if index + 1 >= len(args):
                raise SystemExit(f"{token} requires a value")
            value = args[index + 1]
            index += 2
            continue
        result.append(token)
        index += 1
    return result, value


def _pop_flag(args: list[str], name: str) -> tuple[list[str], bool]:
    removed = False
    result: list[str] = []
    for token in args:
        if token == name:
            removed = True
            continue
        result.append(token)
    return result, removed


def _parse_optional_int(value: str | None, *, option: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{option} must be an integer") from exc
    if parsed <= 0:
        raise SystemExit(f"{option} must be positive")
    return parsed


def _generate_session_api_key() -> str:
    return "oi_" + secrets.token_urlsafe(24)


def _parse_serve_plan(service_args: list[str]) -> _ServePlan:
    args = list(service_args)
    action: str | None = None
    if args and args[0] in {"status", "stop"}:
        action = args.pop(0)

    args, backend = _pop_option(args, {"--backend"})
    args, model = _pop_option(args, {"-m", "--model"})
    args, mmproj = _pop_option(args, {"-mm", "--mmproj"})
    args, ctx_text = _pop_option(args, {"--ctx-size"})
    args, api_key = _pop_option(args, {"--api-key"})
    args, detach = _pop_flag(args, "--detach")
    args, smoke_test = _pop_flag(args, "--smoke-test")
    args, no_smoke_test = _pop_flag(args, "--no-smoke-test")
    if smoke_test and no_smoke_test:
        raise SystemExit("Use either --smoke-test or --no-smoke-test, not both.")
    if no_smoke_test:
        smoke_test = False

    remote_request = _flag_in_argv(args, "--cloudflare") or _flag_in_argv(args, "--lan")
    allow_insecure = _flag_in_argv(args, "--allow-insecure-lan")
    api_key_generated = False
    env_key = os.environ.get("OMNIINFER_API_KEY", "").strip()
    if api_key is not None and api_key.strip().lower() == "auto":
        api_key = _generate_session_api_key()
        api_key_generated = True
    elif api_key is None and remote_request and not allow_insecure:
        if env_key:
            api_key = env_key
        else:
            api_key = _generate_session_api_key()
            api_key_generated = True

    return _ServePlan(
        action=action,
        service_args=args,
        backend=backend,
        model=model,
        mmproj=mmproj,
        ctx_size=_parse_optional_int(ctx_text, option="--ctx-size"),
        detach=detach,
        smoke_test=smoke_test,
        api_key=api_key,
        api_key_generated=api_key_generated,
        print_api_key=not _flag_in_argv(args, "--cloudflare-no-print-key"),
    )


def _serve_plan_needs_orchestration(plan: _ServePlan) -> bool:
    return bool(
        plan.action
        or plan.backend
        or plan.model
        or plan.mmproj
        or plan.ctx_size
        or plan.detach
        or plan.smoke_test
    )


def _serve_port(service_args: list[str]) -> int:
    config = load_app_config(APP_ROOT)
    text = _value_from_argv(service_args, "--port") or str(config.get("port", "9000"))
    try:
        return int(text)
    except ValueError as exc:
        raise SystemExit("--port must be an integer") from exc


def _serve_log_path(port: int) -> Path:
    return Path(APP_ROOT) / ".local" / "logs" / f"serve-{port}.log"


def _serve_pid_path(port: int) -> Path:
    return Path(APP_ROOT) / ".local" / "run" / f"serve-{port}.json"


def _read_json_path(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_serve_pid_file(
    *,
    port: int,
    process: subprocess.Popen[Any],
    log_path: Path,
    public_url: str | None,
    state: dict[str, Any] | None,
) -> None:
    path = _serve_pid_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "pid": process.pid,
        "port": port,
        "log": str(log_path),
        "public_url": public_url,
        "openai_base_url": f"{public_url.rstrip('/')}/v1" if public_url else None,
    }
    if isinstance(state, dict):
        payload.update(
            {
                "backend": state.get("backend"),
                "model": state.get("model"),
                "mmproj": state.get("mmproj"),
                "ctx_size": state.get("ctx_size"),
                "backend_ready": state.get("backend_ready"),
                "backend_pid": state.get("backend_pid"),
                "backend_port": state.get("backend_port"),
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _serve_child_command(service_args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "serve", *service_args]
    return [sys.executable, str(Path(REPO_ROOT) / "omniinfer.py"), "serve", *service_args]


def _start_serve_child(plan: _ServePlan, *, port: int, log_path: Path) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env["OMNIINFER_SERVE_DIRECT"] = "1"
    if plan.api_key:
        env["OMNIINFER_API_KEY"] = plan.api_key
    command = _serve_child_command(plan.service_args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launching OmniInfer serve: {' '.join(command)}\n")
        log_handle.flush()
        popen_kwargs: dict[str, Any] = {
            "cwd": str(APP_ROOT),
            "env": env,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        else:  # pragma: no cover - Windows process-group behavior
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(command, **popen_kwargs)
    print(f"Starting OmniInfer service on port {port}...")
    print(f"Log: {log_path}")
    return process


def _cleanup_serve_child(process: subprocess.Popen[Any], *, port: int) -> None:
    global _cli_port_override
    _cli_port_override = port
    commands.set_port_override(port)
    try:
        commands.shutdown_service(wait_timeout_s=5.0)
    except BaseException:
        pass

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    elif process.poll() is None:  # pragma: no cover - Windows process cleanup
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        else:  # pragma: no cover - Windows process cleanup
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _wait_for_cloudflare_url(log_path: Path, *, timeout_s: int) -> str | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = _tail_file(log_path, max_chars=8000)
        for line in reversed(text.splitlines()):
            url = parse_trycloudflare_url(line)
            if url:
                return url
        time.sleep(0.25)
    return None


def _request_json_url(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url=url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.getcode(), try_parse_json(raw)
    except urllib.error.HTTPError as exc:
        parsed = try_parse_json(exc.read())
        raise SystemExit(f"{method} {url} failed with status {exc.code}: {parsed}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"{method} {url} failed: {exc}") from exc


def _serve_state(host: str, port: int) -> dict[str, Any]:
    _code, payload = _request_json_url("GET", f"http://{host}:{port}/health?deep=true", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Service health response has an unexpected format.")
    omni = payload.get("omni")
    return omni if isinstance(omni, dict) else payload


def _serve_smoke(base_url: str, *, api_key: str | None) -> str:
    payload = {
        "model": "omniinfer",
        "messages": [{"role": "user", "content": PUBLIC_SMOKE_PROMPT}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    _code, response = _request_json_url(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload=payload,
        api_key=api_key,
        timeout=120.0,
    )
    if not isinstance(response, dict):
        raise SystemExit("Smoke test response has an unexpected format.")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit("Smoke test response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise SystemExit("Smoke test response did not include a message.")
    content = str(message.get("content") or "").strip()
    if not content:
        raise SystemExit("Smoke test returned an empty response.")
    return content


def _print_serve_ready(
    *,
    port: int,
    state: dict[str, Any],
    public_url: str | None,
    api_key: str | None,
    print_api_key: bool,
    log_path: Path,
    smoke_text: str | None,
) -> None:
    print()
    print("OmniInfer service is ready")
    local_base_url = f"http://127.0.0.1:{port}/v1"
    if public_url:
        print(f"OpenAI Base URL: {public_url.rstrip('/')}/v1")
        print(f"Health URL: {public_url.rstrip('/')}/health")
    print(f"Local Base URL: {local_base_url}")
    if api_key and print_api_key:
        print(f"API Key: {api_key}")
    print(f"Backend: {state.get('backend') or '-'}")
    print(f"Backend ready: {'yes' if state.get('backend_ready') else 'no'}")
    print(f"Model: {state.get('model') or '-'}")
    print(f"mmproj: {state.get('mmproj') or '-'}")
    print(f"ctx-size: {state.get('ctx_size') or '-'}")
    if smoke_text is not None:
        print(f"Smoke: {smoke_text}")
    print(f"Log: {log_path}")
    print(f"Stop: ./omniinfer serve stop --port {port}")
    if public_url:
        print("Curl:")
        auth = f" -H 'Authorization: Bearer {api_key}'" if api_key and print_api_key else ""
        print(
            "  curl -sS"
            f"{auth} -H 'Content-Type: application/json' "
            f"{public_url.rstrip('/')}/v1/chat/completions "
            "-d '{\"model\":\"omniinfer\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}'"
        )


def _configure_serve_runtime(plan: _ServePlan, *, port: int) -> dict[str, Any]:
    global _cli_port_override
    _cli_port_override = port
    commands.set_port_override(port)
    if plan.backend:
        selected = commands.select_backend(plan.backend)
        print(f"Selected backend: {selected.backend}")
    if plan.model:
        response, selection = commands.load_model(
            commands.ModelLoadOptions(
                model=plan.model,
                mmproj=plan.mmproj,
                ctx_size=plan.ctx_size,
            )
        )
        if selection.auto_selected:
            print(f"No backend selected. Auto-selected: {selection.backend}")
        print(f"Model loaded: {response.get('selected_model') or plan.model}")
    host, _port, _timeout = _detached_health_target(plan.service_args)
    return _serve_state(host, port)


def serve_status(service_args: list[str]) -> int:
    port = _serve_port(service_args)
    host, _port, _timeout = _detached_health_target(service_args)
    try:
        state = _serve_state(host, port)
    except SystemExit as exc:
        print(f"OmniInfer service is not running on port {port}: {exc}")
        return 0
    pid_info = _read_json_path(_serve_pid_path(port))
    print("OmniInfer Serve Status")
    print(f"Port: {port}")
    if pid_info.get("pid"):
        print(f"PID: {pid_info.get('pid')}")
    if pid_info.get("public_url"):
        print(f"OpenAI Base URL: {str(pid_info['public_url']).rstrip('/')}/v1")
    print(f"Backend: {state.get('backend') or '-'}")
    print(f"Backend ready: {'yes' if state.get('backend_ready') else 'no'}")
    print(f"Model: {state.get('model') or '-'}")
    print(f"ctx-size: {state.get('ctx_size') or '-'}")
    if pid_info.get("log"):
        print(f"Log: {pid_info.get('log')}")
    return 0


def serve_stop(service_args: list[str]) -> int:
    global _cli_port_override
    port = _serve_port(service_args)
    _cli_port_override = port
    commands.set_port_override(port)
    stopped = commands.shutdown_service(wait_timeout_s=10.0)
    pid_path = _serve_pid_path(port)
    if pid_path.exists():
        try:
            pid_path.unlink()
        except OSError:
            pass
    if stopped:
        print(f"OmniInfer service stopped on port {port}")
    else:
        print(f"OmniInfer service is not running on port {port}")
    return 0


def serve_orchestrated(plan: _ServePlan) -> int:
    port = _serve_port(plan.service_args)
    health_host, health_port, startup_timeout = _detached_health_target(plan.service_args)
    log_path = _serve_log_path(port)
    process = _start_serve_child(plan, port=port, log_path=log_path)
    try:
        _wait_for_detached_health(
            process,
            host=health_host,
            port=health_port,
            timeout_s=startup_timeout,
            log_path=log_path,
        )
        public_url = _wait_for_cloudflare_url(log_path, timeout_s=startup_timeout) if _flag_in_argv(plan.service_args, "--cloudflare") else None
        state = _configure_serve_runtime(plan, port=port)
        smoke_text = None
        smoke_failed = False
        if plan.smoke_test:
            base_url = public_url or f"http://127.0.0.1:{port}"
            try:
                smoke_text = _serve_smoke(base_url, api_key=plan.api_key if public_url else None)
            except (Exception, SystemExit) as exc:
                smoke_failed = True
                smoke_text = f"failed: {exc}"
        _write_serve_pid_file(
            port=port,
            process=process,
            log_path=log_path,
            public_url=public_url,
            state=state,
        )
        _print_serve_ready(
            port=port,
            state=state,
            public_url=public_url,
            api_key=plan.api_key,
            print_api_key=plan.print_api_key,
            log_path=log_path,
            smoke_text=smoke_text,
        )
        if plan.detach:
            return 1 if smoke_failed else 0
        print("Press Ctrl+C to stop.")
        try:
            return int(process.wait())
        except KeyboardInterrupt:
            commands.shutdown_service(wait_timeout_s=10.0)
            return 0
    except BaseException:
        _cleanup_serve_child(process, port=port)
        raise


def serve_command(service_args: list[str]) -> int:
    if _is_help_request(service_args):
        return print_serve_help()
    plan = _parse_serve_plan(service_args)
    if plan.action == "status":
        return serve_status(plan.service_args)
    if plan.action == "stop":
        return serve_stop(plan.service_args)
    if _serve_plan_needs_orchestration(plan):
        return serve_orchestrated(plan)
    return serve_interactive_or_foreground(service_args)


def _requested_window_mode(argv: list[str]) -> str:
    mode = "hidden"
    for idx, token in enumerate(argv):
        if token == "--window-mode" and idx + 1 < len(argv):
            value = argv[idx + 1].strip().lower()
            if value in {"visible", "hidden"}:
                mode = value
        elif token.startswith("--window-mode="):
            value = token.split("=", 1)[1].strip().lower()
            if value in {"visible", "hidden"}:
                mode = value
    return mode


def _value_from_argv(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for idx, token in enumerate(argv):
        if token == name and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return None


def _flag_in_argv(argv: list[str], name: str) -> bool:
    return any(token == name for token in argv)


def _detached_health_target(argv: list[str]) -> tuple[str, int, int]:
    config = load_app_config(APP_ROOT)
    host = _value_from_argv(argv, "--host") or str(config.get("host", "127.0.0.1"))
    if _flag_in_argv(argv, "--lan") and _value_from_argv(argv, "--host") is None:
        host = "0.0.0.0"

    port_text = _value_from_argv(argv, "--port") or str(config.get("port", "9000"))
    timeout_text = _value_from_argv(argv, "--startup-timeout") or str(config.get("startup_timeout", "60"))
    try:
        port = int(port_text)
    except ValueError:
        port = 9000
    try:
        timeout = max(1, int(timeout_text))
    except ValueError:
        timeout = 60

    health_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    return health_host, port, timeout


def _detached_child_log_path() -> Path:
    return Path(APP_ROOT) / ".local" / "logs" / "serve-child.log"


def _tail_file(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def _wait_for_detached_health(
    process: subprocess.Popen[Any],
    *,
    host: str,
    port: int,
    timeout_s: int,
    log_path: Path,
) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            message = f"OmniInfer detached server exited early with code {returncode}; expected health at {url}."
            tail = _tail_file(log_path)
            if tail:
                message += f"\nChild log tail ({log_path}):\n{tail}"
            raise SystemExit(message)
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)

    message = f"OmniInfer detached server did not become healthy at {url} within {timeout_s}s."
    if last_error:
        message += f" Last error: {last_error}."
    tail = _tail_file(log_path)
    if tail:
        message += f"\nChild log tail ({log_path}):\n{tail}"
    raise SystemExit(message)


def _is_help_request(argv: list[str]) -> bool:
    return any(token in {"-h", "--help"} for token in argv)


def _is_cloudflare_request(argv: list[str]) -> bool:
    return any(token == "--cloudflare" for token in argv)


def _is_direct_serve_request() -> bool:
    return os.environ.get("OMNIINFER_SERVE_DIRECT") == "1"


def _has_console() -> bool:
    try:
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return False


def _attach_console_streams() -> None:
    sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
    sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)


def _ensure_window_mode(argv: list[str]) -> None:
    if os.name != "nt":
        return
    if _is_help_request(argv):
        return
    if _is_direct_serve_request():
        return
    if _is_cloudflare_request(argv):
        return

    mode = _requested_window_mode(argv)
    child_flag = "OMNIINFER_WINDOW_MODE_CHILD"
    if mode == "hidden" and os.environ.get(child_flag) != "1" and _has_console():
        env = os.environ.copy()
        env[child_flag] = "1"

        if getattr(sys, "frozen", False):
            command = [sys.executable, *sys.argv[1:]]
        else:
            command = [sys.executable, str(Path(REPO_ROOT) / "omniinfer.py"), *sys.argv[1:]]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        health_host, health_port, startup_timeout = _detached_health_target(argv)
        log_path = _detached_child_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launching detached OmniInfer: {' '.join(command)}\n"
            )
            log_handle.flush()
            process = subprocess.Popen(
                command,
                env=env,
                creationflags=creationflags,
                cwd=str(APP_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        _wait_for_detached_health(
            process,
            host=health_host,
            port=health_port,
            timeout_s=startup_timeout,
            log_path=log_path,
        )
        print(f"OmniInfer detached server is healthy at http://{health_host}:{health_port}/health")
        print(f"Detached server log: {log_path}")
        raise SystemExit(0)

    if mode == "visible" and not _has_console():
        if ctypes.windll.kernel32.AllocConsole():
            _attach_console_streams()


def serve_foreground(service_args: list[str]) -> int:
    _ensure_window_mode(service_args)
    from service_core.service import main as service_main

    return service_main(service_args)


def _should_run_server_tui(service_args: list[str]) -> bool:
    if os.environ.get("OMNIINFER_SERVE_DIRECT") == "1":
        return False
    if _is_help_request(service_args):
        return False
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


def serve_interactive_or_foreground(service_args: list[str]) -> int:
    if _should_run_server_tui(service_args):
        from service_core.tui import run_server_tui

        return run_server_tui(service_args)
    return serve_foreground(service_args)


def _service_args_from_cli(args: argparse.Namespace, unknown_args: list[str]) -> list[str]:
    service_args = list(unknown_args)
    if args.port is not None and not any(item == "--port" or item.startswith("--port=") for item in service_args):
        service_args.extend(["--port", str(args.port)])
    return service_args


def print_completion(shell: str) -> int:
    if shell != "bash":
        raise SystemExit("Only bash completion is currently available. Use: omniinfer completion bash")
    print(BASH_COMPLETION_SCRIPT)
    return 0


def print_serve_help() -> int:
    print(SERVE_HELP_TEXT)
    return 0


def complete_backend_name(prefix: str) -> list[str]:
    return [item for item in local_backend_ids() if item.startswith(prefix)]


def handle_hidden_completion(argv: list[str]) -> int:
    cword = int(os.environ.get("COMP_CWORD", "0")) - 1
    words = argv[:]
    current = words[cword] if 0 <= cword < len(words) else ""
    previous = words[cword - 1] if cword - 1 >= 0 else ""

    top_level = ["advisor", "backend", "status", "ps", "model", "load", "thinking", "chat", "shutdown", "serve", "completion"]
    if commands.is_backend_build_supported():
        top_level.insert(1, "build")
    backend_sub = ["list", "select", "stop"]
    advisor_sub = ["system", "inspect", "fit", "plan", "recommend"]
    model_sub = ["list", "load"]
    thinking_sub = ["show", "set"]
    serve_sub = ["status", "stop"]

    suggestions: list[str] = []
    if cword <= 0:
        suggestions = top_level
    elif words and words[0] == "build":
        if cword == 1:
            suggestions = complete_backend_name(current)
    elif words and words[0] == "backend":
        if cword == 1:
            suggestions = backend_sub
        elif len(words) > 1 and words[1] == "select":
            suggestions = complete_backend_name(current)
    elif words and words[0] == "advisor":
        if cword == 1:
            suggestions = advisor_sub
        elif previous in {"--backend"}:
            suggestions = complete_backend_name(current)
    elif words and words[0] == "model":
        if cword == 1:
            suggestions = model_sub
        elif previous in {"--system"}:
            suggestions = [item for item in SYSTEM_CHOICES if item.startswith(current)]
    elif words and words[0] == "thinking":
        if cword == 1:
            suggestions = thinking_sub
        elif previous == "set":
            suggestions = [item for item in ("on", "off") if item.startswith(current)]
    elif words and words[0] == "serve":
        if cword == 1 and not current.startswith("-"):
            suggestions = serve_sub
        elif previous == "--backend":
            suggestions = complete_backend_name(current)
    elif previous in {"--think"}:
        suggestions = [item for item in ("on", "off") if item.startswith(current)]
    elif previous in {"--system"}:
        suggestions = [item for item in SYSTEM_CHOICES if item.startswith(current)]
    elif words and words[0] == "completion" and cword == 1:
        suggestions = ["bash"]

    print(" ".join(sorted(set(suggestions))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    return build_parser_bundle()["parser"]


def build_parser_bundle() -> dict[str, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        prog="omniinfer",
        description="OmniInfer local CLI",
        epilog=_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=None, help="Gateway port (overrides config file)")
    sub = parser.add_subparsers(dest="command", parser_class=CommandGroupParser)

    backend = sub.add_parser(
        "backend",
        help="Backend commands",
        description="OmniInfer backend commands",
        epilog=BACKEND_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    backend_sub = backend.add_subparsers(dest="backend_command", metavar="COMMAND", parser_class=CommandGroupParser)
    backend_list = backend_sub.add_parser("list", help="List backends available on this system")
    backend_list.add_argument("--scope", choices=("installed", "compatible", "all"), default="compatible", help="Filter backends by scope (default: compatible)")
    backend_list.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON (for scripting)")
    backend_select = backend_sub.add_parser("select", help="Select a backend")
    backend_select.add_argument("backend_name", help="Backend name")
    backend_sub.add_parser("stop", help="Stop the current backend process")

    if commands.is_backend_build_supported():
        build = sub.add_parser("build", help="Build a backend from this source checkout")
        build.add_argument("backend_name", help="Backend name")
        build_mode = build.add_mutually_exclusive_group()
        build_mode.add_argument("--prebuilt", action="store_true", help="Install the backend from a configured prebuilt archive")
        build_mode.add_argument("--from-source", action="store_true", help="Build the backend from the checked-out source submodule")

    sub.add_parser("status", help="Show current status")

    ps = sub.add_parser("ps", help="List all running OmniInfer services")
    ps.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    model = sub.add_parser(
        "model",
        help="Model commands",
        description="OmniInfer model commands",
        epilog=MODEL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_sub = model.add_subparsers(dest="model_command", metavar="COMMAND", parser_class=CommandGroupParser)
    model_list = model_sub.add_parser("list", help="List supported models")
    model_list.add_argument("--system", choices=SYSTEM_CHOICES, default=detect_system_name(), help="Target system, defaults to the current system")
    model_list.add_argument("--all-backends", action="store_true", help="Show the raw backend-grouped view")
    model_load = model_sub.add_parser("load", help="Load a model")
    add_model_load_arguments(model_load)

    load_alias = sub.add_parser("load", help="Load a model")
    add_model_load_arguments(load_alias)

    advisor = sub.add_parser(
        "advisor",
        help="Hardware and model advisor commands",
        description="OmniInfer advisor commands",
        epilog=ADVISOR_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    advisor_sub = advisor.add_subparsers(dest="advisor_command", metavar="COMMAND", parser_class=CommandGroupParser)
    advisor_system = advisor_sub.add_parser("system", help="Inspect local hardware and OmniInfer runtimes")
    advisor_system.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    advisor_inspect = advisor_sub.add_parser("inspect", help="Inspect a model reference or local model artifact")
    advisor_inspect.add_argument("model", help="Model path, model directory, or backend reference")
    advisor_inspect.add_argument("-mm", "--mmproj", help="Optional mmproj path")
    advisor_inspect.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    advisor_fit = advisor_sub.add_parser("fit", help="Recommend a backend and launch shape for a model")
    advisor_fit.add_argument("model", help="Model path, model directory, or backend reference")
    advisor_fit.add_argument("-mm", "--mmproj", help="Optional mmproj path")
    advisor_fit.add_argument("--ctx-size", type=int, help="Context length used for estimation")
    advisor_fit.add_argument("--backend", help="Restrict analysis to a single backend")
    advisor_fit.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    advisor_plan = advisor_sub.add_parser("plan", help="Estimate hardware requirements for a model")
    advisor_plan.add_argument("model", help="Model path, model directory, or backend reference")
    advisor_plan.add_argument("-mm", "--mmproj", help="Optional mmproj path")
    advisor_plan.add_argument("--ctx-size", type=int, help="Context length used for estimation")
    advisor_plan.add_argument("--gpu-vram", type=float, dest="gpu_vram_gib", help="Simulate available GPU VRAM in GiB")
    advisor_plan.add_argument("--ram", type=float, dest="ram_gib", help="Simulate available system RAM in GiB")
    advisor_plan.add_argument("--cpu-cores", type=int, help="Simulate CPU core count")
    advisor_plan.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    advisor_recommend = advisor_sub.add_parser("recommend", help="Recommend from locally managed model files")
    advisor_recommend.add_argument("--task", help="Optional task filter: chat, coding, vision, embedding")
    advisor_recommend.add_argument("-n", "--limit", type=int, default=5, help="Maximum recommendations to show")
    advisor_recommend.add_argument("--ctx-size", type=int, help="Context length used for estimation")
    advisor_recommend.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    thinking = sub.add_parser(
        "thinking",
        help="Default thinking controls",
        description="OmniInfer thinking commands",
        epilog=THINKING_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    thinking_sub = thinking.add_subparsers(dest="thinking_command", metavar="COMMAND", parser_class=CommandGroupParser)
    thinking_sub.add_parser("show", help="Show the default thinking state")
    thinking_set = thinking_sub.add_parser("set", help="Set the default thinking state")
    thinking_set.add_argument("value", choices=("on", "off"), help="on or off")

    chat_cmd = sub.add_parser("chat", help="Run inference on the loaded model")
    chat_cmd.add_argument("prompt", nargs="?", help="User message")
    chat_cmd.add_argument("--message", help="User message")
    stream_group = chat_cmd.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=None, help="Stream tokens to stdout")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false", help="Wait for the final response before printing")
    chat_cmd.add_argument("--image", help="Optional image path for multimodal inference")
    chat_cmd.add_argument("--think", choices=("on", "off"), help="Thinking mode for this request")
    chat_cmd.add_argument("--temperature", type=float, help="Sampling temperature")
    chat_cmd.add_argument("--max-tokens", type=int, help="Maximum output tokens for this request")

    sub.add_parser("shutdown", help="Stop the OmniInfer service")
    serve = sub.add_parser("serve", aliases=("server",), help="Start the OmniInfer service in the foreground")
    serve.set_defaults(command="serve")

    completion = sub.add_parser("completion", help="Print shell completion")
    completion.add_argument("shell", choices=("bash",), help="Currently supported shell: bash")

    return {
        "parser": parser,
        "backend": backend,
        "model": model,
        "advisor": advisor,
        "thinking": thinking,
    }


def _group_error(parser: argparse.ArgumentParser, message: str) -> None:
    parser.print_help(sys.stderr)
    parser.exit(2, f"Error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    global _cli_port_override

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "__complete":
        return handle_hidden_completion(argv[1:])

    parser_bundle = build_parser_bundle()
    parser = parser_bundle["parser"]
    if argv and argv[0] in {"serve", "server"}:
        args, unknown_args = parser.parse_known_args(argv[:1])
        _cli_port_override = args.port
        commands.set_port_override(args.port)
        return serve_command(_service_args_from_cli(args, argv[1:]))

    from service_core.logger import setup_logging

    setup_logging(level="DEBUG", console=False, log_to_file=True)
    cli_logger = logging.getLogger("cli")
    cli_logger.debug("CLI invoked: %s", " ".join(sys.argv))

    args, unknown_args = parser.parse_known_args(argv)
    unknown_args = [item for item in unknown_args if item != "--"]
    setattr(args, "backend_extra_args", unknown_args)

    if args.port is not None:
        _cli_port_override = args.port
    commands.set_port_override(args.port)

    if args.command is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            from service_core.tui import run_tui

            return run_tui()
        parser.print_help()
        return 0

    if args.command == "backend":
        if unknown_args:
            _group_error(parser_bundle["backend"], f"unrecognized arguments: {' '.join(unknown_args)}")
        if args.backend_command == "list":
            return print_backend_list(scope=args.scope, json_output=getattr(args, "json_output", False))
        if args.backend_command == "select":
            return select_backend(args.backend_name)
        if args.backend_command == "stop":
            return backend_stop()
        _group_error(parser_bundle["backend"], "missing subcommand. Try one of: list, select, stop")

    if args.command == "build":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return build_backend(
            args.backend_name,
            prebuilt=getattr(args, "prebuilt", False),
            from_source=getattr(args, "from_source", False),
        )

    if args.command == "advisor":
        if unknown_args:
            _group_error(parser_bundle["advisor"], f"unrecognized arguments: {' '.join(unknown_args)}")
        if args.advisor_command == "system":
            return print_advisor_system(json_output=getattr(args, "json_output", False))
        if args.advisor_command == "inspect":
            return print_advisor_inspect(args.model, mmproj=getattr(args, "mmproj", None), json_output=getattr(args, "json_output", False))
        if args.advisor_command == "fit":
            return print_advisor_fit(
                args.model,
                mmproj=getattr(args, "mmproj", None),
                ctx_size=getattr(args, "ctx_size", None),
                backend=getattr(args, "backend", None),
                json_output=getattr(args, "json_output", False),
            )
        if args.advisor_command == "plan":
            return print_advisor_plan(
                args.model,
                mmproj=getattr(args, "mmproj", None),
                ctx_size=getattr(args, "ctx_size", None),
                gpu_vram_gib=getattr(args, "gpu_vram_gib", None),
                ram_gib=getattr(args, "ram_gib", None),
                cpu_cores=getattr(args, "cpu_cores", None),
                json_output=getattr(args, "json_output", False),
            )
        if args.advisor_command == "recommend":
            return print_advisor_recommend(
                task=getattr(args, "task", None),
                limit=getattr(args, "limit", 5),
                ctx_size=getattr(args, "ctx_size", None),
                json_output=getattr(args, "json_output", False),
            )
        _group_error(parser_bundle["advisor"], "missing subcommand. Try one of: system, inspect, fit, plan, recommend")

    if args.command == "status":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return print_status()
    if args.command == "ps":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return print_ps(json_output=getattr(args, "json_output", False))
    if args.command == "model":
        if args.model_command == "list":
            if unknown_args:
                _group_error(parser_bundle["model"], f"unrecognized arguments: {' '.join(unknown_args)}")
            return print_model_list(system_name=args.system, best=not args.all_backends)
        if args.model_command == "load":
            return print_model_load(args)
        _group_error(parser_bundle["model"], "missing subcommand. Try one of: list, load")
    if args.command == "load":
        return print_model_load(args)
    if args.command == "thinking":
        if unknown_args:
            _group_error(parser_bundle["thinking"], f"unrecognized arguments: {' '.join(unknown_args)}")
        if args.thinking_command == "show":
            return print_thinking_show()
        if args.thinking_command == "set":
            return print_thinking_set(args.value)
        _group_error(parser_bundle["thinking"], "missing subcommand. Try one of: show, set")
    if args.command == "chat":
        return chat(args)
    if args.command == "shutdown":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return shutdown_service()
    if args.command == "serve":
        return serve_command(_service_args_from_cli(args, unknown_args))
    if args.command == "completion":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return print_completion(args.shell)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
