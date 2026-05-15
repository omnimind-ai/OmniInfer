from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import re
import sys
import threading
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from service_core import commands
from service_core.service import APP_ROOT, REPO_ROOT, load_app_config


_cli_port_override: int | None = None
DEFAULT_IMAGE_PATH = Path(REPO_ROOT) / "tests" / "pictures" / "test1.png"
SYSTEM_CHOICES = ("linux", "mac", "windows")


HELP_TEXT = """\
OmniInfer CLI

Common commands:
  omniinfer
  omniinfer backend list
  omniinfer backend select <backend>
  omniinfer status
  omniinfer load -m /path/to/model.gguf
  omniinfer load -m /path/to/model.gguf --config
  omniinfer chat "Introduce yourself in one sentence."
  omniinfer chat "Describe this image." --image photo.png
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
  status                    -> show service status, selected backend, and loaded model
  load                      -> load a model
  model list                -> show models supported on the current system
  model load                -> load a model, optionally with a backend config JSON
  thinking show/set         -> show or change the default think setting
  chat                      -> run text or multimodal inference on the loaded model
  backend stop              -> stop the currently running backend process
  shutdown                  -> stop the OmniInfer service
  serve                     -> start the service in the foreground
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


def build_backend(name: str) -> int:
    options = commands.BackendBuildOptions(backend=name)
    command, _script_path = commands.backend_build_command(options)
    print(f"Building backend: {name}")
    print("Build type: Release")
    print("Command: " + " ".join(command))

    result = commands.build_backend(options)
    print(f"Backend build completed: {result.backend}")
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

    mode = _requested_window_mode(argv)
    child_flag = "OMNIINFER_WINDOW_MODE_CHILD"
    if mode == "hidden" and os.environ.get(child_flag) != "1" and _has_console():
        env = os.environ.copy()
        env[child_flag] = "1"
        import subprocess

        if getattr(sys, "frozen", False):
            command = [sys.executable, *sys.argv[1:]]
        else:
            command = [sys.executable, str(Path(REPO_ROOT) / "omniinfer.py"), *sys.argv[1:]]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            command,
            env=env,
            creationflags=creationflags,
            cwd=str(APP_ROOT),
        )
        raise SystemExit(0)

    if mode == "visible" and not _has_console():
        if ctypes.windll.kernel32.AllocConsole():
            _attach_console_streams()


def serve_foreground(service_args: list[str]) -> int:
    _ensure_window_mode(service_args)
    from service_core.service import main as service_main

    return service_main(service_args)


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


def complete_backend_name(prefix: str) -> list[str]:
    return [item for item in local_backend_ids() if item.startswith(prefix)]


def handle_hidden_completion(argv: list[str]) -> int:
    cword = int(os.environ.get("COMP_CWORD", "0")) - 1
    words = argv[:]
    current = words[cword] if 0 <= cword < len(words) else ""
    previous = words[cword - 1] if cword - 1 >= 0 else ""

    top_level = ["backend", "status", "ps", "model", "load", "thinking", "chat", "shutdown", "serve", "completion"]
    if commands.is_backend_build_supported():
        top_level.insert(1, "build")
    backend_sub = ["list", "select", "stop"]
    model_sub = ["list", "load"]
    thinking_sub = ["show", "set"]

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
    elif previous in {"--think"}:
        suggestions = [item for item in ("on", "off") if item.startswith(current)]
    elif previous in {"--system"}:
        suggestions = [item for item in SYSTEM_CHOICES if item.startswith(current)]
    elif words and words[0] == "completion" and cword == 1:
        suggestions = ["bash"]

    print(" ".join(sorted(set(suggestions))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniinfer",
        description="OmniInfer local CLI",
        epilog=_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=None, help="Gateway port (overrides config file)")
    sub = parser.add_subparsers(dest="command")

    backend = sub.add_parser("backend", help="Backend commands")
    backend_sub = backend.add_subparsers(dest="backend_command")
    backend_list = backend_sub.add_parser("list", help="List backends available on this system")
    backend_list.add_argument("--scope", choices=("installed", "compatible", "all"), default="compatible", help="Filter backends by scope (default: compatible)")
    backend_list.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON (for scripting)")
    backend_select = backend_sub.add_parser("select", help="Select a backend")
    backend_select.add_argument("backend_name", help="Backend name")
    backend_sub.add_parser("stop", help="Stop the current backend process")

    if commands.is_backend_build_supported():
        build = sub.add_parser("build", help="Build a backend from this source checkout")
        build.add_argument("backend_name", help="Backend name")

    sub.add_parser("status", help="Show current status")

    ps = sub.add_parser("ps", help="List all running OmniInfer services")
    ps.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    model = sub.add_parser("model", help="Model commands")
    model_sub = model.add_subparsers(dest="model_command")
    model_list = model_sub.add_parser("list", help="List supported models")
    model_list.add_argument("--system", choices=SYSTEM_CHOICES, default=detect_system_name(), help="Target system, defaults to the current system")
    model_list.add_argument("--all-backends", action="store_true", help="Show the raw backend-grouped view")
    model_load = model_sub.add_parser("load", help="Load a model")
    add_model_load_arguments(model_load)

    load_alias = sub.add_parser("load", help="Load a model")
    add_model_load_arguments(load_alias)

    thinking = sub.add_parser("thinking", help="Default thinking controls")
    thinking_sub = thinking.add_subparsers(dest="thinking_command")
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
    sub.add_parser("serve", help="Start the OmniInfer service in the foreground")

    completion = sub.add_parser("completion", help="Print shell completion")
    completion.add_argument("shell", choices=("bash",), help="Currently supported shell: bash")

    return parser


def main(argv: list[str] | None = None) -> int:
    global _cli_port_override

    from service_core.logger import setup_logging

    setup_logging(level="DEBUG", console=False, log_to_file=True)
    cli_logger = logging.getLogger("cli")
    cli_logger.debug("CLI invoked: %s", " ".join(sys.argv))

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "__complete":
        return handle_hidden_completion(argv[1:])

    parser = build_parser()
    if argv and argv[0] == "serve":
        args, unknown_args = parser.parse_known_args(argv[:1])
        _cli_port_override = args.port
        commands.set_port_override(args.port)
        return serve_foreground(_service_args_from_cli(args, argv[1:]))

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
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        if args.backend_command == "list":
            return print_backend_list(scope=args.scope, json_output=getattr(args, "json_output", False))
        if args.backend_command == "select":
            return select_backend(args.backend_name)
        if args.backend_command == "stop":
            return backend_stop()
        parser.error("backend requires a subcommand: list / select / stop")

    if args.command == "build":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return build_backend(args.backend_name)

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
                parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
            return print_model_list(system_name=args.system, best=not args.all_backends)
        if args.model_command == "load":
            return print_model_load(args)
        parser.error("model requires a subcommand: list / load")
    if args.command == "load":
        return print_model_load(args)
    if args.command == "thinking":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        if args.thinking_command == "show":
            return print_thinking_show()
        if args.thinking_command == "set":
            return print_thinking_set(args.value)
        parser.error("thinking requires a subcommand: show / set")
    if args.command == "chat":
        return chat(args)
    if args.command == "shutdown":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return shutdown_service()
    if args.command == "serve":
        return serve_foreground(_service_args_from_cli(args, unknown_args))
    if args.command == "completion":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return print_completion(args.shell)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
