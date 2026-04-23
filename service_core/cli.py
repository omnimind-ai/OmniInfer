from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from service_core.backend_cli_args import (
    ParsedBackendExtraArgs,
    parse_backend_chat_extra_args,
    parse_backend_load_extra_args,
)
from service_core.backend_configs import (
    BackendProfile,
    ensure_backend_profile_template,
    load_backend_profile,
    profile_path_for_backend,
)
from service_core.runtime import RuntimeManager
from service_core.service import APP_ROOT, REPO_ROOT, load_app_config


CLI_STATE_DIR = Path.home() / ".config" / "omniinfer"
CLI_STATE_FILE = CLI_STATE_DIR / "cli_state.json"
CLI_LOG_DIR = Path.home() / ".cache" / "omniinfer"
CLI_LOG_FILE = CLI_LOG_DIR / "gateway.log"

_cli_port_override: int | None = None
DEFAULT_IMAGE_PATH = Path(REPO_ROOT) / "tests" / "pictures" / "test1.png"
SYSTEM_CHOICES = ("linux", "mac", "windows")


HELP_TEXT = """\
OmniInfer CLI

Common commands:
  omniinfer backend list
  omniinfer select <backend>
  omniinfer status
  omniinfer model load -m /path/to/model.gguf
  omniinfer model load -m /path/to/model-directory --config
  omniinfer select mlx-mac
  omniinfer model load -m /path/to/mlx-model-directory
  omniinfer chat --message "Introduce yourself in one sentence."
  omniinfer chat --message "Introduce yourself in one sentence." --config
  omniinfer chat -m /path/to/model.gguf --message "Introduce yourself."
  omniinfer chat -m /path/to/model.gguf -mm /path/to/mmproj.gguf --image tests/pictures/test1.png --message "Describe this image."
  omniinfer shutdown

Design notes:
  1. The CLI automatically checks whether the local OmniInfer service is running and starts it when needed.
  2. Host and port details are hidden by default; the CLI uses local app configuration automatically.
  3. `omniinfer select <backend>` persists the current backend selection.
  4. `omniinfer chat` uses the selected backend by default. If none is selected, it prompts you to run `omniinfer backend list` and `omniinfer select <backend>`.
  5. Output is designed for a local inference workflow rather than exposing raw HTTP JSON.
  6. `omniinfer chat` streams tokens to stdout by default. Use `--no-stream` if you want the final response after completion instead.

Command map:
  backend list              -> show available backends
  select <backend>          -> choose a backend
  status                    -> show service status, selected backend, and loaded model
  model list                -> show models supported on the current system
  model load                -> choose and load a model, optionally from a backend config JSON
  thinking show/set         -> show or change the default think setting
  chat                      -> run text or multimodal inference, optionally from a backend config JSON
  backend stop              -> stop the currently running backend process
  shutdown                  -> stop the OmniInfer service
  serve                     -> start the service in the foreground
  completion bash           -> print the bash completion script
"""


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


def parse_boolish(value: str) -> bool:
    low = value.strip().lower()
    if low in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if low in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    raise SystemExit(f"invalid boolean value: {value!r}")


def load_cli_state() -> dict[str, Any]:
    if not CLI_STATE_FILE.is_file():
        return {}
    try:
        with CLI_STATE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_cli_state(payload: dict[str, Any]) -> None:
    CLI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CLI_STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(CLI_STATE_FILE)


def get_service_config() -> dict[str, Any]:
    config = load_app_config(Path(APP_ROOT))
    if _cli_port_override is not None:
        config["port"] = _cli_port_override
    return config


def service_base_url() -> str:
    config = get_service_config()
    return f"http://{config['host']}:{int(config['port'])}"


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
    return list(local_backends().keys())


def local_backends() -> dict[str, Any]:
    config = get_service_config()
    manager = RuntimeManager(
        repo_root=str(REPO_ROOT),
        app_root=str(APP_ROOT),
        backend_host="127.0.0.1",
        backend_port=0,
        startup_timeout_s=int(config.get("startup_timeout", 60)),
        backend_window_mode=str(config.get("window_mode", "hidden")),
        runtime_root=str(config.get("runtime_root", "runtime")),
        backend_overrides=config.get("backends"),
        default_backend_id=str(config.get("default_backend", "")),
    )
    return manager.backends


def start_service_background() -> None:
    config = get_service_config()
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 9000))
    startup_timeout = int(config.get("startup_timeout", 60))
    window_mode = str(config.get("window_mode", "hidden"))
    default_backend = str(config.get("default_backend", ""))
    default_thinking = str(config.get("default_thinking", "off"))

    command = gateway_launch_command(
        host=host,
        port=port,
        startup_timeout=startup_timeout,
        window_mode=window_mode,
        default_thinking=default_thinking,
        default_backend=default_backend,
    )

    logging.getLogger("cli").info("Starting gateway service in background: %s", " ".join(command))
    CLI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = CLI_LOG_FILE.open("a", encoding="utf-8")
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
    else:
        subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    log_handle.close()


def wait_for_service_ready(timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_service_running():
            return
        time.sleep(0.5)
    tail = ""
    if CLI_LOG_FILE.is_file():
        tail = CLI_LOG_FILE.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise SystemExit(f"failed to start local OmniInfer service in time\n{tail}")


def resolve_gateway_binary() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None

    current_exe = Path(sys.executable).resolve()
    candidates = [
        current_exe.with_name("OmniInfer.exe"),
        current_exe.with_name("OmniInfer"),
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != current_exe:
            return candidate
    return None


def gateway_launch_command(
    *,
    host: str,
    port: int,
    startup_timeout: int,
    window_mode: str,
    default_thinking: str,
    default_backend: str,
) -> list[str]:
    if getattr(sys, "frozen", False):
        gateway_binary = resolve_gateway_binary()
        if gateway_binary is None:
            raise SystemExit("unable to locate the packaged OmniInfer gateway binary next to the CLI")
        command = [str(gateway_binary)]
    else:
        command = [sys.executable, str(Path(REPO_ROOT) / "omniinfer_gateway.py")]

    command.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--startup-timeout",
            str(startup_timeout),
            "--window-mode",
            window_mode,
            "--default-thinking",
            default_thinking,
        ]
    )
    if default_backend:
        command.extend(["--default-backend", default_backend])
    return command


def ensure_service_running() -> None:
    if is_service_running():
        return
    start_service_background()
    wait_for_service_ready(timeout_s=max(int(get_service_config().get("startup_timeout", 60)), 10))


def sync_selected_backend() -> str | None:
    state = load_cli_state()
    selected_backend = state.get("selected_backend")
    if not selected_backend:
        return None

    status, payload, _ = request_json("GET", "/omni/state", timeout=10.0)
    if isinstance(payload, dict) and payload.get("backend") == selected_backend:
        return str(selected_backend)

    request_json("POST", "/omni/backend/select", payload={"backend": selected_backend}, timeout=30.0)
    return str(selected_backend)


def ensure_service_and_selection() -> str | None:
    ensure_service_running()
    return sync_selected_backend()


def format_bool(value: bool | None) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "unknown"


def print_backend_list(scope: str = "all") -> int:
    ensure_service_running()
    state = load_cli_state()
    saved_backend = state.get("selected_backend")
    _status, payload, _ = request_json("GET", f"/omni/backends?scope={scope}", timeout=10.0)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No backends are available on this system.")

    print("Available backends")
    for item in rows:
        backend_id = str(item.get("id", ""))
        marker = "* " if backend_id == saved_backend else "  "
        runtime_available = "yes" if item.get("binary_exists") else "no"
        selected = "yes" if item.get("selected") else "no"
        capabilities = ", ".join(item.get("capabilities") or [])
        print(f"{marker}{backend_id}")
        print(f"    Selected in CLI: {'yes' if backend_id == saved_backend else 'no'}")
        print(f"    Active in service: {selected}")
        print(f"    Runtime available: {runtime_available}")
        if capabilities:
            print(f"    Capabilities: {capabilities}")
        description = str(item.get('description', '')).strip()
        if description:
            print(f"    Description: {description}")
    return 0


def select_backend(name: str) -> int:
    ensure_service_running()
    available_backends = local_backends()
    available = list(available_backends.keys())
    if name not in available_backends:
        raise SystemExit(f"Unsupported backend: {name}\nAvailable backends: {', '.join(available)}")
    _status, payload, _ = request_json("POST", "/omni/backend/select", payload={"backend": name}, timeout=30.0)
    save_selected_backend_name(name)
    print(f"Selected backend: {name}")
    models_dir = payload.get("models_dir") if isinstance(payload, dict) else None
    if models_dir:
        print(f"Models directory: {models_dir}")
    profile_path, created = ensure_backend_profile_template(available_backends[name])
    print(
        "Backend config: "
        f"{profile_path} ({'created' if created else 'already exists'})"
    )
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
    print(f"Active backend: {payload.get('backend') or 'unknown'}")
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


def require_selected_backend(allow_auto: bool) -> str | None:
    selected_backend = ensure_service_and_selection()
    if selected_backend:
        return selected_backend
    if allow_auto:
        return None
    raise SystemExit(
        "No backend has been selected yet.\n"
        "Run the following first:\n"
        "  omniinfer backend list\n"
        "  omniinfer select <backend>\n"
        "If you want OmniInfer to auto-select a backend for this command, use --auto."
    )


def resolve_model_reference(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Model path does not exist: {path}")
    return path


def resolve_existing_path(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def resolve_backend_profile_arg(path_text: str | None, selected_backend: str | None) -> BackendProfile | None:
    if path_text is None:
        return None
    target_path = path_text
    if path_text == "":
        if not selected_backend:
            raise SystemExit(
                "Using --config without a path requires a selected backend. "
                "Run `omniinfer select <backend>` first."
            )
        target_path = str(profile_path_for_backend(selected_backend))
    try:
        profile = load_backend_profile(target_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if profile.backend_id and selected_backend and profile.backend_id != selected_backend:
        raise SystemExit(
            f"Backend config {profile.path} belongs to {profile.backend_id}, "
            f"but the current selected backend is {selected_backend}."
        )
    return profile


def save_selected_backend_name(name: str | None) -> None:
    if not name:
        return
    state = load_cli_state()
    state["selected_backend"] = name
    save_cli_state(state)


def merge_backend_request_overrides(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = list(merged[key]) + list(value)
        else:
            merged[key] = value
    return merged


def resolve_backend_spec_for_native_args(
    *,
    allow_auto: bool,
    needs_native_args: bool,
) -> tuple[str | None, Any | None]:
    if allow_auto and needs_native_args:
        raise SystemExit(
            "Backend-native extra args cannot be combined with --auto. "
            "Run `omniinfer select <backend>` first, then use --config or backend-specific extra args."
        )
    selected_backend = require_selected_backend(allow_auto=allow_auto)
    if not selected_backend:
        if needs_native_args:
            raise SystemExit(
                "Backend-native extra args require a selected backend. "
                "Run `omniinfer select <backend>` first."
            )
        return None, None
    backends = local_backends()
    backend = backends.get(selected_backend)
    if backend is None:
        raise SystemExit(f"Selected backend is no longer available locally: {selected_backend}")
    return selected_backend, backend


def combine_backend_extra_args(
    *,
    backend: Any,
    command_name: str,
    profile: BackendProfile | None,
    cli_tokens: list[str],
) -> ParsedBackendExtraArgs:
    profile_tokens: list[str] = []
    if profile is not None:
        profile_tokens = profile.load_extra_args if command_name == "load" else profile.infer_extra_args
    merged_tokens = [*profile_tokens, *cli_tokens]
    try:
        if command_name == "load":
            return parse_backend_load_extra_args(backend, merged_tokens)
        return parse_backend_chat_extra_args(backend, merged_tokens)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def print_model_load(args: argparse.Namespace) -> int:
    verbose = getattr(args, "verbose", False)
    spinner = _Spinner()
    if not verbose:
        spinner.start()

    try:
        config_arg = getattr(args, "config", None)
        backend_extra_args = list(getattr(args, "backend_extra_args", []))
        selected_backend, backend = resolve_backend_spec_for_native_args(
            allow_auto=bool(getattr(args, "auto", False)),
            needs_native_args=bool(config_arg is not None or backend_extra_args),
        )
        profile = resolve_backend_profile_arg(config_arg, selected_backend)
        if profile and profile.backend_id and not selected_backend:
            selected_backend = profile.backend_id
            backend = local_backends().get(selected_backend)
        backend_extras = combine_backend_extra_args(
            backend=backend,
            command_name="load",
            profile=profile,
            cli_tokens=backend_extra_args,
        ) if backend is not None else ParsedBackendExtraArgs()

        model_input = args.model
        if not model_input:
            spinner.stop()
            raise SystemExit("Please specify a model path with -m or --model.")
        model_ref = resolve_model_reference(model_input)
        mmproj_input = args.mmproj
        mmproj_file = resolve_existing_path(mmproj_input, "mmproj file") if mmproj_input else None
        effective_ctx_size = args.ctx_size if args.ctx_size is not None else backend_extras.ctx_size
        if effective_ctx_size is not None and effective_ctx_size <= 0:
            spinner.stop()
            raise SystemExit("--ctx-size must be a positive integer")

        payload: dict[str, Any] = {"model": str(model_ref)}
        if mmproj_file:
            payload["mmproj"] = str(mmproj_file)
        if effective_ctx_size is not None:
            payload["ctx_size"] = effective_ctx_size
        if selected_backend:
            payload["backend"] = selected_backend
        if backend_extras.launch_args:
            payload["launch_args"] = backend_extras.launch_args
        if profile is not None and backend is not None:
            profile_chat_extras = combine_backend_extra_args(
                backend=backend,
                command_name="chat",
                profile=None,
                cli_tokens=list(profile.infer_extra_args),
            )
            if profile_chat_extras.request_overrides:
                payload["request_defaults"] = profile_chat_extras.request_overrides

        response = _stream_model_load(payload, timeout=600.0, verbose=verbose, spinner=spinner)
    except SystemExit:
        spinner.stop()
        raise

    if not isinstance(response, dict):
        raise SystemExit("Failed to load the model.")
    save_selected_backend_name(str(response.get("selected_backend") or selected_backend or ""))
    print("Model loaded")
    print(f"Backend: {response.get('selected_backend') or '-'}")
    print(f"Model: {response.get('selected_model') or '-'}")
    print(f"mmproj: {response.get('selected_mmproj') or '-'}")
    print(f"ctx-size: {response.get('selected_ctx_size') or '-'}")
    return 0


_SPINNER_FRAMES = "⠋⠙⠸⢰⣠⣄⡆⠇"
_SPINNER_INTERVAL = 0.08  # ~12 fps


class _Spinner:
    """Thread-driven spinner that renders independently of the event stream."""

    def __init__(self) -> None:
        self._text = "Loading model..."
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


def _stream_model_load(
    payload: dict[str, Any],
    timeout: float = 600.0,
    verbose: bool = False,
    spinner: _Spinner | None = None,
) -> dict[str, Any]:
    """POST /omni/model/select with SSE progress, falling back to JSON."""
    url = f"{service_base_url()}/omni/model/select"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Accept": "text/event-stream, application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    if spinner is None:
        spinner = _Spinner()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                spinner.stop()
                raw = response.read()
                parsed = try_parse_json(raw)
                if isinstance(parsed, dict):
                    return parsed
                raise SystemExit("Failed to load the model.")

            result: dict[str, Any] = {}
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type", "")
                if event_type == "status":
                    msg = event.get("message", "")
                    if verbose:
                        print(msg)
                    elif msg:
                        spinner.update(msg)
                elif event_type == "log":
                    msg = event.get("message", "")
                    if verbose and msg:
                        print(f"  {msg}")
                elif event_type == "done":
                    spinner.stop()
                    elapsed = event.get("elapsed_s")
                    if elapsed is not None:
                        print(f"Backend ready ({elapsed}s)")
                    result = event
                elif event_type == "error":
                    spinner.stop()
                    raise SystemExit(event.get("message", "model loading failed"))
            spinner.stop()
            return result
    except urllib.error.HTTPError as exc:
        spinner.stop()
        raw = exc.read().decode("utf-8", errors="replace")
        parsed = try_parse_json(raw.encode("utf-8"))
        if isinstance(parsed, dict):
            error = parsed.get("error", {})
            message = error.get("message", raw) if isinstance(error, dict) else raw
        else:
            message = raw
        raise SystemExit(f"Model loading failed (HTTP {exc.code}): {message}") from exc
    except urllib.error.URLError as exc:
        spinner.stop()
        raise SystemExit(f"Unable to reach local OmniInfer service: {exc}") from exc


def print_thinking_show() -> int:
    ensure_service_running()
    _status, payload, _ = request_json("GET", "/omni/thinking", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to read the thinking state.")
    print(f"Default thinking: {format_bool(payload.get('default_enabled'))}")
    return 0


def print_thinking_set(value: str) -> int:
    ensure_service_running()
    enabled = value == "on"
    _status, payload, _ = request_json("POST", "/omni/thinking/select", payload={"enabled": enabled}, timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to update the thinking state.")
    print(f"Default thinking set to: {format_bool(payload.get('default_enabled'))}")
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


def current_runtime_state() -> dict[str, Any]:
    _status, payload, _ = request_json("GET", "/omni/state", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to read the current runtime state.")
    return payload


def stream_chat(payload: dict[str, Any]) -> int:
    request = urllib.request.Request(
        url=f"{service_base_url()}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "text/event-stream, application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    print("Response")
    final_payload: dict[str, Any] | None = None
    prefix_buffer = ""
    prefix_flushed = False
    try:
        with urllib.request.urlopen(request, timeout=3600.0) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            if prefix_flushed:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            else:
                                prefix_buffer += content
                                out, prefix_flushed = flush_stream_prefix(prefix_buffer)
                                if prefix_flushed and out:
                                    sys.stdout.write(out)
                                    sys.stdout.flush()
                if isinstance(event, dict) and "usage" in event:
                    final_payload = event
            if not prefix_flushed and prefix_buffer:
                sys.stdout.write(prefix_buffer)
                sys.stdout.flush()
            print()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Streaming inference failed with status {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Streaming inference failed: {exc}") from exc

    if final_payload:
        print_performance(final_payload)
    return 0


def chat(args: argparse.Namespace) -> int:
    config_arg = getattr(args, "config", None)
    backend_extra_args = list(getattr(args, "backend_extra_args", []))
    selected_backend, backend = resolve_backend_spec_for_native_args(
        allow_auto=bool(getattr(args, "auto", False)),
        needs_native_args=bool(config_arg is not None or backend_extra_args),
    )
    profile = resolve_backend_profile_arg(config_arg, selected_backend)
    if profile and profile.backend_id and not selected_backend:
        selected_backend = profile.backend_id
        backend = local_backends().get(selected_backend)
    backend_extras = combine_backend_extra_args(
        backend=backend,
        command_name="chat",
        profile=profile,
        cli_tokens=backend_extra_args,
    ) if backend is not None else ParsedBackendExtraArgs()
    state = current_runtime_state()

    model_input = args.model
    model_path = resolve_model_reference(model_input) if model_input else None
    mmproj_input = args.mmproj
    mmproj_path = resolve_existing_path(mmproj_input, "mmproj file") if mmproj_input else None
    effective_ctx_size = args.ctx_size if args.ctx_size is not None else backend_extras.ctx_size
    if effective_ctx_size is not None and effective_ctx_size <= 0:
        raise SystemExit("--ctx-size must be a positive integer")

    if model_path is None and not state.get("model"):
        raise SystemExit(
            "No model is currently loaded.\n"
            "Run `omniinfer model load -m <model>` first, or pass a model directly to this chat command with -m/--model."
        )

    runtime_request_defaults = state.get("request_defaults") if isinstance(state.get("request_defaults"), dict) else {}
    effective_request_defaults = dict(runtime_request_defaults)
    effective_request_defaults = merge_backend_request_overrides(
        effective_request_defaults,
        backend_extras.request_overrides,
    )

    message_text = args.message if args.message is not None else backend_extras.message
    if not message_text:
        raise SystemExit("Please provide --message.")

    image_input = args.image if args.image is not None else backend_extras.image
    messages = [{"role": "user", "content": message_text}]
    if image_input:
        image_file = resolve_existing_path(image_input, "image file")
        mime = "image/png"
        if image_file.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif image_file.suffix.lower() == ".webp":
            mime = "image/webp"
        image_b64 = base64.b64encode(image_file.read_bytes()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ]

    payload: dict[str, Any] = dict(effective_request_defaults)
    payload["messages"] = messages
    payload["temperature"] = args.temperature if args.temperature is not None else payload.get("temperature", 0.2)
    payload["max_tokens"] = args.max_tokens if args.max_tokens is not None else payload.get("max_tokens", 128)
    payload["stream"] = args.stream if args.stream is not None else payload.get("stream", True)
    payload["think"] = (
        parse_boolish(args.think)
        if args.think is not None
        else payload.get("think", False)
    )
    if model_path:
        payload["model"] = str(model_path)
    if mmproj_path:
        payload["mmproj"] = str(mmproj_path)
    if effective_ctx_size is not None:
        payload["ctx_size"] = effective_ctx_size
    if selected_backend:
        payload["backend"] = selected_backend
    if profile is not None and backend is not None:
        load_extras = combine_backend_extra_args(
            backend=backend,
            command_name="load",
            profile=None,
            cli_tokens=list(profile.load_extra_args),
        )
        if load_extras.launch_args:
            payload["launch_args"] = load_extras.launch_args
        if load_extras.ctx_size is not None and args.ctx_size is None and "ctx_size" not in payload:
            payload["ctx_size"] = load_extras.ctx_size
    if backend_extras.request_overrides:
        payload["request_defaults"] = backend_extras.request_overrides

    effective_stream = bool(payload.get("stream"))
    if effective_stream:
        payload["stream_options"] = {"include_usage": True}
        result = stream_chat(payload)
        save_selected_backend_name(selected_backend)
        return result

    _status, response, _ = request_json("POST", "/v1/chat/completions", payload=payload, timeout=600.0)
    if not isinstance(response, dict):
        raise SystemExit("Inference response has an unexpected format.")
    save_selected_backend_name(selected_backend)
    print_chat_output(response)
    return 0


def backend_stop() -> int:
    ensure_service_running()
    request_json("POST", "/omni/backend/stop", timeout=30.0)
    print("Current backend process stopped")
    return 0


def shutdown_service() -> int:
    if not is_service_running():
        print("OmniInfer service is not running")
        return 0
    request_json("POST", "/omni/shutdown", timeout=30.0)
    print("OmniInfer service stopped")
    return 0


def serve_foreground() -> int:
    config = get_service_config()
    command = gateway_launch_command(
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 9000)),
        startup_timeout=int(config.get("startup_timeout", 60)),
        window_mode=str(config.get("window_mode", "hidden")),
        default_thinking=str(config.get("default_thinking", "off")),
        default_backend=str(config.get("default_backend", "")),
    )
    os.execv(command[0], command)
    return 0


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

    top_level = ["backend", "select", "status", "model", "thinking", "chat", "shutdown", "serve", "completion"]
    backend_sub = ["list", "select", "stop"]
    model_sub = ["list", "load"]
    thinking_sub = ["show", "set"]

    suggestions: list[str] = []
    if cword <= 0:
        suggestions = top_level
    elif words and words[0] == "backend":
        if cword == 1:
            suggestions = backend_sub
        elif len(words) > 1 and words[1] == "select":
            suggestions = complete_backend_name(current)
    elif words and words[0] == "select":
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
        epilog=HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=None, help="Gateway port (overrides config file)")
    sub = parser.add_subparsers(dest="command")

    backend = sub.add_parser("backend", help="Backend commands")
    backend_sub = backend.add_subparsers(dest="backend_command")
    backend_list = backend_sub.add_parser("list", help="List backends available on this system")
    backend_list.add_argument("--scope", choices=("installed", "compatible", "all"), default="all", help="Filter backends by scope (default: all)")
    backend_select = backend_sub.add_parser("select", help="Select a backend")
    backend_select.add_argument("backend_name", help="Backend name")
    backend_sub.add_parser("stop", help="Stop the current backend process")

    select_alias = sub.add_parser("select", help="Select a backend")
    select_alias.add_argument("backend_name", help="Backend name")

    sub.add_parser("status", help="Show current status")

    model = sub.add_parser("model", help="Model commands")
    model_sub = model.add_subparsers(dest="model_command")
    model_list = model_sub.add_parser("list", help="List supported models")
    model_list.add_argument("--system", choices=SYSTEM_CHOICES, default=detect_system_name(), help="Target system, defaults to the current system")
    model_list.add_argument("--all-backends", action="store_true", help="Show the raw backend-grouped view")
    model_load = model_sub.add_parser("load", help="Load a model")
    model_load.add_argument("-m", "--model", help="Path to the model file or model directory")
    model_load.add_argument("-mm", "--mmproj", help="Path to the mmproj file")
    model_load.add_argument("--ctx-size", type=int, help="Optional llama.cpp context length override for this load")
    model_load.add_argument(
        "--config",
        nargs="?",
        const="",
        help="Use the selected backend config JSON, or pass an explicit config path",
    )
    model_load.add_argument("--auto", action="store_true", help="Let OmniInfer auto-select a backend for this command instead of using the saved selection")
    model_load.add_argument("--verbose", action="store_true", help="Show full backend log output during loading")

    thinking = sub.add_parser("thinking", help="Default thinking controls")
    thinking_sub = thinking.add_subparsers(dest="thinking_command")
    thinking_sub.add_parser("show", help="Show the default thinking state")
    thinking_set = thinking_sub.add_parser("set", help="Set the default thinking state")
    thinking_set.add_argument("value", choices=("on", "off"), help="on or off")

    chat_cmd = sub.add_parser("chat", help="Run model inference")
    chat_cmd.add_argument("--message", help="User message")
    stream_group = chat_cmd.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=None, help="Stream tokens to stdout")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false", help="Wait for the final response before printing")
    chat_cmd.add_argument("-m", "--model", help="Model path or model directory for this request")
    chat_cmd.add_argument("-mm", "--mmproj", help="mmproj path for this request")
    chat_cmd.add_argument("--ctx-size", type=int, help="Optional llama.cpp context length override for this request")
    chat_cmd.add_argument("--image", help="Optional image path for multimodal inference")
    chat_cmd.add_argument("--think", choices=("on", "off"), help="Thinking mode for this request")
    chat_cmd.add_argument("--temperature", type=float, help="Sampling temperature")
    chat_cmd.add_argument("--max-tokens", type=int, help="Maximum output tokens for this request")
    chat_cmd.add_argument(
        "--config",
        nargs="?",
        const="",
        help="Use the selected backend config JSON, or pass an explicit config path",
    )
    chat_cmd.add_argument("--auto", action="store_true", help="Let OmniInfer auto-select a backend for this command instead of using the saved selection")

    sub.add_parser("shutdown", help="Stop the OmniInfer service")
    sub.add_parser("serve", help="Start the OmniInfer service in the foreground")

    completion = sub.add_parser("completion", help="Print shell completion")
    completion.add_argument("shell", choices=("bash",), help="Currently supported shell: bash")

    return parser


def main(argv: list[str] | None = None) -> int:
    from service_core.logger import setup_logging

    setup_logging(level="DEBUG", console=False, log_to_file=True)
    cli_logger = logging.getLogger("cli")
    cli_logger.debug("CLI invoked: %s", " ".join(sys.argv))

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "__complete":
        return handle_hidden_completion(argv[1:])

    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    unknown_args = [item for item in unknown_args if item != "--"]
    setattr(args, "backend_extra_args", unknown_args)

    global _cli_port_override
    if args.port is not None:
        _cli_port_override = args.port

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "backend":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        if args.backend_command == "list":
            return print_backend_list(scope=args.scope)
        if args.backend_command == "select":
            return select_backend(args.backend_name)
        if args.backend_command == "stop":
            return backend_stop()
        parser.error("backend requires a subcommand: list / select / stop")

    if args.command == "select":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return select_backend(args.backend_name)
    if args.command == "status":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return print_status()
    if args.command == "model":
        if args.model_command == "list":
            if unknown_args:
                parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
            return print_model_list(system_name=args.system, best=not args.all_backends)
        if args.model_command == "load":
            return print_model_load(args)
        parser.error("model requires a subcommand: list / load")
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
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return serve_foreground()
    if args.command == "completion":
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        return print_completion(args.shell)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
