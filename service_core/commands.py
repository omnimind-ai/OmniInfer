from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from service_core.backend_cli_args import (
    ParsedBackendExtraArgs,
    parse_backend_chat_extra_args,
    parse_backend_load_extra_args,
)
from service_core.advisor import (
    fit_model as advisor_fit_model,
    inspect_model as advisor_inspect_model,
    recommend_models as advisor_recommend_models,
    system_snapshot as advisor_system_snapshot,
)
from service_core.backend_configs import (
    BackendProfile,
    ensure_backend_profile_template,
    load_backend_profile,
    profile_path_for_backend,
)
from service_core.local_state import (
    load_selected_model,
    load_tui_show_reasoning,
    local_dir,
    local_logs_dir,
    save_default_thinking,
    save_selected_backend,
    save_selected_model,
    save_tui_show_reasoning,
)
from service_core.runtime import RuntimeManager
from service_core.service import APP_ROOT, REPO_ROOT, load_app_config


CLI_LOG_DIR = local_logs_dir(Path(APP_ROOT))
CLI_LOG_FILE = CLI_LOG_DIR / "gateway.log"
SYSTEM_CHOICES = ("linux", "mac", "windows")
MODEL_SUFFIXES = (".gguf",)
DEFAULT_CHAT_MAX_TOKENS = 2048

_port_override: int | None = None


def _is_windows() -> bool:
    return os.name == "nt"


@dataclass(frozen=True)
class BackendSelectResult:
    backend: str
    models_dir: str | None
    profile_path: Path
    profile_created: bool


@dataclass(frozen=True)
class AutoSelectResult:
    backend: str
    auto_selected: bool


@dataclass(frozen=True)
class LocalModel:
    path: Path
    label: str


@dataclass(frozen=True)
class ModelLoadOptions:
    model: str
    mmproj: str | None = None
    ctx_size: int | None = None
    config: str | None = None
    backend_extra_args: list[str] | None = None
    verbose: bool = False


@dataclass(frozen=True)
class ChatOptions:
    message: str
    image: str | None = None
    stream: bool | None = None
    think: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class BackendBuildOptions:
    backend: str
    prebuilt: bool = False
    from_source: bool = False


@dataclass(frozen=True)
class BackendBuildResult:
    backend: str
    command: list[str]
    script_path: Path
    binary_path: Path | None
    returncode: int


@dataclass(frozen=True)
class ChatStreamChunk:
    text: str = ""
    reasoning_text: str = ""
    final_payload: dict[str, Any] | None = None


ProgressCallback = Callable[[dict[str, Any]], None]


def set_port_override(port: int | None) -> None:
    global _port_override
    _port_override = port


def detect_system_name() -> str:
    import platform

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


def get_service_config() -> dict[str, Any]:
    config = load_app_config(Path(APP_ROOT))
    if _port_override is not None:
        config["port"] = _port_override
    return config


def service_base_url() -> str:
    config = get_service_config()
    host = str(config["host"]).strip()
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{int(config['port'])}"


def try_parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except json.JSONDecodeError:
        return None


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


def is_service_running() -> bool:
    try:
        status, payload, _ = request_json("GET", "/health", timeout=2.0, allow_http_error=True)
    except SystemExit:
        return False
    return status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"


def _local_runtime_manager() -> RuntimeManager:
    config = get_service_config()
    return RuntimeManager(
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


def local_backends() -> dict[str, Any]:
    manager = _local_runtime_manager()
    return manager.backends


def local_backend_ids() -> list[str]:
    return list(local_backends().keys())


def advisor_system() -> dict[str, Any]:
    manager = _local_runtime_manager()
    return advisor_system_snapshot(manager.platform, manager.backends)


def advisor_inspect(model: str, *, mmproj: str | None = None) -> dict[str, Any]:
    return advisor_inspect_model(model, mmproj=mmproj)


def advisor_fit(
    model: str,
    *,
    mmproj: str | None = None,
    ctx_size: int | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    manager = _local_runtime_manager()
    if backend is not None and backend not in manager.backends:
        available = ", ".join(sorted(manager.backends))
        raise SystemExit(f"Unsupported backend: {backend}\nAvailable backends: {available}")
    return advisor_fit_model(
        model,
        platform_obj=manager.platform,
        backends=manager.backends,
        mmproj=mmproj,
        ctx_size=ctx_size,
        backend_filter=backend,
    )


def advisor_recommend(
    *,
    task: str | None = None,
    limit: int = 5,
    ctx_size: int | None = None,
) -> dict[str, Any]:
    manager = _local_runtime_manager()
    return advisor_recommend_models(
        platform_obj=manager.platform,
        backends=manager.backends,
        task=task,
        limit=limit,
        ctx_size=ctx_size,
    )


def is_backend_build_supported() -> bool:
    """Return true when this process is running from a source checkout with build scripts."""
    if getattr(sys, "frozen", False):
        return False
    scripts_root = Path(REPO_ROOT) / "scripts" / "platforms"
    return scripts_root.is_dir()


def backend_build_platform_dir(system_name: str | None = None) -> str:
    system = system_name or detect_system_name()
    if system == "mac":
        return "macos"
    if system in {"linux", "windows"}:
        return system
    raise SystemExit(f"backend builds are not supported on this platform: {system}")


def backend_build_script_path(backend: str, *, system_name: str | None = None) -> Path:
    platform_dir = backend_build_platform_dir(system_name)
    extension = "ps1" if platform_dir == "windows" else "sh"
    return Path(REPO_ROOT) / "scripts" / "platforms" / platform_dir / backend / f"build.{extension}"


def backend_build_command(options: BackendBuildOptions) -> tuple[list[str], Path]:
    if not is_backend_build_supported():
        raise SystemExit("Backend builds are only available from a source checkout, not packaged releases.")

    backends = local_backends()
    available = sorted(backends)
    if options.backend not in backends:
        raise SystemExit(f"Unsupported backend: {options.backend}\nAvailable backends: {', '.join(available)}")

    script_path = backend_build_script_path(options.backend)
    if not script_path.is_file():
        raise SystemExit(
            f"No build script found for backend: {options.backend}\n"
            f"Expected: {script_path}"
        )

    if _is_windows():
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-BuildType",
            "Release",
        ]
        if options.prebuilt:
            command.append("-Prebuilt")
    else:
        command = ["bash", str(script_path), "--build-type", "Release"]
        if options.prebuilt:
            command.append("--prebuilt")
        if options.from_source:
            command.append("--from-source")
    return command, script_path


def build_backend(options: BackendBuildOptions) -> BackendBuildResult:
    command, script_path = backend_build_command(options)
    backends = local_backends()
    backend = backends[options.backend]
    binary_path = Path(backend.launcher_path) if backend.launcher_path else None

    if is_service_running():
        try:
            state = current_runtime_state()
            if state.get("backend") == options.backend and state.get("backend_ready"):
                request_json("POST", "/omni/backend/stop", timeout=30.0)
        except SystemExit:
            pass

    result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise SystemExit(f"Backend build failed for {options.backend} with exit code {result.returncode}.")

    refreshed = local_backends().get(options.backend)
    if refreshed is None:
        raise SystemExit(f"Backend disappeared after build: {options.backend}")
    refreshed_binary = Path(refreshed.launcher_path) if refreshed.launcher_path else None
    if not refreshed.binary_exists:
        expected = str(refreshed_binary) if refreshed_binary else "backend launcher"
        raise SystemExit(f"Backend build completed, but the expected binary was not found: {expected}")

    return BackendBuildResult(
        backend=options.backend,
        command=command,
        script_path=script_path,
        binary_path=refreshed_binary,
        returncode=result.returncode,
    )


def resolve_gateway_binary() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None

    current_exe = Path(sys.executable).resolve()
    candidates = [
        current_exe.with_name("omniinfer.exe"),
        current_exe.with_name("omniinfer"),
        current_exe.with_name("omniinfer-cli.exe"),
        current_exe.with_name("omniinfer-cli"),
        current_exe,
    ]
    for candidate in candidates:
        if candidate.is_file():
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
            raise SystemExit("unable to locate the packaged OmniInfer CLI binary")
        command = [str(gateway_binary), "serve"]
    else:
        command = [sys.executable, str(Path(REPO_ROOT) / "omniinfer.py"), "serve"]

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


def start_service_background() -> None:
    config = get_service_config()
    command = gateway_launch_command(
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 9000)),
        startup_timeout=int(config.get("startup_timeout", 60)),
        window_mode=str(config.get("window_mode", "hidden")),
        default_thinking=str(config.get("default_thinking", "off")),
        default_backend=str(config.get("default_backend", "")),
    )

    logging.getLogger("cli").info("Starting gateway service in background: %s", " ".join(command))
    CLI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = CLI_LOG_FILE.open("a", encoding="utf-8")
    if _is_windows():
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        )
        subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
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


def wait_for_service_ready(timeout_s: int, progress: ProgressCallback | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_service_running():
            if progress is not None:
                progress({"type": "status", "message": "OmniInfer gateway ready."})
            return
        time.sleep(0.5)
    tail = ""
    if CLI_LOG_FILE.is_file():
        tail = CLI_LOG_FILE.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise SystemExit(f"failed to start local OmniInfer service in time\n{tail}")


def ensure_service_running(progress: ProgressCallback | None = None) -> None:
    if is_service_running():
        if progress is not None:
            progress({"type": "status", "message": "Connected to OmniInfer gateway."})
        return
    if progress is not None:
        progress({"type": "status", "message": "Starting OmniInfer gateway..."})
    start_service_background()
    if progress is not None:
        progress({"type": "status", "message": "Waiting for OmniInfer gateway..."})
    wait_for_service_ready(
        timeout_s=max(int(get_service_config().get("startup_timeout", 60)), 10),
        progress=progress,
    )


def shutdown_service(wait_timeout_s: float = 10.0) -> bool:
    if not is_service_running():
        return False
    request_json("POST", "/omni/shutdown", timeout=30.0)
    deadline = time.time() + max(wait_timeout_s, 0.0)
    while time.time() < deadline:
        if not is_service_running():
            return True
        time.sleep(0.2)
    return True


def current_runtime_state() -> dict[str, Any]:
    _status, payload, _ = request_json("GET", "/omni/state", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to read the current runtime state.")
    return payload


def current_backend_props() -> dict[str, Any]:
    _status, payload, _ = request_json("GET", "/omni/backend/props", timeout=10.0)
    return payload if isinstance(payload, dict) else {}


def get_default_thinking() -> bool:
    ensure_service_running()
    _status, payload, _ = request_json("GET", "/omni/thinking", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to read the thinking state.")
    return bool(payload.get("default_enabled"))


def set_default_thinking(enabled: bool) -> bool:
    ensure_service_running()
    _status, payload, _ = request_json(
        "POST",
        "/omni/thinking/select",
        payload={"enabled": enabled},
        timeout=10.0,
    )
    if not isinstance(payload, dict):
        raise SystemExit("Unable to update the thinking state.")
    saved_value = bool(payload.get("default_enabled"))
    save_default_thinking(saved_value, Path(APP_ROOT))
    return saved_value


def get_tui_show_reasoning() -> bool:
    return load_tui_show_reasoning(Path(APP_ROOT))


def set_tui_show_reasoning(enabled: bool) -> bool:
    save_tui_show_reasoning(enabled, Path(APP_ROOT))
    return enabled


def selected_backend() -> str | None:
    ensure_service_running()
    payload = current_runtime_state()
    if payload.get("backend"):
        return str(payload["backend"])
    return None


def list_backends(scope: str = "compatible") -> dict[str, Any]:
    ensure_service_running()
    _status, payload, _ = request_json("GET", f"/omni/backends?scope={scope}", timeout=10.0)
    if not isinstance(payload, dict):
        raise SystemExit("Unable to read backend list.")
    data = payload.get("data")
    if not isinstance(data, list):
        payload["data"] = []
    return payload


def select_backend(name: str) -> BackendSelectResult:
    ensure_service_running()
    available_backends = local_backends()
    available = list(available_backends.keys())
    if name not in available_backends:
        raise SystemExit(f"Unsupported backend: {name}\nAvailable backends: {', '.join(available)}")
    _status, payload, _ = request_json("POST", "/omni/backend/select", payload={"backend": name}, timeout=30.0)
    save_selected_backend(name, Path(APP_ROOT))
    models_dir = payload.get("models_dir") if isinstance(payload, dict) else None
    profile_path, created = ensure_backend_profile_template(available_backends[name])
    return BackendSelectResult(
        backend=name,
        models_dir=str(models_dir) if models_dir else None,
        profile_path=profile_path,
        profile_created=created,
    )


def require_selected_backend() -> AutoSelectResult:
    current = selected_backend()
    if current:
        return AutoSelectResult(backend=current, auto_selected=False)

    payload = list_backends(scope="installed")
    recommended = payload.get("recommended")
    if not recommended:
        raise SystemExit(
            "No installed backend found.\n"
            "Build or install a backend first, then run:\n"
            "  omniinfer backend list\n"
            "  omniinfer backend select <backend>"
        )
    backend_id = str(recommended)
    request_json("POST", "/omni/backend/select", payload={"backend": backend_id}, timeout=30.0)
    save_selected_backend(backend_id, Path(APP_ROOT))
    return AutoSelectResult(backend=backend_id, auto_selected=True)


def resolve_model_reference(path_text: str, *, backend: Any | None = None) -> Path | str:
    if backend is not None and getattr(backend, "family", None) == "vllm":
        text = normalize_path_text(path_text)
        if not text:
            raise SystemExit("Model reference must not be empty.")
        path = absolute_path_from_text(text)
        if path.exists():
            return path
        return text

    path = _absolute_existing_path(path_text)
    if not path.exists():
        raise SystemExit(f"Model path does not exist: {path}")
    return path


def resolve_existing_path(path_text: str, label: str) -> Path:
    path = _absolute_existing_path(path_text)
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def normalize_path_text(path_text: str) -> str:
    text = path_text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def absolute_path_from_text(path_text: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(normalize_path_text(path_text))))


def _absolute_existing_path(path_text: str) -> Path:
    return absolute_path_from_text(path_text)


def resolve_backend_profile_arg(path_text: str | None, selected_backend_id: str | None) -> BackendProfile | None:
    if path_text is None:
        return None
    target_path = path_text
    if path_text == "":
        if not selected_backend_id:
            raise SystemExit(
                "Using --config without a path requires a selected backend. "
                "Run `omniinfer backend select <backend>` first."
            )
        target_path = str(profile_path_for_backend(selected_backend_id))
    try:
        profile = load_backend_profile(target_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if profile.backend_id and selected_backend_id and profile.backend_id != selected_backend_id:
        raise SystemExit(
            f"Backend config {profile.path} belongs to {profile.backend_id}, "
            f"but the current selected backend is {selected_backend_id}."
        )
    return profile


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


def build_model_load_payload(options: ModelLoadOptions) -> tuple[dict[str, Any], AutoSelectResult]:
    backend_selection = require_selected_backend()
    selected_backend_id = backend_selection.backend
    backends = local_backends()
    backend = backends.get(selected_backend_id)
    if backend is None:
        raise SystemExit(f"Selected backend is no longer available locally: {selected_backend_id}")

    profile = resolve_backend_profile_arg(options.config, selected_backend_id)
    if profile and profile.backend_id and profile.backend_id != selected_backend_id:
        selected_backend_id = profile.backend_id
        backend = backends.get(selected_backend_id)
    backend_extras = combine_backend_extra_args(
        backend=backend,
        command_name="load",
        profile=profile,
        cli_tokens=list(options.backend_extra_args or []),
    )

    model_ref = resolve_model_reference(options.model, backend=backend)
    mmproj_file = resolve_existing_path(options.mmproj, "mmproj file") if options.mmproj else None
    effective_ctx_size = options.ctx_size if options.ctx_size is not None else backend_extras.ctx_size
    if effective_ctx_size is not None and effective_ctx_size <= 0:
        raise SystemExit("--ctx-size must be a positive integer")

    payload: dict[str, Any] = {"model": str(model_ref)}
    if mmproj_file:
        payload["mmproj"] = str(mmproj_file)
    if effective_ctx_size is not None:
        payload["ctx_size"] = effective_ctx_size
    if selected_backend_id:
        payload["backend"] = selected_backend_id
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
    return payload, backend_selection


def load_model(
    options: ModelLoadOptions,
    *,
    progress: ProgressCallback | None = None,
    timeout: float = 600.0,
) -> tuple[dict[str, Any], AutoSelectResult]:
    ensure_service_running(progress=progress)
    if progress is not None:
        progress({"type": "status", "message": "Preparing model request..."})
    payload, backend_selection = build_model_load_payload(options)
    if progress is not None:
        progress({"type": "status", "message": "Sending model load request..."})
    response = stream_model_load(payload, progress=progress, timeout=timeout)
    if not isinstance(response, dict):
        raise SystemExit("Failed to load the model.")
    selected_backend = response.get("selected_backend")
    if isinstance(selected_backend, str) and selected_backend.strip():
        save_selected_backend(selected_backend, Path(APP_ROOT))
    save_selected_model(
        str(payload["model"]),
        Path(APP_ROOT),
        mmproj=str(payload["mmproj"]) if payload.get("mmproj") else None,
        ctx_size=response.get("selected_ctx_size") if isinstance(response.get("selected_ctx_size"), int) else None,
    )
    return response, backend_selection


def remembered_model_load_options() -> ModelLoadOptions | None:
    remembered = load_selected_model(Path(APP_ROOT))
    if not remembered:
        return None
    model = Path(remembered["model"]).expanduser()
    if not model.exists():
        return None
    mmproj = remembered.get("mmproj")
    if mmproj and not Path(mmproj).expanduser().exists():
        mmproj = None
    return ModelLoadOptions(
        model=str(model),
        mmproj=mmproj,
        ctx_size=remembered.get("ctx_size") if isinstance(remembered.get("ctx_size"), int) else None,
    )


def stream_model_load(
    payload: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
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

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
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
                if progress is not None:
                    progress(event)
                event_type = event.get("type", "")
                if event_type == "done":
                    result = event
                elif event_type == "error":
                    raise SystemExit(event.get("message", "model loading failed"))
            return result
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        parsed = try_parse_json(raw.encode("utf-8"))
        if isinstance(parsed, dict):
            error = parsed.get("error", {})
            message = error.get("message", raw) if isinstance(error, dict) else raw
        else:
            message = raw
        raise SystemExit(f"Model loading failed (HTTP {exc.code}): {message}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Unable to reach local OmniInfer service: {exc}") from exc


def build_chat_payload(options: ChatOptions) -> dict[str, Any]:
    ensure_service_running()
    state = current_runtime_state()
    if not state.get("model"):
        raise SystemExit(
            "No model is currently loaded.\n"
            "Run `omniinfer load -m <model>` first."
        )

    message_text = options.message
    if not message_text:
        raise SystemExit("Please provide a message, for example: omniinfer chat \"Hello\".")

    messages: list[dict[str, Any]] = [{"role": "user", "content": message_text}]
    if options.image:
        image_file = resolve_existing_path(options.image, "image file")
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

    runtime_request_defaults = state.get("request_defaults") if isinstance(state.get("request_defaults"), dict) else {}
    payload: dict[str, Any] = dict(runtime_request_defaults)
    payload["messages"] = messages
    payload["temperature"] = options.temperature if options.temperature is not None else payload.get("temperature", 0.2)
    payload["max_tokens"] = options.max_tokens if options.max_tokens is not None else payload.get("max_tokens", DEFAULT_CHAT_MAX_TOKENS)
    payload["stream"] = options.stream if options.stream is not None else payload.get("stream", True)
    if options.think is not None:
        payload["think"] = parse_boolish(options.think)
    return payload


def request_chat(options: ChatOptions) -> dict[str, Any]:
    payload = build_chat_payload(options)
    payload["stream"] = False
    return request_chat_payload(payload)


def request_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _status, response, _ = request_json("POST", "/v1/chat/completions", payload=payload, timeout=600.0)
    if not isinstance(response, dict):
        raise SystemExit("Inference response has an unexpected format.")
    return response


def iter_chat_stream(options: ChatOptions) -> Iterator[ChatStreamChunk]:
    payload = build_chat_payload(options)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    yield from iter_chat_stream_payload(payload)


def iter_chat_stream_payload(payload: dict[str, Any]) -> Iterator[ChatStreamChunk]:
    request = urllib.request.Request(
        url=f"{service_base_url()}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "text/event-stream, application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
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
                            yield ChatStreamChunk(text=content)
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str):
                            yield ChatStreamChunk(reasoning_text=reasoning)
                if isinstance(event, dict) and "usage" in event:
                    yield ChatStreamChunk(final_payload=event)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Streaming inference failed with status {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Streaming inference failed: {exc}") from exc


def discover_local_models() -> list[LocalModel]:
    backends = local_backends()
    roots: list[Path] = []
    for backend in backends.values():
        models_dir = getattr(backend, "models_dir", None)
        if not models_dir:
            continue
        path = Path(str(models_dir)).expanduser().resolve()
        if _is_under_local_dir(path):
            roots.append(path)
    return discover_models_in_roots(roots)


def managed_models_dir() -> Path:
    return local_dir(Path(APP_ROOT)) / "models"


def detect_model_files_in_directory(directory: Path) -> list[LocalModel]:
    root = Path(os.path.abspath(os.path.expanduser(str(directory))))
    if not root.is_dir():
        return []
    models: list[LocalModel] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in MODEL_SUFFIXES:
            continue
        if _is_mmproj_file(candidate):
            continue
        models.append(LocalModel(path=candidate, label=_model_label(root, candidate)))
    return sorted(models, key=lambda item: (_model_file_rank(item.path), item.label.lower()))


def infer_managed_model_root(model_path: Path, search_root: Path) -> Path:
    source = Path(os.path.abspath(os.path.expanduser(str(model_path)))).resolve()
    root = Path(os.path.abspath(os.path.expanduser(str(search_root)))).resolve()
    parent = source.parent
    try:
        parent.relative_to(root)
    except ValueError:
        return parent

    for candidate in [parent, *parent.parents]:
        if candidate == root.parent:
            break
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        name = _safe_path_component(candidate.name)
        if _is_useful_model_directory_name(name):
            return candidate
    return parent


def link_model_into_managed_models(
    model_path: Path,
    *,
    model_root: Path | None = None,
    preserve_relative_path: bool = True,
) -> Path:
    requested = Path(os.path.abspath(os.path.expanduser(str(model_path))))
    source = requested.resolve() if requested.is_symlink() else requested
    if not source.exists():
        raise FileNotFoundError(f"model path does not exist: {source}")
    if not source.is_file():
        return source

    target_root = managed_models_dir().resolve()
    if requested.parent != target_root:
        try:
            requested.relative_to(target_root)
            return requested
        except ValueError:
            pass

    target_root.mkdir(parents=True, exist_ok=True)
    target = _managed_model_target(
        source,
        target_root,
        model_root=model_root,
        preserve_relative_path=preserve_relative_path,
    )
    if _link_points_to(target, source):
        return target
    if target.exists() or target.is_symlink():
        index = 2
        while True:
            candidate = _managed_model_target(
                source,
                target_root,
                model_root=model_root,
                preserve_relative_path=preserve_relative_path,
                root_suffix=f"-{index}",
            )
            if _link_points_to(candidate, source):
                return candidate
            if not candidate.exists() and not candidate.is_symlink():
                target = candidate
                break
            index += 1

    _link_model_file(source, target)
    return target


def discover_models_in_roots(roots: list[Path]) -> list[LocalModel]:
    seen: set[Path] = set()
    models: list[LocalModel] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in MODEL_SUFFIXES:
                continue
            if _is_mmproj_file(candidate):
                continue
            if _is_flat_managed_model(root, candidate):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            models.append(LocalModel(path=candidate, label=_model_label(root, candidate)))
    return sorted(models, key=lambda item: item.label.lower())


def _is_under_local_dir(path: Path) -> bool:
    root = local_dir(Path(APP_ROOT)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _model_label(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _managed_model_directory_name(source: Path) -> str:
    parent_name = _safe_path_component(source.parent.name)
    if parent_name:
        return parent_name
    stem_name = _safe_path_component(source.stem)
    return stem_name or "model"


def _managed_model_target(
    source: Path,
    target_root: Path,
    *,
    model_root: Path | None,
    preserve_relative_path: bool,
    root_suffix: str = "",
) -> Path:
    if model_root is None:
        return target_root / f"{_managed_model_directory_name(source)}{root_suffix}" / source.name

    root = Path(os.path.abspath(os.path.expanduser(str(model_root))))
    root_name = _safe_path_component(root.name) or _managed_model_directory_name(source)
    if not preserve_relative_path:
        return target_root / f"{root_name}{root_suffix}" / source.name
    try:
        relative_parent = source.parent.resolve().relative_to(root.resolve())
    except ValueError:
        relative_parent = Path()
    return target_root / f"{root_name}{root_suffix}" / relative_parent / source.name


def _is_useful_model_directory_name(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    generic_names = {
        "model",
        "models",
        "gguf",
        "ggml",
        "weights",
        "weight",
        "checkpoint",
        "checkpoints",
        "snapshot",
        "snapshots",
        "resolve",
        "main",
    }
    if lowered in generic_names:
        return False
    if re.fullmatch(r"[0-9a-f]{8,}", lowered):
        return False
    return True


def _is_mmproj_file(path: Path) -> bool:
    return "mmproj" in path.name.lower()


def _model_file_rank(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    quant_order = ["q4_k_m", "q4_0", "q5_k_m", "q6_k", "q8_0", "f16", "q3_k_m", "q2_k"]
    quant_rank = next((index for index, quant in enumerate(quant_order) if quant in name), len(quant_order))
    return (len(path.parts), quant_rank, name)


def _safe_path_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return text.strip(" .-_")


def _is_flat_managed_model(root: Path, candidate: Path) -> bool:
    try:
        if root.resolve() != managed_models_dir().resolve():
            return False
    except OSError:
        return False
    return candidate.parent.resolve() == root.resolve()


def _link_points_to(path: Path, source: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    try:
        if path.exists() and source.exists() and path.samefile(source):
            return True
    except OSError:
        pass
    try:
        return path.resolve(strict=True) == source
    except FileNotFoundError:
        return False


def _is_windows_symlink_privilege_error(exc: OSError) -> bool:
    return _is_windows() and getattr(exc, "winerror", None) == 1314


def _create_directory_junction_for_model(source: Path, target: Path) -> None:
    if not _is_windows():
        raise OSError("directory junction fallback is only supported on Windows")

    junction = target.parent
    if junction.exists() and not junction.is_symlink() and not any(junction.iterdir()):
        junction.rmdir()
    if junction.exists() or junction.is_symlink():
        raise OSError(f"cannot create junction because target directory already exists: {junction}")

    junction.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(source.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "mklink failed").strip()
        raise OSError(f"could not create junction {junction} -> {source.parent}: {message}")
    if not target.exists():
        raise OSError(f"created junction {junction}, but linked model is not visible at {target}")


def _link_model_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target)
        return
    except OSError as symlink_exc:
        if not _is_windows_symlink_privilege_error(symlink_exc):
            raise

    try:
        os.link(source, target)
        return
    except OSError:
        _create_directory_junction_for_model(source, target)
