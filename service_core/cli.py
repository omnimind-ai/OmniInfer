from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from service_core.runtime import RuntimeManager
from service_core.service import APP_ROOT, REPO_ROOT, load_app_config


CLI_STATE_DIR = Path.home() / ".config" / "omniinfer"
CLI_STATE_FILE = CLI_STATE_DIR / "cli_state.json"
CLI_LOG_DIR = Path.home() / ".cache" / "omniinfer"
CLI_LOG_FILE = CLI_LOG_DIR / "gateway.log"
DEFAULT_IMAGE_PATH = Path(REPO_ROOT) / "tests" / "pictures" / "test1.png"
SYSTEM_CHOICES = ("linux", "mac", "windows")


HELP_TEXT = """\
OmniInfer CLI

Common commands:
  omniinfer backend list
  omniinfer select <backend>
  omniinfer status
  omniinfer model load -m /path/to/model.gguf
  omniinfer chat --message "Introduce yourself in one sentence."
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
  model load                -> choose and load a model
  thinking show/set         -> show or change the default think setting
  chat                      -> run text or multimodal inference with streaming enabled by default
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
    return load_app_config(Path(APP_ROOT))


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
    return [item["id"] for item in manager.list_backends()]


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


def print_backend_list() -> int:
    ensure_service_running()
    state = load_cli_state()
    saved_backend = state.get("selected_backend")
    _status, payload, _ = request_json("GET", "/omni/backends", timeout=10.0)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No backends are available on this system.")

    print("Available backends")
    for item in rows:
        backend_id = str(item.get("id", ""))
        marker = "* " if backend_id == saved_backend else "  "
        binary_exists = "yes" if item.get("binary_exists") else "no"
        selected = "yes" if item.get("selected") else "no"
        capabilities = ", ".join(item.get("capabilities") or [])
        print(f"{marker}{backend_id}")
        print(f"    Selected in CLI: {'yes' if backend_id == saved_backend else 'no'}")
        print(f"    Active in service: {selected}")
        print(f"    Binary available: {binary_exists}")
        if capabilities:
            print(f"    Capabilities: {capabilities}")
        description = str(item.get('description', '')).strip()
        if description:
            print(f"    Description: {description}")
    return 0


def select_backend(name: str) -> int:
    ensure_service_running()
    available = local_backend_ids()
    if name not in available:
        raise SystemExit(f"Unsupported backend: {name}\nAvailable backends: {', '.join(available)}")
    _status, payload, _ = request_json("POST", "/omni/backend/select", payload={"backend": name}, timeout=30.0)
    state = load_cli_state()
    state["selected_backend"] = name
    save_cli_state(state)
    print(f"Selected backend: {name}")
    models_dir = payload.get("models_dir") if isinstance(payload, dict) else None
    if models_dir:
        print(f"Models directory: {models_dir}")
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


def print_model_load(args: argparse.Namespace) -> int:
    if not args.model:
        raise SystemExit("Please specify a model path with -m or --model.")
    model_file = Path(args.model).expanduser().resolve()
    if not model_file.is_file():
        raise SystemExit(f"Model file does not exist: {model_file}")
    mmproj_file = Path(args.mmproj).expanduser().resolve() if args.mmproj else None
    if mmproj_file and not mmproj_file.is_file():
        raise SystemExit(f"mmproj file does not exist: {mmproj_file}")
    if args.ctx_size is not None and args.ctx_size <= 0:
        raise SystemExit("--ctx-size must be a positive integer")

    selected_backend = require_selected_backend(allow_auto=args.auto)
    payload: dict[str, Any] = {"model": str(model_file)}
    if mmproj_file:
        payload["mmproj"] = str(mmproj_file)
    if args.ctx_size is not None:
        payload["ctx_size"] = args.ctx_size
    if selected_backend:
        payload["backend"] = selected_backend

    _status, response, _ = request_json("POST", "/omni/model/select", payload=payload, timeout=600.0)
    if not isinstance(response, dict):
        raise SystemExit("Failed to load the model.")
    print("Model loaded")
    print(f"Backend: {response.get('selected_backend') or '-'}")
    print(f"Model: {response.get('selected_model') or '-'}")
    print(f"mmproj: {response.get('selected_mmproj') or '-'}")
    print(f"ctx-size: {response.get('selected_ctx_size') or '-'}")
    return 0


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
    selected_backend = require_selected_backend(allow_auto=args.auto)
    state = current_runtime_state()

    model_path = Path(args.model).expanduser().resolve() if args.model else None
    if model_path and not model_path.is_file():
        raise SystemExit(f"Model file does not exist: {model_path}")
    mmproj_path = Path(args.mmproj).expanduser().resolve() if args.mmproj else None
    if mmproj_path and not mmproj_path.is_file():
        raise SystemExit(f"mmproj file does not exist: {mmproj_path}")
    if args.ctx_size is not None and args.ctx_size <= 0:
        raise SystemExit("--ctx-size must be a positive integer")

    if model_path is None and not state.get("model"):
        raise SystemExit(
            "No model is currently loaded.\n"
            "Run `omniinfer model load -m <model>` first, or pass a model directly to this chat command with -m/--model."
        )

    messages = [{"role": "user", "content": args.message}]
    if args.image:
        image_file = Path(args.image).expanduser().resolve()
        if not image_file.is_file():
            raise SystemExit(f"Image file does not exist: {image_file}")
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
                    {"type": "text", "text": args.message},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ]

    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens if args.max_tokens is not None else 128,
        "stream": bool(args.stream),
        "think": parse_boolish(args.think) if args.think is not None else False,
    }
    if model_path:
        payload["model"] = str(model_path)
    if mmproj_path:
        payload["mmproj"] = str(mmproj_path)
    if args.ctx_size is not None:
        payload["ctx_size"] = args.ctx_size
    if selected_backend:
        payload["backend"] = selected_backend

    if args.stream:
        payload["stream_options"] = {"include_usage": True}
        return stream_chat(payload)

    _status, response, _ = request_json("POST", "/v1/chat/completions", payload=payload, timeout=600.0)
    if not isinstance(response, dict):
        raise SystemExit("Inference response has an unexpected format.")
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
    sub = parser.add_subparsers(dest="command")

    backend = sub.add_parser("backend", help="Backend commands")
    backend_sub = backend.add_subparsers(dest="backend_command")
    backend_sub.add_parser("list", help="List backends available on this system")
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
    model_load.add_argument("-m", "--model", required=True, help="Path to the model file")
    model_load.add_argument("-mm", "--mmproj", help="Path to the mmproj file")
    model_load.add_argument("--ctx-size", type=int, help="Optional llama.cpp context length override for this load")
    model_load.add_argument("--auto", action="store_true", help="Let OmniInfer auto-select a backend for this command instead of using the saved selection")

    thinking = sub.add_parser("thinking", help="Default thinking controls")
    thinking_sub = thinking.add_subparsers(dest="thinking_command")
    thinking_sub.add_parser("show", help="Show the default thinking state")
    thinking_set = thinking_sub.add_parser("set", help="Set the default thinking state")
    thinking_set.add_argument("value", choices=("on", "off"), help="on or off")

    chat_cmd = sub.add_parser("chat", help="Run model inference")
    chat_cmd.add_argument("--message", required=True, help="User message")
    stream_group = chat_cmd.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=True, help="Stream tokens to stdout (default)")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false", help="Wait for the final response before printing")
    chat_cmd.add_argument("-m", "--model", help="Model path for this request")
    chat_cmd.add_argument("-mm", "--mmproj", help="mmproj path for this request")
    chat_cmd.add_argument("--ctx-size", type=int, help="Optional llama.cpp context length override for this request")
    chat_cmd.add_argument("--image", help="Optional image path for multimodal inference")
    chat_cmd.add_argument("--think", choices=("on", "off"), help="Thinking mode for this request")
    chat_cmd.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature, defaults to 0.2")
    chat_cmd.add_argument("--max-tokens", type=int, help="Maximum output tokens for this request")
    chat_cmd.add_argument("--auto", action="store_true", help="Let OmniInfer auto-select a backend for this command instead of using the saved selection")

    sub.add_parser("shutdown", help="Stop the OmniInfer service")
    sub.add_parser("serve", help="Start the OmniInfer service in the foreground")

    completion = sub.add_parser("completion", help="Print shell completion")
    completion.add_argument("shell", choices=("bash",), help="Currently supported shell: bash")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "__complete":
        return handle_hidden_completion(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "backend":
        if args.backend_command == "list":
            return print_backend_list()
        if args.backend_command == "select":
            return select_backend(args.backend_name)
        if args.backend_command == "stop":
            return backend_stop()
        parser.error("backend requires a subcommand: list / select / stop")

    if args.command == "select":
        return select_backend(args.backend_name)
    if args.command == "status":
        return print_status()
    if args.command == "model":
        if args.model_command == "list":
            return print_model_list(system_name=args.system, best=not args.all_backends)
        if args.model_command == "load":
            return print_model_load(args)
        parser.error("model requires a subcommand: list / load")
    if args.command == "thinking":
        if args.thinking_command == "show":
            return print_thinking_show()
        if args.thinking_command == "set":
            return print_thinking_set(args.value)
        parser.error("thinking requires a subcommand: show / set")
    if args.command == "chat":
        return chat(args)
    if args.command == "shutdown":
        return shutdown_service()
    if args.command == "serve":
        return serve_foreground()
    if args.command == "completion":
        return print_completion(args.shell)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
