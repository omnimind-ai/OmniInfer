from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

try:
    import readline as _readline
except ImportError:  # pragma: no cover - platform dependent
    _readline = None  # type: ignore[assignment]

from service_core import commands

_LOADING_FRAMES = "⠋⠙⠸⢰⣠⣄⡆⠇"
_LOADING_INTERVAL = 0.08
_MIN_WIDTH = 64
_MAX_WIDTH = 100


class _Theme:
    def __init__(self) -> None:
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")

    def paint(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def dim(self, text: str) -> str:
        return self.paint(text, "2")

    def brand(self, text: str) -> str:
        return self.paint(text, "1;36")

    def accent(self, text: str) -> str:
        return self.paint(text, "36")

    def success(self, text: str) -> str:
        return self.paint(text, "32")

    def warning(self, text: str) -> str:
        return self.paint(text, "33")

    def error(self, text: str) -> str:
        return self.paint(text, "31")

    def role_user(self, text: str) -> str:
        return self.paint(text, "1;35")

    def role_assistant(self, text: str) -> str:
        return self.paint(text, "1;34")


_THEME = _Theme()
_CHAT_HISTORY: list[str] = []

if _readline is not None:
    try:
        _readline.set_history_length(200)
    except (AttributeError, OSError):
        pass


def run_tui() -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("OmniInfer TUI requires an interactive terminal.")

    interrupted = False
    try:
        _clear()
        _print_header("OmniInfer", "Local inference console")
        remembered = commands.remembered_model_load_options()
        if remembered is not None:
            backend = _try_load_remembered_model(remembered)
            if backend is None:
                backend = _choose_backend()
                if backend is None:
                    _print_notice("No backend selected.", kind="warning")
                    return 1
                model = _choose_model()
                if model is None:
                    _print_notice("No model selected.", kind="warning")
                    return 1
                backend = _load_model(commands.ModelLoadOptions(model=str(model))) or backend
        else:
            backend = _choose_backend()
            if backend is None:
                _print_notice("No backend selected.", kind="warning")
                return 1
            model = _choose_model()
            if model is None:
                _print_notice("No model selected.", kind="warning")
                return 1
            backend = _load_model(commands.ModelLoadOptions(model=str(model))) or backend
        _chat_loop(backend)
    except KeyboardInterrupt:
        interrupted = True
        print()
    finally:
        _shutdown_service_for_tui()
    return 130 if interrupted else 0


def _choose_backend() -> str | None:
    while True:
        payload = commands.list_backends(scope="installed")
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        if not rows:
            raise SystemExit("No installed backends are available.")

        items: list[_MenuItem] = []
        for item in rows:
            installed = "installed" if item.get("binary_exists") else "not installed"
            capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), list) else []
            details = [installed, *[str(value) for value in capabilities[:4]]]
            items.append(
                _MenuItem(
                    label=str(item.get("id") or ""),
                    details=details,
                    selected=bool(item.get("selected")),
                )
            )
        index = _select_menu(
            title="Backends",
            subtitle="Choose the runtime used for model loading",
            items=items,
            default_index=_default_selected_index(rows),
        )
        if index is None:
            return None
        backend_id = str(rows[index].get("id", ""))
        if not backend_id:
            _print_notice("Invalid backend.", kind="warning")
            continue
        result = commands.select_backend(backend_id)
        _print_notice(f"Selected backend: {result.backend}", kind="success")
        if result.models_dir:
            _print_kv("Models directory", result.models_dir)
        print()
        return backend_id


def _choose_model() -> Path | None:
    while True:
        models = commands.discover_local_models()
        items: list[_MenuItem] = []
        if models:
            for model in models:
                items.append(_MenuItem(label=model.label))
            items.append(_MenuItem(label="Enter path manually", details=["link into .local/models"]))
            index = _select_menu(
                title="Models",
                subtitle="Pick a managed model or link a new local file",
                items=items,
                default_index=0,
            )
            if index is None:
                return None
            if index == len(models):
                return _prompt_model_path()
            return models[index].path

        _print_notice("No models found in OmniInfer .local model directories.", kind="warning")
        index = _select_menu(
            title="Models",
            subtitle="No managed models were found",
            items=[_MenuItem(label="Enter path manually", details=["link into .local/models"])],
            default_index=0,
        )
        if index is None:
            return None
        return _prompt_model_path()


def _prompt_model_path() -> Path | None:
    while True:
        text = _prompt("Model path")
        path = Path(os.path.abspath(os.path.expanduser(text)))
        if path.exists():
            model_root: Path | None = path.parent if path.is_file() else None
            if path.is_dir():
                directory = path
                selected = _select_model_from_directory(directory)
                if selected is None:
                    return None
                model_root = commands.infer_managed_model_root(selected, directory)
                path = selected
            try:
                linked = commands.link_model_into_managed_models(
                    path,
                    model_root=model_root,
                    preserve_relative_path=False,
                )
            except OSError as exc:
                _print_notice(f"Could not link model into {commands.managed_models_dir()}: {exc}", kind="warning")
                return path
            if linked != path:
                _print_notice(f"Linked model: {linked}", kind="success")
            return linked
        _print_notice(f"Model path does not exist: {path}", kind="warning")


def _select_model_from_directory(directory: Path) -> Path | None:
    candidates = commands.detect_model_files_in_directory(directory)
    if not candidates:
        _print_notice(f"No GGUF model files found under {directory}", kind="warning")
        return None
    if len(candidates) == 1:
        model = candidates[0]
        _print_notice(f"Detected model: {model.label}", kind="success")
        return model.path

    index = _select_menu(
        title="Models",
        subtitle=f"Detected {len(candidates)} model files under {directory.name}",
        items=[_MenuItem(label=model.label) for model in candidates],
        default_index=0,
    )
    if index is None:
        return None
    return candidates[index].path


def _try_load_remembered_model(options: commands.ModelLoadOptions) -> str | None:
    _print_section("Resume", "Loading your last selected backend and model")
    _print_kv("Model", options.model)
    try:
        backend = _load_model(options)
    except SystemExit as exc:
        _print_notice(f"Could not load previous model: {exc}", kind="warning")
        print()
        return None
    if backend:
        return backend
    return commands.selected_backend() or ""


def _load_model_after_backend_switch() -> str | None:
    remembered = commands.remembered_model_load_options()
    if remembered is not None:
        try:
            return _load_model(remembered)
        except SystemExit as exc:
            _print_notice(f"Could not reload previous model: {exc}", kind="warning")
            print()

    model = _choose_model()
    if model is None:
        return None
    return _load_model(commands.ModelLoadOptions(model=str(model)))


def _load_model(options: commands.ModelLoadOptions) -> str | None:
    print()
    _print_kv("Loading model", options.model)
    spinner = _LoadingSpinner("Loading model...")
    ready_printed = False

    def on_progress(event: dict[str, Any]) -> None:
        nonlocal ready_printed
        event_type = event.get("type")
        message = event.get("message")
        if event_type == "status" and message:
            spinner.update(_model_load_progress_text(str(message)))
        elif event_type == "log" and message:
            return
        elif event_type == "done":
            spinner.stop()
            ready_printed = True
            _print_notice("Backend ready", kind="success")

    spinner.start()
    try:
        response, selection = commands.load_model(
            options,
            progress=on_progress,
        )
    finally:
        spinner.stop()
    if not ready_printed:
        _print_notice("Backend ready", kind="success")
    if selection.auto_selected:
        _print_notice(f"Auto-selected backend: {selection.backend}", kind="success")
    _print_kv("Model loaded", str(response.get("selected_model") or options.model))
    print()
    backend = response.get("selected_backend")
    return str(backend) if backend else None


def _chat_loop(backend: str) -> None:
    current_backend = backend
    last_usage: dict[str, Any] | None = None
    messages: list[dict[str, str]] = []
    _print_chat_header(current_backend)
    while True:
        message = _prompt(_THEME.role_user("You"), history=True)
        if not message:
            continue
        if message == "/exit":
            return
        if message == "/backend":
            selected_backend = _choose_backend()
            if selected_backend:
                current_backend = selected_backend
                messages.clear()
                last_usage = None
                loaded_backend = _load_model_after_backend_switch()
                if loaded_backend:
                    current_backend = loaded_backend
            _print_chat_header(current_backend)
            continue
        if message == "/model":
            model = _choose_model()
            if model is None:
                _print_chat_header(current_backend)
                continue
            loaded_backend = _load_model(commands.ModelLoadOptions(model=str(model)))
            if loaded_backend:
                current_backend = loaded_backend
                messages.clear()
                last_usage = None
            _print_chat_header(current_backend)
            continue
        if message == "/clear":
            _clear()
            _print_header("OmniInfer", "Local inference console")
            _print_chat_header(current_backend)
            continue
        if message == "/status":
            _print_status(last_usage=last_usage, messages=messages)
            print()
            continue
        if message == "/help":
            _print_help()
            print()
            continue
        if message == "/think" or message == "/thinking" or message.startswith("/think ") or message.startswith("/thinking "):
            _handle_thinking_command(message)
            print()
            _print_chat_header(current_backend)
            continue

        print(_THEME.role_assistant("Assistant"))
        final_payload: dict[str, Any] | None = None
        buffer = ""
        assistant_text = ""
        visible_started = False
        try:
            payload = _build_conversation_payload(message, messages)
            for chunk in commands.iter_chat_stream_payload(payload):
                if chunk.text:
                    buffer += chunk.text
                    output, buffer, visible_started = _consume_visible_text(buffer, visible_started)
                    if output:
                        assistant_text += output
                        sys.stdout.write(output)
                        sys.stdout.flush()
                if chunk.final_payload:
                    final_payload = chunk.final_payload
        except SystemExit as exc:
            _print_notice(str(exc), kind="warning")
            print()
            continue
        if buffer:
            output, _buffer, _visible_started = _consume_visible_text(buffer, visible_started, final=True)
            if output:
                assistant_text += output
                sys.stdout.write(output)
                sys.stdout.flush()
        print()
        if final_payload:
            usage = final_payload.get("usage") if isinstance(final_payload.get("usage"), dict) else None
            if usage:
                last_usage = usage
            _print_short_performance(final_payload)
        if assistant_text:
            messages.append({"role": "user", "content": message})
            messages.append({"role": "assistant", "content": assistant_text})
        print()
        _ = current_backend


def _build_conversation_payload(message: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload = commands.build_chat_payload(commands.ChatOptions(message=message))
    payload["messages"] = [*messages, {"role": "user", "content": message}]
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    return payload


def _print_short_performance(payload: dict[str, Any]) -> None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    pieces: list[str] = []
    if usage:
        pieces.append(f"tokens={usage.get('total_tokens', '-')}")
    if isinstance(timings, dict) and "predicted_per_second" in timings:
        pieces.append(f"speed={_format_speed(timings.get('predicted_per_second'))} tok/s")
    if pieces:
        print(_THEME.dim("[" + ", ".join(pieces) + "]"))


def _format_speed(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _print_status(*, last_usage: dict[str, Any] | None, messages: list[dict[str, str]]) -> None:
    try:
        state = commands.current_runtime_state()
    except SystemExit as exc:
        _print_notice(f"Could not read status: {exc}", kind="warning")
        return

    _print_section("Status", "Current OmniInfer session")
    _print_kv("Backend", str(state.get("backend") or "-"))
    _print_kv("Backend ready", "yes" if state.get("backend_ready") else "no")
    _print_kv("Model", str(state.get("model") or "-"))
    if state.get("mmproj"):
        _print_kv("MMProj", str(state.get("mmproj")))
    ctx_size = _resolve_context_size(state)
    _print_kv("Context size", _format_context_size(ctx_size, loaded=bool(state.get("backend_ready"))))

    defaults = state.get("request_defaults") if isinstance(state.get("request_defaults"), dict) else {}
    thinking = state.get("thinking") if isinstance(state.get("thinking"), dict) else {}
    if "default_enabled" in thinking:
        _print_kv("Thinking", _format_on_off(bool(thinking.get("default_enabled"))))
    if defaults:
        default_bits = [
            f"{key}={defaults[key]}"
            for key in ("temperature", "max_tokens", "stream", "think")
            if key in defaults
        ]
        if default_bits:
            _print_kv("Request defaults", ", ".join(default_bits))

    _print_kv("Conversation messages", str(len(messages)))
    usage_text = _format_context_usage(last_usage, ctx_size)
    _print_kv("Context usage", usage_text)


def _resolve_context_size(state: dict[str, Any]) -> int | None:
    state_ctx = _positive_int(state.get("ctx_size"))
    if state_ctx:
        return state_ctx
    if not state.get("backend_ready"):
        return None
    try:
        props = commands.current_backend_props()
    except SystemExit:
        return None
    return _context_size_from_runtime_props(props)


def _context_size_from_runtime_props(props: dict[str, Any]) -> int | None:
    direct = _positive_int(props.get("n_ctx"))
    if direct:
        return direct
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict):
        nested = _positive_int(settings.get("n_ctx"))
        if nested:
            return nested
        params = settings.get("params")
        if isinstance(params, dict):
            return _positive_int(params.get("n_ctx"))
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _format_context_size(ctx_size: int | None, *, loaded: bool) -> str:
    if ctx_size:
        return str(ctx_size)
    return "backend default (unreported)" if loaded else "not loaded"


def _format_context_usage(usage: dict[str, Any] | None, ctx_size: Any) -> str:
    if not usage:
        return "not available yet"
    input_tokens = usage.get("prompt_tokens", "-")
    output_tokens = usage.get("completion_tokens", "-")
    total_tokens = usage.get("total_tokens")
    pieces = [f"input={input_tokens}", f"output={output_tokens}"]
    if isinstance(total_tokens, int):
        try:
            ctx_int = int(ctx_size)
        except (TypeError, ValueError):
            ctx_int = 0
        if ctx_int > 0:
            percent = total_tokens / ctx_int * 100
            pieces.append(f"total={total_tokens}/{ctx_int} ({percent:.1f}%)")
        else:
            pieces.append(f"total={total_tokens}")
    return ", ".join(pieces)


def _shutdown_service_for_tui() -> None:
    try:
        stopped = commands.shutdown_service()
    except SystemExit as exc:
        _print_notice(f"Could not stop OmniInfer service: {exc}", kind="warning")
        return
    if stopped:
        _print_notice("OmniInfer service stopped", kind="success")


def _print_header(title: str, subtitle: str) -> None:
    width = _content_width()
    print(_THEME.brand(title))
    print(_THEME.dim(subtitle))
    print(_THEME.dim("─" * width))
    print()


def _print_section(title: str, subtitle: str | None = None) -> None:
    width = _content_width()
    print(_THEME.accent(title))
    if subtitle:
        print(_THEME.dim(_truncate(subtitle, width)))
    print(_THEME.dim("─" * min(width, max(len(title), 24))))


def _print_menu_item(
    *,
    index: int,
    label: str,
    details: list[str] | None = None,
    selected: bool = False,
) -> None:
    width = _content_width()
    marker = _THEME.success("●") if selected else _THEME.dim("○")
    prefix = f"{index:>2}. {marker} "
    detail_text = ""
    if details:
        clean_details = [item for item in details if item]
        if clean_details:
            detail_text = "  " + _THEME.dim(" · ".join(clean_details))
    plain_budget = max(width - len(_strip_ansi(prefix)) - len(_strip_ansi(detail_text)), 18)
    print(f"{prefix}{_truncate(label, plain_budget)}{detail_text}")


def _print_notice(message: str, *, kind: str = "info") -> None:
    colors = {
        "success": _THEME.success,
        "warning": _THEME.warning,
        "error": _THEME.error,
        "info": _THEME.accent,
    }
    paint = colors.get(kind, _THEME.accent)
    print(f"  {paint('•')} {_truncate(message, _content_width() - 4)}")


def _print_kv(label: str, value: str) -> None:
    width = _content_width()
    prefix = f"{_THEME.dim(label + ':')} "
    print(f"  {prefix}{_truncate(value, max(width - len(label) - 4, 16))}")


def _print_command_bar() -> None:
    commands_text = " /backend  /model  /think  /status  /clear  /help  /exit "
    print(_THEME.dim("Commands") + _THEME.accent(commands_text))


def _print_chat_header(backend: str) -> None:
    thinking = _current_thinking_label()
    suffix = f" · Thinking: {thinking}" if thinking else ""
    _print_section("Chat", f"Backend: {backend}{suffix}")
    _print_command_bar()
    print()


def _print_help() -> None:
    _print_section("Help", "Conversation commands")
    commands_table = [
        ("/backend", "switch the selected runtime"),
        ("/model", "load a different managed model"),
        ("/think", "toggle thinking mode; use /think on or /think off to set it"),
        ("/status", "show backend, model, request defaults, and conversation context usage"),
        ("/clear", "clear the terminal and redraw the chat header"),
        ("/help", "show this command reference"),
        ("/exit", "stop the OmniInfer service and leave the TUI"),
    ]
    for name, description in commands_table:
        _print_kv(name, description)


def _handle_thinking_command(message: str) -> None:
    parts = message.split()
    if len(parts) == 1:
        enabled = not commands.get_default_thinking()
    elif len(parts) == 2 and parts[1].lower() in {"on", "off"}:
        enabled = parts[1].lower() == "on"
    else:
        _print_notice("Usage: /think, /think on, or /think off", kind="warning")
        return
    new_value = commands.set_default_thinking(enabled)
    _print_notice(f"Thinking mode: {_format_on_off(new_value)}", kind="success")


def _current_thinking_label() -> str | None:
    try:
        return _format_on_off(commands.get_default_thinking())
    except SystemExit:
        return None


def _format_on_off(enabled: bool) -> str:
    return "on" if enabled else "off"


class _MenuItem:
    def __init__(self, label: str, details: list[str] | None = None, selected: bool = False) -> None:
        self.label = label
        self.details = details or []
        self.selected = selected


def _select_menu(
    *,
    title: str,
    subtitle: str,
    items: list[_MenuItem],
    default_index: int = 0,
) -> int | None:
    if not items:
        return None
    default_index = max(0, min(default_index, len(items) - 1))
    if _can_use_interactive_menu():
        return _select_menu_interactive(title=title, subtitle=subtitle, items=items, default_index=default_index)
    return _select_menu_prompt(title=title, subtitle=subtitle, items=items, default_index=default_index)


def _select_menu_prompt(
    *,
    title: str,
    subtitle: str,
    items: list[_MenuItem],
    default_index: int,
) -> int | None:
    _print_section(title, subtitle)
    for index, item in enumerate(items, 1):
        _print_menu_item(index=index, label=item.label, details=item.details, selected=item.selected)
    print(_THEME.dim("Press Enter to keep the default, or type Esc to cancel."))
    print()
    while True:
        choice = _prompt(f"Select {title.lower()[:-1] if title.endswith('s') else title.lower()}", default=str(default_index + 1))
        if choice.lower() in {"esc", "cancel", "q"}:
            return None
        index = _parse_choice(choice, len(items))
        if index is not None:
            return index - 1
        _print_notice("Invalid selection.", kind="warning")


def _select_menu_interactive(
    *,
    title: str,
    subtitle: str,
    items: list[_MenuItem],
    default_index: int,
) -> int | None:
    index = default_index
    rendered_lines = 0
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        while True:
            if rendered_lines:
                sys.stdout.write(f"\033[{rendered_lines}A\033[J")
            rendered = _render_menu(title, subtitle, items, index)
            rendered_lines = len(rendered.splitlines())
            sys.stdout.write(rendered)
            sys.stdout.flush()
            key = _read_menu_key()
            if key == "up":
                index = (index - 1) % len(items)
            elif key == "down":
                index = (index + 1) % len(items)
            elif key == "home":
                index = 0
            elif key == "end":
                index = len(items) - 1
            elif key == "enter":
                sys.stdout.write(f"\033[{rendered_lines}A\033[J")
                return index
            elif key == "esc":
                sys.stdout.write(f"\033[{rendered_lines}A\033[J")
                _print_notice("Selection cancelled.", kind="warning")
                print()
                return None
            elif key and key.isdigit():
                number = int(key)
                if 1 <= number <= min(len(items), 9):
                    index = number - 1
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def _render_menu(title: str, subtitle: str, items: list[_MenuItem], active_index: int) -> str:
    lines: list[str] = []
    width = _content_width()
    lines.append(_THEME.accent(title))
    if subtitle:
        lines.append(_THEME.dim(_truncate(subtitle, width)))
    lines.append(_THEME.dim("─" * min(width, max(len(title), 24))))
    for index, item in enumerate(items):
        selected = "●" if item.selected else "○"
        cursor = "›" if index == active_index else " "
        detail = ""
        if item.details:
            clean_details = [value for value in item.details if value]
            if clean_details:
                detail = "  " + _THEME.dim(" · ".join(clean_details))
        prefix = f" {cursor} {selected} "
        budget = max(width - len(_strip_ansi(prefix)) - len(_strip_ansi(detail)), 18)
        label = _truncate(item.label, budget)
        if index == active_index:
            label = _THEME.paint(label, "7")
        lines.append(f"{prefix}{label}{detail}")
    lines.append(_THEME.dim("  ↑/↓ move   Enter select   Esc back"))
    lines.append("")
    return "\n".join(lines)


def _can_use_interactive_menu() -> bool:
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


def _read_menu_key() -> str:
    if os.name == "nt":
        return _read_menu_key_windows()
    return _read_menu_key_posix()


def _read_menu_key_windows() -> str:
    import msvcrt

    char = msvcrt.getwch()
    if char == "\x03":
        raise KeyboardInterrupt
    if char in {"\r", "\n"}:
        return "enter"
    if char == "\x1b":
        return "esc"
    if char in {"\x00", "\xe0"}:
        code = msvcrt.getwch()
        return {
            "H": "up",
            "P": "down",
            "G": "home",
            "O": "end",
        }.get(code, "")
    return char


def _read_menu_key_posix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = os.read(fd, 1).decode(errors="ignore")
        if char == "\x03":
            raise KeyboardInterrupt
        if char in {"\r", "\n"}:
            return "enter"
        if char != "\x1b":
            return char
        seq = _read_escape_sequence(fd, initial_timeout_s=0.2)
        return _decode_escape_sequence(seq) or "esc"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_escape_sequence(fd: int, *, initial_timeout_s: float = 0.2, continuation_timeout_s: float = 0.03) -> str:
    import select

    parts: list[str] = []
    while len(parts) < 8:
        timeout_s = initial_timeout_s if not parts else continuation_timeout_s
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            break
        data = os.read(fd, 1)
        if not data:
            break
        char = data.decode(errors="ignore")
        parts.append(char)
        if char.isalpha() or char == "~":
            break
    return "".join(parts)


def _decode_escape_sequence(seq: str) -> str:
    return {
        "[A": "up",
        "OA": "up",
        "[B": "down",
        "OB": "down",
        "[H": "home",
        "OH": "home",
        "[1~": "home",
        "[F": "end",
        "OF": "end",
        "[4~": "end",
    }.get(seq, "")


def _model_load_progress_text(message: str) -> str:
    if re.fullmatch(r"Starting backend .+ and loading model\.\.\.", message):
        return "Loading model..."
    return message


def _content_width() -> int:
    size = shutil.get_terminal_size(fallback=(80, 24))
    return max(_MIN_WIDTH, min(_MAX_WIDTH, size.columns))


def _truncate(text: str, width: int) -> str:
    if width <= 1:
        return "…"
    if len(text) <= width:
        return text
    return text[: max(width - 1, 1)] + "…"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


class _LoadingSpinner:
    def __init__(self, text: str) -> None:
        self._text = text
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active = False
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

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
            self._thread = None
        if self._active:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._active = False

    def _run(self) -> None:
        index = 0
        while not self._stop_event.is_set():
            with self._lock:
                text = self._text
            frame = _LOADING_FRAMES[index % len(_LOADING_FRAMES)]
            index += 1
            sys.stdout.write(f"\r  {_THEME.accent(frame)} {_THEME.dim(text)}\033[K")
            sys.stdout.flush()
            self._stop_event.wait(_LOADING_INTERVAL)


def _consume_visible_text(buffer: str, visible_started: bool, final: bool = False) -> tuple[str, str, bool]:
    if visible_started:
        return buffer, "", True

    stripped = buffer.lstrip()
    leading_len = len(buffer) - len(stripped)
    if not stripped.startswith("<think>"):
        return buffer, "", True

    close_index = stripped.find("</think>")
    if close_index < 0:
        return ("", "" if final else buffer, False)

    after = stripped[close_index + len("</think>") :]
    after = re.sub(r"^\s*", "", after)
    return after, "", bool(after)


def _prompt(label: str, default: str | None = None, *, history: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    _load_readline_history(_CHAT_HISTORY if history else [])
    try:
        value = input(f"{label}{suffix}: ").strip()
    finally:
        if not history:
            _load_readline_history([])
    result = value or (default or "")
    if history and result:
        _remember_chat_input(result)
    return result


def _remember_chat_input(value: str) -> None:
    if _CHAT_HISTORY and _CHAT_HISTORY[-1] == value:
        _load_readline_history(_CHAT_HISTORY)
        return
    _CHAT_HISTORY.append(value)
    del _CHAT_HISTORY[:-200]
    _load_readline_history(_CHAT_HISTORY)


def _load_readline_history(items: list[str]) -> None:
    if _readline is None:
        return
    try:
        _readline.clear_history()
        for item in items:
            _readline.add_history(item)
    except (AttributeError, OSError):
        return


def _parse_choice(value: str, count: int) -> int | None:
    try:
        index = int(value)
    except ValueError:
        return None
    if 1 <= index <= count:
        return index
    return None


def _default_selected_index(rows: list[dict[str, Any]]) -> int:
    for index, item in enumerate(rows, 1):
        if item.get("selected"):
            return index - 1
    return 0


def _clear() -> None:
    command = "cls" if os.name == "nt" else "clear"
    if os.environ.get("TERM"):
        os.system(command)
