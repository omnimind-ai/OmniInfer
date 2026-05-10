from __future__ import annotations

import codecs
import os
import re
import shutil
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
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
_PROMPT_BOX_ROWS = 4
_ACTIVE_NOTICE_CENTER: _NoticeCenter | None = None


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


@dataclass
class _Notice:
    message: str
    kind: str = "info"


class _NoticeCenter:
    def __init__(self) -> None:
        self.current: _Notice | None = None

    def push(self, message: str, *, kind: str = "info") -> None:
        self.current = _Notice(message=message, kind=kind)

    def clear(self) -> None:
        self.current = None

    def status_text(self) -> str:
        if not self.current:
            return ""
        prefix = {
            "success": "ok",
            "warning": "warn",
            "error": "error",
            "info": "info",
        }.get(self.current.kind, "info")
        return f"{prefix}: {self.current.message}"

    def capture(self) -> "_NoticeCapture":
        return _NoticeCapture(self)


class _NoticeCapture:
    def __init__(self, center: _NoticeCenter) -> None:
        self._center = center
        self._previous: _NoticeCenter | None = None

    def __enter__(self) -> _NoticeCenter:
        global _ACTIVE_NOTICE_CENTER
        self._previous = _ACTIVE_NOTICE_CENTER
        _ACTIVE_NOTICE_CENTER = self._center
        return self._center

    def __exit__(self, *_args: object) -> None:
        global _ACTIVE_NOTICE_CENTER
        _ACTIVE_NOTICE_CENTER = self._previous


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


@dataclass
class _ChatSessionState:
    backend: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_usage: dict[str, Any] | None = None
    notices: _NoticeCenter = field(default_factory=_NoticeCenter)

    def switch_backend(self, backend: str) -> None:
        self.backend = backend
        self.clear_conversation()

    def clear_conversation(self) -> None:
        self.messages.clear()
        self.last_usage = None


class _TranscriptView:
    def render_header(self, session: _ChatSessionState) -> None:
        _print_chat_header(session.backend)

    def render_assistant_header(self) -> None:
        sys.stdout.write(f"\r{_THEME.role_assistant('Assistant')}:\n")
        sys.stdout.flush()

    def render_status(self, session: _ChatSessionState) -> None:
        _print_status(last_usage=session.last_usage, messages=session.messages)
        print()

    def render_help(self) -> None:
        _print_help()
        print()


class _StatusLine:
    def text(self, session: _ChatSessionState) -> str:
        state: dict[str, Any] = {}
        try:
            state = commands.current_runtime_state()
        except SystemExit:
            pass
        model = _format_status_model(state.get("model"))
        thinking = _current_thinking_label() or "-"
        pieces = [
            f"backend {session.backend}",
            f"model {model}",
            f"think {thinking}",
        ]
        notice = session.notices.status_text()
        if notice:
            pieces.insert(0, notice)
        usage = _format_status_usage(session.last_usage, _resolve_context_size(state) if state else None)
        if usage:
            pieces.append(usage)
        return "  ".join(pieces)


class _FixedPromptDuringOutput:
    def __init__(self, label: str, status: str) -> None:
        self._label = label
        self._status = status
        self._active = False

    def __enter__(self) -> "_FixedPromptDuringOutput":
        if not _can_use_fixed_input_box():
            return self
        rows = shutil.get_terminal_size(fallback=(80, 24)).lines
        scroll_bottom = max(1, rows - _PROMPT_BOX_ROWS)
        sys.stdout.write(f"\0337\033[1;{scroll_bottom}r\0338")
        self._active = True
        self.redraw(self._status)
        return self

    def __exit__(self, *_args: object) -> None:
        if not self._active:
            return
        sys.stdout.write("\0337\033[r\0338\033[?25h")
        sys.stdout.flush()

    def redraw(self, status: str | None = None) -> None:
        if not self._active:
            return
        if status is not None:
            self._status = status
        _draw_prompt_placeholder(self._label, status=self._status)


def _chat_loop(backend: str) -> None:
    session = _ChatSessionState(backend=backend)
    transcript = _TranscriptView()
    status_line = _StatusLine()
    transcript.render_header(session)
    while True:
        message = _prompt(_THEME.role_user("You"), history=True, status=status_line.text(session))
        if not message:
            continue
        if message == "/exit":
            return
        if message == "/backend":
            with session.notices.capture():
                selected_backend = _choose_backend()
                if selected_backend:
                    session.switch_backend(selected_backend)
                    loaded_backend = _load_model_after_backend_switch()
                    if loaded_backend:
                        session.backend = loaded_backend
            continue
        if message == "/model":
            with session.notices.capture():
                model = _choose_model()
                if model is None:
                    continue
                loaded_backend = _load_model(commands.ModelLoadOptions(model=str(model)))
                if loaded_backend:
                    session.switch_backend(loaded_backend)
            continue
        if message == "/clear":
            _clear()
            _print_header("OmniInfer", "Local inference console")
            transcript.render_header(session)
            continue
        if message == "/status":
            transcript.render_status(session)
            continue
        if message == "/help":
            transcript.render_help()
            continue
        if message == "/think" or message == "/thinking" or message.startswith("/think ") or message.startswith("/thinking "):
            with session.notices.capture():
                _handle_thinking_command(message)
            continue

        session.notices.clear()
        with _FixedPromptDuringOutput(_THEME.role_user("You"), status_line.text(session)) as fixed_prompt:
            final_payload: dict[str, Any] | None = None
            buffer = ""
            assistant_text = ""
            visible_started = False
            assistant_header_printed = False
            thinking_spinner = _LoadingSpinner("Thinking...")
            try:
                payload = _build_conversation_payload(message, session.messages)
                for chunk in commands.iter_chat_stream_payload(payload):
                    if chunk.reasoning_text and not visible_started and not thinking_spinner.active:
                        thinking_spinner.start()
                    if chunk.text:
                        buffer += chunk.text
                        output, buffer, visible_started = _consume_visible_text(buffer, visible_started)
                        if output and thinking_spinner.active:
                            thinking_spinner.stop()
                        elif _is_hidden_thinking_pending(buffer, visible_started) and not thinking_spinner.active:
                            thinking_spinner.start()
                        if output:
                            if not assistant_header_printed:
                                transcript.render_assistant_header()
                                assistant_header_printed = True
                            assistant_text += output
                            sys.stdout.write(output)
                            sys.stdout.flush()
                    if chunk.final_payload:
                        final_payload = chunk.final_payload
            except SystemExit as exc:
                thinking_spinner.stop()
                session.notices.push(str(exc), kind="warning")
                fixed_prompt.redraw(status_line.text(session))
                continue
            if buffer:
                output, _buffer, _visible_started = _consume_visible_text(buffer, visible_started, final=True)
                if output:
                    thinking_spinner.stop()
                    if not assistant_header_printed:
                        transcript.render_assistant_header()
                        assistant_header_printed = True
                    assistant_text += output
                    sys.stdout.write(output)
                    sys.stdout.flush()
            thinking_spinner.stop()
            print()
            if final_payload:
                usage = final_payload.get("usage") if isinstance(final_payload.get("usage"), dict) else None
                if usage:
                    session.last_usage = usage
                _print_short_performance(final_payload)
            if assistant_text:
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": assistant_text})
            fixed_prompt.redraw(status_line.text(session))


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


def _format_status_model(value: Any) -> str:
    if not value:
        return "-"
    return Path(str(value)).name


def _format_status_usage(usage: dict[str, Any] | None, ctx_size: Any) -> str:
    if not usage:
        return ""
    input_tokens = usage.get("prompt_tokens", "-")
    output_tokens = usage.get("completion_tokens", "-")
    total_tokens = usage.get("total_tokens")
    pieces = [f"in {input_tokens}", f"out {output_tokens}"]
    if isinstance(total_tokens, int):
        try:
            ctx_int = int(ctx_size)
        except (TypeError, ValueError):
            ctx_int = 0
        if ctx_int > 0:
            percent = total_tokens / ctx_int * 100
            pieces.append(f"ctx {total_tokens}/{ctx_int} {percent:.1f}%")
        else:
            pieces.append(f"ctx {total_tokens}")
    return "  ".join(pieces)


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
    if _ACTIVE_NOTICE_CENTER is not None:
        _ACTIVE_NOTICE_CENTER.push(message, kind=kind)
        return
    colors = {
        "success": _THEME.success,
        "warning": _THEME.warning,
        "error": _THEME.error,
        "info": _THEME.accent,
    }
    paint = colors.get(kind, _THEME.accent)
    print(f"  {paint('•')} {_truncate(message, _content_width() - 4)}")


def _print_kv(label: str, value: str) -> None:
    if _ACTIVE_NOTICE_CENTER is not None:
        _ACTIVE_NOTICE_CENTER.push(f"{label}: {value}", kind="info")
        return
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
        "[C": "right",
        "OC": "right",
        "[D": "left",
        "OD": "left",
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
        if self._active:
            return
        self._stop_event.clear()
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

    @property
    def active(self) -> bool:
        return self._active


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


def _is_hidden_thinking_pending(buffer: str, visible_started: bool) -> bool:
    if visible_started:
        return False
    stripped = buffer.lstrip()
    return stripped.startswith("<think>")


def _prompt(label: str, default: str | None = None, *, history: bool = False, status: str | None = None) -> str:
    if history and default is None and _can_use_fixed_input_box():
        result = _prompt_chat_box(label, status=status)
        if result:
            _remember_chat_input(result)
        return result
    return _prompt_basic(label, default, history=history)


def _prompt_basic(label: str, default: str | None = None, *, history: bool = False) -> str:
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


def _can_use_fixed_input_box() -> bool:
    return bool(
        os.name != "nt"
        and getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


def _prompt_chat_box(label: str, *, status: str | None = None) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    state = _InputBoxState(history=list(_CHAT_HISTORY))
    decoder = codecs.getincrementaldecoder("utf-8")()
    finished = False
    sys.stdout.write("\033[s\033[?25l")
    sys.stdout.flush()
    _draw_input_box(label, state, status=status)
    try:
        tty.setraw(fd)
        while True:
            data = os.read(fd, 1)
            if not data:
                continue
            if data == b"\x03":
                raise KeyboardInterrupt
            if data in {b"\r", b"\n"}:
                result = state.text.strip()
                _clear_input_box(restore=True)
                sys.stdout.write(f"\r{label}: {result}\n")
                sys.stdout.flush()
                finished = True
                return result
            if data == b"\x1b":
                decoder.reset()
                key = _decode_escape_sequence(_read_escape_sequence(fd, initial_timeout_s=0.05)) or "esc"
                if key == "esc":
                    state.text = ""
                    state.cursor = 0
                elif key == "up":
                    state.history_previous()
                elif key == "down":
                    state.history_next()
                elif key == "left":
                    state.cursor = max(0, state.cursor - 1)
                elif key == "right":
                    state.cursor = min(len(state.text), state.cursor + 1)
                elif key == "home":
                    state.cursor = 0
                elif key == "end":
                    state.cursor = len(state.text)
                _draw_input_box(label, state, status=status)
                continue
            if data in {b"\x7f", b"\b"}:
                decoder.reset()
                state.backspace()
                _draw_input_box(label, state, status=status)
                continue
            if data == b"\x04":
                decoder.reset()
                if state.text:
                    state.delete()
                    _draw_input_box(label, state, status=status)
                    continue
                _clear_input_box(restore=True)
                finished = True
                return "/exit"

            char = decoder.decode(data)
            if char and char.isprintable():
                state.insert(char)
                _draw_input_box(label, state, status=status)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if not finished:
            _clear_input_box(restore=True)
        else:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()


class _InputBoxState:
    def __init__(self, history: list[str]) -> None:
        self.text = ""
        self.cursor = 0
        self._history = history
        self._history_index: int | None = None
        self._draft = ""

    def insert(self, value: str) -> None:
        self.text = self.text[: self.cursor] + value + self.text[self.cursor :]
        self.cursor += len(value)
        self._history_index = None

    def backspace(self) -> None:
        if self.cursor <= 0:
            return
        self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
        self.cursor -= 1
        self._history_index = None

    def delete(self) -> None:
        if self.cursor >= len(self.text):
            return
        self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        self._history_index = None

    def history_previous(self) -> None:
        if not self._history:
            return
        if self._history_index is None:
            self._draft = self.text
            self._history_index = len(self._history) - 1
        else:
            self._history_index = max(0, self._history_index - 1)
        self.text = self._history[self._history_index]
        self.cursor = len(self.text)

    def history_next(self) -> None:
        if self._history_index is None:
            return
        if self._history_index >= len(self._history) - 1:
            self._history_index = None
            self.text = self._draft
        else:
            self._history_index += 1
            self.text = self._history[self._history_index]
        self.cursor = len(self.text)


def _draw_input_box(label: str, state: _InputBoxState, *, status: str | None = None) -> None:
    columns, rows = shutil.get_terminal_size(fallback=(80, 24))
    width = max(20, columns)
    top = max(1, rows - _PROMPT_BOX_ROWS + 1)
    label_text = _strip_ansi(label)
    max_value_width = max(width - _display_width(label_text) - 8, 8)
    rendered = _render_input_value(state.text, state.cursor, max_value_width)
    prompt = f"{label}: {rendered}"
    prompt = _pad_display(prompt, width - 4)
    status_text = _render_prompt_status(status or "", width - 4)
    border_width = max(width - 2, 1)
    sys.stdout.write(f"\033[{top};1H\033[2K{_THEME.dim('╭' + '─' * border_width + '╮')}")
    sys.stdout.write(f"\033[{top + 1};1H\033[2K{_THEME.dim('│')} {prompt} {_THEME.dim('│')}")
    sys.stdout.write(f"\033[{top + 2};1H\033[2K{_THEME.dim('│')} {status_text} {_THEME.dim('│')}")
    sys.stdout.write(f"\033[{top + 3};1H\033[2K{_THEME.dim('╰' + '─' * border_width + '╯')}")
    cursor_col = min(width, 3 + _display_width(f"{label_text}: ") + _input_cursor_prefix_width(rendered))
    sys.stdout.write(f"\033[{top + 1};{cursor_col}H")
    sys.stdout.flush()


def _draw_prompt_placeholder(label: str, *, status: str | None = None) -> None:
    sys.stdout.write("\0337\033[?25l")
    _draw_input_box(label, _InputBoxState(history=[]), status=status)
    sys.stdout.write("\0338")
    sys.stdout.flush()


def _clear_input_box(*, restore: bool = False) -> None:
    columns, rows = shutil.get_terminal_size(fallback=(80, 24))
    del columns
    top = max(1, rows - _PROMPT_BOX_ROWS + 1)
    for row in range(top, top + _PROMPT_BOX_ROWS):
        sys.stdout.write(f"\033[{row};1H\033[2K")
    if restore:
        sys.stdout.write("\033[u")
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _render_input_value(text: str, cursor: int, max_width: int) -> str:
    cursor = max(0, min(cursor, len(text)))
    left_chars = list(text[:cursor])
    right_chars = list(text[cursor:])
    cursor_marker = _THEME.accent("▌")
    left_width = _display_width("".join(left_chars))
    right_width = _display_width("".join(right_chars))
    cursor_width = _display_width(cursor_marker)
    if max_width <= cursor_width:
        return cursor_marker

    if left_width + cursor_width + right_width <= max_width:
        return "".join(left_chars) + cursor_marker + "".join(right_chars)

    needs_left_ellipsis = bool(left_chars)
    needs_right_ellipsis = bool(right_chars)
    available = max_width - cursor_width
    if needs_left_ellipsis:
        available -= 1
    if needs_right_ellipsis:
        available -= 1
    available = max(available, 0)

    left_target = min(left_width, available // 2)
    right_target = min(right_width, available - left_target)
    remaining = available - left_target - right_target
    if remaining > 0:
        extra_left = min(left_width - left_target, remaining)
        left_target += extra_left
        remaining -= extra_left
        if remaining > 0:
            right_target += min(right_width - right_target, remaining)

    left_rendered, left_rendered_width = _take_display_suffix(left_chars, left_target)
    right_rendered, right_rendered_width = _take_display_prefix(right_chars, right_target)
    rendered = []
    if needs_left_ellipsis and left_rendered_width < left_width:
        rendered.append("…")
    rendered.append(left_rendered)
    rendered.append(cursor_marker)
    rendered.append(right_rendered)
    if needs_right_ellipsis and right_rendered_width < right_width:
        rendered.append("…")
    return "".join(rendered)


def _render_prompt_status(status: str, max_width: int) -> str:
    text = _THEME.dim(_truncate(status, max_width)) if status else ""
    return _pad_display(text, max_width)


def _input_cursor_prefix_width(rendered: str) -> int:
    clean = _strip_ansi(rendered)
    marker_index = clean.find("▌")
    if marker_index < 0:
        return _display_width(clean)
    return _display_width(clean[:marker_index])


def _take_display_prefix(chars: list[str], max_width: int) -> tuple[str, int]:
    width = 0
    rendered: list[str] = []
    for char in chars:
        char_width = _display_width(char)
        if width + char_width > max_width:
            break
        rendered.append(char)
        width += char_width
    return "".join(rendered), width


def _take_display_suffix(chars: list[str], max_width: int) -> tuple[str, int]:
    width = 0
    rendered: list[str] = []
    for char in reversed(chars):
        char_width = _display_width(char)
        if width + char_width > max_width:
            break
        rendered.append(char)
        width += char_width
    rendered.reverse()
    return "".join(rendered), width


def _pad_display(text: str, width: int) -> str:
    current = _display_width(text)
    if current >= width:
        return text
    return text + " " * (width - current)


def _display_width(text: str) -> int:
    clean = _strip_ansi(text)
    width = 0
    for char in clean:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


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
