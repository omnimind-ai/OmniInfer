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
                model = _choose_model()
                backend = _load_model(commands.ModelLoadOptions(model=str(model))) or backend
        else:
            backend = _choose_backend()
            model = _choose_model()
            backend = _load_model(commands.ModelLoadOptions(model=str(model))) or backend
        _chat_loop(backend)
    except KeyboardInterrupt:
        interrupted = True
        print()
    finally:
        _shutdown_service_for_tui()
    return 130 if interrupted else 0


def _choose_backend() -> str:
    while True:
        payload = commands.list_backends(scope="installed")
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        if not rows:
            raise SystemExit("No installed backends are available.")

        _print_section("Backends", "Choose the runtime used for model loading")
        for index, item in enumerate(rows, 1):
            installed = "installed" if item.get("binary_exists") else "not installed"
            capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), list) else []
            details = [installed, *[str(value) for value in capabilities[:4]]]
            _print_menu_item(
                index=index,
                label=str(item.get("id") or ""),
                details=details,
                selected=bool(item.get("selected")),
            )
        print()
        choice = _prompt("Select backend", default=_default_selected_index(rows))
        index = _parse_choice(choice, len(rows))
        if index is None:
            _print_notice("Invalid selection.", kind="warning")
            continue
        backend_id = str(rows[index - 1].get("id", ""))
        if not backend_id:
            _print_notice("Invalid backend.", kind="warning")
            continue
        result = commands.select_backend(backend_id)
        _print_notice(f"Selected backend: {result.backend}", kind="success")
        if result.models_dir:
            _print_kv("Models directory", result.models_dir)
        print()
        return backend_id


def _choose_model() -> Path:
    while True:
        models = commands.discover_local_models()
        _print_section("Models", "Pick a managed model or link a new local file")
        if models:
            for index, model in enumerate(models, 1):
                _print_menu_item(index=index, label=model.label)
            _print_menu_item(index=0, label="Enter path manually", details=["link into .local/models"])
            choice = _prompt("Select model", default="1")
            if choice == "0":
                return _prompt_model_path()
            index = _parse_choice(choice, len(models))
            if index is not None:
                return models[index - 1].path
            _print_notice("Invalid selection.", kind="warning")
            continue

        _print_notice("No models found in OmniInfer .local model directories.", kind="warning")
        return _prompt_model_path()


def _prompt_model_path() -> Path:
    while True:
        text = _prompt("Model path")
        path = Path(os.path.abspath(os.path.expanduser(text)))
        if path.exists():
            try:
                linked = commands.link_model_into_managed_models(path)
            except OSError as exc:
                _print_notice(f"Could not link model into {commands.managed_models_dir()}: {exc}", kind="warning")
                return path
            if linked != path:
                _print_notice(f"Linked model: {linked}", kind="success")
            return linked
        _print_notice(f"Model path does not exist: {path}", kind="warning")


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


def _load_model(options: commands.ModelLoadOptions) -> str | None:
    print()
    _print_kv("Loading model", options.model)
    spinner = _LoadingSpinner("Preparing model...")
    ready_printed = False

    def on_progress(event: dict[str, Any]) -> None:
        nonlocal ready_printed
        event_type = event.get("type")
        message = event.get("message")
        if event_type == "status" and message:
            spinner.update(str(message))
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
    _print_chat_header(current_backend)
    while True:
        message = _prompt(_THEME.role_user("You"), history=True)
        if not message:
            continue
        if message == "/exit":
            return
        if message == "/backend":
            current_backend = _choose_backend()
            _print_chat_header(current_backend)
            continue
        if message == "/model":
            model = _choose_model()
            loaded_backend = _load_model(commands.ModelLoadOptions(model=str(model)))
            if loaded_backend:
                current_backend = loaded_backend
            _print_chat_header(current_backend)
            continue
        if message == "/clear":
            _clear()
            _print_header("OmniInfer", "Local inference console")
            _print_chat_header(current_backend)
            continue
        if message == "/help":
            _print_help()
            print()
            continue

        print(_THEME.role_assistant("Assistant"))
        final_payload: dict[str, Any] | None = None
        buffer = ""
        visible_started = False
        for chunk in commands.iter_chat_stream(commands.ChatOptions(message=message)):
            if chunk.text:
                buffer += chunk.text
                output, buffer, visible_started = _consume_visible_text(buffer, visible_started)
                if output:
                    sys.stdout.write(output)
                    sys.stdout.flush()
            if chunk.final_payload:
                final_payload = chunk.final_payload
        if buffer:
            output, _buffer, _visible_started = _consume_visible_text(buffer, visible_started, final=True)
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()
        print()
        if final_payload:
            _print_short_performance(final_payload)
        print()
        _ = current_backend


def _print_short_performance(payload: dict[str, Any]) -> None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    pieces: list[str] = []
    if usage:
        pieces.append(f"tokens={usage.get('total_tokens', '-')}")
    if isinstance(timings, dict) and "predicted_per_second" in timings:
        pieces.append(f"speed={timings.get('predicted_per_second')} tok/s")
    if pieces:
        print(_THEME.dim("[" + ", ".join(pieces) + "]"))


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
    commands_text = " /backend  /model  /clear  /help  /exit "
    print(_THEME.dim("Commands") + _THEME.accent(commands_text))


def _print_chat_header(backend: str) -> None:
    _print_section("Chat", f"Backend: {backend}")
    _print_command_bar()
    print()


def _print_help() -> None:
    _print_section("Help", "Conversation commands")
    commands_table = [
        ("/backend", "switch the selected runtime"),
        ("/model", "load a different managed model"),
        ("/clear", "clear the terminal and redraw the chat header"),
        ("/help", "show this command reference"),
        ("/exit", "stop the OmniInfer service and leave the TUI"),
    ]
    for name, description in commands_table:
        _print_kv(name, description)


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


def _default_selected_index(rows: list[dict[str, Any]]) -> str:
    for index, item in enumerate(rows, 1):
        if item.get("selected"):
            return str(index)
    return "1"


def _clear() -> None:
    command = "cls" if os.name == "nt" else "clear"
    if os.environ.get("TERM"):
        os.system(command)
