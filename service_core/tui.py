from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from service_core import commands


def run_tui() -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("OmniInfer TUI requires an interactive terminal.")

    _clear()
    print("OmniInfer")
    print()
    backend = _choose_backend()
    model = _choose_model()
    _load_model(model)
    _chat_loop(backend)
    return 0


def _choose_backend() -> str:
    while True:
        payload = commands.list_backends(scope="installed")
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        if not rows:
            raise SystemExit("No installed backends are available.")

        print("Backends")
        for index, item in enumerate(rows, 1):
            marker = "*" if item.get("selected") else " "
            installed = "installed" if item.get("binary_exists") else "not installed"
            print(f"{index:>2}. {marker} {item.get('id')} ({installed})")
        print()
        choice = _prompt("Select backend", default=_default_selected_index(rows))
        index = _parse_choice(choice, len(rows))
        if index is None:
            print("Invalid selection.")
            continue
        backend_id = str(rows[index - 1].get("id", ""))
        if not backend_id:
            print("Invalid backend.")
            continue
        result = commands.select_backend(backend_id)
        print(f"Selected backend: {result.backend}")
        if result.models_dir:
            print(f"Models directory: {result.models_dir}")
        print()
        return backend_id


def _choose_model() -> Path:
    while True:
        models = commands.discover_local_models()
        print("Models")
        if models:
            for index, model in enumerate(models, 1):
                print(f"{index:>2}. {model.label}")
            print(" 0. Enter path manually")
            choice = _prompt("Select model", default="1")
            if choice == "0":
                return _prompt_model_path()
            index = _parse_choice(choice, len(models))
            if index is not None:
                return models[index - 1].path
            print("Invalid selection.")
            continue

        print("No models found in OmniInfer .local model directories.")
        return _prompt_model_path()


def _prompt_model_path() -> Path:
    while True:
        text = _prompt("Model path")
        path = Path(text).expanduser().resolve()
        if path.exists():
            try:
                linked = commands.link_model_into_managed_models(path)
            except OSError as exc:
                print(f"Could not link model into {commands.managed_models_dir()}: {exc}")
                return path
            if linked != path:
                print(f"Linked model: {linked}")
            return linked
        print(f"Model path does not exist: {path}")


def _load_model(model: Path) -> None:
    print()
    print(f"Loading model: {model}")

    def on_progress(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        message = event.get("message")
        if event_type == "status" and message:
            print(f"  {message}")
        elif event_type == "log" and message:
            return
        elif event_type == "done":
            elapsed = event.get("elapsed_s")
            print(f"  Backend ready ({elapsed}s)" if elapsed is not None else "  Backend ready")

    response, selection = commands.load_model(
        commands.ModelLoadOptions(model=str(model)),
        progress=on_progress,
    )
    if selection.auto_selected:
        print(f"Auto-selected backend: {selection.backend}")
    print(f"Model loaded: {response.get('selected_model') or model}")
    print()


def _chat_loop(backend: str) -> None:
    print("Chat")
    print("Commands: /backend, /model, /exit")
    print()
    current_backend = backend
    while True:
        message = _prompt("You")
        if not message:
            continue
        if message == "/exit":
            return
        if message == "/backend":
            current_backend = _choose_backend()
            continue
        if message == "/model":
            model = _choose_model()
            _load_model(model)
            continue

        print("Assistant")
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
        print("[" + ", ".join(pieces) + "]")


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


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


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
