#!/usr/bin/env python3
"""OmniInfer unified API service entrypoint."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


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
        if getattr(sys, "frozen", False):
            command = [sys.executable, *argv[1:]]
            workdir = str(Path(sys.executable).resolve().parent)
        else:
            script_path = str(Path(__file__).resolve())
            command = [sys.executable, script_path, *argv[1:]]
            workdir = str(Path(__file__).resolve().parent)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            command,
            env=env,
            creationflags=creationflags,
            cwd=workdir,
        )
        raise SystemExit(0)

    if mode == "visible" and not _has_console():
        if ctypes.windll.kernel32.AllocConsole():
            _attach_console_streams()


def _run() -> int:
    _ensure_window_mode(sys.argv)
    from service_core.service import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run())
