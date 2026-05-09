from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def local_dir(app_root: Path | None = None) -> Path:
    return (app_root or project_root()) / ".local"


def local_config_dir(app_root: Path | None = None) -> Path:
    return local_dir(app_root) / "config"


def local_logs_dir(app_root: Path | None = None) -> Path:
    return local_dir(app_root) / "logs"


def backend_profile_dir(app_root: Path | None = None) -> Path:
    return local_config_dir(app_root) / "backend_profiles"


def state_file(app_root: Path | None = None) -> Path:
    return local_config_dir(app_root) / "cli_state.json"


def load_state(app_root: Path | None = None) -> dict[str, Any]:
    path = state_file(app_root)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(payload: dict[str, Any], app_root: Path | None = None) -> None:
    target = state_file(app_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(target)


def load_selected_backend(app_root: Path | None = None) -> str | None:
    value = load_state(app_root).get("selected_backend")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def save_selected_backend(name: str, app_root: Path | None = None) -> None:
    text = str(name).strip()
    if not text:
        return
    payload = load_state(app_root)
    payload["selected_backend"] = text
    save_state(payload, app_root)
