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
    return local_config_dir(app_root) / "state.json"


def legacy_state_file(app_root: Path | None = None) -> Path:
    return local_config_dir(app_root) / "cli_state.json"


def load_state(app_root: Path | None = None) -> dict[str, Any]:
    path = state_file(app_root)
    legacy_path = legacy_state_file(app_root)
    source = path
    should_migrate = False
    if not source.is_file():
        if not legacy_path.is_file():
            return {}
        source = legacy_path
        should_migrate = True
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if should_migrate:
        save_state(payload, app_root)
    return payload


def save_state(payload: dict[str, Any], app_root: Path | None = None) -> None:
    target = state_file(app_root)
    legacy = legacy_state_file(app_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        tmp.replace(target)
        if legacy.is_file():
            legacy.unlink()
    finally:
        if tmp.exists():
            tmp.unlink()


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


def load_selected_model(app_root: Path | None = None) -> dict[str, Any] | None:
    payload = load_state(app_root)
    model = payload.get("selected_model")
    if model is None or not str(model).strip():
        return None
    result: dict[str, Any] = {"model": str(model).strip()}
    mmproj = payload.get("selected_mmproj")
    if mmproj is not None and str(mmproj).strip():
        result["mmproj"] = str(mmproj).strip()
    ctx_size = payload.get("selected_ctx_size")
    if isinstance(ctx_size, int) and ctx_size > 0:
        result["ctx_size"] = ctx_size
    return result


def save_selected_model(
    model: str,
    app_root: Path | None = None,
    *,
    mmproj: str | None = None,
    ctx_size: int | None = None,
) -> None:
    text = str(model).strip()
    if not text:
        return
    payload = load_state(app_root)
    payload["selected_model"] = text
    mmproj_text = str(mmproj).strip() if mmproj is not None else ""
    if mmproj_text:
        payload["selected_mmproj"] = mmproj_text
    else:
        payload.pop("selected_mmproj", None)
    if ctx_size is not None and ctx_size > 0:
        payload["selected_ctx_size"] = int(ctx_size)
    else:
        payload.pop("selected_ctx_size", None)
    save_state(payload, app_root)
