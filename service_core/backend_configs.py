from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from service_core.backends import BackendSpec
from service_core.platforms.common import parse_extra_args


PROFILE_SCHEMA_VERSION = 2
PROFILE_DIR = Path.home() / ".config" / "omniinfer" / "backend_profiles"


@dataclass
class BackendProfile:
    path: Path
    backend_id: str | None
    family: str | None
    load_extra_args: list[str]
    infer_extra_args: list[str]


def profile_path_for_backend(backend_id: str) -> Path:
    return PROFILE_DIR / f"{backend_id}.json"


def ensure_backend_profile_template(backend: BackendSpec) -> tuple[Path, bool]:
    path = profile_path_for_backend(backend.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path, False
    path.write_text(
        json.dumps(build_backend_profile_template(backend), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, True


def load_backend_profile(path_text: str) -> BackendProfile:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"backend config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"backend config must be a JSON object: {path}")

    load_section = payload.get("load") if isinstance(payload.get("load"), dict) else {}
    infer_section = payload.get("infer") if isinstance(payload.get("infer"), dict) else {}

    load_extra_args = parse_extra_args(load_section.get("extra_args"))
    infer_extra_args = parse_extra_args(infer_section.get("extra_args"))

    if not load_extra_args and "launcher_args" in load_section:
        load_extra_args = parse_extra_args(load_section.get("launcher_args"))

    infer_legacy_args: list[str] = []
    for key in ("temperature", "max_tokens", "stream", "think"):
        if key in infer_section:
            value = infer_section.get(key)
            if isinstance(value, bool):
                infer_legacy_args.append(f"--{key.replace('_', '-')}")
                if key in {"stream", "think"}:
                    infer_legacy_args.append("true" if value else "false")
            else:
                infer_legacy_args.extend([f"--{key.replace('_', '-')}", str(value)])
    request_overrides = infer_section.get("request_overrides")
    if isinstance(request_overrides, dict):
        for key, value in request_overrides.items():
            flag = f"--{str(key).replace('_', '-')}"
            if isinstance(value, bool):
                infer_legacy_args.extend([flag, "true" if value else "false"])
            elif isinstance(value, list):
                for item in value:
                    infer_legacy_args.extend([flag, str(item)])
            else:
                infer_legacy_args.extend([flag, str(value)])
    if not infer_extra_args and infer_legacy_args:
        infer_extra_args = infer_legacy_args

    _validate_profile_extra_args(path, load_extra_args, infer_extra_args)

    return BackendProfile(
        path=path,
        backend_id=_optional_string(payload.get("backend")),
        family=_optional_string(payload.get("family")),
        load_extra_args=load_extra_args,
        infer_extra_args=infer_extra_args,
    )


def build_backend_profile_template(backend: BackendSpec) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "backend": backend.id,
        "family": backend.family,
        "description": (
            "Advanced backend-native parameters for OmniInfer. Keep basic user inputs such as "
            "model path, message, image, and mmproj on the CLI, and only store backend-specific "
            "extra parameters here."
        ),
        "load": {
            "extra_args": [],
        },
        "infer": {
            "extra_args": [],
        },
    }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_profile_extra_args(path: Path, load_extra_args: list[str], infer_extra_args: list[str]) -> None:
    reserved_load = {"-m", "--model", "-mm", "--mmproj"}
    reserved_infer = {"-m", "--model", "-mm", "--mmproj", "-p", "--prompt", "--message", "--image"}
    for token in load_extra_args:
        flag = token.split("=", 1)[0]
        if flag in reserved_load:
            raise ValueError(
                f"{path}: backend config load.extra_args must not contain {flag}; "
                "keep basic model and mmproj inputs on the CLI"
            )
    for token in infer_extra_args:
        flag = token.split("=", 1)[0]
        if flag in reserved_infer:
            raise ValueError(
                f"{path}: backend config infer.extra_args must not contain {flag}; "
                "keep basic message/image/model inputs on the CLI"
            )
