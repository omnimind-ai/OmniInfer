"""Version helpers for OmniInfer."""

from __future__ import annotations

import os
import re
import sys
from importlib import metadata
from pathlib import Path

_FALLBACK_VERSION = "0.3.0"


def _version_from_pyproject() -> str | None:
    if getattr(sys, "frozen", False):
        return None
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        return None
    return match.group(1)


def get_omniinfer_version() -> str:
    env_version = os.environ.get("OMNIINFER_VERSION", "").strip()
    if env_version:
        return env_version

    source_version = _version_from_pyproject()
    if source_version:
        return source_version

    try:
        return metadata.version("omniinfer")
    except metadata.PackageNotFoundError:
        return _FALLBACK_VERSION
