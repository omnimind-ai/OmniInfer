"""Centralized logging for OmniInfer.

Usage in every module::

    import logging
    logger = logging.getLogger("runtime")  # "gateway", "platform", "driver.mnn", etc.

Call ``setup_logging()`` once at process startup before any log calls.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import Any

_LOG_FILE = "omniinfer.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False
_active_log_file: Path | None = None

# Project root: resolved from this file's location (service_core/logger.py -> ..),
# or from the executable path when frozen (PyInstaller).
if getattr(sys, "frozen", False):
    _PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_log_dir() -> Path:
    """Return the default log directory under the project tree.

    Uses ``{PROJECT_ROOT}/.local/logs/``, consistent with the existing
    ``.local/runtime/`` convention.  Works on all platforms (Linux, macOS,
    Windows, Android/Termux) because it lives inside the project directory
    which the user already has write access to.
    """
    return _PROJECT_ROOT / ".local" / "logs"


def setup_logging(
    *,
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Path | None = None,
    console: bool = True,
) -> Path | None:
    """Initialize the root logger.  Returns the log file path, or *None*."""
    global _initialized, _active_log_file
    if _initialized:
        return _active_log_file
    _initialized = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root captures everything; handlers filter
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(numeric_level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if log_to_file:
        try:
            target_dir = log_dir or _default_log_dir()
            target_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = target_dir / _LOG_FILE
            fh = logging.handlers.RotatingFileHandler(
                str(log_file_path),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)  # file always captures full detail
            fh.setFormatter(formatter)
            root.addHandler(fh)
            _active_log_file = log_file_path
        except OSError:
            # Android, sandboxed, or restricted FS -- fall back to console only
            pass

    return _active_log_file


def log_session_header(
    *,
    version: str = "0.2",
    config: dict[str, Any] | None = None,
    backends: list[str] | None = None,
) -> None:
    """Emit a startup diagnostics block."""
    lg = logging.getLogger("startup")
    lg.info("=" * 60)
    lg.info("OmniInfer %s starting", version)
    lg.info("Python %s", sys.version.replace("\n", " "))
    lg.info("OS: %s %s %s", platform.system(), platform.release(), platform.machine())

    # RAM
    try:
        from service_core.platforms.common import (
            bytes_to_gib,
            get_available_memory_bytes,
        )

        avail = get_available_memory_bytes()
        if avail is not None:
            lg.info("RAM available: %.2f GiB", bytes_to_gib(avail))
    except Exception:
        pass

    _log_gpu_info(lg)

    if config:
        lg.info(
            "Config: host=%s port=%s default_backend=%s",
            config.get("host"),
            config.get("port"),
            config.get("default_backend"),
        )
    if backends:
        lg.info("Available backends: %s", ", ".join(backends))

    if _active_log_file:
        lg.info("Log file: %s", _active_log_file)
    else:
        lg.info("Log file: (disabled or unavailable)")
    lg.info("=" * 60)


def _log_gpu_info(lg: logging.Logger) -> None:
    try:
        from service_core.platforms.common import (
            bytes_to_gib,
            get_available_cuda_memory_bytes,
        )

        cuda_bytes = get_available_cuda_memory_bytes()
        if cuda_bytes is not None:
            lg.info("CUDA GPU memory free: %.2f GiB", bytes_to_gib(cuda_bytes))
    except Exception:
        pass

    try:
        from service_core.platforms.common import (
            bytes_to_gib,
            get_available_rocm_memory_bytes,
        )

        rocm_bytes = get_available_rocm_memory_bytes()
        if rocm_bytes is not None:
            lg.info("ROCm GPU memory free: %.2f GiB", bytes_to_gib(rocm_bytes))
    except Exception:
        pass

    if platform.system() == "Darwin":
        lg.info("GPU: Apple Metal (unified memory)")


def resolve_log_level(
    *,
    verbose: bool = False,
    debug_body: bool = False,
    env_var: str = "OMNIINFER_LOG_LEVEL",
) -> str:
    """Resolve effective log level from CLI flags and env var."""
    env_level = os.environ.get(env_var, "").strip().upper()
    if env_level and hasattr(logging, env_level):
        return env_level
    if debug_body:
        return "DEBUG"
    if verbose:
        return "DEBUG"
    return "INFO"
