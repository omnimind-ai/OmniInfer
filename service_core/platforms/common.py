from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_MODEL_LIST_URLS: dict[str, str] = {
    "windows": "https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/windows/model_list.json",
    "mac": "https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/mac/model_list.json",
    "linux": "https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/linux/model_list.json",
}


def resolve_input_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def display_path_reference(path: str, root_dir: str | None) -> str:
    target = Path(path).resolve()
    if not root_dir:
        return str(target)
    root = Path(root_dir).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    current = env.get(key, "").strip()
    env[key] = value if not current else f"{value}{os.pathsep}{current}"


def wait_http_ready(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.getcode() in (200, 503):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def pick_available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def hidden_subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def get_available_memory_bytes() -> int:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("sullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            page_size = 4096
            counts: dict[str, int] = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "page size of" in line:
                    match = re.search(r"page size of (\d+) bytes", line)
                    if match:
                        page_size = int(match.group(1))
                    continue
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                digits = "".join(ch for ch in raw_value if ch.isdigit())
                if digits:
                    counts[key.strip()] = int(digits)

            available_pages = (
                counts.get("Pages free", 0)
                + counts.get("Pages inactive", 0)
                + counts.get("Pages speculative", 0)
            )
            if available_pages > 0:
                return int(page_size * available_pages)

    page_size = os.sysconf("SC_PAGE_SIZE")
    avail_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * avail_pages)


def get_available_cuda_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=True,
            **hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    free_mib_values: list[int] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            free_mib_values.append(int(value))
        except ValueError:
            continue

    if not free_mib_values:
        return None

    return max(free_mib_values) * 1024 * 1024


def get_available_rocm_memory_bytes() -> int | None:
    candidate_commands = [
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        ["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram", "--json"],
    ]
    for rocm_dir in sorted(Path("/opt").glob("rocm-*")):
        candidate_commands.append([str(rocm_dir / "bin" / "rocm-smi"), "--showmeminfo", "vram", "--json"])

    for command in candidate_commands:
        tool = Path(command[0])
        if "/" in command[0] and not tool.is_file():
            continue
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=True,
                **hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            continue

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        free_bytes: list[int] = []
        for row in payload.values():
            if not isinstance(row, dict):
                continue
            total = parse_optional_int(row.get("VRAM Total Memory (B)"))
            used = parse_optional_int(row.get("VRAM Total Used Memory (B)"))
            if total is None or used is None:
                continue
            free_bytes.append(max(total - used, 0))

        if free_bytes:
            return max(free_bytes)

    return None


def bytes_to_gib(value: int) -> float:
    return round(float(value) / float(1024 ** 3), 2)


def parse_size_gib(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_extra_args(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return shlex.split(value, posix=False)
    return []


def current_system_name() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system.startswith("darwin") or system.startswith("mac"):
        return "mac"
    if system.startswith("linux"):
        return "linux"
    raise ValueError(f"unsupported host system: {platform.system()}")


def maybe_auto_mmproj(models_dir: str | None, model_path: str) -> str | None:
    model_file = Path(model_path)
    sibling_candidates = [
        model_file.with_name("mmproj-F32.gguf"),
        model_file.with_name("mmproj-f32.gguf"),
        model_file.with_name("mmproj-F16.gguf"),
        model_file.with_name("mmproj-f16.gguf"),
    ]
    for candidate in sibling_candidates:
        if candidate.is_file():
            return str(candidate)

    if not models_dir:
        return None

    root = Path(models_dir)
    flat_candidates = [
        root / "mmproj-F32.gguf",
        root / "mmproj-f32.gguf",
        root / "mmproj-F16.gguf",
        root / "mmproj-f16.gguf",
    ]
    for candidate in flat_candidates:
        if candidate.is_file():
            return str(candidate)
    return None
