from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


TRYCLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
DEFAULT_QUICK_TUNNEL_TIMEOUT_S = 30


class RemoteAccessError(RuntimeError):
    """Raised when a remote access tunnel cannot be started safely."""


@dataclass
class QuickTunnelHandle:
    process: subprocess.Popen[str]
    public_url: str
    log_lines: list[str] = field(default_factory=list)

    def stop(self, timeout_s: float = 5.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout_s)


def parse_trycloudflare_url(line: str) -> str | None:
    match = TRYCLOUDFLARE_URL_RE.search(line)
    if not match:
        return None
    return match.group(0)


def _candidate_cloudflared_paths(explicit_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("OMNIINFER_CLOUDFLARED", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    path_match = shutil.which("cloudflared")
    if path_match:
        candidates.append(Path(path_match))

    if os.name == "nt":
        candidates.extend(
            [
                Path(r"D:\Program\Network\cloudflared\cloudflared.exe"),
                Path(r"C:\Program Files\cloudflared\cloudflared.exe"),
            ]
        )
    return candidates


def find_cloudflared(explicit_path: str | None = None) -> Path:
    for candidate in _candidate_cloudflared_paths(explicit_path):
        if candidate.is_file():
            return candidate
    raise RemoteAccessError(
        "cloudflared was not found. Install cloudflared, add it to PATH, "
        "set OMNIINFER_CLOUDFLARED, or pass --cloudflared-path."
    )


def default_cloudflared_config_path() -> Path:
    if os.name == "nt":
        home = Path(os.environ.get("USERPROFILE") or str(Path.home()))
    else:
        home = Path.home()
    return home / ".cloudflared" / "config.yaml"


def quick_tunnel_config_warning() -> str | None:
    config_path = default_cloudflared_config_path()
    if not config_path.exists():
        return None
    return (
        f"Cloudflare Quick Tunnel may fail because {config_path} exists. "
        "Quick Tunnel is intended to run without a default cloudflared config file."
    )


def _reader_thread(stream: object, output: queue.Queue[str], sink: list[str]) -> None:
    try:
        for raw_line in stream:  # type: ignore[operator]
            line = str(raw_line).rstrip()
            sink.append(line)
            output.put(line)
    finally:
        try:
            stream.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def start_cloudflare_quick_tunnel(
    cloudflared: Path,
    local_url: str,
    timeout_s: int = DEFAULT_QUICK_TUNNEL_TIMEOUT_S,
) -> QuickTunnelHandle:
    command = [str(cloudflared), "tunnel", "--url", local_url]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_lines: list[str] = []
    line_queue: queue.Queue[str] = queue.Queue()
    threads = [
        threading.Thread(target=_reader_thread, args=(process.stdout, line_queue, log_lines), daemon=True),
        threading.Thread(target=_reader_thread, args=(process.stderr, line_queue, log_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_lines[-10:])
            raise RemoteAccessError(f"cloudflared exited before creating a Quick Tunnel.\n{tail}".rstrip())
        try:
            line = line_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        public_url = parse_trycloudflare_url(line)
        if public_url:
            return QuickTunnelHandle(process=process, public_url=public_url, log_lines=log_lines)

    handle = QuickTunnelHandle(process=process, public_url="", log_lines=log_lines)
    handle.stop()
    tail = "\n".join(log_lines[-10:])
    raise RemoteAccessError(f"Timed out waiting for Cloudflare Quick Tunnel URL.\n{tail}".rstrip())
