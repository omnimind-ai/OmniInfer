from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import platform
import re
import tarfile
import tempfile
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TRYCLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
CLOUDFLARED_VERSION_RE = re.compile(r"cloudflared version ([^\s]+)")
CLOUDFLARED_RELEASE_API_URL = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
DEFAULT_QUICK_TUNNEL_TIMEOUT_S = 30
DEFAULT_CLOUDFLARED_DOWNLOAD_TIMEOUT_S = 120


class RemoteAccessError(RuntimeError):
    """Raised when a remote access tunnel cannot be started safely."""


@dataclass(frozen=True)
class CloudflaredReleaseInfo:
    tag_name: str
    asset_name: str
    download_url: str
    digest: str | None


@dataclass(frozen=True)
class CloudflaredBinary:
    path: Path
    source: str
    version: str | None = None
    asset_name: str | None = None
    digest: str | None = None
    updated: bool = False


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


def _normalized_machine() -> str:
    return platform.machine().strip().lower().replace(" ", "")


def cloudflared_asset_name_for_current_platform() -> str:
    system = platform.system().strip().lower()
    machine = _normalized_machine()

    if system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "cloudflared-linux-amd64"
        if machine in {"i386", "i686", "x86"}:
            return "cloudflared-linux-386"
        if machine in {"aarch64", "arm64"}:
            return "cloudflared-linux-arm64"
        if machine in {"armv7l", "armv7", "armv8l"}:
            return "cloudflared-linux-armhf"
        if machine.startswith("arm"):
            return "cloudflared-linux-arm"
    elif system == "darwin":
        if machine in {"arm64", "aarch64"}:
            return "cloudflared-darwin-arm64.tgz"
        if machine in {"x86_64", "amd64"}:
            return "cloudflared-darwin-amd64.tgz"
    elif system == "windows":
        if machine in {"amd64", "x86_64"}:
            return "cloudflared-windows-amd64.exe"
        if machine in {"x86", "i386", "i686"}:
            return "cloudflared-windows-386.exe"

    raise RemoteAccessError(f"unsupported cloudflared platform: {platform.system()} {platform.machine()}")


def _managed_cloudflared_dir(app_root: Path) -> Path:
    return app_root / ".local" / "tools" / "cloudflared"


def _managed_cloudflared_path(app_root: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _managed_cloudflared_dir(app_root) / f"cloudflared{suffix}"


def _managed_cloudflared_manifest_path(app_root: Path) -> Path:
    return _managed_cloudflared_dir(app_root) / "manifest.json"


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _cloudflared_version(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = f"{result.stdout}\n{result.stderr}"
    match = CLOUDFLARED_VERSION_RE.search(output)
    return match.group(1) if match else None


def _load_managed_cloudflared(app_root: Path) -> CloudflaredBinary | None:
    path = _managed_cloudflared_path(app_root)
    if not path.is_file():
        return None
    manifest = _read_json_file(_managed_cloudflared_manifest_path(app_root))
    version = str(manifest.get("version") or "").strip() or _cloudflared_version(path)
    asset_name = str(manifest.get("asset_name") or "").strip() or None
    digest = str(manifest.get("digest") or "").strip() or None
    return CloudflaredBinary(path=path, source="managed", version=version, asset_name=asset_name, digest=digest)


def _github_request(url: str, *, timeout_s: int) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "OmniInfer",
        },
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        raise RemoteAccessError(f"unable to query cloudflared releases: {exc}") from exc


def fetch_latest_cloudflared_release(timeout_s: int = 30) -> CloudflaredReleaseInfo:
    data = _github_request(CLOUDFLARED_RELEASE_API_URL, timeout_s=timeout_s)
    if not isinstance(data, dict):
        raise RemoteAccessError("invalid cloudflared release response")
    tag_name = str(data.get("tag_name") or "").strip()
    assets = data.get("assets")
    if not tag_name or not isinstance(assets, list):
        raise RemoteAccessError("invalid cloudflared release metadata")

    expected_name = cloudflared_asset_name_for_current_platform()
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("name") != expected_name:
            continue
        download_url = str(asset.get("browser_download_url") or "").strip()
        if not download_url:
            break
        digest = str(asset.get("digest") or "").strip() or None
        return CloudflaredReleaseInfo(
            tag_name=tag_name,
            asset_name=expected_name,
            download_url=download_url,
            digest=digest,
        )

    raise RemoteAccessError(f"cloudflared release {tag_name} has no asset named {expected_name}")


def _download_bytes(url: str, timeout_s: int = DEFAULT_CLOUDFLARED_DOWNLOAD_TIMEOUT_S) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "OmniInfer",
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return response.read()
        except (HTTPError, URLError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RemoteAccessError(f"unable to download cloudflared: {last_error}") from last_error


def _verify_digest(data: bytes, digest: str | None) -> None:
    if not digest:
        return
    if not digest.startswith("sha256:"):
        raise RemoteAccessError(f"unsupported cloudflared digest format: {digest}")
    expected = digest.split(":", 1)[1].lower()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RemoteAccessError("cloudflared download failed SHA-256 verification")


def _write_executable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), prefix=".cloudflared-") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    if os.name != "nt":
        tmp_path.chmod(0o755)
    tmp_path.replace(path)


def _extract_cloudflared_from_tgz(path: Path, data: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = next((item for item in archive.getmembers() if item.isfile() and Path(item.name).name == "cloudflared"), None)
        if member is None:
            raise RemoteAccessError("cloudflared archive did not contain a cloudflared binary")
        stream = archive.extractfile(member)
        if stream is None:
            raise RemoteAccessError("cloudflared archive member could not be read")
        _write_executable(path, stream.read())


def install_managed_cloudflared(app_root: Path, release: CloudflaredReleaseInfo) -> CloudflaredBinary:
    data = _download_bytes(release.download_url)
    _verify_digest(data, release.digest)

    path = _managed_cloudflared_path(app_root)
    if release.asset_name.endswith(".tgz"):
        _extract_cloudflared_from_tgz(path, data)
    else:
        _write_executable(path, data)

    version = _cloudflared_version(path) or release.tag_name
    manifest = {
        "version": version,
        "release": release.tag_name,
        "asset_name": release.asset_name,
        "download_url": release.download_url,
        "digest": release.digest,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = _managed_cloudflared_manifest_path(app_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CloudflaredBinary(
        path=path,
        source="managed",
        version=version,
        asset_name=release.asset_name,
        digest=release.digest,
        updated=True,
    )


def _managed_cloudflared_matches(release: CloudflaredReleaseInfo, current: CloudflaredBinary | None) -> bool:
    if current is None or not current.path.is_file():
        return False
    if current.version != release.tag_name and current.version != release.tag_name.lstrip("v"):
        return False
    if current.asset_name and current.asset_name != release.asset_name:
        return False
    if current.digest and release.digest and current.digest != release.digest:
        return False
    return True


def resolve_cloudflared_for_quick_tunnel(app_root: Path, explicit_path: str | None = None) -> CloudflaredBinary:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise RemoteAccessError(f"cloudflared was not found at {path}")
        return CloudflaredBinary(path=path, source="explicit", version=_cloudflared_version(path))

    current = _load_managed_cloudflared(app_root)
    try:
        latest = fetch_latest_cloudflared_release()
    except RemoteAccessError:
        if current is not None and current.path.is_file():
            return current
        raise

    if _managed_cloudflared_matches(latest, current):
        return current  # type: ignore[return-value]

    return install_managed_cloudflared(app_root, latest)


def find_cloudflared(explicit_path: str | None = None, *, app_root: Path | None = None) -> Path:
    root = Path.cwd() if app_root is None else app_root
    return resolve_cloudflared_for_quick_tunnel(root, explicit_path).path


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
