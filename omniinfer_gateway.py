#!/usr/bin/env python3
"""
OmniInfer Gateway

Based on the validated OmniServer MVP from E:\\Coding\\repository\\llama.cpp\\dist.

Goals:
1) Select backend via API (currently only: llama.cpp(CPU))
2) Select model via API and load it
3) Call OpenAI-compatible /v1/chat/completions

Implementation note:
- This gateway controls a local llama-server process as backend runtime.
- It exposes management APIs under /omni/* and inference APIs under /v1/*.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "platform" / "Windows" / "llama.cpp-CPU"
DEFAULT_LLAMA_SERVER_PATH = PACKAGE_ROOT / "bin" / "llama-server.exe"
DEFAULT_MODELS_DIR = PACKAGE_ROOT / "models"

SUPPORTED_BACKENDS = ["llama.cpp(CPU)"]
DEFAULT_BACKEND = SUPPORTED_BACKENDS[0]


@dataclass
class ServerState:
    lock: threading.Lock
    selected_backend: str | None
    selected_model: str | None
    selected_mmproj: str | None
    backend_proc: subprocess.Popen[Any] | None
    backend_host: str
    backend_port: int
    backend_llama_path: str
    models_dir: str
    request_timeout_s: int


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def resolve_input_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def display_path_reference(path: str, root_dir: str) -> str:
    target = Path(path).resolve()
    root = Path(root_dir).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[int, bytes]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return 599, json_dumps({"error": {"message": f"backend unreachable: {e}"}})


def stream_http_request(
    handler: BaseHTTPRequestHandler,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> None:
    headers = {"Accept": "text/event-stream, application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, data=body, method=method, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            handler.send_response(resp.getcode())
            handler.send_header(
                "Content-Type",
                resp.headers.get("Content-Type", "text/event-stream; charset=utf-8"),
            )
            handler.send_header("Cache-Control", resp.headers.get("Cache-Control", "no-cache"))
            handler.send_header("Connection", "close")
            handler.send_header("X-Accel-Buffering", "no")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            handler.end_headers()

            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except urllib.error.HTTPError as e:
        body = e.read()
        handler.send_response(e.code)
        handler.send_header(
            "Content-Type",
            e.headers.get("Content-Type", "application/json; charset=utf-8"),
        )
        handler.send_header("Connection", "close")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
    except urllib.error.URLError as e:
        body = json_dumps({"error": {"message": f"backend unreachable: {e}"}})
        handler.send_response(599)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Connection", "close")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return


def is_gguf_model(filename: str) -> bool:
    low = filename.lower()
    return low.endswith(".gguf") and "mmproj" not in low


def list_model_entries(models_dir: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    root = Path(models_dir)
    if not root.is_dir():
        return entries

    for path in sorted(root.rglob("*.gguf")):
        if not path.is_file():
            continue
        if "mmproj" in path.name.lower():
            continue
        rel = path.relative_to(root).as_posix()
        entries.append({"id": rel, "path": str(path)})
    return entries


def maybe_auto_mmproj(models_dir: str, model_name: str, model_path: str) -> str | None:
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

    low = model_name.lower()
    root = Path(models_dir)
    flat_candidates: list[Path] = []
    if "qwen3.5-4b" in low:
        flat_candidates.append(root / "mmproj-Qwen3.5-4B-F32.gguf")
    if "qwen3.5-0.8b" in low:
        flat_candidates.append(root / "mmproj-Qwen3.5-0.8B-F32.gguf")
    flat_candidates.append(root / "mmproj-F32.gguf")

    for candidate in flat_candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def stop_backend(state: ServerState) -> None:
    if state.backend_proc is None:
        return

    proc = state.backend_proc
    state.backend_proc = None

    if proc.poll() is None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass


def wait_backend_ready(host: str, port: int, timeout_s: int) -> bool:
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


def start_backend(state: ServerState, model_path: str, mmproj_path: str | None) -> tuple[bool, str]:
    if not os.path.isfile(state.backend_llama_path):
        return False, f"llama-server not found: {state.backend_llama_path}"

    cmd = [
        state.backend_llama_path,
        "-m",
        model_path,
        "--host",
        state.backend_host,
        "--port",
        str(state.backend_port),
        "--no-webui",
    ]
    if mmproj_path:
        cmd.extend(["-mm", mmproj_path])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(state.backend_llama_path).resolve().parent),
        )
    except OSError as e:
        return False, f"failed to start backend: {e}"

    state.backend_proc = proc
    if not wait_backend_ready(state.backend_host, state.backend_port, state.request_timeout_s):
        msg = "backend did not become ready in time"
        if proc.poll() is not None and proc.stdout:
            lines: list[str] = []
            for _ in range(60):
                line = proc.stdout.readline()
                if not line:
                    break
                lines.append(line.rstrip())
            if lines:
                msg += "; backend output: " + " | ".join(lines[-6:])
        stop_backend(state)
        return False, msg

    return True, "ok"


class OmniHandler(BaseHTTPRequestHandler):
    server_version = "OmniInferGateway/0.1"

    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return {}

    def _omni_state_payload(self) -> dict[str, Any]:
        with self.state.lock:
            proc = self.state.backend_proc
            running = bool(proc and proc.poll() is None)
            return {
                "backend": self.state.selected_backend,
                "model": self.state.selected_model,
                "mmproj": self.state.selected_mmproj,
                "backend_runtime": {
                    "host": self.state.backend_host,
                    "port": self.state.backend_port,
                    "running": running,
                    "pid": proc.pid if running and proc else None,
                },
            }

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = {"status": "ok", "omni": self._omni_state_payload()}
            self._send_json(200, payload)
            return

        if self.path == "/omni/state":
            self._send_json(200, self._omni_state_payload())
            return

        if self.path == "/omni/backends":
            with self.state.lock:
                selected = self.state.selected_backend
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": x, "selected": x == selected} for x in SUPPORTED_BACKENDS],
                },
            )
            return

        if self.path.startswith("/omni/models"):
            try:
                data = list_model_entries(self.state.models_dir)
            except OSError as e:
                self._send_json(500, {"error": {"message": f"cannot list models dir: {e}"}})
                return
            self._send_json(200, {"object": "list", "data": data})
            return

        if self.path == "/v1/models":
            with self.state.lock:
                running = bool(self.state.backend_proc and self.state.backend_proc.poll() is None)
                host, port = self.state.backend_host, self.state.backend_port
            if running:
                code, body = http_json("GET", f"http://{host}:{port}/v1/models", timeout=10)
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                return

            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [],
                    "meta": {"message": "no model loaded yet; call /omni/model/select first"},
                },
            )
            return

        self._send_json(404, {"error": {"message": f"not found: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/omni/backend/select":
            payload = self._read_json()
            backend = str(payload.get("backend", "")).strip()
            if backend not in SUPPORTED_BACKENDS:
                self._send_json(400, {"error": {"message": f"unsupported backend: {backend}"}})
                return
            with self.state.lock:
                self.state.selected_backend = backend
            self._send_json(200, {"ok": True, "selected_backend": backend})
            return

        if self.path == "/omni/backend/stop":
            with self.state.lock:
                stop_backend(self.state)
            self._send_json(200, {"ok": True, "stopped": True})
            return

        if self.path == "/omni/model/select":
            payload = self._read_json()
            model = str(payload.get("model", "")).strip()
            mmproj = str(payload.get("mmproj", "")).strip()
            if not model:
                self._send_json(400, {"error": {"message": "field 'model' is required"}})
                return

            with self.state.lock:
                if self.state.selected_backend != DEFAULT_BACKEND:
                    self._send_json(409, {"error": {"message": f"select backend first: {DEFAULT_BACKEND}"}})
                    return

                model_path = model if os.path.isabs(model) else os.path.join(self.state.models_dir, model)
                if not os.path.isfile(model_path):
                    self._send_json(400, {"error": {"message": f"model file not found: {model_path}"}})
                    return

                if mmproj:
                    mmproj_path = mmproj if os.path.isabs(mmproj) else os.path.join(self.state.models_dir, mmproj)
                    if not os.path.isfile(mmproj_path):
                        self._send_json(400, {"error": {"message": f"mmproj file not found: {mmproj_path}"}})
                        return
                else:
                    mmproj_path = maybe_auto_mmproj(
                        self.state.models_dir,
                        os.path.basename(model_path),
                        model_path,
                    )

                stop_backend(self.state)
                ok, msg = start_backend(self.state, model_path, mmproj_path)
                if not ok:
                    self._send_json(500, {"error": {"message": msg}})
                    return

                self.state.selected_model = display_path_reference(model_path, self.state.models_dir)
                self.state.selected_mmproj = (
                    display_path_reference(mmproj_path, self.state.models_dir) if mmproj_path else None
                )

            self._send_json(
                200,
                {
                    "ok": True,
                    "selected_backend": DEFAULT_BACKEND,
                    "selected_model": self.state.selected_model,
                    "selected_mmproj": self.state.selected_mmproj,
                    "backend_url": f"http://{self.state.backend_host}:{self.state.backend_port}",
                },
            )
            return

        if self.path == "/v1/chat/completions":
            payload = self._read_json()
            with self.state.lock:
                proc = self.state.backend_proc
                running = bool(proc and proc.poll() is None)
                host, port = self.state.backend_host, self.state.backend_port
                selected_model = self.state.selected_model
            if not running:
                self._send_json(409, {"error": {"message": "no backend model loaded; call /omni/model/select first"}})
                return

            if "model" not in payload or not payload.get("model"):
                payload["model"] = selected_model

            if payload.get("stream") is True:
                stream_http_request(
                    self,
                    "POST",
                    f"http://{host}:{port}/v1/chat/completions",
                    payload=payload,
                    timeout=3600,
                )
                return

            code, body = http_json(
                "POST",
                f"http://{host}:{port}/v1/chat/completions",
                payload=payload,
                timeout=600,
            )
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        self._send_json(404, {"error": {"message": f"not found: {self.path}"}})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniInfer Gateway")
    p.add_argument("--host", default="127.0.0.1", help="Gateway bind host")
    p.add_argument("--port", type=int, default=9000, help="Gateway bind port")
    p.add_argument("--backend-host", default="127.0.0.1", help="Managed backend bind host")
    p.add_argument("--backend-port", type=int, default=18080, help="Managed backend bind port")
    p.add_argument(
        "--llama-server-path",
        default=os.environ.get("OMNIINFER_LLAMA_SERVER_PATH", str(DEFAULT_LLAMA_SERVER_PATH)),
        help="Path to llama-server executable. Relative paths are resolved from the repository root.",
    )
    p.add_argument(
        "--models-dir",
        default=os.environ.get("OMNIINFER_MODELS_DIR", str(DEFAULT_MODELS_DIR)),
        help="Directory containing gguf model files. Relative paths are resolved from the repository root.",
    )
    p.add_argument("--startup-timeout", type=int, default=60, help="Backend startup timeout seconds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    llama_server_path = resolve_input_path(args.llama_server_path, REPO_ROOT)
    models_dir = resolve_input_path(args.models_dir, REPO_ROOT)
    state = ServerState(
        lock=threading.Lock(),
        selected_backend=DEFAULT_BACKEND,
        selected_model=None,
        selected_mmproj=None,
        backend_proc=None,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        backend_llama_path=llama_server_path,
        models_dir=models_dir,
        request_timeout_s=args.startup_timeout,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), OmniHandler)
    httpd.state = state  # type: ignore[attr-defined]

    print(f"OmniInfer Gateway listening on http://{args.host}:{args.port}")
    print(f"Default backend: {DEFAULT_BACKEND}")
    print(f"Managed backend target: {DEFAULT_BACKEND} on http://{args.backend_host}:{args.backend_port}")
    print(f"llama-server path: {llama_server_path}")
    print(f"models dir: {models_dir}")
    print("Use POST /omni/model/select -> POST /v1/chat/completions")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with state.lock:
            stop_backend(state)
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
