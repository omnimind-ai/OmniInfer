from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from service_core.runtime import RuntimeManager


if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
    REPO_ROOT = APP_ROOT
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    APP_ROOT = REPO_ROOT

INTERNAL_BACKEND_HOST = "127.0.0.1"
INTERNAL_BACKEND_PORT = 0


def default_backend_for_host() -> str:
    if sys.platform == "darwin":
        return "llama.cpp-mac"
    return "llama.cpp-cpu"


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_config(app_root: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 9000,
        "default_backend": default_backend_for_host(),
        "default_thinking": "off",
        "window_mode": "hidden",
        "startup_timeout": 60,
        "runtime_root": "runtime",
        "backends": {
            "llama.cpp-mac": {
                "ngl": "999",
            },
            "llama.cpp-cpu": {},
            "llama.cpp-cuda": {
                "ngl": "999",
            },
        },
    }
    config_path = app_root / "config" / "omniinfer.json"
    if not config_path.is_file():
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        return defaults
    return deep_merge(defaults, loaded)


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if low in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
    raise ValueError(f"cannot parse boolean value from {value!r}")


def apply_thinking_mode(payload: dict[str, Any], default_enabled: bool) -> None:
    requested = payload.pop("think", None)
    enabled: bool | None = None
    if requested is not None:
        enabled = parse_boolish(requested)

    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
        payload["chat_template_kwargs"] = chat_template_kwargs
    elif not isinstance(chat_template_kwargs, dict):
        return

    if enabled is not None:
        chat_template_kwargs["enable_thinking"] = enabled
    elif "enable_thinking" not in chat_template_kwargs:
        chat_template_kwargs["enable_thinking"] = default_enabled

    final_enabled = chat_template_kwargs.get("enable_thinking")
    if isinstance(final_enabled, bool) and final_enabled is False and "reasoning_format" not in payload:
        payload["reasoning_format"] = "none"


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


class OmniHandler(BaseHTTPRequestHandler):
    server_version = "OmniInfer/0.2"

    @property
    def manager(self) -> RuntimeManager:
        return self.server.manager  # type: ignore[attr-defined]

    @property
    def default_thinking(self) -> bool:
        return self.server.default_thinking  # type: ignore[attr-defined]

    @default_thinking.setter
    def default_thinking(self, value: bool) -> None:
        self.server.default_thinking = value  # type: ignore[attr-defined]

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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            obj = json.loads(raw.decode("utf-8-sig"))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return {}

    def _parse_request_target(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        return parsed.path, parse_qs(parsed.query)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path, query = self._parse_request_target()

        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "omni": self.manager.snapshot(),
                    "thinking": {"default_enabled": self.default_thinking},
                },
            )
            return

        if path == "/omni/state":
            payload = self.manager.snapshot()
            payload["available_backends"] = self.manager.list_backends()
            payload["thinking"] = {"default_enabled": self.default_thinking}
            self._send_json(200, payload)
            return

        if path == "/omni/backends":
            self._send_json(200, {"object": "list", "data": self.manager.list_backends()})
            return

        if path == "/omni/thinking":
            self._send_json(200, {"default_enabled": self.default_thinking})
            return

        if path == "/omni/models":
            self._send_json(
                410,
                {
                    "error": {
                        "message": "GET /omni/models has been deprecated and is no longer maintained"
                    }
                },
            )
            return

        if path == "/omni/supported-models":
            system_name = query.get("system", [None])[0]
            try:
                payload = self.manager.list_supported_models(system_name=system_name or "")
            except (ValueError, RuntimeError, urllib.error.URLError) as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            self._send_json(200, payload)
            return

        if path == "/omni/supported-models/best":
            system_name = query.get("system", [None])[0]
            try:
                payload = self.manager.list_supported_models_best(system_name=system_name or "")
            except (ValueError, RuntimeError, urllib.error.URLError) as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            self._send_json(200, payload)
            return

        if path == "/v1/models":
            self._send_json(
                410,
                {
                    "error": {
                        "message": "GET /v1/models is not maintained in OmniInfer right now"
                    }
                },
            )
            return

        self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path, _query = self._parse_request_target()

        if path == "/omni/backend/select":
            payload = self._read_json()
            backend = str(payload.get("backend", "")).strip()
            if not backend:
                self._send_json(400, {"error": {"message": "field 'backend' is required"}})
                return
            try:
                result = self.manager.select_backend(backend)
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            self._send_json(200, result)
            return

        if path == "/omni/backend/stop":
            self._send_json(200, self.manager.stop_runtime())
            return

        if path == "/omni/shutdown":
            self._send_json(200, {"ok": True, "message": "shutdown requested"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if path == "/omni/thinking/select":
            payload = self._read_json()
            raw_enabled = payload.get("enabled", payload.get("think"))
            if raw_enabled is None:
                self._send_json(400, {"error": {"message": "field 'enabled' is required"}})
                return
            try:
                self.default_thinking = parse_boolish(raw_enabled)
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            self._send_json(200, {"ok": True, "default_enabled": self.default_thinking})
            return

        if path == "/omni/model/select":
            payload = self._read_json()
            model = str(payload.get("model", "")).strip()
            mmproj = str(payload.get("mmproj", "")).strip() or None
            backend = str(payload.get("backend", "")).strip() or None
            if not model:
                self._send_json(400, {"error": {"message": "field 'model' is required"}})
                return
            try:
                result = self.manager.select_model(model=model, mmproj=mmproj, backend_id=backend)
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                status = 400 if isinstance(e, (ValueError, FileNotFoundError)) else 409
                self._send_json(status, {"error": {"message": str(e)}})
                return
            self._send_json(200, result)
            return

        if path == "/v1/chat/completions":
            payload = self._read_json()
            requested_backend = str(payload.pop("backend", "")).strip() or None
            requested_model = str(payload.get("model", "")).strip() or None
            requested_mmproj = str(payload.pop("mmproj", "")).strip() or None
            try:
                apply_thinking_mode(payload, default_enabled=self.default_thinking)
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return

            try:
                runtime = self.manager.ensure_model_loaded(
                    model=requested_model,
                    mmproj=requested_mmproj,
                    backend_id=requested_backend,
                )
            except (ValueError, FileNotFoundError) as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            except RuntimeError as e:
                self._send_json(409, {"error": {"message": str(e)}})
                return

            if not payload.get("model"):
                payload["model"] = runtime.model_ref

            target = self.manager.current_proxy_target()
            if not target:
                self._send_json(409, {"error": {"message": "selected backend is not ready"}})
                return
            host, port = target

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

        self._send_json(404, {"error": {"message": f"not found: {path}"}})


def parse_args(config: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniInfer unified API service")
    p.add_argument("--host", default=config["host"], help="Gateway bind host")
    p.add_argument("--port", type=int, default=int(config["port"]), help="Gateway bind port")
    p.add_argument(
        "--backend-host",
        default=str(config.get("backend_host", INTERNAL_BACKEND_HOST)),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--backend-port",
        type=int,
        default=int(config.get("backend_port", INTERNAL_BACKEND_PORT)),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--default-backend",
        default=config["default_backend"],
        help="Backend selected on startup (llama.cpp-mac, llama.cpp-cpu, or llama.cpp-cuda)",
    )
    p.add_argument(
        "--default-thinking",
        choices=("on", "off"),
        default=str(config["default_thinking"]),
        help="Default thinking mode for chat requests when not overridden per request",
    )
    p.add_argument(
        "--window-mode",
        choices=("visible", "hidden"),
        default=str(config.get("window_mode", config.get("backend_window_mode", "hidden"))),
        help="Whether OmniInfer and managed backend console windows should be visible or hidden",
    )
    p.add_argument("--startup-timeout", type=int, default=int(config["startup_timeout"]), help="Backend startup timeout seconds")
    return p.parse_args()


def main() -> int:
    config = load_app_config(APP_ROOT)
    args = parse_args(config)
    manager = RuntimeManager(
        repo_root=str(REPO_ROOT),
        app_root=str(APP_ROOT),
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        startup_timeout_s=args.startup_timeout,
        backend_window_mode=args.window_mode,
        runtime_root=str(config.get("runtime_root", "runtime")),
        backend_overrides=config.get("backends"),
        default_backend_id=args.default_backend,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), OmniHandler)
    httpd.manager = manager  # type: ignore[attr-defined]
    httpd.default_thinking = args.default_thinking == "on"  # type: ignore[attr-defined]

    print(f"OmniInfer listening on http://{args.host}:{args.port}")
    print(f"Selected backend on startup: {manager.snapshot()['backend']}")
    print(f"Default thinking mode: {'on' if httpd.default_thinking else 'off'}")
    print("Use GET /omni/backends -> GET /omni/supported-models/best -> POST /omni/model/select")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_runtime()
        httpd.server_close()
    return 0
