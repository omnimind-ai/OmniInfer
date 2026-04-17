from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from service_core.logger import log_session_header, resolve_log_level, setup_logging
from service_core.platforms import current_host_platform, default_backend_for_current_host, parse_extra_args
from service_core.runtime import RuntimeManager

logger = logging.getLogger("gateway")


if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
    REPO_ROOT = APP_ROOT
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    APP_ROOT = REPO_ROOT

INTERNAL_BACKEND_HOST = "127.0.0.1"
INTERNAL_BACKEND_PORT = 0


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_config(app_root: Path) -> dict[str, Any]:
    host_platform = current_host_platform()
    defaults: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 9000,
        "default_backend": default_backend_for_current_host(),
        "default_thinking": "off",
        "window_mode": "hidden",
        "startup_timeout": 60,
        "runtime_root": "runtime",
        "backends": host_platform.default_config_backends(),
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


def summarize_for_log(value: Any, max_len: int = 800) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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


def parse_optional_positive_int_field(
    payload: dict[str, Any],
    *names: str,
    pop: bool = False,
) -> int | None:
    raw_value: Any = None
    found = False
    for name in names:
        if name in payload:
            raw_value = payload.pop(name) if pop else payload.get(name)
            found = True
            break
    if not found or raw_value in (None, ""):
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"field '{names[0]}' must be a positive integer")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field '{names[0]}' must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"field '{names[0]}' must be a positive integer")
    return parsed


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


def _log_completion(status: int, body: bytes | None, elapsed: float, mode: str) -> None:
    """Log chat completion result with usage metrics when available."""
    parts = [f"POST /v1/chat/completions -> {status} ({elapsed:.2f}s, {mode})"]
    if status == 200 and body:
        try:
            data = json.loads(body) if isinstance(body, (bytes, bytearray)) else body
            if not isinstance(data, dict):
                raise ValueError
            segs: list[str] = []
            usage = data.get("usage")
            timings = data.get("timings")
            if isinstance(usage, dict):
                segs.append(f"prompt={usage.get('prompt_tokens', 0)}")
                segs.append(f"completion={usage.get('completion_tokens', 0)}")
                ptd = usage.get("prompt_tokens_details", {})
                if isinstance(ptd, dict) and ptd.get("cached_tokens") is not None:
                    segs.append(f"cached={ptd['cached_tokens']}")
            elif isinstance(timings, dict):
                # Stream mode: timings has prompt_n / predicted_n instead of usage
                segs.append(f"prompt={timings.get('prompt_n', 0)}")
                segs.append(f"completion={timings.get('predicted_n', 0)}")
                if timings.get("cache_n"):
                    segs.append(f"cached={timings['cache_n']}")
            if isinstance(timings, dict):
                pps = timings.get("prompt_per_second")
                dps = timings.get("predicted_per_second")
                if pps is not None:
                    segs.append(f"prefill={pps:.1f}t/s")
                if dps is not None:
                    segs.append(f"decode={dps:.1f}t/s")
            if segs:
                parts.append(" ".join(segs))
        except Exception:
            pass
    logger.info("  ".join(parts))


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
) -> bytes | None:
    """Stream an HTTP request to the client, returning the last SSE data line (for usage extraction)."""
    headers = {"Accept": "text/event-stream, application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    tail_buf = b""
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
                tail_buf = (tail_buf + chunk)[-4096:]
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
        pass  # client disconnected, but tail_buf may still have usage data

    # Extract last SSE data line containing usage/timings info
    for line in reversed(tail_buf.decode("utf-8", errors="replace").splitlines()):
        if line.startswith("data: {") and ('"usage"' in line or '"timings"' in line):
            return line[6:].encode("utf-8")
    return None


def stream_embedded_events(
    handler: BaseHTTPRequestHandler,
    events: list[dict[str, Any]],
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    try:
        for event in events:
            body = json.dumps(event, ensure_ascii=False).encode("utf-8")
            handler.wfile.write(b"data: " + body + b"\n\n")
            handler.wfile.flush()
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
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
        logging.getLogger("http").debug("%s %s", self.address_string(), fmt % args)

    def _debug_body_enabled(self) -> bool:
        return bool(getattr(self.server, "debug_body", False))  # type: ignore[attr-defined]

    def _debug(self, message: str) -> None:
        logger.debug(message)

    @property
    def forced_backend(self) -> str | None:
        value = getattr(self.server, "forced_backend", "")  # type: ignore[attr-defined]
        return str(value).strip() or None

    def _send_json(self, status: int, payload: Any) -> None:
        self._debug(f"{self.command} {self.path} -> {status}")
        if status >= 400:
            error_msg = ""
            if isinstance(payload, dict) and "error" in payload:
                error_msg = payload["error"].get("message", "")
            log_fn = logger.error if status >= 500 else logger.warning
            log_fn("%s %s -> %d: %s", self.command, self.path, status, error_msg)
        if self._debug_body_enabled():
            self._debug(f"response body: {summarize_for_log(payload)}")
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
                if self._debug_body_enabled():
                    self._debug(f"request body: {summarize_for_log(obj)}")
                return obj
        except json.JSONDecodeError:
            pass
        if self._debug_body_enabled() and raw:
            self._debug(f"request body (invalid json): {raw[:400]!r}")
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
        self._debug(f"{self.command} {self.path} from {self.client_address[0]}")
        try:
            self._do_GET_impl()
        except Exception as e:  # pragma: no cover - defensive server-side fallback
            logger.exception("Unhandled error in %s %s", self.command, self.path)
            self._send_json(500, {"error": {"message": f"internal server error: {e}"}})

    def _do_GET_impl(self) -> None:
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
            payload["available_backends"] = self.manager.list_backends(scope="all")[0]
            payload["thinking"] = {"default_enabled": self.default_thinking}
            self._send_json(200, payload)
            return

        if path == "/omni/backends":
            scope = query.get("scope", ["installed"])[0]
            if scope not in ("installed", "compatible", "all"):
                self._send_json(400, {"error": {"message": f"invalid scope: {scope}. Must be one of: installed, compatible, all"}})
                return
            backends_data, recommended = self.manager.list_backends(scope=scope)
            self._send_json(200, {"object": "list", "data": backends_data, "recommended": recommended})
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
            snapshot = self.manager.snapshot()
            models: list[dict[str, Any]] = []
            if snapshot.get("backend_ready") and snapshot.get("model"):
                model_id = snapshot["model"]
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "omniinfer",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                })
            self._send_json(200, {"object": "list", "data": models})
            return

        self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        self._debug(f"{self.command} {self.path} from {self.client_address[0]}")
        try:
            self._do_POST_impl()
        except Exception as e:  # pragma: no cover - defensive server-side fallback
            logger.exception("Unhandled error in %s %s", self.command, self.path)
            self._send_json(500, {"error": {"message": f"internal server error: {e}"}})

    def _do_POST_impl(self) -> None:
        path, _query = self._parse_request_target()

        if path == "/omni/backend/select":
            payload = self._read_json()
            backend = self.forced_backend or str(payload.get("backend", "")).strip()
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

        if path == "/omni/cache/clear":
            try:
                result = self.manager.clear_kv_cache()
            except RuntimeError as e:
                self._send_json(409, {"error": {"message": str(e)}})
                return
            self._send_json(200, result)
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
            backend = self.forced_backend or (str(payload.get("backend", "")).strip() or None)
            launch_args = parse_extra_args(payload.get("launch_args")) if "launch_args" in payload else None
            raw_request_defaults = payload.get("request_defaults")
            request_defaults = dict(raw_request_defaults) if isinstance(raw_request_defaults, dict) else None
            try:
                ctx_size = parse_optional_positive_int_field(payload, "ctx_size", "ctx-size")
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            if not model:
                self._send_json(400, {"error": {"message": "field 'model' is required"}})
                return
            try:
                result = self.manager.select_model(
                    model=model,
                    mmproj=mmproj,
                    backend_id=backend,
                    ctx_size=ctx_size,
                    launch_args=launch_args,
                    request_defaults=request_defaults,
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                status = 400 if isinstance(e, (ValueError, FileNotFoundError)) else 409
                self._send_json(status, {"error": {"message": str(e)}})
                return
            self._send_json(200, result)
            return

        if path == "/v1/chat/completions":
            _req_start = time.perf_counter()
            payload = self._read_json()
            logger.info(
                "POST /v1/chat/completions model=%s messages=%d stream=%s",
                payload.get("model", ""),
                len(payload.get("messages", [])),
                payload.get("stream", False),
            )
            requested_backend = self.forced_backend or (str(payload.pop("backend", "")).strip() or None)
            requested_model = str(payload.get("model", "")).strip() or None
            requested_mmproj = str(payload.pop("mmproj", "")).strip() or None
            requested_launch_args = parse_extra_args(payload.pop("launch_args", None)) if "launch_args" in payload else None
            raw_request_defaults = payload.pop("request_defaults", None) if "request_defaults" in payload else None
            requested_request_defaults = dict(raw_request_defaults) if isinstance(raw_request_defaults, dict) else None
            try:
                requested_ctx_size = parse_optional_positive_int_field(
                    payload,
                    "ctx_size",
                    "ctx-size",
                    pop=True,
                )
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
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
                    ctx_size=requested_ctx_size,
                    launch_args=requested_launch_args,
                    request_defaults=requested_request_defaults,
                )
            except (ValueError, FileNotFoundError) as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
            except RuntimeError as e:
                self._send_json(409, {"error": {"message": str(e)}})
                return

            effective_payload = dict(runtime.request_defaults)
            effective_payload.update(payload)
            payload = effective_payload

            if not payload.get("model"):
                payload["model"] = runtime.model_ref

            runtime_mode = self.manager.current_runtime_mode()
            if runtime_mode == "embedded":
                try:
                    if payload.get("stream") is True:
                        events = self.manager.stream_chat_completion(payload)
                        stream_embedded_events(self, events)
                        usage_event = next((e for e in reversed(events) if "usage" in e), None)
                        _log_completion(200, json_dumps(usage_event) if usage_event else None, time.perf_counter() - _req_start, "embedded stream")
                        return
                    response = self.manager.chat_completion(payload)
                except ValueError as e:
                    self._send_json(400, {"error": {"message": str(e)}})
                    return
                except RuntimeError as e:
                    self._send_json(409, {"error": {"message": str(e)}})
                    return
                _log_completion(200, json_dumps(response), time.perf_counter() - _req_start, "embedded")
                self._send_json(200, response)
                return

            target = self.manager.current_proxy_target()
            if not target:
                self._send_json(409, {"error": {"message": "selected backend is not ready"}})
                return
            host, port = target

            if payload.get("stream") is True:
                self._debug(f"proxy stream -> http://{host}:{port}/v1/chat/completions")
                last_data = stream_http_request(
                    self,
                    "POST",
                    f"http://{host}:{port}/v1/chat/completions",
                    payload=payload,
                    timeout=3600,
                )
                _log_completion(200, last_data, time.perf_counter() - _req_start, "proxy stream")
                return

            code, body = http_json(
                "POST",
                f"http://{host}:{port}/v1/chat/completions",
                payload=payload,
                timeout=600,
            )
            _log_completion(code, body, time.perf_counter() - _req_start, "proxy")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        self._send_json(404, {"error": {"message": f"not found: {path}"}})


def parse_args(config: dict[str, Any]) -> argparse.Namespace:
    backend_names = ", ".join(template.id for template in current_host_platform().backend_templates)
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
        help=f"Backend selected on startup ({backend_names})",
    )
    p.add_argument(
        "--default-thinking",
        choices=("on", "off"),
        default=str(config["default_thinking"]),
        help="Default thinking mode for chat requests when not overridden per request",
    )
    p.add_argument(
        "--force-backend",
        default=str(config.get("force_backend", "")),
        help="Force all model selection and chat requests to use a specific backend",
    )
    p.add_argument(
        "--window-mode",
        choices=("visible", "hidden"),
        default=str(config.get("window_mode", config.get("backend_window_mode", "hidden"))),
        help="Whether OmniInfer and managed backend console windows should be visible or hidden",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print request routing and response status logs to the console",
    )
    p.add_argument(
        "--debug-body",
        action="store_true",
        help="With --verbose, also print truncated request/response JSON bodies",
    )
    p.add_argument("--startup-timeout", type=int, default=int(config["startup_timeout"]), help="Backend startup timeout seconds")
    p.add_argument("--log-level", default=None, choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Log level override")
    return p.parse_args()


def main() -> int:
    config = load_app_config(APP_ROOT)
    args = parse_args(config)

    # --- Initialize logging ---
    level = args.log_level or resolve_log_level(verbose=args.verbose, debug_body=args.debug_body)
    log_file = setup_logging(level=level, console=True, log_to_file=True)

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

    log_session_header(
        config=config,
        backends=[b["id"] for b in manager.list_backends(scope="installed")[0]],
    )

    httpd = ThreadingHTTPServer((args.host, args.port), OmniHandler)
    httpd.manager = manager  # type: ignore[attr-defined]
    httpd.default_thinking = args.default_thinking == "on"  # type: ignore[attr-defined]
    httpd.debug_http = bool(args.verbose)  # type: ignore[attr-defined]
    httpd.debug_body = bool(args.debug_body)  # type: ignore[attr-defined]
    httpd.forced_backend = args.force_backend.strip()  # type: ignore[attr-defined]

    logger.info("OmniInfer listening on http://%s:%s", args.host, args.port)
    logger.info("Selected backend on startup: %s", manager.snapshot()["backend"])
    logger.info("Default thinking mode: %s", "on" if httpd.default_thinking else "off")
    if httpd.forced_backend:
        logger.info("Forced backend: %s", httpd.forced_backend)
    if log_file:
        logger.info("Log file: %s", log_file)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_runtime()
        httpd.server_close()
    return 0
