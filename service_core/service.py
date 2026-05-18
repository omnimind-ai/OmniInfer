from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from service_core.logger import log_session_header, resolve_log_level, setup_logging
from service_core.local_state import load_default_thinking, save_default_thinking
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
PUBLIC_GET_ENDPOINTS = frozenset({"/health", "/v1/models"})
PUBLIC_POST_ENDPOINTS = frozenset({"/v1/chat/completions", "/v1/messages"})


@dataclass(frozen=True)
class GatewayAccessPolicy:
    api_key: str = ""
    allow_insecure_lan: bool = False
    allow_remote_management: bool = False


@dataclass(frozen=True)
class LineStreamOptions:
    enabled: bool = False
    max_line_chars: int = 240
    include_reasoning: bool = False


@dataclass(frozen=True)
class ResolvedApiKey:
    value: str
    generated: bool = False


def is_loopback_address(host: str) -> bool:
    value = host.strip().strip("[]")
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def is_loopback_bind_host(host: str) -> bool:
    value = host.strip().strip("[]")
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def is_all_interfaces_host(host: str) -> bool:
    value = host.strip().strip("[]")
    return value in {"0.0.0.0", "::", ""}


def should_require_remote_api_key(host: str) -> bool:
    return not is_loopback_bind_host(host)


def resolve_api_key(cli_value: str | None, config: dict[str, Any], *, lan_mode: bool) -> ResolvedApiKey:
    raw = (cli_value if cli_value is not None else str(config.get("api_key", ""))).strip()
    if not raw:
        raw = os.environ.get("OMNIINFER_API_KEY", "").strip()
    if raw:
        return ResolvedApiKey(raw, generated=False)
    if lan_mode:
        return ResolvedApiKey("oi_" + secrets.token_urlsafe(24), generated=True)
    return ResolvedApiKey("")


def local_lan_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not is_loopback_address(ip) and not ip.startswith("169.254."):
                addresses.add(ip)
    except OSError:
        pass

    # This avoids DNS-only setups where getaddrinfo(hostname) misses the active route.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if not is_loopback_address(ip) and not ip.startswith("169.254."):
                addresses.add(ip)
    except OSError:
        pass

    return sorted(addresses)


def log_gateway_access_urls(host: str, port: int, api_key: str, *, lan_enabled: bool) -> None:
    logger.info("Local API: http://127.0.0.1:%d/v1", port)
    if not lan_enabled:
        return

    addresses = local_lan_ipv4_addresses()
    if not addresses:
        logger.warning("LAN API enabled, but no non-loopback IPv4 address was detected")
    else:
        for address in addresses:
            logger.info("LAN API: http://%s:%d/v1", address, port)
    if api_key:
        logger.info("LAN API key is configured")
    else:
        logger.warning("LAN API is running without an API key")


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
        saved_thinking = load_default_thinking(app_root)
        if saved_thinking is not None:
            defaults["default_thinking"] = "on" if saved_thinking else "off"
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        return defaults
    merged = deep_merge(defaults, loaded)
    saved_thinking = load_default_thinking(app_root)
    if saved_thinking is not None:
        merged["default_thinking"] = "on" if saved_thinking else "off"
    _validate_config(merged, config_path)
    return merged


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    port = config.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"{config_path}: invalid port {port!r} (must be 1-65535)")
    timeout = config.get("startup_timeout")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"{config_path}: invalid startup_timeout {timeout!r} (must be positive)")
    host = config.get("host", "")
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"{config_path}: invalid host {host!r} (must be non-empty string)")


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


_DISABLED_REASONING_EFFORTS = {"none", "off", "disabled", "false", "0"}
_THINK_START = "<think>"
_THINK_END = "</think>"


def reasoning_effort_enabled(payload: dict[str, Any]) -> bool | None:
    effort = payload.pop("reasoning_effort", None)
    if effort not in (None, ""):
        return str(effort).strip().lower() not in _DISABLED_REASONING_EFFORTS

    reasoning = payload.pop("reasoning", None)
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if effort in (None, ""):
        return None
    return str(effort).strip().lower() not in _DISABLED_REASONING_EFFORTS


def apply_thinking_mode(payload: dict[str, Any], default_enabled: bool) -> None:
    requested = payload.pop("think", None)
    reasoning_enabled = reasoning_effort_enabled(payload)

    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
        payload["chat_template_kwargs"] = chat_template_kwargs
    elif not isinstance(chat_template_kwargs, dict):
        return

    enabled: bool | None = None
    if requested is not None:
        enabled = parse_boolish(requested)
    elif "enable_thinking" not in chat_template_kwargs:
        enabled = reasoning_enabled

    if enabled is not None:
        chat_template_kwargs["enable_thinking"] = enabled
    elif "enable_thinking" not in chat_template_kwargs:
        chat_template_kwargs["enable_thinking"] = default_enabled

    final_enabled = chat_template_kwargs.get("enable_thinking")
    if isinstance(final_enabled, bool) and final_enabled is False and "reasoning_format" not in payload:
        payload["reasoning_format"] = "none"


def split_thinking_text(text: str) -> tuple[str | None, str] | None:
    start = text.find(_THINK_START)
    if start < 0:
        return None

    before = text[:start]
    remainder = text[start + len(_THINK_START) :]
    end = remainder.find(_THINK_END)
    if end < 0:
        return remainder.strip(), before.rstrip()

    reasoning = remainder[:end].strip()
    content = (before + remainder[end + len(_THINK_END) :]).lstrip()
    return reasoning, content


def normalize_chat_completion_reasoning(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        parsed = split_thinking_text(content)
        if parsed is None:
            continue
        reasoning, answer = parsed
        message["reasoning_content"] = reasoning
        message["content"] = answer
    return payload


def normalize_chat_completion_body(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return body
    if not isinstance(payload, dict):
        return body
    return json_dumps(normalize_chat_completion_reasoning(payload))


class ChatReasoningStreamNormalizer:
    def __init__(self) -> None:
        self._buffer = ""
        self._in_reasoning = False

    def _consume_text(self, text: str, *, final: bool = False) -> list[tuple[str, str]]:
        self._buffer += text
        out: list[tuple[str, str]] = []

        while self._buffer:
            if self._in_reasoning:
                end = self._buffer.find(_THINK_END)
                if end >= 0:
                    reasoning = self._buffer[:end]
                    if reasoning.strip():
                        out.append(("reasoning", reasoning))
                    self._buffer = self._buffer[end + len(_THINK_END) :].lstrip()
                    self._in_reasoning = False
                    continue

                keep = 0 if final else max(len(_THINK_END) - 1, 0)
                if len(self._buffer) > keep:
                    split_at = len(self._buffer) - keep
                    reasoning = self._buffer[:split_at]
                    if reasoning.strip():
                        out.append(("reasoning", reasoning))
                    self._buffer = self._buffer[split_at:]
                break

            start = self._buffer.find(_THINK_START)
            if start >= 0:
                if start > 0:
                    out.append(("content", self._buffer[:start]))
                self._buffer = self._buffer[start + len(_THINK_START) :]
                self._in_reasoning = True
                continue

            keep = 0 if final else max(len(_THINK_START) - 1, 0)
            if len(self._buffer) > keep:
                split_at = len(self._buffer) - keep
                out.append(("content", self._buffer[:split_at]))
                self._buffer = self._buffer[split_at:]
            break

        return out

    def _with_delta(self, event: dict[str, Any], kind: str, text: str) -> dict[str, Any]:
        copied = dict(event)
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return event
        copied_choices = list(choices)
        choice = dict(choices[0]) if isinstance(choices[0], dict) else {}
        delta = dict(choice.get("delta")) if isinstance(choice.get("delta"), dict) else {}
        if kind == "reasoning":
            delta["reasoning_content"] = text
            delta["content"] = None
        else:
            delta["content"] = text
        choice["delta"] = delta
        copied_choices[0] = choice
        copied["choices"] = copied_choices
        return copied

    def transform_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return [event]
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return [event]
        content = delta.get("content")
        if not isinstance(content, str) or not content:
            if choice.get("finish_reason") is not None:
                flushed = self.flush(event)
                return flushed if flushed else [event]
            return [event]

        parts = self._consume_text(content)
        transformed = [self._with_delta(event, kind, text) for kind, text in parts if text]
        if choice.get("finish_reason") is not None:
            transformed.extend(self.flush(event))
        return transformed

    def flush(self, template: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        parts = self._consume_text("", final=True)
        if not template:
            return []
        return [self._with_delta(template, kind, text) for kind, text in parts if text]


class ChatUsageStreamNormalizer:
    def __init__(self) -> None:
        self._latest_usage_event: dict[str, Any] | None = None
        self._emitted_usage_event = False

    def transform_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        if "usage" not in event and "timings" not in event:
            return [event]

        choices = event.get("choices")
        if isinstance(choices, list) and choices:
            self._latest_usage_event = self._to_usage_event(event)
            stripped = dict(event)
            stripped.pop("usage", None)
            stripped.pop("timings", None)
            return [stripped]

        self._latest_usage_event = self._to_usage_event(event)
        self._emitted_usage_event = True
        return [self._latest_usage_event]

    def flush(self) -> list[dict[str, Any]]:
        if self._emitted_usage_event or self._latest_usage_event is None:
            return []
        self._emitted_usage_event = True
        return [self._latest_usage_event]

    def _to_usage_event(self, event: dict[str, Any]) -> dict[str, Any]:
        usage_event = dict(event)
        usage_event["choices"] = []
        return usage_event


_request_log_counter = 0
_request_log_lock = threading.Lock()


def _save_request_response(
    request_json: str,
    response_body: bytes | None,
    status: int,
    elapsed: float,
    mode: str,
) -> None:
    """Save request/response pair to .local/logs/requests/{date}/{seq}-chat.json."""
    global _request_log_counter
    try:
        now = datetime.now()
        log_dir = APP_ROOT / ".local" / "logs" / "requests" / now.strftime("%Y-%m-%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        with _request_log_lock:
            _request_log_counter += 1
            seq = _request_log_counter
        filename = f"{now.strftime('%H%M%S')}-{seq:04d}-chat.json"
        resp_str = ""
        if response_body:
            resp_str = response_body.decode("utf-8", errors="replace")
        # Assemble JSON manually to avoid re-parsing request_json
        with open(log_dir / filename, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write(f'  "timestamp": {json.dumps(now.isoformat())},\n')
            f.write(f'  "status": {status},\n')
            f.write(f'  "mode": {json.dumps(mode)},\n')
            f.write(f'  "elapsed_s": {round(elapsed, 3)},\n')
            f.write(f'  "request": {request_json},\n')
            f.write(f'  "response": {resp_str if resp_str.startswith("{") else json.dumps(resp_str)}\n')
            f.write('}\n')
    except OSError:
        logger.debug("Failed to save request/response log", exc_info=True)


def _enrich_context_overflow_error(body: bytes, n_messages: int) -> bytes:
    """Enrich a context-overflow 400 error with actionable detail."""
    try:
        data = json.loads(body)
        err = data.get("error", {})
        if err.get("type") != "exceed_context_size_error":
            return body
        n_prompt = err.get("n_prompt_tokens", 0)
        n_ctx = err.get("n_ctx", 0)
        overflow = n_prompt - n_ctx
        err["message"] = (
            f"Prompt ({n_prompt} tokens) exceeds context window ({n_ctx} tokens) "
            f"by {overflow} tokens. "
            f"The request contains {n_messages} messages. "
            f"To fix: trim older messages from the conversation history, "
            f"reduce max_tokens, or reload the model with a larger ctx_size."
        )
        err["overflow_tokens"] = overflow
        err["n_messages"] = n_messages
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (json.JSONDecodeError, KeyError, TypeError):
        return body


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
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    logger.info("  ".join(parts))


def resolve_line_stream_options(payload: dict[str, Any]) -> LineStreamOptions:
    raw = payload.pop("omni_stream", None)
    legacy_format = payload.pop("stream_format", None)
    if raw is None and legacy_format is None:
        return LineStreamOptions()

    if raw is None:
        raw = {"format": legacy_format}
    if not isinstance(raw, dict):
        raise ValueError("omni_stream must be an object")

    raw_format = raw.get("format", legacy_format)
    if raw_format in (None, "", "tokens", "openai"):
        return LineStreamOptions()
    if raw_format != "lines":
        raise ValueError("omni_stream.format must be 'lines'")

    max_line_chars = raw.get("max_line_chars", 240)
    if not isinstance(max_line_chars, int) or isinstance(max_line_chars, bool) or max_line_chars <= 0:
        raise ValueError("omni_stream.max_line_chars must be a positive integer")

    include_reasoning = raw.get("include_reasoning", False)
    if not isinstance(include_reasoning, bool):
        raise ValueError("omni_stream.include_reasoning must be a boolean")

    return LineStreamOptions(
        enabled=True,
        max_line_chars=max_line_chars,
        include_reasoning=include_reasoning,
    )


class ChatLineStreamNormalizer:
    def __init__(self, options: LineStreamOptions):
        self.options = options
        self._buffers: dict[str, str] = {"content": "", "reasoning": ""}
        self._line_index = 0
        self._finish_reason: str | None = None
        self._usage_event: dict[str, Any] | None = None

    def transform_event(self, event: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        output: list[tuple[str, dict[str, Any]]] = []
        if isinstance(event.get("usage"), dict) or isinstance(event.get("timings"), dict):
            self._usage_event = {key: event[key] for key in ("usage", "timings") if key in event}

        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self._finish_reason = str(finish_reason)

            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                output.extend(self._append_text("content", content))

            reasoning = delta.get("reasoning_content")
            if self.options.include_reasoning and isinstance(reasoning, str) and reasoning:
                output.extend(self._append_text("reasoning", reasoning))

        return output

    def flush(self) -> list[tuple[str, dict[str, Any]]]:
        output: list[tuple[str, dict[str, Any]]] = []
        for text_type in ("reasoning", "content"):
            output.extend(self._flush_buffer(text_type))

        final: dict[str, Any] = {"finish_reason": self._finish_reason or "stop"}
        if self._usage_event:
            final.update(self._usage_event)
        output.append(("done", final))
        return output

    def _append_text(self, text_type: str, text: str) -> list[tuple[str, dict[str, Any]]]:
        self._buffers[text_type] += text
        output: list[tuple[str, dict[str, Any]]] = []

        while "\n" in self._buffers[text_type]:
            line, rest = self._buffers[text_type].split("\n", 1)
            output.append(self._line_event(text_type, line, newline=True))
            self._buffers[text_type] = rest

        while len(self._buffers[text_type]) >= self.options.max_line_chars:
            chunk = self._buffers[text_type][: self.options.max_line_chars]
            self._buffers[text_type] = self._buffers[text_type][self.options.max_line_chars :]
            output.append(self._line_event(text_type, chunk, newline=False))

        return output

    def _flush_buffer(self, text_type: str) -> list[tuple[str, dict[str, Any]]]:
        text = self._buffers[text_type]
        self._buffers[text_type] = ""
        if not text:
            return []
        return [self._line_event(text_type, text, newline=False)]

    def _line_event(self, text_type: str, text: str, *, newline: bool) -> tuple[str, dict[str, Any]]:
        event = {
            "index": self._line_index,
            "type": text_type,
            "role": "assistant",
            "text": text,
            "newline": newline,
        }
        self._line_index += 1
        return "line", event


def _write_sse_event(handler: BaseHTTPRequestHandler, event_name: str, payload: dict[str, Any]) -> None:
    handler.wfile.write(f"event: {event_name}\n".encode("utf-8"))
    handler.wfile.write(b"data: " + json_dumps(payload) + b"\n\n")


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
    normalize_reasoning: bool = False,
    line_stream_options: LineStreamOptions | None = None,
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

            reasoning_normalizer = ChatReasoningStreamNormalizer() if normalize_reasoning else None
            usage_normalizer = ChatUsageStreamNormalizer()
            line_normalizer = (
                ChatLineStreamNormalizer(line_stream_options)
                if line_stream_options is not None and line_stream_options.enabled
                else None
            )
            last_event: dict[str, Any] | None = None
            for chunk in resp:
                if not chunk:
                    continue
                tail_buf = (tail_buf + chunk)[-4096:]
                if not chunk.startswith(b"data:"):
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                    continue

                prefix, _sep, data_bytes = chunk.partition(b":")
                data = data_bytes.strip()
                if data == b"[DONE]":
                    if reasoning_normalizer is not None:
                        for event in reasoning_normalizer.flush(last_event):
                            for transformed in usage_normalizer.transform_event(event):
                                if line_normalizer is None:
                                    handler.wfile.write(b"data: " + json_dumps(transformed) + b"\n\n")
                                else:
                                    for event_name, line_event in line_normalizer.transform_event(transformed):
                                        _write_sse_event(handler, event_name, line_event)
                    for event in usage_normalizer.flush():
                        if line_normalizer is None:
                            handler.wfile.write(b"data: " + json_dumps(event) + b"\n\n")
                        else:
                            for event_name, line_event in line_normalizer.transform_event(event):
                                _write_sse_event(handler, event_name, line_event)
                    if line_normalizer is None:
                        handler.wfile.write(chunk)
                    else:
                        for event_name, line_event in line_normalizer.flush():
                            _write_sse_event(handler, event_name, line_event)
                    handler.wfile.flush()
                    continue

                try:
                    event = json.loads(data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                    continue

                if not isinstance(event, dict):
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                    continue

                last_event = event
                pending = [event] if reasoning_normalizer is None else reasoning_normalizer.transform_event(event)
                for item in pending:
                    for transformed in usage_normalizer.transform_event(item):
                        if line_normalizer is None:
                            handler.wfile.write(prefix + b": " + json_dumps(transformed) + b"\n\n")
                        else:
                            for event_name, line_event in line_normalizer.transform_event(transformed):
                                _write_sse_event(handler, event_name, line_event)
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
    normalize_reasoning: bool = True,
    line_stream_options: LineStreamOptions | None = None,
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
        reasoning_normalizer = ChatReasoningStreamNormalizer() if normalize_reasoning else None
        usage_normalizer = ChatUsageStreamNormalizer()
        line_normalizer = (
            ChatLineStreamNormalizer(line_stream_options)
            if line_stream_options is not None and line_stream_options.enabled
            else None
        )
        last_event: dict[str, Any] | None = None
        for event in events:
            pending = [event] if reasoning_normalizer is None else reasoning_normalizer.transform_event(event)
            last_event = event
            for item in pending:
                for transformed in usage_normalizer.transform_event(item):
                    if line_normalizer is None:
                        handler.wfile.write(b"data: " + json_dumps(transformed) + b"\n\n")
                    else:
                        for event_name, line_event in line_normalizer.transform_event(transformed):
                            _write_sse_event(handler, event_name, line_event)
            handler.wfile.flush()
        if reasoning_normalizer is not None:
            for event in reasoning_normalizer.flush(last_event):
                for transformed in usage_normalizer.transform_event(event):
                    if line_normalizer is None:
                        handler.wfile.write(b"data: " + json_dumps(transformed) + b"\n\n")
                    else:
                        for event_name, line_event in line_normalizer.transform_event(transformed):
                            _write_sse_event(handler, event_name, line_event)
        for event in usage_normalizer.flush():
            if line_normalizer is None:
                handler.wfile.write(b"data: " + json_dumps(event) + b"\n\n")
            else:
                for event_name, line_event in line_normalizer.transform_event(event):
                    _write_sse_event(handler, event_name, line_event)
        if line_normalizer is None:
            handler.wfile.write(b"data: [DONE]\n\n")
        else:
            for event_name, line_event in line_normalizer.flush():
                _write_sse_event(handler, event_name, line_event)
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

    @property
    def access_policy(self) -> GatewayAccessPolicy:
        policy = getattr(self.server, "access_policy", None)  # type: ignore[attr-defined]
        if isinstance(policy, GatewayAccessPolicy):
            return policy
        return GatewayAccessPolicy()

    def _is_remote_client(self) -> bool:
        try:
            return not is_loopback_address(str(self.client_address[0]))
        except (IndexError, TypeError):
            return True

    def _remote_request_authenticated(self) -> bool:
        policy = self.access_policy
        if not policy.api_key:
            return bool(policy.allow_insecure_lan)

        authorization = self.headers.get("Authorization", "").strip()
        token = ""
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token:
            token = self.headers.get("x-api-key", "").strip()
        return hmac.compare_digest(token, policy.api_key)

    def _authorize_request(self, method: str, path: str) -> bool:
        if not self._is_remote_client():
            return True

        policy = self.access_policy
        if method == "GET" and path in PUBLIC_GET_ENDPOINTS:
            public_endpoint = True
        elif method == "POST" and path in PUBLIC_POST_ENDPOINTS:
            public_endpoint = True
        else:
            public_endpoint = False

        if not public_endpoint and not policy.allow_remote_management:
            self._send_json(
                403,
                {
                    "error": {
                        "message": (
                            "remote clients may only access inference endpoints; "
                            "management endpoints are local-only"
                        )
                    }
                },
            )
            return False

        if not self._remote_request_authenticated():
            status = 401 if policy.api_key else 403
            message = "missing or invalid API key" if policy.api_key else "remote access requires an API key"
            self._send_json(status, {"error": {"message": message}})
            return False

        return True

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
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def _stream_model_select_sse(
        self,
        model: str,
        mmproj: str | None,
        backend: str | None,
        ctx_size: int | None,
        launch_args: list[str] | None,
        request_defaults: dict[str, Any] | None,
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

        client_gone = False

        def emit(event: dict[str, Any]) -> None:
            nonlocal client_gone
            if client_gone:
                return
            try:
                body = json.dumps(event, ensure_ascii=False).encode("utf-8")
                self.wfile.write(b"data: " + body + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                client_gone = True

        _load_start = time.perf_counter()
        try:
            result = self.manager.select_model(
                model=model,
                mmproj=mmproj,
                backend_id=backend,
                ctx_size=ctx_size,
                launch_args=launch_args,
                request_defaults=request_defaults,
                on_progress=emit,
            )
            _elapsed = round(time.perf_counter() - _load_start, 1)
            emit({"type": "done", "elapsed_s": _elapsed, **result})
        except (ValueError, FileNotFoundError) as e:
            emit({"type": "error", "message": str(e)})
        except RuntimeError as e:
            emit({"type": "error", "message": str(e)})
        except Exception as e:
            logger.exception("Unhandled error in streaming model select")
            emit({"type": "error", "message": f"internal error: {e}"})

        if not client_gone:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

    _MAX_REQUEST_BODY = 100 * 1024 * 1024  # 100 MB (multimodal requests may include base64 images)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n > self._MAX_REQUEST_BODY:
            raise ValueError(f"Request body too large ({n} bytes, max {self._MAX_REQUEST_BODY})")
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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, anthropic-version, x-api-key",
        )
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
        if not self._authorize_request("GET", path):
            return

        if path == "/health":
            payload: dict[str, Any] = {
                "status": "ok",
                "omni": self.manager.snapshot(),
                "thinking": {"default_enabled": self.default_thinking},
            }
            if query.get("deep", [""])[0].lower() in ("true", "1", "yes"):
                payload["backend_health"] = self.manager.backend_health()
            self._send_json(200, payload)
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

        if path == "/omni/backend/props":
            self._send_json(200, self.manager.backend_props())
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
        except ValueError as e:
            self._send_json(400, {"error": {"message": str(e)}})
        except Exception as e:  # pragma: no cover - defensive server-side fallback
            logger.exception("Unhandled error in %s %s", self.command, self.path)
            self._send_json(500, {"error": {"message": f"internal server error: {e}"}})

    def _do_POST_impl(self) -> None:
        path, _query = self._parse_request_target()
        if not self._authorize_request("POST", path):
            return

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
            save_default_thinking(self.default_thinking, APP_ROOT)
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
            accept = self.headers.get("Accept", "")
            if "text/event-stream" in accept:
                self._stream_model_select_sse(
                    model=model,
                    mmproj=mmproj,
                    backend=backend,
                    ctx_size=ctx_size,
                    launch_args=launch_args,
                    request_defaults=request_defaults,
                )
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
            _original_request_json = json.dumps(payload, ensure_ascii=False)
            _n_messages = len(payload.get("messages", []))
            try:
                from service_core.platforms.common import (
                    bytes_to_gib,
                    get_available_cuda_memory_bytes,
                    get_available_memory_bytes,
                    get_available_rocm_memory_bytes,
                )
                gpu_mem = get_available_cuda_memory_bytes() or get_available_rocm_memory_bytes()
                if gpu_mem is not None:
                    mem_str = f" vram={bytes_to_gib(gpu_mem):.2f}GiB"
                else:
                    mem_str = f" ram={bytes_to_gib(get_available_memory_bytes()):.2f}GiB"
            except (ImportError, OSError):
                mem_str = ""
            logger.info(
                "POST /v1/chat/completions model=%s messages=%d stream=%s%s",
                payload.get("model", ""),
                len(payload.get("messages", [])),
                payload.get("stream", False),
                mem_str,
            )
            try:
                line_stream_options = resolve_line_stream_options(payload)
            except ValueError as e:
                self._send_json(400, {"error": {"message": str(e)}})
                return
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
                # Chat completions never triggers model loading; use /omni/model/select instead.
                # model/mmproj/ctx_size/launch_args from the request are ignored for loading decisions.
                runtime = self.manager.ensure_model_loaded(
                    model=None,
                    mmproj=None,
                    backend_id=None,
                    ctx_size=None,
                    launch_args=None,
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
                        stream_embedded_events(self, events, line_stream_options=line_stream_options)
                        usage_event = next((e for e in reversed(events) if "usage" in e), None)
                        _elapsed = time.perf_counter() - _req_start
                        _resp_body = json_dumps(usage_event) if usage_event else None
                        _log_completion(200, _resp_body, _elapsed, "embedded stream")
                        _save_request_response(_original_request_json, _resp_body, 200, _elapsed, "embedded stream")
                        return
                    response = normalize_chat_completion_reasoning(self.manager.chat_completion(payload))
                except ValueError as e:
                    self._send_json(400, {"error": {"message": str(e)}})
                    return
                except RuntimeError as e:
                    self._send_json(409, {"error": {"message": str(e)}})
                    return
                _elapsed = time.perf_counter() - _req_start
                _resp_body = json_dumps(response)
                _log_completion(200, _resp_body, _elapsed, "embedded")
                _save_request_response(_original_request_json, _resp_body, 200, _elapsed, "embedded")
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
                    normalize_reasoning=True,
                    line_stream_options=line_stream_options,
                )
                _elapsed = time.perf_counter() - _req_start
                _log_completion(200, last_data, _elapsed, "proxy stream")
                _save_request_response(_original_request_json, last_data, 200, _elapsed, "proxy stream")
                return

            code, body = http_json(
                "POST",
                f"http://{host}:{port}/v1/chat/completions",
                payload=payload,
                timeout=600,
            )
            if code == 400:
                body = _enrich_context_overflow_error(body, _n_messages)
            elif code == 200:
                body = normalize_chat_completion_body(body)
            _elapsed = time.perf_counter() - _req_start
            _log_completion(code, body, _elapsed, "proxy")
            _save_request_response(_original_request_json, body, code, _elapsed, "proxy")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/v1/messages":
            self._handle_anthropic_messages()
            return

        self._send_json(404, {"error": {"message": f"not found: {path}"}})

    # ── Anthropic Messages API (/v1/messages) ────────────────────────────

    def _handle_anthropic_messages(self) -> None:
        from service_core.anthropic_adapter import (
            anthropic_request_to_openai,
            openai_response_to_anthropic,
            stream_anthropic_from_embedded,
            stream_anthropic_proxy,
        )

        _req_start = time.perf_counter()
        body = self._read_json()
        requested_model = str(body.get("model", "")).strip() or None
        is_stream = body.get("stream") is True
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json(
                400,
                {"error": {"type": "invalid_request_error", "message": "messages is required"}},
            )
            return

        logger.info(
            "POST /v1/messages model=%s messages=%d stream=%s",
            body.get("model", ""),
            len(messages),
            is_stream,
        )

        # Convert Anthropic → OpenAI format
        try:
            payload = anthropic_request_to_openai(body)
        except Exception as e:
            self._send_json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            return

        # Apply thinking mode (adapter already sets "think" field)
        try:
            apply_thinking_mode(payload, default_enabled=self.default_thinking)
        except ValueError as e:
            self._send_json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            return

        # Ensure a model is loaded
        try:
            runtime = self.manager.ensure_model_loaded(
                model=None,
                mmproj=None,
                backend_id=None,
                ctx_size=None,
                launch_args=None,
                request_defaults=None,
            )
        except (ValueError, FileNotFoundError) as e:
            self._send_json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            return
        except RuntimeError as e:
            self._send_json(409, {"error": {"type": "invalid_request_error", "message": str(e)}})
            return

        # Merge runtime defaults and set model
        effective_payload = dict(runtime.request_defaults)
        effective_payload.update(payload)
        payload = effective_payload
        if not payload.get("model"):
            payload["model"] = runtime.model_ref

        response_model = requested_model or payload.get("model", "")

        runtime_mode = self.manager.current_runtime_mode()

        # ── Embedded mode ────────────────────────────────────────────
        if runtime_mode == "embedded":
            try:
                if is_stream:
                    events = self.manager.stream_chat_completion(payload)
                    stream_anthropic_from_embedded(self, events, response_model)
                    _elapsed = time.perf_counter() - _req_start
                    logger.info("POST /v1/messages -> 200 (%.2fs, embedded stream)", _elapsed)
                    return
                response = self.manager.chat_completion(payload)
            except ValueError as e:
                self._send_json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            except RuntimeError as e:
                self._send_json(409, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return

            anthropic_resp = openai_response_to_anthropic(response, response_model)
            _elapsed = time.perf_counter() - _req_start
            logger.info("POST /v1/messages -> 200 (%.2fs, embedded)", _elapsed)
            self._send_json(200, anthropic_resp)
            return

        # ── Proxy mode ───────────────────────────────────────────────
        target = self.manager.current_proxy_target()
        if not target:
            self._send_json(409, {"error": {"type": "invalid_request_error", "message": "selected backend is not ready"}})
            return
        host, port = target

        if is_stream:
            self._debug(f"anthropic proxy stream -> http://{host}:{port}/v1/chat/completions")
            stream_anthropic_proxy(self, host, port, payload, response_model)
            _elapsed = time.perf_counter() - _req_start
            logger.info("POST /v1/messages -> 200 (%.2fs, proxy stream)", _elapsed)
            return

        code, resp_body = http_json(
            "POST",
            f"http://{host}:{port}/v1/chat/completions",
            payload=payload,
            timeout=600,
        )
        _elapsed = time.perf_counter() - _req_start

        if code != 200:
            logger.warning("POST /v1/messages -> %d (%.2fs, proxy)", code, _elapsed)
            try:
                err = json.loads(resp_body)
            except (json.JSONDecodeError, ValueError):
                err = {"error": {"type": "api_error", "message": resp_body.decode("utf-8", errors="replace")}}
            self._send_json(code, err)
            return

        try:
            oai_resp = json.loads(resp_body)
        except json.JSONDecodeError:
            self._send_json(502, {"error": {"type": "api_error", "message": "invalid JSON from backend"}})
            return

        anthropic_resp = openai_response_to_anthropic(oai_resp, response_model)
        logger.info("POST /v1/messages -> 200 (%.2fs, proxy)", _elapsed)
        self._send_json(200, anthropic_resp)


def parse_args(config: dict[str, Any], argv: list[str] | None = None) -> argparse.Namespace:
    backend_names = ", ".join(template.id for template in current_host_platform().backend_templates)
    argv_list = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="OmniInfer unified API service")
    p.add_argument("--host", default=config["host"], help="Gateway bind host")
    p.add_argument("--port", type=int, default=int(config["port"]), help="Gateway bind port")
    p.add_argument(
        "--lan",
        action="store_true",
        help="Expose inference endpoints on the local network with API key protection",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key required for remote LAN clients (or set OMNIINFER_API_KEY)",
    )
    p.add_argument(
        "--allow-insecure-lan",
        action="store_true",
        help="Allow remote LAN inference without an API key (not recommended)",
    )
    p.add_argument(
        "--allow-remote-management",
        action="store_true",
        help="Allow authenticated remote clients to call /omni/* management endpoints",
    )
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
    args = p.parse_args(argv)
    host_was_explicit = any(item == "--host" or item.startswith("--host=") for item in argv_list)
    if args.lan and not host_was_explicit:
        args.host = "0.0.0.0"
    return args


def _shutdown_existing_gateway(host: str, port: int) -> None:
    """Try to shut down an already-running gateway on the same address."""
    shutdown_host = "127.0.0.1" if is_all_interfaces_host(host) else host
    url = f"http://{shutdown_host}:{port}/omni/shutdown"
    req = urllib.request.Request(url=url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        return
    logger.info("Shut down existing gateway on %s:%d", shutdown_host, port)
    time.sleep(2)


def main(argv: list[str] | None = None) -> int:
    config = load_app_config(APP_ROOT)
    args = parse_args(config, argv)
    resolved_api_key = resolve_api_key(args.api_key, config, lan_mode=bool(args.lan))
    api_key = resolved_api_key.value
    remote_bind = should_require_remote_api_key(args.host)
    if remote_bind and not api_key and not args.allow_insecure_lan:
        raise SystemExit(
            "Refusing to expose OmniInfer on a non-loopback host without an API key. "
            "Use --lan to generate a session key, --api-key/OMNIINFER_API_KEY to set one, "
            "or --allow-insecure-lan for trusted test networks."
        )
    if args.allow_remote_management and not api_key:
        raise SystemExit("--allow-remote-management requires --api-key or OMNIINFER_API_KEY")

    # --- Initialize logging ---
    level = args.log_level or resolve_log_level(verbose=args.verbose, debug_body=args.debug_body)
    log_file = setup_logging(level=level, console=True, log_to_file=True)

    # Shut down any existing gateway on the same port to avoid orphan processes
    _shutdown_existing_gateway(args.host, args.port)

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
    httpd.access_policy = GatewayAccessPolicy(  # type: ignore[attr-defined]
        api_key=api_key,
        allow_insecure_lan=bool(args.allow_insecure_lan),
        allow_remote_management=bool(args.allow_remote_management),
    )

    logger.info("OmniInfer listening on http://%s:%s", args.host, args.port)
    log_gateway_access_urls(args.host, args.port, api_key, lan_enabled=remote_bind)
    if resolved_api_key.generated:
        print(f"LAN API key: {api_key}")
        logger.info("Generated a session LAN API key")
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
