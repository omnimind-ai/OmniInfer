"""Anthropic Messages API <-> OpenAI Chat Completions format converter.

Handles both request/response conversion and SSE stream conversion.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler
from typing import Any


def _make_msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def _normalize_system(system: Any) -> str | None:
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts = [
            b["text"]
            for b in system
            if isinstance(b, dict) and b.get("type") == "text" and "text" in b
        ]
        return "\n".join(parts) if parts else None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Request conversion: Anthropic → OpenAI
# ──────────────────────────────────────────────────────────────────────────────

def anthropic_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages API request body to OpenAI chat/completions format."""
    oai: dict[str, Any] = {}

    if "model" in body:
        oai["model"] = body["model"]

    messages: list[dict[str, Any]] = []

    system_text = _normalize_system(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages") or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": ""})
            continue

        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        tool_use_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        tool_result_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]

        if role == "assistant" and tool_use_blocks:
            # Assistant message with tool calls
            text = text_blocks[0].get("text", "") if text_blocks else ""
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {
                        "id": b.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    }
                    for i, b in enumerate(tool_use_blocks)
                ],
            })
        elif role == "user" and tool_result_blocks:
            # Tool result messages
            for b in tool_result_blocks:
                rc = b.get("content", "")
                if isinstance(rc, list):
                    rc = " ".join(
                        x.get("text", "")
                        for x in rc
                        if isinstance(x, dict) and x.get("type") == "text"
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id", ""),
                    "content": str(rc),
                })
            if text_blocks:
                messages.append({"role": "user", "content": text_blocks[0].get("text", "")})
        else:
            # Build OAI content list (text + images)
            oai_parts: list[dict[str, Any]] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    oai_parts.append({"type": "text", "text": b.get("text", "")})
                elif btype == "image":
                    source = b.get("source") or {}
                    if source.get("type") == "base64":
                        oai_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                            },
                        })
                    elif source.get("type") == "url":
                        oai_parts.append({
                            "type": "image_url",
                            "image_url": {"url": source.get("url", "")},
                        })

            if len(oai_parts) == 1 and oai_parts[0].get("type") == "text":
                messages.append({"role": role, "content": oai_parts[0]["text"]})
            elif oai_parts:
                messages.append({"role": role, "content": oai_parts})
            else:
                messages.append({"role": role, "content": ""})

    oai["messages"] = messages

    # Scalar parameters
    if "max_tokens" in body:
        oai["max_tokens"] = body["max_tokens"]
    if "stop_sequences" in body:
        oai["stop"] = body["stop_sequences"]
    for field in ("temperature", "top_p"):
        if field in body:
            oai[field] = body[field]
    if "top_k" in body:
        oai["top_k"] = body["top_k"]
    if "stream" in body:
        oai["stream"] = body["stream"]

    # thinking → think (OmniInfer's field)
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            oai["think"] = True
            if "budget_tokens" in thinking:
                oai["thinking_budget"] = thinking["budget_tokens"]
        elif thinking.get("type") == "disabled":
            oai["think"] = False

    # tools: input_schema → function.parameters
    tools = body.get("tools")
    if tools:
        oai["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {},
                },
            }
            for t in tools
            if isinstance(t, dict)
        ]

    # tool_choice — NEVER use "required" (llama.cpp crashes)
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type in ("auto", "any"):
            oai["tool_choice"] = "auto"
        elif tc_type == "tool":
            oai["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
        elif tc_type == "none":
            oai["tool_choice"] = "none"

    return oai


# ──────────────────────────────────────────────────────────────────────────────
# Response conversion: OpenAI → Anthropic (non-streaming)
# ──────────────────────────────────────────────────────────────────────────────

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def openai_response_to_anthropic(response: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an OpenAI chat/completions response to Anthropic Messages API format."""
    choices = response.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or "stop"
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    content: list[dict[str, Any]] = []

    msg_content = message.get("content")
    if msg_content:
        content.append({"type": "text", "text": msg_content})

    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        try:
            input_data = json.loads(func.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:24]),
            "name": func.get("name", ""),
            "input": input_data,
        })

    usage = response.get("usage") or {}
    return {
        "id": _make_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Streaming conversion: OpenAI SSE → Anthropic SSE
# ──────────────────────────────────────────────────────────────────────────────

def _write_sse(wfile: Any, event_type: str, data: dict[str, Any]) -> None:
    line = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    wfile.write(line.encode("utf-8"))
    wfile.flush()


class _OAIToAnthropicStreamConverter:
    """Stateful converter: consumes OpenAI chunk dicts, emits Anthropic SSE events."""

    def __init__(self, wfile: Any, model: str, msg_id: str) -> None:
        self._wfile = wfile
        self._model = model
        self._msg_id = msg_id
        # index → {"id": str, "name": str} for started tool_use blocks
        self._tool_blocks: dict[int, dict[str, str]] = {}
        self._text_started = False
        self._finish_reason = "end_turn"
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        _write_sse(self._wfile, event_type, data)

    def send_preamble(self) -> None:
        self._write("message_start", {
            "type": "message_start",
            "message": {
                "id": self._msg_id,
                "type": "message",
                "role": "assistant",
                "model": self._model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        self._write("ping", {"type": "ping"})

    def process_chunk(self, data: dict[str, Any]) -> None:
        choices = data.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}

            # Text delta
            text = delta.get("content")
            if text:
                if not self._text_started:
                    self._write("content_block_start", {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    })
                    self._text_started = True
                self._write("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                })

            # Tool call deltas
            for tc_delta in delta.get("tool_calls") or []:
                tc_idx = tc_delta.get("index", 0)
                # Offset by 1 if a text block occupies index 0
                block_idx = tc_idx + (1 if self._text_started else 0)

                if block_idx not in self._tool_blocks:
                    func = tc_delta.get("function") or {}
                    block_id = tc_delta.get("id") or ("toolu_" + uuid.uuid4().hex[:24])
                    block_name = func.get("name", "")
                    self._tool_blocks[block_idx] = {"id": block_id, "name": block_name}
                    self._write("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": block_id,
                            "name": block_name,
                            "input": {},
                        },
                    })
                    # Emit any initial argument fragment
                    func = tc_delta.get("function") or {}
                    args_frag = func.get("arguments") or ""
                    if args_frag:
                        self._write("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args_frag},
                        })
                else:
                    func = tc_delta.get("function") or {}
                    args_frag = func.get("arguments") or ""
                    if args_frag:
                        self._write("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args_frag},
                        })

            fr = choice.get("finish_reason")
            if fr:
                self._finish_reason = _FINISH_REASON_MAP.get(fr, "end_turn")

        # Extract token counts from usage or timings
        usage = data.get("usage") or {}
        if usage:
            self._prompt_tokens = usage.get("prompt_tokens", self._prompt_tokens)
            self._completion_tokens = usage.get("completion_tokens", self._completion_tokens)
        timings = data.get("timings") or {}
        if timings:
            self._prompt_tokens = timings.get("prompt_n", self._prompt_tokens)
            self._completion_tokens = timings.get("predicted_n", self._completion_tokens)

    def send_epilogue(self) -> None:
        # Ensure at least one content block was opened
        if not self._text_started and not self._tool_blocks:
            self._write("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            self._write("content_block_stop", {"type": "content_block_stop", "index": 0})
        else:
            if self._text_started:
                self._write("content_block_stop", {"type": "content_block_stop", "index": 0})
            for block_idx in sorted(self._tool_blocks):
                self._write("content_block_stop", {"type": "content_block_stop", "index": block_idx})

        self._write("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": self._finish_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": self._completion_tokens},
        })
        self._write("message_stop", {"type": "message_stop"})


def _send_sse_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization, anthropic-version, x-api-key",
    )
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()


def stream_anthropic_proxy(
    handler: BaseHTTPRequestHandler,
    host: str,
    port: int,
    payload: dict[str, Any],
    model: str,
) -> None:
    """Open a streaming connection to the llama.cpp proxy and re-emit as Anthropic SSE."""
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "text/event-stream, application/json",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"http://{host}:{port}/v1/chat/completions"
    req = urllib.request.Request(url=url, data=body, method="POST", headers=headers)

    msg_id = _make_msg_id()
    converter = _OAIToAnthropicStreamConverter(handler.wfile, model, msg_id)

    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            _send_sse_headers(handler)
            converter.send_preamble()

            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    line_b = line_b.strip()
                    if not line_b or line_b == b"data: [DONE]":
                        continue
                    if line_b.startswith(b"data: "):
                        try:
                            data = json.loads(line_b[6:])
                            converter.process_chunk(data)
                        except json.JSONDecodeError:
                            pass

            converter.send_epilogue()

    except urllib.error.HTTPError as e:
        err_body = e.read()
        try:
            err_data = json.loads(err_body)
        except Exception:
            err_data = {"error": {"message": f"backend error {e.code}"}}
        handler.send_response(e.code)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps(err_data, ensure_ascii=False).encode("utf-8"))
    except urllib.error.URLError as e:
        err_data = {"error": {"message": f"backend unreachable: {e}"}}
        handler.send_response(599)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps(err_data, ensure_ascii=False).encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass


def stream_anthropic_from_embedded(
    handler: BaseHTTPRequestHandler,
    events: list[dict[str, Any]],
    model: str,
) -> None:
    """Convert embedded-mode OpenAI event list to Anthropic SSE and write to client."""
    msg_id = _make_msg_id()
    converter = _OAIToAnthropicStreamConverter(handler.wfile, model, msg_id)

    _send_sse_headers(handler)
    try:
        converter.send_preamble()
        for event in events:
            converter.process_chunk(event)
        converter.send_epilogue()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass
