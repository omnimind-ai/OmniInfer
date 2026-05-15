#!/usr/bin/env python3
"""HTTP handler tests: exercises OmniHandler via real HTTP against a mock RuntimeManager."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from service_core.service import (
    ChatReasoningStreamNormalizer,
    ChatUsageStreamNormalizer,
    GatewayAccessPolicy,
    OmniHandler,
    apply_thinking_mode,
    normalize_chat_completion_reasoning,
    parse_args,
    split_thinking_text,
)


def _create_test_server() -> tuple[ThreadingHTTPServer, str]:
    """Start a real HTTP server on a random port with a mock RuntimeManager."""
    manager = MagicMock()
    manager.snapshot.return_value = {
        "backend": "llama.cpp-cpu",
        "model": None,
        "mmproj": None,
        "ctx_size": None,
        "request_defaults": {},
        "backend_ready": False,
    }
    manager.list_backends.return_value = ([], None)
    manager.backend_props.return_value = {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), OmniHandler)
    server.manager = manager
    server.default_thinking = False
    server.forced_backend = ""
    server.access_policy = GatewayAccessPolicy()

    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def _get(base_url: str, path: str, headers: dict | None = None) -> tuple[int, Any]:
    req = urllib.request.Request(f"{base_url}{path}", method="GET", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.getcode(), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(base_url: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, Any]:
    data = json.dumps(body or {}).encode() if body is not None else b"{}"
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(f"{base_url}{path}", data=data, method="POST", headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.getcode(), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class HttpHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _create_test_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    # --- GET endpoints ---

    def test_health(self) -> None:
        code, body = _get(self.base_url, "/health")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIn("omni", body)

    def test_state(self) -> None:
        code, body = _get(self.base_url, "/omni/state")
        self.assertEqual(code, 200)
        self.assertIn("backend", body)

    def test_backends(self) -> None:
        code, body = _get(self.base_url, "/omni/backends")
        self.assertEqual(code, 200)
        self.assertEqual(body["object"], "list")

    def test_backends_invalid_scope(self) -> None:
        code, body = _get(self.base_url, "/omni/backends?scope=invalid")
        self.assertEqual(code, 400)
        self.assertIn("invalid scope", body["error"]["message"])

    def test_thinking(self) -> None:
        code, body = _get(self.base_url, "/omni/thinking")
        self.assertEqual(code, 200)
        self.assertIn("default_enabled", body)

    def test_backend_props(self) -> None:
        self.server.manager.backend_props.return_value = {"n_ctx": 4096}
        code, body = _get(self.base_url, "/omni/backend/props")
        self.assertEqual(code, 200)
        self.assertEqual(body["n_ctx"], 4096)
        self.server.manager.backend_props.return_value = {}

    def test_v1_models_empty(self) -> None:
        code, body = _get(self.base_url, "/v1/models")
        self.assertEqual(code, 200)
        self.assertEqual(body["data"], [])

    def test_health_deep_no_backend(self) -> None:
        self.server.manager.backend_health.return_value = {"status": "no_backend"}
        code, body = _get(self.base_url, "/health?deep=true")
        self.assertEqual(code, 200)
        self.assertEqual(body["backend_health"]["status"], "no_backend")

    def test_health_shallow_has_no_backend_health_key(self) -> None:
        code, body = _get(self.base_url, "/health")
        self.assertEqual(code, 200)
        self.assertNotIn("backend_health", body)

    def test_not_found(self) -> None:
        code, body = _get(self.base_url, "/nonexistent")
        self.assertEqual(code, 404)

    def test_remote_public_endpoint_requires_api_key(self) -> None:
        self.server.access_policy = GatewayAccessPolicy(api_key="secret")
        try:
            with patch("service_core.service.is_loopback_address", return_value=False):
                code, body = _get(self.base_url, "/health")
        finally:
            self.server.access_policy = GatewayAccessPolicy()

        self.assertEqual(code, 401)
        self.assertIn("API key", body["error"]["message"])

    def test_remote_public_endpoint_accepts_bearer_key(self) -> None:
        self.server.access_policy = GatewayAccessPolicy(api_key="secret")
        try:
            with patch("service_core.service.is_loopback_address", return_value=False):
                code, body = _get(self.base_url, "/health", {"Authorization": "Bearer secret"})
        finally:
            self.server.access_policy = GatewayAccessPolicy()

        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")

    def test_remote_management_endpoint_is_local_only(self) -> None:
        self.server.access_policy = GatewayAccessPolicy(api_key="secret")
        try:
            with patch("service_core.service.is_loopback_address", return_value=False):
                code, body = _post(
                    self.base_url,
                    "/omni/shutdown",
                    {},
                    {"Authorization": "Bearer secret"},
                )
        finally:
            self.server.access_policy = GatewayAccessPolicy()

        self.assertEqual(code, 403)
        self.assertIn("management endpoints are local-only", body["error"]["message"])

    # --- POST error cases ---

    def test_oversized_content_length_rejected(self) -> None:
        import socket as _socket
        _, _, port_str = self.base_url.rpartition(":")
        with _socket.create_connection(("127.0.0.1", int(port_str)), timeout=5) as sock:
            raw_request = (
                "POST /v1/chat/completions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 999999999999\r\n"
                "\r\n"
            )
            sock.sendall(raw_request.encode())
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks).decode("utf-8", errors="replace")
        self.assertIn("400", response)
        self.assertIn("too large", response)

    def test_backend_select_missing_field(self) -> None:
        code, body = _post(self.base_url, "/omni/backend/select", {})
        self.assertEqual(code, 400)
        self.assertIn("required", body["error"]["message"])

    def test_thinking_select_missing_field(self) -> None:
        code, body = _post(self.base_url, "/omni/thinking/select", {})
        self.assertEqual(code, 400)
        self.assertIn("required", body["error"]["message"])

    def test_thinking_select_persists_state(self) -> None:
        with patch("service_core.service.save_default_thinking") as save:
            code, body = _post(self.base_url, "/omni/thinking/select", {"enabled": True})

        self.assertEqual(code, 200)
        self.assertTrue(body["default_enabled"])
        save.assert_called_once()
        self.assertTrue(save.call_args.args[0])
        self.server.default_thinking = False

    def test_model_select_missing_model(self) -> None:
        code, body = _post(self.base_url, "/omni/model/select", {})
        self.assertEqual(code, 400)
        self.assertIn("required", body["error"]["message"])

    def test_post_not_found(self) -> None:
        code, body = _post(self.base_url, "/nonexistent", {})
        self.assertEqual(code, 404)

    # --- OPTIONS (CORS preflight) ---

    def test_options_cors(self) -> None:
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.getcode(), 204)


class ThinkingModeTests(unittest.TestCase):
    def test_reasoning_effort_enables_thinking(self) -> None:
        payload: dict[str, Any] = {"reasoning": {"effort": "high"}}
        apply_thinking_mode(payload, default_enabled=False)
        self.assertNotIn("reasoning", payload)
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("reasoning_format", payload)

    def test_top_level_reasoning_effort_enables_thinking(self) -> None:
        payload: dict[str, Any] = {"reasoning_effort": "high"}
        apply_thinking_mode(payload, default_enabled=False)
        self.assertNotIn("reasoning_effort", payload)
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("reasoning_format", payload)

    def test_reasoning_effort_none_disables_thinking(self) -> None:
        payload: dict[str, Any] = {"reasoning": {"effort": "none"}}
        apply_thinking_mode(payload, default_enabled=True)
        self.assertNotIn("reasoning", payload)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["reasoning_format"], "none")

    def test_think_overrides_reasoning_effort(self) -> None:
        payload: dict[str, Any] = {"think": False, "reasoning": {"effort": "high"}}
        apply_thinking_mode(payload, default_enabled=True)
        self.assertNotIn("reasoning", payload)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])

    def test_chat_template_kwargs_override_reasoning_effort(self) -> None:
        payload: dict[str, Any] = {
            "reasoning": {"effort": "high"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        apply_thinking_mode(payload, default_enabled=True)
        self.assertNotIn("reasoning", payload)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])


class ReasoningContentTests(unittest.TestCase):
    def test_split_empty_thinking_block(self) -> None:
        self.assertEqual(split_thinking_text("<think>\n\n</think>\n\nhello"), ("", "hello"))

    def test_normalizes_non_stream_message_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nplan\n</think>\n\nanswer",
                    }
                }
            ]
        }
        normalize_chat_completion_reasoning(payload)
        message = payload["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "plan")
        self.assertEqual(message["content"], "answer")

    def test_stream_normalizer_splits_reasoning_delta(self) -> None:
        normalizer = ChatReasoningStreamNormalizer()
        event = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>plan</think>answer"},
                    "finish_reason": None,
                }
            ],
        }
        chunks = normalizer.transform_event(event)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["reasoning_content"], "plan")
        self.assertIsNone(chunks[0]["choices"][0]["delta"]["content"])
        flushed = normalizer.flush(event)
        self.assertEqual(flushed[0]["choices"][0]["delta"]["content"], "answer")

    def test_stream_normalizer_suppresses_empty_thinking_block(self) -> None:
        normalizer = ChatReasoningStreamNormalizer()
        event = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "<think>\n\n</think>\n\nanswer"},
                    "finish_reason": None,
                }
            ],
        }
        chunks = normalizer.transform_event(event)
        flushed = normalizer.flush(event)
        self.assertEqual(chunks, [])
        self.assertEqual(flushed[0]["choices"][0]["delta"]["content"], "answer")

    def test_usage_stream_normalizer_moves_usage_to_final_event(self) -> None:
        normalizer = ChatUsageStreamNormalizer()
        first = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        second = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }

        transformed = [*normalizer.transform_event(first), *normalizer.transform_event(second), *normalizer.flush()]

        self.assertEqual(len(transformed), 3)
        self.assertNotIn("usage", transformed[0])
        self.assertEqual(transformed[0]["choices"][0]["delta"]["content"], "a")
        self.assertNotIn("usage", transformed[1])
        self.assertEqual(transformed[1]["choices"][0]["delta"]["content"], "b")
        self.assertEqual(transformed[2]["choices"], [])
        self.assertEqual(transformed[2]["usage"]["total_tokens"], 4)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class ConfigValidationTests(unittest.TestCase):
    def _validate(self, overrides: dict) -> None:
        from service_core.service import _validate_config
        config = {"host": "127.0.0.1", "port": 9000, "startup_timeout": 60}
        config.update(overrides)
        _validate_config(config, Path("test.json"))

    def test_valid_config_passes(self) -> None:
        self._validate({})  # should not raise

    def test_port_zero_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"port": 0})

    def test_port_negative_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"port": -1})

    def test_port_too_high_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"port": 70000})

    def test_port_string_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"port": "9000"})

    def test_timeout_zero_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"startup_timeout": 0})

    def test_timeout_negative_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"startup_timeout": -10})

    def test_host_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._validate({"host": ""})

    def test_lan_mode_defaults_to_all_interfaces(self) -> None:
        args = parse_args(
            {"host": "127.0.0.1", "port": 9000, "default_backend": "", "default_thinking": "off", "startup_timeout": 60},
            ["--lan"],
        )
        self.assertEqual(args.host, "0.0.0.0")


if __name__ == "__main__":
    unittest.main()
