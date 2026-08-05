#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from service_core.service import OmniHandler, normalize_prompt_cache_key


class RecordingBackendHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        self.server.payloads.append(payload)  # type: ignore[attr-defined]

        if payload.get("stream") is True:
            body = (
                b'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n'
                b"data: [DONE]\n\n"
            )
            content_type = "text/event-stream; charset=utf-8"
        else:
            body = json.dumps(
                {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeRuntimeManager:
    def __init__(
        self,
        backend_address: tuple[str, int],
        request_defaults: dict[str, object] | None = None,
    ) -> None:
        self.backend_address = backend_address
        self.request_defaults = dict(request_defaults or {})
        self.select_calls: list[dict[str, object]] = []

    def ensure_model_loaded(self, **kwargs: object) -> SimpleNamespace:
        request_defaults = kwargs.get("request_defaults")
        if request_defaults is not None:
            self.request_defaults = dict(request_defaults)  # type: ignore[arg-type]
        return SimpleNamespace(request_defaults=self.request_defaults, model_ref="test-model")

    def current_runtime_mode(self) -> str:
        return "external_server"

    def current_proxy_target(self) -> tuple[str, int]:
        return self.backend_address

    def select_model(self, **kwargs: object) -> dict[str, object]:
        self.select_calls.append(kwargs)
        return {"ok": True}


class GatewayPromptCacheKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), RecordingBackendHandler)
        self.backend.payloads = []  # type: ignore[attr-defined]
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()

        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), OmniHandler)
        self.gateway.manager = FakeRuntimeManager(self.backend.server_address)  # type: ignore[attr-defined]
        self.gateway.default_thinking = False  # type: ignore[attr-defined]
        self.gateway.debug_body = False  # type: ignore[attr-defined]
        self.gateway.forced_backend = ""  # type: ignore[attr-defined]
        self.gateway_thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.request_log_patch = patch("service_core.service._save_request_response")
        self.request_log_patch.start()
        self.gateway_thread.start()

    def tearDown(self) -> None:
        self.gateway.shutdown()
        self.gateway.server_close()
        self.gateway_thread.join(timeout=5)
        self.request_log_patch.stop()
        self.backend.shutdown()
        self.backend.server_close()
        self.backend_thread.join(timeout=5)

    def post_chat(self, payload: dict[str, object]) -> tuple[int, bytes]:
        return self.post_json("/v1/chat/completions", payload)

    def post_json(self, path: str, payload: dict[str, object]) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url=f"http://127.0.0.1:{self.gateway.server_port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_non_stream_request_preserves_prompt_cache_key(self) -> None:
        status, body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": "openclaw/session-a",
                "stream": False,
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.backend.payloads[-1]["prompt_cache_key"], "openclaw/session-a")  # type: ignore[attr-defined]

    def test_stream_request_preserves_prompt_cache_key(self) -> None:
        status, body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": "openclaw/session-b",
                "stream": True,
            }
        )

        self.assertEqual(status, 200)
        self.assertIn(b"data: [DONE]", body)
        self.assertEqual(self.backend.payloads[-1]["prompt_cache_key"], "openclaw/session-b")  # type: ignore[attr-defined]

    def test_runtime_default_prompt_cache_key_is_forwarded(self) -> None:
        self.gateway.manager.request_defaults = {  # type: ignore[attr-defined]
            "prompt_cache_key": "openclaw/default-session"
        }

        status, _body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(  # type: ignore[attr-defined]
            self.backend.payloads[-1]["prompt_cache_key"],
            "openclaw/default-session",
        )

    def test_request_prompt_cache_key_overrides_runtime_default(self) -> None:
        self.gateway.manager.request_defaults = {  # type: ignore[attr-defined]
            "prompt_cache_key": "openclaw/default-session"
        }

        status, _body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": "openclaw/request-session",
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(  # type: ignore[attr-defined]
            self.backend.payloads[-1]["prompt_cache_key"],
            "openclaw/request-session",
        )

    def test_empty_or_null_key_disables_runtime_default(self) -> None:
        self.gateway.manager.request_defaults = {  # type: ignore[attr-defined]
            "prompt_cache_key": "openclaw/default-session"
        }

        for value in (None, ""):
            with self.subTest(value=value):
                status, _body = self.post_chat(
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "prompt_cache_key": value,
                    }
                )

                self.assertEqual(status, 200)
                self.assertNotIn(  # type: ignore[attr-defined]
                    "prompt_cache_key",
                    self.backend.payloads[-1],
                )

    def test_invalid_prompt_cache_key_combinations_are_rejected(self) -> None:
        invalid_payloads = (
            {"prompt_cache_key": 123},
            {"prompt_cache_key": "\ud800"},
            {"prompt_cache_key": "x" * 257},
            {"prompt_cache_key": "session-a", "id_slot": 0},
            {"prompt_cache_key": "session-a", "cache_prompt": False},
            {"prompt_cache_key": "session-a", "n": 2},
            {"prompt_cache_key": "session-a", "n_cmpl": 2},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    normalize_prompt_cache_key(payload)

    def test_multibyte_prompt_cache_key_uses_utf8_byte_limit(self) -> None:
        valid_payload = {"prompt_cache_key": "会" * 85 + "x"}
        normalize_prompt_cache_key(valid_payload)
        self.assertIn("prompt_cache_key", valid_payload)

        with self.assertRaises(ValueError):
            normalize_prompt_cache_key({"prompt_cache_key": "会" * 86})

    def test_invalid_prompt_cache_key_request_does_not_reach_backend(self) -> None:
        status, body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": "openclaw/session-a",
                "id_slot": 0,
            }
        )

        self.assertEqual(status, 400)
        self.assertIn("cannot be combined", json.loads(body)["error"]["message"])
        self.assertEqual(self.backend.payloads, [])  # type: ignore[attr-defined]

    def test_model_select_rejects_invalid_request_defaults_before_loading(self) -> None:
        status, body = self.post_json(
            "/omni/model/select",
            {
                "model": "test-model.gguf",
                "request_defaults": {
                    "prompt_cache_key": "openclaw/session-a",
                    "id_slot": 0,
                },
            },
        )

        self.assertEqual(status, 400)
        self.assertIn("cannot be combined", json.loads(body)["error"]["message"])
        self.assertEqual(self.gateway.manager.select_calls, [])  # type: ignore[attr-defined]

    def test_invalid_request_defaults_do_not_replace_runtime_defaults(self) -> None:
        self.gateway.manager.request_defaults = {"temperature": 0.25}  # type: ignore[attr-defined]

        status, _body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "request_defaults": {
                    "prompt_cache_key": "openclaw/session-a",
                    "id_slot": 0,
                },
            }
        )

        self.assertEqual(status, 400)
        self.assertEqual(  # type: ignore[attr-defined]
            self.gateway.manager.request_defaults,
            {"temperature": 0.25},
        )
        self.assertEqual(self.backend.payloads, [])  # type: ignore[attr-defined]

        status, _body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello again"}],
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(self.backend.payloads[-1]["temperature"], 0.25)  # type: ignore[attr-defined]
        self.assertNotIn("prompt_cache_key", self.backend.payloads[-1])  # type: ignore[attr-defined]
        self.assertNotIn("id_slot", self.backend.payloads[-1])  # type: ignore[attr-defined]

    def test_request_defaults_conflicting_with_request_do_not_replace_runtime_defaults(self) -> None:
        self.gateway.manager.request_defaults = {"temperature": 0.25}  # type: ignore[attr-defined]

        status, _body = self.post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "id_slot": 0,
                "request_defaults": {"prompt_cache_key": "openclaw/session-a"},
            }
        )

        self.assertEqual(status, 400)
        self.assertEqual(  # type: ignore[attr-defined]
            self.gateway.manager.request_defaults,
            {"temperature": 0.25},
        )
        self.assertEqual(self.backend.payloads, [])  # type: ignore[attr-defined]

    def test_empty_prompt_cache_key_disables_session_routing(self) -> None:
        for value in (None, ""):
            with self.subTest(value=value):
                payload = {"prompt_cache_key": value}
                normalize_prompt_cache_key(payload)
                self.assertNotIn("prompt_cache_key", payload)


if __name__ == "__main__":
    unittest.main()
