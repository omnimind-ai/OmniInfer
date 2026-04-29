#!/usr/bin/env python3
"""HTTP handler tests: exercises OmniHandler via real HTTP against a mock RuntimeManager."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock

from service_core.service import OmniHandler


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

    server = ThreadingHTTPServer(("127.0.0.1", 0), OmniHandler)
    server.manager = manager
    server.default_thinking = False
    server.forced_backend = ""

    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def _get(base_url: str, path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as r:
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

    def test_v1_models_empty(self) -> None:
        code, body = _get(self.base_url, "/v1/models")
        self.assertEqual(code, 200)
        self.assertEqual(body["data"], [])

    def test_not_found(self) -> None:
        code, body = _get(self.base_url, "/nonexistent")
        self.assertEqual(code, 404)

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


if __name__ == "__main__":
    unittest.main()
