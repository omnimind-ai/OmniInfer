#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.platforms.mac import MacPlatform
from service_core.runtime import RuntimeManager


class FakeEmbeddedDriver:
    def __init__(self) -> None:
        self.unloaded = False

    def load_model(
        self,
        *,
        model_path: str,
        model_ref: str,
        mmproj_path: str | None,
        ctx_size: int | None,
        load_options=None,
    ):
        return {
            "model_path": model_path,
            "model_ref": model_ref,
            "mmproj_path": mmproj_path,
            "ctx_size": ctx_size,
            "load_options": dict(load_options or {}),
        }

    def unload_model(self, state):
        self.unloaded = True
        self.state = state

    def chat_completion(self, state, payload):
        return {
            "object": "chat.completion",
            "model": payload.get("model") or state["model_ref"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "embedded hello"}, "finish_reason": "stop"}],
        }

    def stream_chat_completion(self, state, payload):
        yield {
            "object": "chat.completion.chunk",
            "model": payload.get("model") or state["model_ref"],
            "choices": [{"index": 0, "delta": {"content": "embedded "}, "finish_reason": None}],
        }
        yield {
            "object": "chat.completion.chunk",
            "model": payload.get("model") or state["model_ref"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


class EmbeddedRuntimeTests(unittest.TestCase):
    def test_runtime_manager_supports_embedded_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            model_dir = repo_root / "models" / "demo-model"
            model_dir.mkdir(parents=True)
            fake_driver = FakeEmbeddedDriver()

            with patch("service_core.runtime.current_host_platform", return_value=MacPlatform()):
                with patch("service_core.runtime.get_embedded_backend_driver", return_value=fake_driver):
                    manager = RuntimeManager(
                        repo_root=str(repo_root),
                        app_root=str(repo_root),
                        backend_host="127.0.0.1",
                        backend_port=0,
                        startup_timeout_s=10,
                        default_backend_id="mlx-mac",
                    )
                    manager.backends["mlx-mac"].python_modules = ()

                    result = manager.select_model(model=str(model_dir), backend_id="mlx-mac")
                    self.assertEqual(result["selected_backend"], "mlx-mac")
                    self.assertEqual(manager.current_runtime_mode(), "embedded")

                    response = manager.chat_completion({"messages": [{"role": "user", "content": "hello"}]})
                    self.assertEqual(response["choices"][0]["message"]["content"], "embedded hello")

                    events = manager.stream_chat_completion({"messages": [{"role": "user", "content": "hello"}]})
                    self.assertEqual(events[0]["choices"][0]["delta"]["content"], "embedded ")

                    snapshot = manager.snapshot()
                    self.assertEqual(snapshot["backend"], "mlx-mac")
                    self.assertTrue(snapshot["backend_ready"])

                    manager.stop_runtime()
                    self.assertTrue(fake_driver.unloaded)
