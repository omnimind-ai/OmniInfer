#!/usr/bin/env python3
"""RuntimeManager: path resolution, embedded backend lifecycle."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.platforms.linux import LinuxPlatform
from service_core.platforms.mac import MacPlatform
from service_core.runtime import RuntimeManager


# ---------------------------------------------------------------------------
# Runtime root discovery
# ---------------------------------------------------------------------------


class RuntimeRootResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo_root = self.root / "repo"
        self.app_root = self.root / "app"
        self.repo_root.mkdir()
        self.app_root.mkdir()
        self.platform = LinuxPlatform()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prefers_existing_requested_runtime_root(self) -> None:
        requested_root = self.app_root / "custom-runtime"
        requested_root.mkdir()

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
            requested_runtime_root=str(requested_root),
        )

        self.assertEqual(resolved, requested_root.resolve())

    def test_prefers_portable_runtime_root_under_app(self) -> None:
        portable_root = self.app_root / "runtime"
        portable_root.mkdir()

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
        )

        self.assertEqual(resolved, portable_root.resolve())

    def test_prefers_existing_local_runtime_root_under_repo(self) -> None:
        local_root = self.repo_root / ".local" / "runtime" / "linux"
        local_root.mkdir(parents=True)

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
        )

        self.assertEqual(resolved, local_root.resolve())

    def test_returns_canonical_local_runtime_root_when_missing(self) -> None:
        legacy_root = self.repo_root / "platform" / "Linux"
        legacy_root.mkdir(parents=True)

        resolved = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
        )

        self.assertEqual(resolved, (self.repo_root / ".local" / "runtime" / "linux").resolve())


# ---------------------------------------------------------------------------
# Embedded backend lifecycle
# ---------------------------------------------------------------------------


class _FakeEmbeddedDriver:
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
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        repo_root = Path(self.temp_dir.name)
        self.model_dir = repo_root / "models" / "demo-model"
        self.model_dir.mkdir(parents=True)
        self.fake_driver = _FakeEmbeddedDriver()

        self._platform_patch = patch("service_core.runtime.current_host_platform", return_value=MacPlatform())
        self._driver_patch = patch("service_core.runtime.get_embedded_backend_driver", return_value=self.fake_driver)
        self._platform_patch.start()
        self._driver_patch.start()

        self.manager = RuntimeManager(
            repo_root=str(repo_root),
            app_root=str(repo_root),
            backend_host="127.0.0.1",
            backend_port=0,
            startup_timeout_s=10,
            default_backend_id="mlx-mac",
        )
        self.manager.backends["mlx-mac"].python_modules = ()

    def tearDown(self) -> None:
        self._driver_patch.stop()
        self._platform_patch.stop()
        self.temp_dir.cleanup()

    def test_select_model(self) -> None:
        result = self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        self.assertEqual(result["selected_backend"], "mlx-mac")
        self.assertEqual(self.manager.current_runtime_mode(), "embedded")

    def test_chat_completion(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        response = self.manager.chat_completion({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(response["choices"][0]["message"]["content"], "embedded hello")

    def test_stream_chat_completion(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        events = self.manager.stream_chat_completion({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "embedded ")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_snapshot_with_loaded_model(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["backend"], "mlx-mac")
        self.assertTrue(snapshot["backend_ready"])

    def test_stop_runtime(self) -> None:
        self.manager.select_model(model=str(self.model_dir), backend_id="mlx-mac")
        self.manager.stop_runtime()
        self.assertTrue(self.fake_driver.unloaded)

    def test_select_model_invalid_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.select_model(model=str(self.model_dir), backend_id="nonexistent-backend")

    def test_select_model_missing_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.manager.select_model(model="/tmp/does-not-exist-model-dir", backend_id="mlx-mac")

    def test_snapshot_without_loaded_model(self) -> None:
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["backend"], "mlx-mac")
        self.assertIsNone(snapshot["model"])
        self.assertFalse(snapshot["backend_ready"])


if __name__ == "__main__":
    unittest.main()
