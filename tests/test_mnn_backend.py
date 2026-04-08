#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.platforms.linux import LinuxPlatform
from service_core.runtime import RuntimeManager


class MnnBackendTests(unittest.TestCase):
    def test_linux_platform_registers_mnn_backend(self) -> None:
        backend_ids = [template.id for template in LinuxPlatform().backend_templates]
        self.assertIn("mnn-linux", backend_ids)

    def test_runtime_manager_builds_embedded_mnn_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            model_dir = repo_root / "models" / "demo-model"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with patch("service_core.runtime.current_host_platform", return_value=LinuxPlatform()):
                manager = RuntimeManager(
                    repo_root=str(repo_root),
                    app_root=str(repo_root),
                    backend_host="127.0.0.1",
                    backend_port=0,
                    startup_timeout_s=10,
                    default_backend_id="mnn-linux",
                )

            backend = manager.backends["mnn-linux"]
            self.assertEqual(backend.runtime_mode, "embedded")
            self.assertFalse(backend.supports_mmproj)
            self.assertFalse(backend.supports_ctx_size)
