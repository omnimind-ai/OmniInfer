#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.platforms.mac import MacPlatform
from service_core.runtime import RuntimeManager


class TurboQuantBackendTests(unittest.TestCase):
    def test_mac_platform_registers_turboquant_backend(self) -> None:
        backend_ids = [template.id for template in MacPlatform().backend_templates]
        self.assertIn("turboquant-mac", backend_ids)

    def test_runtime_manager_builds_turboquant_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            runtime_bin = repo_root / ".local" / "runtime" / "macos" / "turboquant-mac" / "bin"
            runtime_bin.mkdir(parents=True)
            launcher_path = runtime_bin / "llama-server"
            launcher_path.write_text("", encoding="utf-8")

            with patch("service_core.runtime.current_host_platform", return_value=MacPlatform()):
                manager = RuntimeManager(
                    repo_root=str(repo_root),
                    app_root=str(repo_root),
                    backend_host="127.0.0.1",
                    backend_port=9100,
                    startup_timeout_s=10,
                    default_backend_id="turboquant-mac",
                )

            backend = manager.backends["turboquant-mac"]
            launch = manager._prepare_external_runtime_launch(
                backend,
                model_path="/tmp/qwen3.5-2b.gguf",
                mmproj_path="/tmp/mmproj.gguf",
                ctx_size=8192,
            )

            self.assertEqual(launch.port, 9100)
            self.assertEqual(launch.ctx_size, 8192)
            self.assertEqual(launch.log_file_name, "turboquant-server.log")
            self.assertEqual(launch.cmd[0], str(launcher_path.resolve()))
            self.assertIn("-fa", launch.cmd)
            self.assertIn("on", launch.cmd)
            self.assertIn("--cache-type-k", launch.cmd)
            self.assertIn("turbo4", launch.cmd)
            self.assertIn("--cache-type-v", launch.cmd)
            self.assertIn("-mm", launch.cmd)
            self.assertEqual(launch.cmd[launch.cmd.index("-c") + 1], "8192")


if __name__ == "__main__":
    unittest.main()
