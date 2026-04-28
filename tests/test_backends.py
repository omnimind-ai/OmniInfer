#!/usr/bin/env python3
"""Backend registration, argument parsing, and launch command construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.backend_cli_args import (
    parse_backend_chat_extra_args,
    parse_backend_load_extra_args,
)
from service_core.backends.base import BackendSpec
from service_core.platforms.linux import LinuxPlatform
from service_core.platforms.mac import MacPlatform
from service_core.runtime import RuntimeManager


def make_backend(backend_id: str, family: str) -> BackendSpec:
    return BackendSpec(
        id=backend_id,
        label=backend_id,
        family=family,
        runtime_dir=".",
        launcher_path=None,
        models_dir=None,
        catalog_url=None,
        description="",
        capabilities=[],
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class CliArgParserTests(unittest.TestCase):
    def test_llama_cpp_load_args_passthrough_with_ctx_size(self) -> None:
        backend = make_backend("llama.cpp-vulkan", "llama.cpp")

        parsed = parse_backend_load_extra_args(
            backend,
            ["-ngl", "99", "-t", "8", "--threads-batch", "4", "-c", "4096"],
        )

        self.assertEqual(parsed.ctx_size, 4096)
        self.assertEqual(parsed.launch_args, ["-ngl", "99", "-t", "8", "--threads-batch", "4"])

    def test_turboquant_chat_args_use_llama_compatible_parser(self) -> None:
        backend = make_backend("turboquant-mac", "turboquant")

        parsed = parse_backend_chat_extra_args(
            backend,
            ["-n", "128", "--top-k", "40", "--stop", "<END>", "--stop", "</END>", "-e"],
        )

        self.assertEqual(parsed.request_overrides["max_tokens"], 128)
        self.assertEqual(parsed.request_overrides["top_k"], 40)
        self.assertEqual(parsed.request_overrides["ignore_eos"], True)
        self.assertEqual(parsed.request_overrides["stop"], ["<END>", "</END>"])

    def test_mlx_chat_args_accept_image_again(self) -> None:
        backend = make_backend("mlx-mac", "mlx-lm")

        parsed = parse_backend_chat_extra_args(
            backend,
            ["--image", "demo.png", "--temperature", "0.3", "--top-p", "0.8", "--stop", "<END>"],
        )

        self.assertEqual(parsed.image, "demo.png")
        self.assertEqual(parsed.request_overrides["temperature"], 0.3)
        self.assertEqual(parsed.request_overrides["top_p"], 0.8)
        self.assertEqual(parsed.request_overrides["stop"], ["<END>"])

    def test_empty_args_returns_defaults(self) -> None:
        backend = make_backend("llama.cpp-cpu", "llama.cpp")
        parsed = parse_backend_load_extra_args(backend, [])
        self.assertIsNone(parsed.ctx_size)
        self.assertEqual(parsed.launch_args, [])

    def test_reserved_flags_rejected_in_load_args(self) -> None:
        backend = make_backend("llama.cpp-cpu", "llama.cpp")
        for flag in ["-m", "--model", "-mm", "--mmproj"]:
            with self.assertRaises(ValueError, msg=f"{flag} should be rejected"):
                parse_backend_load_extra_args(backend, [flag, "value"])

    def test_unsupported_chat_arg_rejected(self) -> None:
        backend = make_backend("llama.cpp-cpu", "llama.cpp")
        with self.assertRaises(ValueError):
            parse_backend_chat_extra_args(backend, ["-Z"])  # single-char flags not in known set

    def test_mlx_load_args_not_supported(self) -> None:
        backend = make_backend("mlx-mac", "mlx-lm")
        with self.assertRaises(ValueError):
            parse_backend_load_extra_args(backend, ["-c", "4096"])


# ---------------------------------------------------------------------------
# Platform registration
# ---------------------------------------------------------------------------


class PlatformRegistrationTests(unittest.TestCase):
    def test_mac_platform_registers_all_expected_backends(self) -> None:
        ids = {t.id for t in MacPlatform().backend_templates}
        for expected in ["llama.cpp-mac", "turboquant-mac", "mlx-mac"]:
            self.assertIn(expected, ids)

    def test_linux_platform_registers_all_expected_backends(self) -> None:
        ids = {t.id for t in LinuxPlatform().backend_templates}
        for expected in ["llama.cpp-linux", "mnn-linux"]:
            self.assertIn(expected, ids)

    def test_all_templates_have_unique_ids(self) -> None:
        for platform_cls in [MacPlatform, LinuxPlatform]:
            templates = platform_cls().backend_templates
            ids = [t.id for t in templates]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate IDs in {platform_cls.__name__}")


# ---------------------------------------------------------------------------
# Launch command construction
# ---------------------------------------------------------------------------


class LaunchCommandTests(unittest.TestCase):
    """Turboquant launch command construction tests."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        repo_root = Path(self.temp_dir.name)
        runtime_bin = repo_root / ".local" / "runtime" / "macos" / "turboquant-mac" / "bin"
        runtime_bin.mkdir(parents=True)
        self.launcher_path = runtime_bin / "llama-server"
        self.launcher_path.write_text("", encoding="utf-8")

        with patch("service_core.runtime.current_host_platform", return_value=MacPlatform()):
            self.manager = RuntimeManager(
                repo_root=str(repo_root),
                app_root=str(repo_root),
                backend_host="127.0.0.1",
                backend_port=9100,
                startup_timeout_s=10,
                default_backend_id="turboquant-mac",
            )

        backend = self.manager.backends["turboquant-mac"]
        self.launch = self.manager._prepare_external_runtime_launch(
            backend,
            model_path="/tmp/qwen3.5-2b.gguf",
            mmproj_path="/tmp/mmproj.gguf",
            ctx_size=8192,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_port_and_ctx_size(self) -> None:
        self.assertEqual(self.launch.port, 9100)
        self.assertEqual(self.launch.ctx_size, 8192)

    def test_launcher_path_and_log_name(self) -> None:
        self.assertEqual(self.launch.cmd[0], str(self.launcher_path.resolve()))
        self.assertEqual(self.launch.log_file_name, "turboquant-server.log")

    def test_flash_attention_flags(self) -> None:
        cmd = self.launch.cmd
        self.assertEqual(cmd[cmd.index("-fa") + 1], "on")

    def test_cache_type_flags(self) -> None:
        cmd = self.launch.cmd
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "turbo4")
        self.assertIn("--cache-type-v", cmd)

    def test_model_and_mmproj_flags(self) -> None:
        cmd = self.launch.cmd
        self.assertEqual(cmd[cmd.index("-mm") + 1], "/tmp/mmproj.gguf")
        self.assertEqual(cmd[cmd.index("-c") + 1], "8192")


class EmbeddedBackendTests(unittest.TestCase):
    def test_mnn_embedded_backend_properties(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
