#!/usr/bin/env python3
"""CLI serve orchestration tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from service_core import cli


class ServePlanTests(unittest.TestCase):
    def test_parse_serve_plan_strips_orchestration_args(self) -> None:
        plan = cli._parse_serve_plan(
            [
                "--cloudflare",
                "--port",
                "19189",
                "--backend",
                "llama.cpp-linux-cuda",
                "--model",
                "/models/qwen.gguf",
                "--admin-api-key",
                "admin-secret",
                "--ctx-size",
                "8192",
                "--api-key",
                "auto",
                "--detach",
                "--smoke-test",
            ]
        )

        self.assertIsNone(plan.action)
        self.assertEqual(plan.service_args, ["--cloudflare", "--port", "19189"])
        self.assertEqual(plan.backend, "llama.cpp-linux-cuda")
        self.assertEqual(plan.model, "/models/qwen.gguf")
        self.assertEqual(plan.ctx_size, 8192)
        self.assertTrue(plan.detach)
        self.assertTrue(plan.smoke_test)
        self.assertTrue(plan.api_key_generated)
        self.assertIsNotNone(plan.api_key)
        self.assertTrue(str(plan.api_key).startswith("oi_"))

    def test_direct_foreground_preserves_api_key_and_strips_admin_keys(self) -> None:
        with patch("service_core.cli.serve_interactive_or_foreground", return_value=0) as foreground:
            result = cli.serve_command(
                [
                    "--port",
                    "19189",
                    "--api-key",
                    "public-secret",
                    "--admin-api-key",
                    "admin-secret",
                    "--admin-api-keys",
                    "ops:admin-secret",
                ]
            )

        self.assertEqual(result, 0)
        foreground.assert_called_once_with(["--port", "19189", "--api-key", "public-secret"])

    def test_parse_serve_plan_supports_status_action(self) -> None:
        plan = cli._parse_serve_plan(["status", "--port", "19189"])

        self.assertEqual(plan.action, "status")
        self.assertEqual(plan.service_args, ["--port", "19189"])


class ServeOrchestrationTests(unittest.TestCase):
    def test_detached_orchestration_starts_loads_smokes_and_prints(self) -> None:
        plan = cli._parse_serve_plan(
            [
                "--cloudflare",
                "--port",
                "19189",
                "--backend",
                "llama.cpp-linux-cuda",
                "--model",
                "/models/qwen.gguf",
                "--ctx-size",
                "8192",
                "--api-key",
                "secret",
                "--detach",
                "--smoke-test",
            ]
        )

        class FakeProcess:
            pid = 12345

            def poll(self) -> None:
                return None

        with (
            patch("service_core.cli._start_serve_child", return_value=FakeProcess()) as start_child,
            patch("service_core.cli._wait_for_detached_health") as wait_health,
            patch("service_core.cli._wait_for_cloudflare_url", return_value="https://example.trycloudflare.com") as wait_url,
            patch(
                "service_core.cli._configure_serve_runtime",
                return_value={
                    "backend": "llama.cpp-linux-cuda",
                    "backend_ready": True,
                    "model": "/models/qwen.gguf",
                    "ctx_size": 8192,
                },
            ) as configure,
            patch("service_core.cli._serve_smoke", return_value="omniinfer-public-ok") as smoke,
            patch("service_core.cli._write_serve_pid_file") as write_pid,
            patch("service_core.cli._print_serve_ready") as print_ready,
        ):
            result = cli.serve_orchestrated(plan)

        self.assertEqual(result, 0)
        start_child.assert_called_once()
        wait_health.assert_called_once()
        wait_url.assert_called_once()
        configure.assert_called_once_with(plan, port=19189)
        smoke.assert_called_once_with("https://example.trycloudflare.com", api_key="secret")
        write_pid.assert_called_once()
        print_ready.assert_called_once()

    def test_detached_smoke_failure_keeps_started_service(self) -> None:
        plan = cli._parse_serve_plan(
            [
                "--cloudflare",
                "--port",
                "19189",
                "--model",
                "/models/qwen.gguf",
                "--api-key",
                "secret",
                "--detach",
                "--smoke-test",
            ]
        )

        class FakeProcess:
            pid = 12345

            def poll(self) -> None:
                return None

        with (
            patch("service_core.cli._start_serve_child", return_value=FakeProcess()),
            patch("service_core.cli._wait_for_detached_health"),
            patch("service_core.cli._wait_for_cloudflare_url", return_value="https://example.trycloudflare.com"),
            patch(
                "service_core.cli._configure_serve_runtime",
                return_value={
                    "backend": "llama.cpp-linux-cuda",
                    "backend_ready": True,
                    "model": "/models/qwen.gguf",
                },
            ),
            patch("service_core.cli._serve_smoke", side_effect=SystemExit("dns failed")),
            patch("service_core.cli._write_serve_pid_file") as write_pid,
            patch("service_core.cli._print_serve_ready") as print_ready,
            patch("service_core.cli._cleanup_serve_child") as cleanup,
        ):
            result = cli.serve_orchestrated(plan)

        self.assertEqual(result, 1)
        write_pid.assert_called_once()
        print_ready.assert_called_once()
        cleanup.assert_not_called()
        self.assertIn("dns failed", print_ready.call_args.kwargs["smoke_text"])


if __name__ == "__main__":
    unittest.main()
