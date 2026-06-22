#!/usr/bin/env python3
"""Advisor model inspection and backend fit planning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.advisor import fit_model, inspect_model
from service_core.backends.base import BackendSpec


class FakePlatform:
    system_name = "linux"

    def is_hardware_compatible(self, backend: BackendSpec) -> bool:
        return True


def make_backend(
    backend_id: str,
    *,
    family: str = "llama.cpp",
    capabilities: list[str] | None = None,
    launcher_path: str | None = None,
    model_artifact: str = "file",
    default_args: list[str] | None = None,
) -> BackendSpec:
    return BackendSpec(
        id=backend_id,
        label=backend_id,
        family=family,
        runtime_dir=".",
        launcher_path=launcher_path,
        models_dir=None,
        catalog_url=None,
        description="",
        capabilities=capabilities or ["chat", "gpu", "cuda"],
        default_args=default_args or [],
        model_artifact=model_artifact,
    )


class AdvisorTests(unittest.TestCase):
    def test_inspect_gguf_infers_quantization_params_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Qwen3.5-2B-Q4_K_M.gguf"
            model.write_bytes(b"x" * 1024 * 1024)

            payload = inspect_model(str(model))

        self.assertEqual(payload["format"], "gguf")
        self.assertEqual(payload["quantization"], "Q4_K_M")
        self.assertEqual(payload["params_b"], 2.0)
        self.assertIn("chat", payload["capabilities"])
        self.assertEqual(payload["estimate"]["estimate_source"], "file_size_heuristic")
        breakdown = payload["estimate"]["breakdown"]
        self.assertIn("weights_gib", breakdown)
        self.assertIn("kv_cache_gib", breakdown)
        self.assertIn("activation_gib", breakdown)
        self.assertIn("framework_overhead_gib", breakdown)
        self.assertIn("allocator_slack_gib", breakdown)

    def test_fit_prefers_official_installed_cuda_and_builds_valid_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Qwen3.5-2B-Q4_K_M.gguf"
            model.write_bytes(b"x" * 1024 * 1024)
            official_launcher = root / "llama-server"
            official_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            ik_launcher = root / "ik-llama-server"
            ik_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            backends = {
                "llama.cpp-linux-cuda": make_backend(
                    "llama.cpp-linux-cuda",
                    launcher_path=str(official_launcher),
                    default_args=["-ngl", "999"],
                ),
                "ik_llama.cpp-linux-cuda": make_backend(
                    "ik_llama.cpp-linux-cuda",
                    launcher_path=str(ik_launcher),
                    default_args=["--jinja", "-ngl", "999"],
                ),
            }

            with patch(
                "service_core.advisor._query_cuda_devices",
                return_value=[{"index": "0", "free_gib": 20.0, "utilization_pct": 0}],
            ):
                payload = fit_model(
                    str(model),
                    platform_obj=FakePlatform(),
                    backends=backends,
                    ctx_size=8192,
                )

        self.assertEqual(payload["recommended"]["backend"], "llama.cpp-linux-cuda")
        self.assertEqual(payload["recommended"]["memory_kind"], "gpu")
        self.assertIn("memory_breakdown", payload["recommended"])
        self.assertGreater(payload["recommended"]["memory_breakdown"]["kv_cache_gib"], 0)
        self.assertIn("omniinfer backend select llama.cpp-linux-cuda", payload["next_command"])
        self.assertIn("--ctx-size 8192 -ngl 999", payload["next_command"])
        self.assertNotIn("--ctx-size 8192 8192", payload["next_command"])

    def test_hf_reference_is_compatible_with_vllm_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "vllm"
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            backends = {
                "llama.cpp-linux-cuda": make_backend("llama.cpp-linux-cuda", launcher_path=str(launcher)),
                "vllm-linux-cuda": make_backend(
                    "vllm-linux-cuda",
                    family="vllm",
                    capabilities=["chat", "gpu", "cuda"],
                    launcher_path=str(launcher),
                    model_artifact="reference",
                ),
            }

            with patch(
                "service_core.advisor._query_cuda_devices",
                return_value=[{"index": "0", "free_gib": 20.0, "utilization_pct": 0}],
            ):
                payload = fit_model(
                    "Qwen/Qwen2.5-7B-Instruct",
                    platform_obj=FakePlatform(),
                    backends=backends,
                )

        self.assertEqual(payload["model"]["format"], "hf-reference")
        self.assertEqual(payload["recommended"]["backend"], "vllm-linux-cuda")
        llama = [item for item in payload["all_backends"] if item["backend"] == "llama.cpp-linux-cuda"][0]
        self.assertFalse(llama["compatible"])


if __name__ == "__main__":
    unittest.main()
