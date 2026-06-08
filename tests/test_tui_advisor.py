#!/usr/bin/env python3
"""TUI advisor integration helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core import commands
from service_core import tui


def fit_payload() -> dict:
    return {
        "model": {
            "model": "/tmp/model.gguf",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "estimate": {
                "confidence": "medium",
                "estimate_source": "file_size_heuristic",
                "notes": ["estimate note"],
            },
        },
        "recommended": {
            "backend": "llama.cpp-linux-cuda",
            "fit": "good",
            "installed": True,
            "memory_required_gib": 2.0,
            "memory_available_gib": 20.0,
        },
        "alternatives": [],
        "warnings": [],
        "next_command": "omniinfer backend select llama.cpp-linux-cuda && omniinfer model load -m /tmp/model.gguf",
    }


class TuiAdvisorTests(unittest.TestCase):
    def test_advisor_model_details_include_fit_backend_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_text("x", encoding="utf-8")
            recommendations = {
                tui._advisor_path_key(model): {
                    "recommended": {"fit": "good", "backend": "llama.cpp-linux-cuda"},
                    "warnings": ["low confidence"],
                }
            }

            details = tui._advisor_model_details(model, recommendations)

        self.assertEqual(details, ["advisor good", "llama.cpp-linux-cuda", "warning"])

    def test_advisor_preflight_enter_applies_recommended_backend(self) -> None:
        options = commands.ModelLoadOptions(model="/tmp/model.gguf")
        payload = fit_payload()

        with (
            patch("service_core.tui.commands.advisor_fit", return_value=payload),
            patch("service_core.tui._prompt_basic", return_value=""),
            patch("service_core.tui._print_advisor_preflight_summary"),
            patch("service_core.tui._apply_advisor_backend") as apply_backend,
        ):
            result = tui._advisor_preflight(options)

        self.assertEqual(result, options)
        apply_backend.assert_called_once_with(payload["recommended"])

    def test_advisor_preflight_details_then_enter(self) -> None:
        options = commands.ModelLoadOptions(model="/tmp/model.gguf")
        payload = fit_payload()

        with (
            patch("service_core.tui.commands.advisor_fit", return_value=payload),
            patch("service_core.tui._prompt_basic", side_effect=["a", ""]),
            patch("service_core.tui._print_advisor_preflight_summary"),
            patch("service_core.tui._print_advisor_fit_details") as print_details,
            patch("service_core.tui._apply_advisor_backend"),
        ):
            result = tui._advisor_preflight(options)

        self.assertEqual(result, options)
        print_details.assert_called_once_with(payload)


if __name__ == "__main__":
    unittest.main()
