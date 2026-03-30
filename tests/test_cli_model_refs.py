#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from service_core import cli


class CliModelReferenceTests(unittest.TestCase):
    def test_model_load_accepts_directory_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "demo-model"
            model_dir.mkdir()
            args = argparse.Namespace(model=str(model_dir), mmproj=None, ctx_size=None, auto=False)

            with patch("service_core.cli.require_selected_backend", return_value="mlx-mac"):
                with patch(
                    "service_core.cli.request_json",
                    return_value=(
                        200,
                        {
                            "selected_backend": "mlx-mac",
                            "selected_model": str(model_dir),
                            "selected_mmproj": None,
                            "selected_ctx_size": None,
                        },
                        b"",
                    ),
                ):
                    with redirect_stdout(io.StringIO()):
                        result = cli.print_model_load(args)

        self.assertEqual(result, 0)
