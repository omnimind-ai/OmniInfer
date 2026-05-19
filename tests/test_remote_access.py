from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core import remote_access


class CloudflareQuickTunnelTests(unittest.TestCase):
    def test_parse_trycloudflare_url(self) -> None:
        line = "INF + https://whole-cod-associates-treasure.trycloudflare.com"
        self.assertEqual(
            remote_access.parse_trycloudflare_url(line),
            "https://whole-cod-associates-treasure.trycloudflare.com",
        )

    def test_parse_ignores_non_quick_tunnel_url(self) -> None:
        self.assertIsNone(remote_access.parse_trycloudflare_url("https://example.com"))

    def test_find_cloudflared_prefers_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "cloudflared"
            binary.write_text("", encoding="utf-8")
            with patch("service_core.remote_access.shutil.which", return_value=None):
                self.assertEqual(remote_access.find_cloudflared(str(binary)), binary)

    def test_find_cloudflared_uses_environment_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "cloudflared"
            binary.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"OMNIINFER_CLOUDFLARED": str(binary)}):
                with patch("service_core.remote_access.shutil.which", return_value=None):
                    self.assertEqual(remote_access.find_cloudflared(), binary)

    def test_find_cloudflared_reports_missing_binary(self) -> None:
        with patch.dict(os.environ, {"OMNIINFER_CLOUDFLARED": ""}):
            with patch("service_core.remote_access.shutil.which", return_value=None):
                with self.assertRaises(remote_access.RemoteAccessError):
                    remote_access.find_cloudflared()

    def test_quick_tunnel_config_warning_when_default_config_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / ".cloudflared" / "config.yaml"
            config.parent.mkdir()
            config.write_text("tunnel: example\n", encoding="utf-8")
            with patch("service_core.remote_access.Path.home", return_value=root):
                warning = remote_access.quick_tunnel_config_warning()
        self.assertIsNotNone(warning)
        self.assertIn("config.yaml", warning or "")

    def test_quick_tunnel_config_warning_absent_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("service_core.remote_access.Path.home", return_value=Path(tmp)):
                self.assertIsNone(remote_access.quick_tunnel_config_warning())


if __name__ == "__main__":
    unittest.main()
