from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
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
            self.assertEqual(remote_access.find_cloudflared(str(binary)), binary)

    def test_platform_asset_selection_linux_amd64(self) -> None:
        with (
            patch("service_core.remote_access.platform.system", return_value="Linux"),
            patch("service_core.remote_access.platform.machine", return_value="x86_64"),
        ):
            self.assertEqual(remote_access.cloudflared_asset_name_for_current_platform(), "cloudflared-linux-amd64")

    def test_platform_asset_selection_macos_arm64(self) -> None:
        with (
            patch("service_core.remote_access.platform.system", return_value="Darwin"),
            patch("service_core.remote_access.platform.machine", return_value="arm64"),
        ):
            self.assertEqual(remote_access.cloudflared_asset_name_for_current_platform(), "cloudflared-darwin-arm64.tgz")

    def test_pinned_release_uses_static_download_url_and_digest(self) -> None:
        with (
            patch("service_core.remote_access.platform.system", return_value="Linux"),
            patch("service_core.remote_access.platform.machine", return_value="x86_64"),
        ):
            release = remote_access.pinned_cloudflared_release()

        self.assertEqual(release.tag_name, "2026.5.0")
        self.assertEqual(release.asset_name, "cloudflared-linux-amd64")
        self.assertEqual(
            release.download_url,
            "https://github.com/cloudflare/cloudflared/releases/download/2026.5.0/cloudflared-linux-amd64",
        )
        self.assertEqual(
            release.digest,
            "sha256:0095e46fdc88855d801c4d304cb1f5dd4bd656116c47ab94c2ad0ae7cda1c7ec",
        )

    def test_install_managed_cloudflared_writes_binary_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp)
            payload = b"fake-cloudflared"
            release = remote_access.CloudflaredReleaseInfo(
                tag_name="2026.5.0",
                asset_name="cloudflared-linux-amd64",
                download_url="https://example.invalid/cloudflared",
                digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            )
            with (
                patch("service_core.remote_access._download_bytes", return_value=payload),
                patch("service_core.remote_access._cloudflared_version", return_value="2026.5.0"),
            ):
                result = remote_access.install_managed_cloudflared(app_root, release)

            self.assertTrue(result.updated)
            self.assertEqual(result.path.read_bytes(), payload)
            manifest = json.loads((app_root / ".local" / "tools" / "cloudflared" / "manifest.json").read_text())
            self.assertEqual(manifest["version"], "2026.5.0")
            self.assertEqual(manifest["asset_name"], "cloudflared-linux-amd64")

    def test_install_managed_cloudflared_rejects_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            release = remote_access.CloudflaredReleaseInfo(
                tag_name="2026.5.0",
                asset_name="cloudflared-linux-amd64",
                download_url="https://example.invalid/cloudflared",
                digest="sha256:" + "0" * 64,
            )
            with patch("service_core.remote_access._download_bytes", return_value=b"fake-cloudflared"):
                with self.assertRaises(remote_access.RemoteAccessError):
                    remote_access.install_managed_cloudflared(Path(tmp), release)

    def test_resolve_cloudflared_reuses_matching_pinned_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp)
            install_dir = app_root / ".local" / "tools" / "cloudflared"
            install_dir.mkdir(parents=True)
            binary = install_dir / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
            binary.write_bytes(b"fake")
            (install_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": "2026.5.0",
                        "asset_name": "cloudflared-linux-amd64",
                        "digest": "sha256:0095e46fdc88855d801c4d304cb1f5dd4bd656116c47ab94c2ad0ae7cda1c7ec",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("service_core.remote_access.platform.system", return_value="Linux"),
                patch("service_core.remote_access.platform.machine", return_value="x86_64"),
                patch("service_core.remote_access._download_bytes") as download,
            ):
                result = remote_access.resolve_cloudflared_for_quick_tunnel(app_root)

            self.assertEqual(result.path, binary)
            self.assertFalse(result.updated)
            download.assert_not_called()

    def test_resolve_cloudflared_installs_pinned_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = b"fake-cloudflared"
            with (
                patch(
                    "service_core.remote_access.pinned_cloudflared_release",
                    return_value=remote_access.CloudflaredReleaseInfo(
                        tag_name="2026.5.0",
                        asset_name="cloudflared-linux-amd64",
                        download_url="https://example.invalid/cloudflared",
                        digest="sha256:" + hashlib.sha256(payload).hexdigest(),
                    ),
                ),
                patch("service_core.remote_access._download_bytes", return_value=payload),
                patch("service_core.remote_access._cloudflared_version", return_value="2026.5.0"),
            ):
                result = remote_access.resolve_cloudflared_for_quick_tunnel(Path(tmp))

            self.assertTrue(result.updated)
            self.assertTrue(result.path.is_file())

    def test_resolve_cloudflared_reinstalls_when_pinned_metadata_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp)
            install_dir = app_root / ".local" / "tools" / "cloudflared"
            install_dir.mkdir(parents=True)
            binary = install_dir / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
            binary.write_bytes(b"old")
            payload = b"new"
            (install_dir / "manifest.json").write_text(
                json.dumps({"version": "2026.5.0", "asset_name": "cloudflared-linux-386"}),
                encoding="utf-8",
            )
            with (
                patch(
                    "service_core.remote_access.pinned_cloudflared_release",
                    return_value=remote_access.CloudflaredReleaseInfo(
                        tag_name="2026.5.0",
                        asset_name="cloudflared-linux-amd64",
                        download_url="https://example.invalid/cloudflared",
                        digest="sha256:" + hashlib.sha256(payload).hexdigest(),
                    ),
                ),
                patch("service_core.remote_access._download_bytes", return_value=payload),
                patch("service_core.remote_access._cloudflared_version", return_value="2026.5.0"),
            ):
                result = remote_access.resolve_cloudflared_for_quick_tunnel(app_root)

            self.assertEqual(result.path, binary)
            self.assertTrue(result.updated)
            self.assertEqual(binary.read_bytes(), payload)

    def test_extract_cloudflared_from_tgz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = b"fake-cloudflared"
            archive_bytes = io.BytesIO()
            with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
                info = tarfile.TarInfo("cloudflared")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            target = Path(tmp) / "cloudflared"
            remote_access._extract_cloudflared_from_tgz(target, archive_bytes.getvalue())
            self.assertEqual(target.read_bytes(), payload)

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
