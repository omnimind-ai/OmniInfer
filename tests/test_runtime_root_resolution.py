#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from service_core.platforms.linux import LinuxPlatform


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


if __name__ == "__main__":
    unittest.main()
