from __future__ import annotations

import importlib.util
import stat
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "platforms" / "linux" / "release_runtime_backends.py"
SPEC = importlib.util.spec_from_file_location("release_runtime_backends", MODULE_PATH)
assert SPEC and SPEC.loader
release_runtime_backends = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_runtime_backends
SPEC.loader.exec_module(release_runtime_backends)


def _write_executable(path: Path, content: str = "#!/usr/bin/env bash\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class LinuxReleaseBackendDiscoveryTests(unittest.TestCase):
    def test_discovers_launcher_and_embedded_python_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_executable(root / "llama.cpp-linux" / "bin" / "llama-server")
            _write_executable(root / "vllm-linux-cuda" / "bin" / "vllm")
            _write_executable(root / "mnn-linux" / "bin" / "python3")

            packages = release_runtime_backends.discover_runtime_packages(root)
            by_id = {package.id: package for package in packages}

            self.assertIn("llama.cpp-linux", by_id)
            self.assertIn("vllm-linux-cuda", by_id)
            self.assertIn("mnn-linux", by_id)
            self.assertEqual(by_id["llama.cpp-linux"].copy_mode, "binary-bin")
            self.assertEqual(by_id["vllm-linux-cuda"].copy_mode, "full-runtime")
            self.assertEqual(by_id["mnn-linux"].copy_mode, "full-runtime")

    def test_skips_incomplete_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vllm-linux-cuda" / "bin").mkdir(parents=True)
            (root / "mnn-linux" / "venv").mkdir(parents=True)

            packages = release_runtime_backends.discover_runtime_packages(root)

            self.assertNotIn("vllm-linux-cuda", {package.id for package in packages})
            self.assertNotIn("mnn-linux", {package.id for package in packages})


class LinuxReleaseBackendCopyTests(unittest.TestCase):
    def test_binary_bin_copy_keeps_launchers_and_shared_libraries_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runtime" / "llama.cpp-linux"
            target = root / "release" / "runtime"
            _write_executable(source / "bin" / "llama-server")
            (source / "bin" / "libllama.so").write_text("so", encoding="utf-8")
            (source / "bin" / "README.txt").write_text("skip", encoding="utf-8")
            (source / "build").mkdir()
            package = release_runtime_backends.RuntimePackage(
                id="llama.cpp-linux",
                runtime_dir_name="llama.cpp-linux",
                source_dir=str(source),
                copy_mode="binary-bin",
                launcher_name="llama-server",
                runtime_mode="external_server",
                priority=1,
            )

            release_runtime_backends.copy_runtime_package(package, target)

            copied = target / "llama.cpp-linux"
            self.assertTrue((copied / "bin" / "llama-server").is_file())
            self.assertTrue((copied / "bin" / "libllama.so").is_file())
            self.assertFalse((copied / "bin" / "README.txt").exists())
            self.assertFalse((copied / "build").exists())
            self.assertTrue((copied / "logs").is_dir())

    def test_full_runtime_copy_rewrites_runtime_path_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runtime" / "vllm-linux-cuda"
            target = root / "release" / "runtime"
            _write_executable(
                source / "bin" / "vllm",
                f"#!{source}/bin/python\nfrom vllm.scripts import main\n",
            )
            (source / "pyvenv.cfg").write_text(f"home = {source}\n", encoding="utf-8")
            (source / "lib" / "python3.10" / "site-packages").mkdir(parents=True)
            (source / "lib" / "python3.10" / "site-packages" / "vllm.py").write_text("ok", encoding="utf-8")
            (source / "build").mkdir()
            (source / "logs").mkdir()
            (source / "models").mkdir()
            package = release_runtime_backends.RuntimePackage(
                id="vllm-linux-cuda",
                runtime_dir_name="vllm-linux-cuda",
                source_dir=str(source),
                copy_mode="full-runtime",
                launcher_name="vllm",
                runtime_mode="external_server",
                priority=2,
            )

            release_runtime_backends.copy_runtime_package(package, target)

            copied = target / "vllm-linux-cuda"
            self.assertTrue((copied / "bin" / "vllm").is_file())
            self.assertTrue((copied / "lib" / "python3.10" / "site-packages" / "vllm.py").is_file())
            self.assertFalse((copied / "build").exists())
            self.assertFalse((copied / "models").exists())
            self.assertTrue((copied / "logs").is_dir())
            self.assertIn(str(copied), (copied / "bin" / "vllm").read_text(encoding="utf-8"))
            self.assertIn(str(copied), (copied / "pyvenv.cfg").read_text(encoding="utf-8"))
            self.assertNotIn(str(source), (copied / "bin" / "vllm").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
