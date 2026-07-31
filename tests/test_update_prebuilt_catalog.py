from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_prebuilt_catalog.py"
SPEC = importlib.util.spec_from_file_location("update_prebuilt_catalog", MODULE_PATH)
assert SPEC and SPEC.loader
update_prebuilt_catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = update_prebuilt_catalog
SPEC.loader.exec_module(update_prebuilt_catalog)

VENDORED_REVISION = "1" * 40
SUBMODULE_REVISION = "2" * 40


class SourceCheckoutRevisionTests(unittest.TestCase):
    def test_reads_vendored_revision_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "framework" / "llama.cpp"
            source_root.mkdir(parents=True)
            (source_root / ".omniinfer-upstream-revision").write_text(
                f"{VENDORED_REVISION}\n",
                encoding="utf-8",
            )

            with mock.patch.object(update_prebuilt_catalog, "REPO_ROOT", root):
                actual = update_prebuilt_catalog.source_checkout_revision(
                    "framework/llama.cpp"
                )

            self.assertEqual(actual, VENDORED_REVISION)

    def test_falls_back_to_submodule_gitlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "framework" / "vllm").mkdir(parents=True)

            with (
                mock.patch.object(update_prebuilt_catalog, "REPO_ROOT", root),
                mock.patch.object(
                    update_prebuilt_catalog,
                    "gitlink_commit",
                    return_value=SUBMODULE_REVISION,
                ) as gitlink_commit,
            ):
                actual = update_prebuilt_catalog.source_checkout_revision("framework/vllm")

            self.assertEqual(actual, SUBMODULE_REVISION)
            gitlink_commit.assert_called_once_with("framework/vllm")

    def test_source_match_uses_vendored_revision_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "framework" / "llama.cpp"
            source_root.mkdir(parents=True)
            (source_root / ".omniinfer-upstream-revision").write_text(
                f"{VENDORED_REVISION}\n",
                encoding="utf-8",
            )
            catalog = {
                "schema_version": 5,
                "sources": {
                    "ggml-org/llama.cpp": {
                        "tag": "b1",
                        "submodule_tag": "b1",
                        "submodule_path": "framework/llama.cpp",
                        "submodule_commit": VENDORED_REVISION,
                    }
                },
                "platforms": {},
                "python_runtimes": {},
            }

            with mock.patch.object(update_prebuilt_catalog, "REPO_ROOT", root):
                errors = update_prebuilt_catalog.validate(
                    catalog,
                    require_source_match=True,
                    verify_upstream_tags=False,
                )

            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
