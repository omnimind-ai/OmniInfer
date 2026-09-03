import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_model_catalog_assets.py"
SPEC = importlib.util.spec_from_file_location("validate_model_catalog_assets", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


class ModelCatalogAssetTests(unittest.TestCase):
    def test_bundled_catalogs_use_shared_exact_asset_metadata(self):
        assets = VALIDATOR.validate_local()
        self.assertGreaterEqual(len(assets), 2)
        qwen4b = {
            url.rsplit("/", 1)[-1]: metadata
            for url, metadata in assets.items()
            if "/unsloth/Qwen3.5-4B-GGUF/" in url
        }
        self.assertEqual(qwen4b["Qwen3.5-4B-Q4_K_M.gguf"]["size_bytes"], 2_740_937_888)
        self.assertEqual(qwen4b["mmproj-F16.gguf"]["size_bytes"], 672_423_616)

    def test_content_range_parser_requires_exact_total(self):
        match = VALIDATOR.CONTENT_RANGE_RE.fullmatch("bytes 0-0/3413361504")
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 3_413_361_504)
        self.assertIsNone(VALIDATOR.CONTENT_RANGE_RE.fullmatch("bytes */3413361504"))


if __name__ == "__main__":
    unittest.main()
