from __future__ import annotations

import unittest
from unittest.mock import patch

from service_core.model_catalog import SupportedModelCatalog


class FakePlatform:
    system_name = "windows"

    def resolve_catalog_backend_id(self, backend_id: str) -> str:
        return backend_id

    def available_memory_gib_for_backend(self, backend_name: str) -> float:
        return 999.0

    def safety_margin_gib_for_backend(self, backend_name: str) -> float:
        return 0.0


class ModelCatalogTests(unittest.TestCase):
    def test_supported_models_load_from_bundled_catalog_without_network(self) -> None:
        catalog = SupportedModelCatalog(FakePlatform(), ["llama.cpp-cpu", "llama.cpp-cuda"])

        with patch("urllib.request.urlopen", side_effect=AssertionError("catalog should be local")):
            payload = catalog.list_supported_models("windows")

        self.assertIn("llama.cpp-cpu", payload)
        self.assertIn("llama.cpp-cuda", payload)

    def test_supported_models_best_uses_installed_backend_filter(self) -> None:
        catalog = SupportedModelCatalog(FakePlatform(), ["llama.cpp-cuda"])

        payload = catalog.list_supported_models_best("windows")

        self.assertIsInstance(payload, dict)
        quantizations = [
            quant_info
            for family_models in payload.values()
            if isinstance(family_models, dict)
            for model_info in family_models.values()
            if isinstance(model_info, dict)
            for quant_info in model_info.get("quantization", {}).values()
            if isinstance(quant_info, dict)
        ]
        self.assertTrue(quantizations)
        self.assertTrue(all(quant_info.get("backend") in {"llama.cpp-cuda", ""} for quant_info in quantizations))

    def test_supported_models_reject_unknown_system(self) -> None:
        catalog = SupportedModelCatalog(FakePlatform(), [])

        with self.assertRaisesRegex(ValueError, "windows, mac, linux"):
            catalog.list_supported_models("android")


if __name__ == "__main__":
    unittest.main()
