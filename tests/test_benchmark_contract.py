import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_benchmark_contract.py"
SPEC = importlib.util.spec_from_file_location("sync_benchmark_contract", SCRIPT)
CONTRACT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONTRACT)


class BenchmarkContractTests(unittest.TestCase):
    def setUp(self):
        self.source = ROOT / "benchmarks" / "contract"
        self.manifest = (self.source / "manifest.json").read_bytes()
        self.artifacts = {
            name: (self.source / name).read_bytes() for name in CONTRACT.ARTIFACT_NAMES
        }

    def test_vendored_contract_passes_offline_check(self):
        CONTRACT.check_contract(self.source)

    def test_install_is_idempotent_when_upstream_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "contract"
            self.assertTrue(
                CONTRACT.install_contract(
                    self.manifest,
                    self.artifacts,
                    destination,
                    CONTRACT.DEFAULT_BASE_URL,
                )
            )
            timestamps = {
                path.name: path.stat().st_mtime_ns for path in destination.iterdir()
            }
            self.assertFalse(
                CONTRACT.install_contract(
                    self.manifest,
                    self.artifacts,
                    destination,
                    CONTRACT.DEFAULT_BASE_URL,
                )
            )
            self.assertEqual(
                timestamps,
                {path.name: path.stat().st_mtime_ns for path in destination.iterdir()},
            )

    def test_offline_check_rejects_tampered_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "contract"
            CONTRACT.install_contract(
                self.manifest,
                self.artifacts,
                destination,
                CONTRACT.DEFAULT_BASE_URL,
            )
            schema = destination / "schema.json"
            schema.write_bytes(schema.read_bytes() + b"\n")
            with self.assertRaisesRegex(CONTRACT.ContractError, "byte count mismatch"):
                CONTRACT.check_contract(destination)

    def test_parser_rejects_duplicate_keys_and_non_finite_numbers(self):
        with self.assertRaisesRegex(CONTRACT.ContractError, "duplicate JSON key"):
            CONTRACT.parse_json(b'{"key": 1, "key": 2}', "duplicate.json")
        with self.assertRaisesRegex(CONTRACT.ContractError, "non-finite JSON number"):
            CONTRACT.parse_json(b'{"value": NaN}', "nan.json")

    def test_source_must_use_https(self):
        with self.assertRaisesRegex(CONTRACT.ContractError, "must use HTTPS"):
            CONTRACT.validate_base_url("http://omnistudio.example/contract")


if __name__ == "__main__":
    unittest.main()
