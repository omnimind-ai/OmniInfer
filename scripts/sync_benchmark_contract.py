#!/usr/bin/env python3
"""Synchronize the reviewed OmniStudio benchmark producer contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://omnistudio.omnimind.com.cn/benchmarks/contract"
DEFAULT_DESTINATION = REPOSITORY_ROOT / "benchmarks" / "contract"
ARTIFACT_NAMES = ("schema.json", "catalog-index.json")
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a benchmark contract is malformed or inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number: {value}")


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"{label} is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ContractError) as error:
        raise ContractError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.rstrip("/"))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ContractError(
            "contract base URL must use HTTPS without credentials, a query, or a fragment"
        )
    return urllib.parse.urlunsplit(parsed).rstrip("/")


def _required_string(value: dict[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ContractError(f"{label}.{key} must be a non-empty string")
    return result


def validate_contract(
    manifest_raw: bytes,
    artifacts: dict[str, bytes],
    base_url: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_url = validate_base_url(base_url)
    manifest = parse_json(manifest_raw, "manifest.json")
    expected_manifest_keys = {
        "artifact_base_url",
        "catalog_schema_version",
        "contract_version",
        "files",
        "policy_version",
        "schema_version",
        "source",
    }
    if set(manifest) != expected_manifest_keys:
        raise ContractError("manifest.json has an unexpected field set")
    if manifest.get("artifact_base_url") != base_url:
        raise ContractError("manifest artifact_base_url does not match the download source")
    contract_version = _required_string(manifest, "contract_version", "manifest")
    schema_version = _required_string(manifest, "schema_version", "manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ContractError("manifest.source must be an object")
    for key in ("catalog_url", "schema_url", "source_commit", "source_timestamp"):
        _required_string(source, key, "manifest.source")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_NAMES):
        raise ContractError("manifest.files must describe exactly the two contract artifacts")

    parsed_artifacts: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_NAMES:
        raw = artifacts.get(name)
        if raw is None:
            raise ContractError(f"missing downloaded artifact: {name}")
        entry = files.get(name)
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
            raise ContractError(f"manifest.files.{name} is malformed")
        expected_bytes = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or not 0 < expected_bytes <= MAX_ARTIFACT_BYTES
        ):
            raise ContractError(f"manifest.files.{name}.bytes is outside the allowed range")
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            raise ContractError(f"manifest.files.{name}.sha256 is invalid")
        if len(raw) != expected_bytes:
            raise ContractError(
                f"{name} byte count mismatch: expected {expected_bytes}, got {len(raw)}"
            )
        actual_hash = sha256(raw)
        if actual_hash != expected_hash:
            raise ContractError(
                f"{name} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        parsed_artifacts[name] = parse_json(raw, name)

    schema = parsed_artifacts["schema.json"]
    if schema.get("$schema") != "http://json-schema.org/draft-07/schema#":
        raise ContractError("schema.json must declare JSON Schema Draft-07")
    declared_schema_version = (
        schema.get("properties", {}).get("schema_version", {}).get("const")
        if isinstance(schema.get("properties"), dict)
        else None
    )
    if declared_schema_version != schema_version:
        raise ContractError("schema.json version does not match manifest.schema_version")

    catalog = parsed_artifacts["catalog-index.json"]
    expected_catalog_keys = {
        "backends",
        "catalog_schema_version",
        "contract_version",
        "data_kind",
        "devices",
        "models",
        "platforms",
        "schema_version",
        "source_commit",
        "source_timestamp",
    }
    if set(catalog) != expected_catalog_keys:
        raise ContractError("catalog-index.json has an unexpected field set")
    if catalog.get("data_kind") != "benchmark-catalog-index":
        raise ContractError("catalog-index.json has the wrong data_kind")
    if catalog.get("contract_version") != contract_version:
        raise ContractError("catalog contract_version does not match the manifest")
    if catalog.get("schema_version") != schema_version:
        raise ContractError("catalog schema_version does not match the manifest")
    if catalog.get("catalog_schema_version") != manifest.get("catalog_schema_version"):
        raise ContractError("catalog schema version does not match the manifest")
    if catalog.get("source_commit") != source.get("source_commit"):
        raise ContractError("catalog source_commit does not match the manifest")
    if catalog.get("source_timestamp") != source.get("source_timestamp"):
        raise ContractError("catalog source_timestamp does not match the manifest")
    for key in ("backends", "devices", "models", "platforms"):
        if not isinstance(catalog.get(key), list):
            raise ContractError(f"catalog-index.json.{key} must be an array")

    snapshot = {
        "artifact_base_url": base_url,
        "catalog_schema_version": manifest["catalog_schema_version"],
        "contract_version": contract_version,
        "manifest": {"bytes": len(manifest_raw), "sha256": sha256(manifest_raw)},
        "schema_version": schema_version,
        "source": source,
    }
    return manifest, snapshot


def _download(url: str, maximum_bytes: int, expected_bytes: int | None = None) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "OmniInfer contract sync"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme != "https" or final.netloc != parsed.netloc:
            raise ContractError("contract download redirected outside the original HTTPS origin")
        content_type = response.headers.get_content_type()
        if content_type not in {"application/json", "application/schema+json"}:
            raise ContractError(f"unexpected content type for {url}: {content_type}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise ContractError(f"invalid Content-Length for {url}") from error
            limit = expected_bytes if expected_bytes is not None else maximum_bytes
            if declared_length > maximum_bytes or (
                expected_bytes is not None and declared_length != limit
            ):
                raise ContractError(f"unsafe Content-Length for {url}: {declared_length}")
        raw = response.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise ContractError(f"download exceeds {maximum_bytes} bytes: {url}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ContractError(
            f"download byte count mismatch for {url}: expected {expected_bytes}, got {len(raw)}"
        )
    return raw


def fetch_contract(base_url: str) -> tuple[bytes, dict[str, bytes]]:
    base_url = validate_base_url(base_url)
    manifest_raw = _download(f"{base_url}/manifest.json", MAX_MANIFEST_BYTES)
    manifest = parse_json(manifest_raw, "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ContractError("manifest.files must be an object")
    artifacts: dict[str, bytes] = {}
    for name in ARTIFACT_NAMES:
        entry = files.get(name)
        expected_bytes = entry.get("bytes") if isinstance(entry, dict) else None
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or not 0 < expected_bytes <= MAX_ARTIFACT_BYTES
        ):
            raise ContractError(f"manifest.files.{name}.bytes is outside the allowed range")
        artifacts[name] = _download(
            f"{base_url}/{name}", MAX_ARTIFACT_BYTES, expected_bytes
        )
    validate_contract(manifest_raw, artifacts, base_url)
    return manifest_raw, artifacts


def _atomic_write(path: Path, raw: bytes) -> None:
    if path.is_symlink():
        raise ContractError(f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_contract(
    manifest_raw: bytes,
    artifacts: dict[str, bytes],
    destination: Path,
    base_url: str,
) -> bool:
    _, snapshot = validate_contract(manifest_raw, artifacts, base_url)
    planned = {
        "schema.json": artifacts["schema.json"],
        "catalog-index.json": artifacts["catalog-index.json"],
        "snapshot.json": canonical_json(snapshot),
        "manifest.json": manifest_raw,
    }
    if destination.is_symlink():
        raise ContractError(f"refusing to use symlink destination: {destination}")
    unchanged = destination.is_dir() and all(
        (destination / name).is_file() and (destination / name).read_bytes() == raw
        for name, raw in planned.items()
    )
    if unchanged:
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("schema.json", "catalog-index.json", "snapshot.json", "manifest.json"):
        _atomic_write(destination / name, planned[name])
    return True


def check_contract(destination: Path) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise ContractError(f"contract directory is missing or unsafe: {destination}")
    files: dict[str, bytes] = {}
    for name in ("manifest.json", *ARTIFACT_NAMES, "snapshot.json"):
        path = destination / name
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"contract file is missing or unsafe: {path}")
        files[name] = path.read_bytes()
    snapshot = parse_json(files["snapshot.json"], "snapshot.json")
    base_url = _required_string(snapshot, "artifact_base_url", "snapshot")
    _, expected_snapshot = validate_contract(
        files["manifest.json"],
        {name: files[name] for name in ARTIFACT_NAMES},
        base_url,
    )
    expected_raw = canonical_json(expected_snapshot)
    if files["snapshot.json"] != expected_raw:
        raise ContractError("snapshot.json provenance does not match the vendored contract")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the vendored snapshot without any network access",
    )
    args = parser.parse_args()
    try:
        if args.check:
            check_contract(args.destination)
            print("benchmark contract is valid and current")
            return 0
        base_url = validate_base_url(args.base_url)
        manifest_raw, artifacts = fetch_contract(base_url)
        changed = install_contract(manifest_raw, artifacts, args.destination, base_url)
        check_contract(args.destination)
        print("benchmark contract updated" if changed else "benchmark contract unchanged")
        return 0
    except (ContractError, OSError, urllib.error.URLError) as error:
        parser.exit(1, f"benchmark contract error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
