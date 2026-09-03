#!/usr/bin/env python3
"""Validate exact, shared download metadata for bundled model catalogs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = REPO_ROOT / "crates" / "omniinfer-core" / "model_catalogs"
ASSET_PATH = CATALOG_DIR / "download_assets.json"
CATALOG_PATHS = tuple(CATALOG_DIR / name for name in ("linux.json", "mac.json", "windows.json"))
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CONTENT_RANGE_RE = re.compile(r"bytes\s+\d+-\d+/(\d+)", re.IGNORECASE)


class CatalogValidationError(RuntimeError):
    """Raised when bundled catalog metadata violates its contract."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"cannot read {path}: {exc}") from exc


def iter_artifacts(value: Any):
    if isinstance(value, dict):
        if "download" in value:
            yield value
        for child in value.values():
            yield from iter_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_artifacts(child)


def validate_local() -> dict[str, dict[str, Any]]:
    manifest = load_json(ASSET_PATH)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CatalogValidationError("download asset schema_version must be 1")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise CatalogValidationError("download asset manifest must contain assets")

    references: Counter[str] = Counter()
    for catalog_path in CATALOG_PATHS:
        catalog = load_json(catalog_path)
        if not isinstance(catalog, dict):
            raise CatalogValidationError(f"{catalog_path.name} root must be an object")
        for artifact in iter_artifacts(catalog):
            download = artifact.get("download")
            estimate = artifact.get("memory_estimate_gib")
            urls = [download] if isinstance(download, str) else download
            if (
                not isinstance(urls, list)
                or not urls
                or any(not isinstance(url, str) or not url.startswith("https://") for url in urls)
            ):
                raise CatalogValidationError(f"{catalog_path.name} has an invalid download URL")
            if not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate <= 0:
                raise CatalogValidationError(
                    f"{catalog_path.name} artifact {urls[0]} needs positive memory_estimate_gib"
                )
            forbidden = {
                "size",
                "size_bytes",
                "size_gib",
                "bundle_size_bytes",
                "bundle_size_gib",
            }.intersection(artifact)
            if forbidden:
                fields = ", ".join(sorted(forbidden))
                raise CatalogValidationError(
                    f"{catalog_path.name} artifact {urls[0]} contains generated fields: {fields}"
                )
            references.update(urls)
            for url in urls:
                parts = urllib.parse.urlparse(url).path.strip("/").split("/")
                if (
                    len(parts) >= 6
                    and parts[0] == "models"
                    and parts[3] == "resolve"
                    and parts[4] not in {"main", "master"}
                    and url not in assets
                ):
                    raise CatalogValidationError(
                        f"revision-pinned download is missing shared metadata: {url}"
                    )

    for url, metadata in assets.items():
        if not isinstance(url, str) or not url.startswith("https://"):
            raise CatalogValidationError("asset keys must be HTTPS URLs")
        if "/resolve/master/" in url or "/resolve/main/" in url:
            raise CatalogValidationError(f"asset URL is not revision-pinned: {url}")
        if not isinstance(metadata, dict):
            raise CatalogValidationError(f"asset metadata must be an object: {url}")
        size_bytes = metadata.get("size_bytes")
        sha256 = metadata.get("sha256")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise CatalogValidationError(f"asset size_bytes must be a positive integer: {url}")
        if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
            raise CatalogValidationError(f"asset sha256 must be lowercase hex: {url}")
        if references[url] == 0:
            raise CatalogValidationError(f"asset is not referenced by any platform catalog: {url}")

    return assets


def remote_size(url: str, timeout: float) -> int:
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "OmniInfer-catalog-validator/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get("Content-Range", "")
            match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
            if match is None:
                raise CatalogValidationError(
                    f"remote endpoint did not honor one-byte Range request: {url}"
                )
            return int(match.group(1))
    except CatalogValidationError:
        raise
    except Exception as exc:
        raise CatalogValidationError(f"remote probe failed for {url}: {exc}") from exc


def validate_remote(assets: dict[str, dict[str, Any]], timeout: float) -> None:
    for url, metadata in assets.items():
        actual = remote_size(url, timeout)
        expected = metadata["size_bytes"]
        if actual != expected:
            raise CatalogValidationError(
                f"remote size mismatch for {url}: expected {expected}, got {actual}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote",
        action="store_true",
        help="probe each pinned URL with a one-byte Range request",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        assets = validate_local()
        if args.remote:
            validate_remote(assets, args.timeout)
    except CatalogValidationError as exc:
        print(f"model catalog validation failed: {exc}", file=sys.stderr)
        return 1
    mode = "local + remote" if args.remote else "local"
    print(f"model catalog validation passed ({mode}, {len(assets)} exact assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
