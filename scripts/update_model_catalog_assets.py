#!/usr/bin/env python3
"""Pin direct ModelScope catalog downloads and regenerate exact asset metadata."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = REPO_ROOT / "crates" / "omniinfer-core" / "model_catalogs"
CATALOG_PATHS = tuple(CATALOG_DIR / name for name in ("linux.json", "mac.json", "windows.json"))
ASSET_PATH = CATALOG_DIR / "download_assets.json"


class UpdateError(RuntimeError):
    """Raised when upstream metadata cannot produce a complete exact entry."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def direct_modelscope_download(url: str) -> tuple[str, str, str] | None:
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc != "modelscope.cn"
        or len(parts) < 6
        or parts[0] != "models"
        or parts[3] != "resolve"
    ):
        return None
    repository = f"{parts[1]}/{parts[2]}"
    revision = parts[4]
    file_path = "/".join(parts[5:])
    return repository, revision, file_path


def iter_download_urls(value: Any):
    if isinstance(value, dict):
        download = value.get("download")
        if isinstance(download, str):
            yield download
        elif isinstance(download, list):
            yield from (url for url in download if isinstance(url, str))
        for child in value.values():
            yield from iter_download_urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_download_urls(child)


def fetch_repository_files(repository: str, revision: str) -> dict[str, dict[str, Any]]:
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "Recursive": "true"})
    url = f"https://modelscope.cn/api/v1/models/{encoded_repo}/repo/files?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "OmniInfer-catalog-updater/1"})
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            payload = json.load(response)
    except Exception as exc:
        raise UpdateError(f"cannot fetch {repository}@{revision}: {exc}") from exc
    files = payload.get("Data", {}).get("Files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise UpdateError(f"unexpected ModelScope response for {repository}@{revision}")
    return {
        file["Path"]: file
        for file in files
        if isinstance(file, dict) and isinstance(file.get("Path"), str)
    }


def pinned_url(repository: str, revision: str, file_path: str) -> str:
    encoded_path = urllib.parse.quote(file_path, safe="/")
    return f"https://modelscope.cn/models/{repository}/resolve/{revision}/{encoded_path}"


def replace_download_urls(value: Any, replacements: dict[str, str]) -> None:
    if isinstance(value, dict):
        download = value.get("download")
        if isinstance(download, str) and download in replacements:
            value["download"] = replacements[download]
        elif isinstance(download, list):
            value["download"] = [replacements.get(url, url) for url in download]
        for child in value.values():
            replace_download_urls(child, replacements)
    elif isinstance(value, list):
        for child in value:
            replace_download_urls(child, replacements)


def atomic_write_json(path: Path, payload: Any, indent: int) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    catalogs = {path: load_json(path) for path in CATALOG_PATHS}
    direct_urls: dict[str, tuple[str, str, str]] = {}
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for catalog in catalogs.values():
        for url in iter_download_urls(catalog):
            parsed = direct_modelscope_download(url)
            if parsed is None:
                continue
            direct_urls[url] = parsed
            repository, revision, file_path = parsed
            groups[(repository, revision)].add(file_path)

    repository_files: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    skipped_groups: set[tuple[str, str]] = set()
    for key in sorted(groups):
        try:
            repository_files[key] = fetch_repository_files(*key)
        except UpdateError as exc:
            skipped_groups.add(key)
            print(f"warning: {exc}", file=sys.stderr)
    replacements: dict[str, str] = {}
    assets: dict[str, dict[str, Any]] = {}
    for url, (repository, requested_revision, file_path) in sorted(direct_urls.items()):
        if (repository, requested_revision) in skipped_groups:
            continue
        metadata = repository_files[(repository, requested_revision)].get(file_path)
        if metadata is None:
            print(
                f"warning: upstream file is missing: "
                f"{repository}@{requested_revision}/{file_path}",
                file=sys.stderr,
            )
            continue
        revision = metadata.get("Revision")
        size = metadata.get("Size")
        sha256 = metadata.get("Sha256")
        if not isinstance(revision, str) or len(revision) != 40:
            raise UpdateError(f"invalid revision for {repository}/{file_path}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise UpdateError(f"invalid size for {repository}/{file_path}")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise UpdateError(f"invalid SHA-256 for {repository}/{file_path}")
        resolved_url = pinned_url(repository, revision, file_path)
        replacements[url] = resolved_url
        assets[resolved_url] = {"size_bytes": size, "sha256": sha256.lower()}

    for path, catalog in catalogs.items():
        replace_download_urls(catalog, replacements)
        atomic_write_json(path, catalog, indent=4)
    atomic_write_json(
        ASSET_PATH,
        {"schema_version": 1, "assets": dict(sorted(assets.items()))},
        indent=2,
    )
    print(
        f"updated {len(catalogs)} catalogs with {len(assets)} exact direct-download assets; "
        f"{len(skipped_groups)} unavailable repositories and repository downloads remain unknown"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, json.JSONDecodeError, UpdateError) as exc:
        print(f"model catalog update failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
