#!/usr/bin/env python3
"""Validate or update prebuilt runtime source metadata and asset digests."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "scripts" / "prebuilt_backends.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_source_assets(catalog: dict[str, Any], source_name: str):
    for platform, entries in catalog.get("platforms", {}).items():
        for backend, entry in entries.items():
            if entry.get("source") != source_name:
                continue
            yield platform, backend, "runtime", entry
            for index, asset in enumerate(entry.get("companion_assets", []), start=1):
                yield platform, backend, f"companion {index}", asset


def validate(catalog: dict[str, Any], *, require_gitlink_match: bool) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != 3:
        errors.append("schema_version must be 3")
    sources = catalog.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        errors.append("sources must be a non-empty object")
        return errors
    for source_name, source in sources.items():
        tag = source.get("tag")
        submodule_path = source.get("submodule_path")
        submodule_commit = source.get("submodule_commit")
        if not isinstance(tag, str) or not tag:
            errors.append(f"{source_name}: tag is required")
        if not isinstance(submodule_path, str) or not submodule_path:
            errors.append(f"{source_name}: submodule_path is required")
        if not isinstance(submodule_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", submodule_commit):
            errors.append(f"{source_name}: submodule_commit must be a 40-character lowercase commit")
        if require_gitlink_match and isinstance(submodule_path, str):
            actual = gitlink_commit(submodule_path)
            if actual != submodule_commit:
                errors.append(
                    f"{source_name}: catalog commit {submodule_commit} does not match gitlink {actual}"
                )
    for platform, entries in catalog.get("platforms", {}).items():
        for backend, entry in entries.items():
            source_name = entry.get("source")
            source = sources.get(source_name)
            if source is None:
                errors.append(f"{platform}/{backend}: unknown source {source_name!r}")
                continue
            tag = source.get("tag")
            validate_asset(errors, platform, backend, "runtime", entry, tag)
            for index, asset in enumerate(entry.get("companion_assets", []), start=1):
                validate_asset(errors, platform, backend, f"companion {index}", asset, tag)
    return errors


def validate_asset(
    errors: list[str],
    platform: str,
    backend: str,
    role: str,
    asset: dict[str, Any],
    tag: Any,
) -> None:
    digest = asset.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append(f"{platform}/{backend} {role}: missing or invalid sha256")
    url = asset.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        errors.append(f"{platform}/{backend} {role}: canonical HTTPS URL is required")
    elif isinstance(tag, str) and f"/download/{tag}/" not in url:
        errors.append(f"{platform}/{backend} {role}: URL does not match tag {tag}")


def gitlink_commit(submodule_path: str) -> str:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", submodule_path],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.split()
    if len(fields) < 2 or fields[0] != "160000":
        raise RuntimeError(f"{submodule_path} is not a staged submodule gitlink")
    return fields[1]


def release_assets(source_name: str, tag: str) -> dict[str, dict[str, Any]]:
    url = f"https://api.github.com/repos/{source_name}/releases/tags/{tag}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "OmniInfer-catalog-updater"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return {asset["name"]: asset for asset in payload.get("assets", [])}


def update_source(
    catalog: dict[str, Any],
    source_name: str,
    tag: str,
    submodule_commit: str,
) -> None:
    source = catalog.get("sources", {}).get(source_name)
    if source is None:
        raise SystemExit(f"unknown catalog source: {source_name}")
    old_tag = source.get("tag")
    if not isinstance(old_tag, str) or not old_tag:
        raise SystemExit(f"catalog source {source_name} has no current tag")
    if submodule_commit == "current":
        submodule_commit = gitlink_commit(source["submodule_path"])
    if not re.fullmatch(r"[0-9a-f]{40}", submodule_commit):
        raise SystemExit("--submodule-commit must be 'current' or a 40-character lowercase commit")
    assets = release_assets(source_name, tag)
    for platform, backend, role, asset in iter_source_assets(catalog, source_name):
        old_url = asset["url"]
        old_name = Path(urlparse(old_url).path).name
        new_name = old_name.replace(old_tag, tag)
        upstream = assets.get(new_name)
        if upstream is None:
            raise SystemExit(f"{platform}/{backend} {role}: release asset {new_name!r} does not exist")
        digest = upstream.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise SystemExit(f"{platform}/{backend} {role}: upstream asset has no SHA256 digest")
        asset["url"] = upstream["browser_download_url"]
        asset["sha256"] = digest.removeprefix("sha256:")
    source["tag"] = tag
    source["submodule_commit"] = submodule_commit


def write_catalog_atomically(path: Path, catalog: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(catalog, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="validate local catalog metadata")
    check_parser.add_argument("--require-gitlink-match", action="store_true")
    update_parser = subparsers.add_parser("update", help="update one source from an upstream release")
    update_parser.add_argument("--source", required=True)
    update_parser.add_argument("--tag", required=True)
    update_parser.add_argument("--submodule-commit", required=True)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    if args.command == "update":
        update_source(catalog, args.source, args.tag, args.submodule_commit)
    errors = validate(
        catalog,
        require_gitlink_match=getattr(args, "require_gitlink_match", False),
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.command == "update":
        write_catalog_atomically(args.catalog, catalog)
    print(f"catalog ok: {args.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
