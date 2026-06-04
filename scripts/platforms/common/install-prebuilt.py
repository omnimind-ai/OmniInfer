#!/usr/bin/env python3
"""Install a prebuilt backend archive into an OmniInfer runtime directory."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid prebuilt catalog: {path}")
    return payload


def backend_entry(catalog: dict[str, Any], platform_name: str, backend: str) -> dict[str, Any]:
    platforms = catalog.get("platforms")
    if not isinstance(platforms, dict):
        raise SystemExit("prebuilt catalog is missing 'platforms'")
    platform_entries = platforms.get(platform_name)
    if not isinstance(platform_entries, dict):
        raise SystemExit(f"no prebuilt catalog entries for platform: {platform_name}")
    entry = platform_entries.get(backend)
    if not isinstance(entry, dict):
        raise SystemExit(f"no prebuilt archive is configured for {platform_name}/{backend}")
    if not entry.get("url") or not entry.get("launcher"):
        raise SystemExit(f"prebuilt entry for {platform_name}/{backend} is missing url or launcher")
    return entry


def mirror_urls(catalog: dict[str, Any], url: str) -> list[str]:
    urls: list[str] = []
    env_prefixes = os.environ.get("OMNIINFER_PREBUILT_MIRROR_PREFIXES", "")
    for prefix in [item.strip() for item in env_prefixes.split(",") if item.strip()]:
        urls.append(f"{prefix}{url}")
    for prefix in catalog.get("mirrors", []):
        if isinstance(prefix, str) and prefix.strip():
            urls.append(f"{prefix.strip()}{url}")
    urls.append(url)
    return urls


def download(urls: list[str], destination: Path) -> str:
    last_error = ""
    for url in urls:
        try:
            print(f"Downloading prebuilt archive: {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "OmniInfer-prebuilt-installer"})
            with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out)
            if destination.stat().st_size < 1024 * 1024:
                last_error = f"downloaded file is unexpectedly small: {destination.stat().st_size} bytes"
                destination.unlink(missing_ok=True)
                continue
            return url
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            destination.unlink(missing_ok=True)
    raise SystemExit(f"failed to download prebuilt archive; last error: {last_error}")


def extract(archive: Path, destination: Path, archive_type: str) -> None:
    destination_resolved = destination.resolve()

    def safe_target(name: str) -> Path:
        normalized = posixpath.normpath(name.replace("\\", "/"))
        if normalized.startswith("../") or normalized == ".." or posixpath.isabs(normalized):
            raise SystemExit(f"unsafe path in prebuilt archive: {name}")
        target = (destination / normalized).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise SystemExit(f"unsafe path in prebuilt archive: {name}")
        return target

    if archive_type == "zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                safe_target(member.filename)
            zf.extractall(destination)
        return
    if archive_type in {"tar.gz", "tgz"}:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                safe_target(member.name)
            tf.extractall(destination)
        return
    raise SystemExit(f"unsupported prebuilt archive type: {archive_type}")


def find_launcher(root: Path, launcher: str) -> Path:
    matches = [path for path in root.rglob(launcher) if path.is_file()]
    if not matches:
        raise SystemExit(f"launcher {launcher!r} was not found in extracted archive")
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def copy_runtime_files(source_dir: Path, runtime_dir: Path) -> None:
    bin_dir = runtime_dir / "bin"
    logs_dir = runtime_dir / "logs"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = bin_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def write_manifest(runtime_dir: Path, *, platform_name: str, backend: str, entry: dict[str, Any], url: str) -> None:
    manifest = {
        "schema_version": 1,
        "installed_at": int(time.time()),
        "platform": platform_name,
        "backend": backend,
        "source": entry.get("source"),
        "tag": entry.get("tag"),
        "url": url,
        "launcher": entry.get("launcher"),
    }
    (runtime_dir / "prebuilt.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    catalog = load_catalog(catalog_path)
    entry = backend_entry(catalog, args.platform, args.backend)
    urls = mirror_urls(catalog, str(entry["url"]))

    print(f"Prebuilt backend: {args.platform}/{args.backend}")
    print(f"  source: {entry.get('source', '-')}")
    print(f"  tag: {entry.get('tag', '-')}")
    print(f"  runtime: {runtime_dir}")
    print(f"  launcher: {entry['launcher']}")
    if args.dry_run:
        for url in urls:
            print(f"  would try: {url}")
        return 0

    runtime_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"omni-prebuilt-{args.backend}-") as temp_name:
        temp_dir = Path(temp_name)
        archive_path = temp_dir / "archive"
        used_url = download(urls, archive_path)
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()
        extract(archive_path, extracted_dir, str(entry.get("archive", "")).lower())
        launcher = find_launcher(extracted_dir, str(entry["launcher"]))
        copy_runtime_files(launcher.parent, runtime_dir)
        installed_launcher = runtime_dir / "bin" / launcher.name
        if not installed_launcher.is_file():
            raise SystemExit(f"prebuilt install failed: {installed_launcher} was not created")
        installed_launcher.chmod(installed_launcher.stat().st_mode | 0o111)
        write_manifest(runtime_dir, platform_name=args.platform, backend=args.backend, entry=entry, url=used_url)
        print(f"Prebuilt backend installed: {installed_launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
