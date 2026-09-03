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
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "scripts" / "prebuilt_backends.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_source_release_assets(catalog: dict[str, Any], source_name: str):
    for platform, entries in catalog.get("platforms", {}).items():
        for backend, entry in entries.items():
            if entry.get("source") != source_name:
                continue
            yield platform, backend, "runtime", entry
            for index, asset in enumerate(entry.get("companion_assets", []), start=1):
                yield platform, backend, f"companion {index}", asset
    for asset in catalog.get("excluded_release_assets", []):
        if asset.get("source") != source_name:
            continue
        yield (
            asset.get("platform", "unknown"),
            asset.get("backend", "unknown"),
            "excluded runtime",
            asset,
        )


def validate(
    catalog: dict[str, Any],
    *,
    require_gitlink_match: bool,
    verify_upstream_tags: bool,
) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != 5:
        errors.append("schema_version must be 5")
    sources = catalog.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        errors.append("sources must be a non-empty object")
        return errors
    for source_name, source in sources.items():
        tag = source.get("tag")
        submodule_tag = source.get("submodule_tag")
        submodule_path = source.get("submodule_path")
        submodule_commit = source.get("submodule_commit")
        if not isinstance(tag, str) or not tag:
            errors.append(f"{source_name}: tag is required")
        if not isinstance(submodule_tag, str) or not submodule_tag:
            errors.append(f"{source_name}: submodule_tag is required")
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
        if (
            verify_upstream_tags
            and isinstance(submodule_tag, str)
            and isinstance(submodule_commit, str)
        ):
            try:
                upstream_commit = github_tag_commit(source_name, submodule_tag)
            except Exception as error:
                errors.append(
                    f"{source_name}: failed to resolve upstream tag {submodule_tag}: {error}"
                )
            else:
                if upstream_commit != submodule_commit:
                    errors.append(
                        f"{source_name}: upstream tag {submodule_tag} resolves to {upstream_commit}, "
                        f"not catalog commit {submodule_commit}"
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
    installable = {
        (platform, backend)
        for platform, entries in catalog.get("platforms", {}).items()
        for backend in entries
    }
    for entry in catalog.get("excluded_release_assets", []):
        source_name = entry.get("source")
        source = sources.get(source_name)
        platform = entry.get("platform")
        backend = entry.get("backend")
        architecture = entry.get("architecture")
        reason = entry.get("reason")
        if source is None:
            errors.append(
                f"excluded {platform}/{backend}: unknown source {source_name!r}"
            )
            continue
        if not all(
            isinstance(value, str) and value.strip()
            for value in (platform, backend, architecture, reason)
        ):
            errors.append("excluded release asset metadata is incomplete")
            continue
        if (platform, backend) in installable:
            errors.append(f"{platform}/{backend}: cannot be both installable and excluded")
        validate_asset(
            errors,
            platform,
            backend,
            "excluded runtime",
            entry,
            source.get("tag"),
        )
    for platform, entries in catalog.get("python_runtimes", {}).items():
        for backend, entry in entries.items():
            required = ("source", "tag", "package", "python", "launcher")
            for key in required:
                if not isinstance(entry.get(key), str) or not entry[key].strip():
                    errors.append(f"{platform}/{backend}: Python runtime {key} is required")
            variants = entry.get("variants")
            uv_assets = entry.get("uv")
            if not isinstance(variants, dict) or not variants:
                errors.append(f"{platform}/{backend}: Python runtime variants are required")
                continue
            if not isinstance(uv_assets, dict) or not uv_assets:
                errors.append(f"{platform}/{backend}: managed uv assets are required")
                continue
            for architecture, variant in variants.items():
                if sources.get(entry.get("source")) is None:
                    errors.append(
                        f"{platform}/{backend}: unknown Python runtime source {entry.get('source')!r}"
                    )
                validate_python_asset(
                    errors, platform, backend, architecture, variant, entry.get("tag")
                )
                version = variant.get("version")
                expected_base = str(entry.get("tag", "")).removeprefix("v")
                if (
                    not isinstance(version, str)
                    or not version
                    or not version.startswith(expected_base)
                ):
                    errors.append(
                        f"{platform}/{backend} {architecture}: Python package version must match {entry.get('tag')!r}"
                    )
                reported_version = variant.get("reported_version", version)
                if (
                    not isinstance(reported_version, str)
                    or not reported_version
                    or not reported_version.startswith(expected_base)
                ):
                    errors.append(
                        f"{platform}/{backend} {architecture}: reported Python package version "
                        f"must match {entry.get('tag')!r}"
                    )
                accelerator = variant.get("accelerator")
                if accelerator not in ("cuda", "rocm"):
                    errors.append(
                        f"{platform}/{backend} {architecture}: accelerator must be cuda or rocm"
                    )
                if not isinstance(variant.get("runtime_version"), str):
                    errors.append(
                        f"{platform}/{backend} {architecture}: runtime_version is required"
                    )
                reported_runtime_version = variant.get(
                    "reported_runtime_version", variant.get("runtime_version")
                )
                if (
                    not isinstance(reported_runtime_version, str)
                    or not reported_runtime_version
                ):
                    errors.append(
                        f"{platform}/{backend} {architecture}: "
                        "reported_runtime_version must be a non-empty string"
                    )
                if accelerator == "cuda" and not isinstance(
                    variant.get("minimum_driver"), str
                ):
                    errors.append(
                        f"{platform}/{backend} {architecture}: minimum_driver is required for CUDA"
                    )
                if accelerator == "rocm":
                    rocm_system = variant.get("rocm_system")
                    if not isinstance(rocm_system, dict):
                        errors.append(
                            f"{platform}/{backend} {architecture}: rocm_system is required for ROCm"
                        )
                    else:
                        packages = rocm_system.get("packages")
                        required_packages = {
                            "comgr",
                            "hipblas",
                            "hipblaslt",
                            "hipfft",
                            "hiprand",
                            "hip-runtime-amd",
                            "hipsolver",
                            "hipsparse",
                            "hipsparselt",
                            "hsa-rocr",
                            "libopenmpi3t64",
                            "libpython3.12-dev",
                            "miopen-hip",
                            "openmp-extras-runtime",
                            "python3.12-dev",
                            "rccl",
                            "rocblas",
                            "rocfft",
                            "rocm-hip-runtime",
                            "rocm-core",
                            "rocm-device-libs",
                            "rocm-language-runtime",
                            "rocm-llvm",
                            "rocm-smi-lib",
                            "rocminfo",
                            "rocprofiler-register",
                            "rocrand",
                            "rocsolver",
                            "rocsparse",
                            "roctracer",
                        }
                        if (
                            not isinstance(packages, dict)
                            or set(packages) != required_packages
                            or not all(
                                isinstance(version, str) and version
                                for version in packages.values()
                            )
                        ):
                            errors.append(
                                f"{platform}/{backend} {architecture}: ROCm system packages must "
                                "exactly match the required PyTorch runtime set"
                            )
                        package_assets = rocm_system.get("package_assets")
                        required_assets = required_packages - {"libopenmpi3t64"}
                        repository = str(rocm_system.get("apt_repository", "")).split()
                        repository_url = (
                            repository[0].rstrip("/") if repository else ""
                        )
                        ubuntu_python_pool = (
                            "https://security.ubuntu.com/ubuntu/pool/main/p/python3.12/"
                        )
                        if (
                            not isinstance(package_assets, dict)
                            or set(package_assets) != required_assets
                        ):
                            errors.append(
                                f"{platform}/{backend} {architecture}: ROCm package assets must "
                                "exactly match the pinned AMD runtime closure"
                            )
                        else:
                            for package, asset in package_assets.items():
                                if not isinstance(asset, dict):
                                    errors.append(
                                        f"{platform}/{backend} {architecture}: invalid ROCm "
                                        f"package asset {package}"
                                    )
                                    continue
                                filename = asset.get("filename")
                                digest = asset.get("sha256")
                                url = asset.get("url")
                                valid_origin = isinstance(url, str) and (
                                    url.startswith(f"{repository_url}/")
                                    or (
                                        package
                                        in {"python3.12-dev", "libpython3.12-dev"}
                                        and url.startswith(ubuntu_python_pool)
                                    )
                                )
                                if (
                                    asset.get("version") != packages.get(package)
                                    or not isinstance(filename, str)
                                    or "/" in filename
                                    or "\\" in filename
                                    or not filename.endswith(".deb")
                                    or not isinstance(asset.get("size"), int)
                                    or asset["size"] <= 0
                                    or not isinstance(digest, str)
                                    or not SHA256_RE.fullmatch(digest)
                                    or not valid_origin
                                    or not url.endswith(filename)
                                ):
                                    errors.append(
                                        f"{platform}/{backend} {architecture}: invalid ROCm "
                                        f"package asset {package}"
                                    )
                uv = uv_assets.get(architecture)
                if not isinstance(uv, dict):
                    errors.append(
                        f"{platform}/{backend} {architecture}: managed uv asset is required"
                    )
                    continue
                validate_asset(
                    errors,
                    platform,
                    backend,
                    f"uv ({architecture})",
                    uv,
                    uv.get("version"),
                )
    return errors


def validate_python_asset(
    errors: list[str],
    platform: str,
    backend: str,
    architecture: str,
    variant: dict[str, Any],
    tag: Any,
) -> None:
    role = f"Python wheel ({architecture})"
    digest = variant.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append(f"{platform}/{backend} {role}: missing or invalid sha256")
    url = variant.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        errors.append(f"{platform}/{backend} {role}: canonical HTTPS URL is required")
        return
    if url.startswith("https://github.com/"):
        if isinstance(tag, str) and f"/download/{tag}/" not in url:
            errors.append(f"{platform}/{backend} {role}: URL does not match tag {tag}")
        return
    build_commit = variant.get("build_commit")
    index_url = variant.get("index_url")
    if (
        not isinstance(build_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", build_commit)
        or not url.startswith("https://wheels.vllm.ai/rocm/")
        or build_commit not in url
        or not isinstance(index_url, str)
        or build_commit not in index_url
    ):
        errors.append(
            f"{platform}/{backend} {role}: invalid independent wheel build provenance"
        )


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


def github_json(url: str) -> Any:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "OmniInfer-catalog-updater",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def github_tag_commit(source_name: str, tag: str) -> str:
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--tags",
            f"https://github.com/{source_name}.git",
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    refs = {
        ref: commit
        for line in result.stdout.splitlines()
        if len(fields := line.split()) == 2
        for commit, ref in [fields]
    }
    commit = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if commit is None or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("upstream tag does not exist or has an invalid target")
    return commit


def release_assets(source_name: str, tag: str) -> dict[str, dict[str, Any]]:
    url = f"https://api.github.com/repos/{source_name}/releases/tags/{tag}"
    payload = github_json(url)
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
    tag_commit = github_tag_commit(source_name, tag)
    if tag_commit != submodule_commit:
        raise SystemExit(
            f"{source_name}: tag {tag} resolves to {tag_commit}, "
            f"but --submodule-commit resolved to {submodule_commit}"
        )
    assets = release_assets(source_name, tag)
    for platform, backend, role, asset in iter_source_release_assets(catalog, source_name):
        old_url = asset["url"]
        old_name = unquote(Path(urlparse(old_url).path).name)
        new_name = old_name.replace(old_tag, tag).replace(
            old_tag.removeprefix("v"), tag.removeprefix("v")
        )
        upstream = assets.get(new_name)
        if upstream is None:
            raise SystemExit(f"{platform}/{backend} {role}: release asset {new_name!r} does not exist")
        digest = upstream.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise SystemExit(f"{platform}/{backend} {role}: upstream asset has no SHA256 digest")
        asset["url"] = upstream["browser_download_url"]
        asset["sha256"] = digest.removeprefix("sha256:")
    source["tag"] = tag
    source["submodule_tag"] = tag
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
    check_parser.add_argument(
        "--verify-upstream-tags",
        action="store_true",
        help="resolve every GitHub tag and verify that it matches the pinned source commit",
    )
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
        verify_upstream_tags=getattr(args, "verify_upstream_tags", False),
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
