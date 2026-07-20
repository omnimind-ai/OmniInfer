#!/usr/bin/env python3
"""Build a CLI-only OmniInfer release archive for the current host platform."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


TARGETS = {
    "linux-x64": {"platform": "linux", "archive": "tar.gz", "system": "Linux", "machine": {"x86_64", "amd64"}},
    "macos-arm64": {"platform": "macos", "archive": "tar.gz", "system": "Darwin", "machine": {"arm64", "aarch64"}},
    "windows-x64": {"platform": "windows", "archive": "zip", "system": "Windows", "machine": {"amd64", "x86_64"}},
}

LINUX_GLIBC_BASELINE = (2, 35)


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def run_capture(command: list[str], cwd: Path) -> str:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout


def normalized_machine() -> str:
    machine = platform.machine().lower()
    if machine in {"x64"}:
        return "x86_64"
    return machine


def assert_host_matches(target: str, expected_platform: str) -> None:
    spec = TARGETS[target]
    actual_system = platform.system()
    actual_machine = normalized_machine()

    if expected_platform != spec["platform"]:
        raise SystemExit(
            f"Target {target} requires platform={spec['platform']}, got platform={expected_platform}."
        )

    if actual_system != spec["system"]:
        raise SystemExit(
            f"Target {target} must be built on {spec['system']}, got {actual_system}."
        )

    if actual_machine not in spec["machine"]:
        expected = ", ".join(sorted(spec["machine"]))
        raise SystemExit(
            f"Target {target} must be built on machine {expected}, got {actual_machine}."
        )


def copy_metadata(repo_root: Path, portable_root: Path, version: str, target: str) -> None:
    for name in ("README.md", "LICENSE"):
        source = repo_root / name
        if source.is_file():
            shutil.copy2(source, portable_root / name)

    (portable_root / "VERSION").write_text(f"{version}\n{target}\n", encoding="utf-8")


def package_cli(args: argparse.Namespace, portable_root: Path) -> None:
    packager = args.repo_root / "scripts" / "platforms" / "common" / "package-rust-cli.py"
    command = [
        "python3" if os.name != "nt" else "python",
        str(packager),
        "--repo-root",
        str(args.repo_root),
        "--portable-root",
        str(portable_root),
        "--platform",
        args.platform,
        "--locked",
    ]
    if args.skip_build:
        command.append("--skip-build")
    run(command, args.repo_root)


def smoke_test(portable_root: Path, platform_name: str) -> None:
    binary = portable_root / ("omniinfer.exe" if platform_name == "windows" else "omniinfer")
    if not binary.is_file():
        raise SystemExit(f"Packaged binary was not found: {binary}")

    run([str(binary), "--version"], portable_root)
    run([str(binary), "--help"], portable_root)
    if platform_name == "windows":
        run(
            [str(binary), "backend", "install", "llama.cpp-cuda", "--dry-run"],
            portable_root,
        )
        with tempfile.TemporaryDirectory(prefix="omniinfer-release-smoke-") as temp:
            state_root = Path(temp) / "state"
            runtime_root = Path(temp) / "runtimes"
            output = run_capture(
                [
                    str(binary),
                    "backend",
                    "install",
                    "llama.cpp-cpu",
                    "--dry-run",
                    "--json",
                    "--state-root",
                    str(state_root),
                    "--runtime-root",
                    str(runtime_root),
                ],
                portable_root,
            )
            events = [json.loads(line) for line in output.splitlines()]
            planned = next(event for event in events if event["event"] == "asset_planned")
            if planned.get("expected_sha256") != "9d04ebc1af723cb11be09a0ec1f9a375934697e8d7fe57439e508a636c197a28":
                raise SystemExit("Packaged Windows CPU catalog digest is missing or incorrect.")
            advisor = json.loads(
                run_capture(
                    [
                        str(binary),
                        "advisor",
                        "system",
                        "--json",
                        "--state-root",
                        str(state_root),
                        "--runtime-root",
                        str(runtime_root),
                    ],
                    portable_root,
                )
            )
            recommendation = advisor["summary"]["recommended_backend_to_install"]
            backend = next(item for item in advisor["backends"] if item["id"] == recommendation)
            if not backend.get("prebuilt_installable") or not backend.get("install_command"):
                raise SystemExit("Advisor recommended a backend that the packaged installer cannot install.")


def parse_glibc_version(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def assert_linux_glibc_baseline(portable_root: Path) -> None:
    binaries = [path for path in portable_root.iterdir() if path.is_file() and os.access(path, os.X_OK)]
    max_required: tuple[int, int] | None = None
    max_path: Path | None = None
    pattern = re.compile(rb"GLIBC_(\d+\.\d+)")

    for binary in binaries:
        data = binary.read_bytes()
        versions = {
            parsed
            for raw in pattern.findall(data)
            if (parsed := parse_glibc_version(raw.decode("ascii"))) is not None
        }
        if not versions:
            continue
        binary_max = max(versions)
        if max_required is None or binary_max > max_required:
            max_required = binary_max
            max_path = binary

    if max_required is None:
        return

    if max_required > LINUX_GLIBC_BASELINE:
        baseline = ".".join(str(part) for part in LINUX_GLIBC_BASELINE)
        required = ".".join(str(part) for part in max_required)
        raise SystemExit(
            f"Linux CLI package requires GLIBC_{required} via {max_path}; "
            f"release baseline is GLIBC_{baseline}. Build linux-x64 on ubuntu-22.04 or older."
        )

    required = ".".join(str(part) for part in max_required)
    print(f"Linux GLIBC baseline check passed: max GLIBC_{required}")


def add_zip_file(zip_file: zipfile.ZipFile, path: Path, arcname: str) -> None:
    info = zipfile.ZipInfo.from_file(path, arcname)
    if os.name != "nt":
        mode = path.stat().st_mode
        info.external_attr = (mode & 0xFFFF) << 16
    with path.open("rb") as handle:
        zip_file.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED)


def create_zip(source_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(source_root.rglob("*")):
            arcname = str(path.relative_to(source_root.parent)).replace(os.sep, "/")
            if path.is_dir():
                continue
            add_zip_file(zip_file, path, arcname)


def create_tar_gz(source_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_root, arcname=source_root.name)


def create_archive(args: argparse.Namespace, portable_root: Path) -> Path:
    archive_kind = TARGETS[args.target]["archive"]
    suffix = ".zip" if archive_kind == "zip" else ".tar.gz"
    archive_path = args.output_dir / f"omniinfer-{args.version}-{args.target}{suffix}"
    if archive_path.exists():
        archive_path.unlink()

    if archive_kind == "zip":
        create_zip(portable_root, archive_path)
    else:
        create_tar_gz(portable_root, archive_path)

    print(f"Created {archive_path}")
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--platform", required=True, choices=["linux", "macos", "windows"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-host-check", action="store_true")
    parser.add_argument("--no-smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.no_host_check:
        assert_host_matches(args.target, args.platform)

    staging_root = args.output_dir / "_stage" / args.target
    portable_root = staging_root / "OmniInfer"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    portable_root.mkdir(parents=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    package_cli(args, portable_root)
    copy_metadata(args.repo_root, portable_root, args.version, args.target)
    if not args.no_smoke_test:
        smoke_test(portable_root, args.platform)
    if args.platform == "linux":
        assert_linux_glibc_baseline(portable_root)
    create_archive(args, portable_root)


if __name__ == "__main__":
    main()
