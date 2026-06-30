#!/usr/bin/env python3
"""Install the Rust OmniInfer CLI into a portable root."""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_unix_launcher(target: Path, binary_name: str) -> None:
    content = rf"""#!/bin/sh
set -eu

SCRIPT_PATH="$0"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$(pwd)/$SCRIPT_PATH" ;;
esac

ROOT="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
exec "$ROOT/{binary_name}" "$@"
"""
    target.write_text(content, encoding="utf-8")
    make_executable(target)


def install_windows_launchers(portable_root: Path) -> None:
    (portable_root / "omniinfer.cmd").write_text(
        """@echo off
setlocal
"%~dp0omniinfer.exe" %*
exit /b %errorlevel%
""",
        encoding="utf-8",
    )
    (portable_root / "omniinfer.ps1").write_text(
        """$ErrorActionPreference = "Stop"

$scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}

& (Join-Path $scriptDir "omniinfer.exe") @args
exit $LASTEXITCODE
""",
        encoding="utf-8",
    )


def install_cli(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    portable_root = args.portable_root.resolve()
    portable_root.mkdir(parents=True, exist_ok=True)

    cargo_args = ["cargo", "build", "--release", "-p", "omniinfer-cli"]
    if args.locked:
        cargo_args.append("--locked")
    if not args.skip_build:
        run(cargo_args, repo_root, args.dry_run)

    exe_suffix = ".exe" if args.platform == "windows" else ""
    built_binary = repo_root / "target" / "release" / f"omniinfer-rs{exe_suffix}"
    if not args.dry_run and not built_binary.is_file():
        raise SystemExit(f"Rust CLI build did not produce {built_binary}")

    print("Python control-plane fallback: removed")

    if args.dry_run:
        print(f"[dry-run] Would copy {built_binary} into {portable_root}")
        return

    if args.platform == "windows":
        shutil.copy2(built_binary, portable_root / "omniinfer.exe")
        install_windows_launchers(portable_root)
    else:
        rust_target = portable_root / "omniinfer-rs"
        shutil.copy2(built_binary, rust_target)
        make_executable(rust_target)
        install_unix_launcher(
            portable_root / "omniinfer",
            "omniinfer-rs",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--portable-root", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=["linux", "macos", "windows"])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    install_cli(parse_args())
