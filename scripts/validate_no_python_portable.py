#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGER = REPO_ROOT / "scripts" / "platforms" / "common" / "package-rust-cli.py"
REMOVED_FALLBACK_MESSAGE = "Python control-plane fallback has been removed"
PACKAGED_BUILD_MESSAGE = "Backend builds are only available from a source checkout"
FORBIDDEN_LAUNCHER_TEXT = (
    "OMNIINFER_FORCE_PYTHON",
    "OMNIINFER_PYTHON",
    "omniinfer.py",
    "python_works",
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def run_command(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "timed_out": False,
            "wall_s": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "wall_s": time.perf_counter() - started,
            "stdout": (exc.stdout or "").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or ""),
            "stderr": (exc.stderr or "").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or ""),
        }


def git_info() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "--short=12", "HEAD"]),
        "status": run(["git", "status", "--short"]),
    }


def file_exists(path: Path) -> CheckResult:
    return CheckResult(
        name=f"exists:{path.name}",
        ok=path.is_file(),
        detail=str(path),
    )


def file_absent(path: Path) -> CheckResult:
    return CheckResult(
        name=f"absent:{path.name}",
        ok=not path.exists(),
        detail=str(path),
    )


def dir_absent(path: Path) -> CheckResult:
    return CheckResult(
        name=f"absent:{path.name}/",
        ok=not path.exists(),
        detail=str(path),
    )


def launcher_has_no_forbidden_text(path: Path) -> CheckResult:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    found = [needle for needle in FORBIDDEN_LAUNCHER_TEXT if needle in text]
    return CheckResult(
        name=f"launcher-clean:{path.name}",
        ok=not found,
        detail="forbidden text: " + ", ".join(found) if found else str(path),
    )


def packaged_binary(platform: str, portable_root: Path) -> Path:
    if platform == "windows":
        return portable_root / "omniinfer.exe"
    return portable_root / "omniinfer"


def expected_files(platform: str, portable_root: Path) -> list[Path]:
    if platform == "windows":
        return [
            portable_root / "omniinfer.exe",
            portable_root / "omniinfer.cmd",
            portable_root / "omniinfer.ps1",
        ]
    return [
        portable_root / "omniinfer",
    ]


def launchers(platform: str, portable_root: Path) -> list[Path]:
    if platform == "windows":
        return [portable_root / "omniinfer.cmd", portable_root / "omniinfer.ps1"]
    return [portable_root / "omniinfer"]


def unported_probe_args(platform: str) -> list[str]:
    if platform == "macos":
        return ["build", "llama.cpp-mac"]
    if platform == "windows":
        return ["build", "llama.cpp-cpu"]
    return ["build", "llama.cpp-linux"]


def validate_portable(platform: str, portable_root: Path, output_dir: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    for path in expected_files(platform, portable_root):
        checks.append(file_exists(path))
    checks.append(file_absent(portable_root / "omniinfer.py"))
    checks.append(dir_absent(portable_root / "service_core"))
    for path in launchers(platform, portable_root):
        checks.append(launcher_has_no_forbidden_text(path))

    (portable_root / "runtime").mkdir(exist_ok=True)
    probe = run_command(
        [str(packaged_binary(platform, portable_root)), *unported_probe_args(platform)],
        timeout_s=60.0,
    )
    probe_text = f"{probe['stdout']}\n{probe['stderr']}"
    checks.append(
        CheckResult(
            name="unported-command-probe",
            ok=probe["returncode"] not in (0, None)
            and (REMOVED_FALLBACK_MESSAGE in probe_text or PACKAGED_BUILD_MESSAGE in probe_text),
            detail=f"exit={probe['returncode']}",
        )
    )
    (output_dir / "unported-command.stdout.txt").write_text(probe["stdout"], encoding="utf-8")
    (output_dir / "unported-command.stderr.txt").write_text(probe["stderr"], encoding="utf-8")
    return checks, probe


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# No-Python Portable Validation",
        "",
        f"- Result: {payload['result']}",
        f"- Platform: `{payload['platform']}`",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Git branch: `{payload['git'].get('branch', '-')}`",
        f"- Git commit: `{payload['git'].get('commit', '-')}`",
        f"- Portable root: `{payload['portable_root']}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| {check['name']} | {'ok' if check['ok'] else 'failed'} | `{check['detail']}` |"
        )
    lines.extend(
        [
            "",
            "## Probe",
            "",
            f"- Exit: `{payload['probe']['returncode']}`",
            "- Stdout: `unported-command.stdout.txt`",
            "- Stderr: `unported-command.stderr.txt`",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate no-Python OmniInfer portable packaging.")
    parser.add_argument("--platform", required=True, choices=["linux", "macos", "windows"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--keep-existing-output", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    portable_root = output_dir / "portable"
    if output_dir.exists() and not args.keep_existing_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    packager = run_command(
        [
            sys.executable,
            str(PACKAGER),
            "--repo-root",
            str(REPO_ROOT),
            "--portable-root",
            str(portable_root),
            "--platform",
            args.platform,
        ],
        timeout_s=900.0,
    )
    (output_dir / "packager.stdout.txt").write_text(packager["stdout"], encoding="utf-8")
    (output_dir / "packager.stderr.txt").write_text(packager["stderr"], encoding="utf-8")

    checks = [
        CheckResult(
            name="package-rust-cli",
            ok=packager["returncode"] == 0 and not packager["timed_out"],
            detail=f"exit={packager['returncode']}",
        )
    ]
    probe: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": ""}
    if checks[0].ok:
        portable_checks, probe = validate_portable(args.platform, portable_root, output_dir)
        checks.extend(portable_checks)

    result = "ok" if all(check.ok for check in checks) else "failed"
    payload = {
        "result": result,
        "platform": args.platform,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "portable_root": str(portable_root),
        "checks": [check.__dict__ for check in checks],
        "packager": packager,
        "probe": probe,
    }
    (output_dir / "raw.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(output_dir, payload)
    print(f"Result: {result}")
    print(f"Wrote {output_dir / 'summary.md'}")
    return 0 if result == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
