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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "test_results"
RUST_BINARY = REPO_ROOT / "target" / "debug" / ("omniinfer-rs.exe" if os.name == "nt" else "omniinfer-rs")


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    timeout_s: float = 300.0
    required: bool = True
    env_overrides: dict[str, str] | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _run_step(step: Step, *, env: dict[str, str], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            step.command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=step.timeout_s,
            env={**env, **(step.env_overrides or {})},
            check=False,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    elapsed = time.perf_counter() - started

    step_dir = output_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    safe_name = step.name.replace("/", "-")
    (step_dir / f"{safe_name}.stdout.txt").write_bytes(stdout)
    (step_dir / f"{safe_name}.stderr.txt").write_bytes(stderr)

    ok = (returncode == 0) and not timed_out
    if not step.required:
        ok = ok or returncode is not None
    return {
        "name": step.name,
        "command": step.command,
        "timeout_s": step.timeout_s,
        "required": step.required,
        "returncode": returncode,
        "timed_out": timed_out,
        "ok": ok,
        "wall_s": elapsed,
        "stdout_path": str(step_dir / f"{safe_name}.stdout.txt"),
        "stderr_path": str(step_dir / f"{safe_name}.stderr.txt"),
        "stdout_preview": stdout.decode("utf-8", errors="replace")[:2000],
        "stderr_preview": stderr.decode("utf-8", errors="replace")[:2000],
    }


def _git_info() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "--short=12", "HEAD"]),
        "status": run(["git", "status", "--short"]),
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Rust Control Plane Validation",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Git branch: `{payload['git'].get('branch', '-')}`",
        f"- Git commit: `{payload['git'].get('commit', '-')}`",
        f"- State root: `{payload['state_root']}`",
        f"- Rust binary: `{payload['rust_binary']}`",
        "",
        "## Steps",
        "",
        "| Step | Result | Exit | Wall |",
        "|---|---:|---:|---:|",
    ]
    for step in payload["steps"]:
        exit_text = "timeout" if step["timed_out"] else str(step["returncode"])
        lines.append(
            f"| {step['name']} | {'ok' if step['ok'] else 'failed'} | {exit_text} | {step['wall_s']:.2f}s |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- No-Python portable validation: `{payload['artifacts']['no_python_portable']}`",
            f"- Rust strict contracts: `{payload['artifacts']['rust_strict_contracts']}`",
            f"- Rust profile: `{payload['artifacts']['rust_profile']}`",
            "",
            "Raw step logs and metadata are stored next to this file.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _python() -> str:
    return sys.executable


def _portable_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def _base_steps(output_dir: Path, state_root: Path, runs: int) -> list[Step]:
    rust_contracts = output_dir / "rust-strict-contracts"
    rust_profile = output_dir / "rust-profile"
    no_python_portable = output_dir / "no-python-portable"
    return [
        Step("cargo-fmt", ["cargo", "fmt", "--all", "--", "--check"], timeout_s=120.0),
        Step("cargo-test", ["cargo", "test", "--workspace"], timeout_s=300.0),
        Step(
            "no-python-portable",
            [
                _python(),
                "scripts/validate_no_python_portable.py",
                "--platform",
                _portable_platform(),
                "--output-dir",
                str(no_python_portable),
            ],
            timeout_s=900.0,
        ),
        Step(
            "rust-strict-contracts",
            [
                _python(),
                "scripts/capture_cli_contracts.py",
                "--binary",
                str(RUST_BINARY),
                "--rust-strict",
                "--state-root",
                str(state_root / "rust-strict-contracts"),
                "--output-dir",
                str(rust_contracts),
            ],
            timeout_s=600.0,
        ),
        Step(
            "rust-profile",
            [
                _python(),
                "scripts/profile_python_cli.py",
                "--runs",
                str(runs),
                "--binary",
                str(RUST_BINARY),
                "--skip-import-trace",
                "--state-root",
                str(state_root / "rust-profile"),
                "--output-dir",
                str(rust_profile),
            ],
            timeout_s=900.0,
        ),
        Step("git-diff-check", ["git", "diff", "--check"], timeout_s=60.0),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Rust OmniInfer control-plane before entrypoint switch.")
    parser.add_argument("--runs", type=int, default=3, help="profile runs per scenario")
    parser.add_argument("--output-dir", type=Path, default=None, help="validation artifact directory")
    parser.add_argument("--keep-state-root", action="store_true", help="keep isolated state root after validation")
    args = parser.parse_args(argv)

    if args.runs <= 0:
        parser.error("--runs must be positive")

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{_utc_stamp()}-rust-control-plane-validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_root = output_dir / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")

    steps = []
    for step in _base_steps(output_dir, state_root, args.runs):
        result = _run_step(step, env=env, output_dir=output_dir)
        steps.append(result)
        print(f"{step.name}: {'ok' if result['ok'] else 'failed'}")
        if step.required and not result["ok"]:
            break

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "state_root": str(state_root),
        "rust_binary": str(RUST_BINARY),
        "steps": steps,
        "artifacts": {
            "no_python_portable": str(output_dir / "no-python-portable" / "summary.md"),
            "rust_strict_contracts": str(output_dir / "rust-strict-contracts" / "summary.md"),
            "rust_profile": str(output_dir / "rust-profile" / "summary.md"),
        },
    }
    (output_dir / "raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", payload)

    if not args.keep_state_root:
        shutil.rmtree(state_root, ignore_errors=True)

    failed = [step for step in steps if step["required"] and not step["ok"]]
    print(f"Wrote {output_dir / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
