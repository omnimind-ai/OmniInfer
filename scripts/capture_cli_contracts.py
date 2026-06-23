#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "test_results"
STATE_PATH = REPO_ROOT / ".local" / "config" / "state.json"


@dataclass(frozen=True)
class ContractScenario:
    name: str
    args: list[str]
    expected_returncodes: tuple[int, ...] = (0,)
    timeout_s: float = 30.0
    read_only: bool = True
    allow_pending: bool = False


SCENARIOS = [
    ContractScenario("help", ["--help"]),
    ContractScenario("advisor-help", ["advisor"], expected_returncodes=(2,)),
    ContractScenario("advisor-system-json", ["advisor", "system", "--json"], timeout_s=60.0),
    ContractScenario("serve-help", ["serve", "--help"]),
    ContractScenario("completion-bash", ["completion", "bash"]),
    ContractScenario("status", ["status"], timeout_s=60.0),
    ContractScenario("thinking-show", ["thinking", "show"], timeout_s=60.0),
    ContractScenario("thinking-set-off", ["thinking", "set", "off"], timeout_s=60.0, read_only=False),
    ContractScenario("serve-status", ["serve", "status", "--port", "9000"], timeout_s=60.0),
    ContractScenario("backend-list", ["backend", "list"], timeout_s=60.0),
    ContractScenario("model-list", ["model", "list"], timeout_s=60.0),
    ContractScenario(
        "chat-no-stream-no-model",
        ["chat", "--no-stream", "hello"],
        expected_returncodes=(1,),
        timeout_s=60.0,
    ),
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _run_scenario(binary: str, scenario: ContractScenario, *, extra_env: dict[str, str]) -> dict[str, Any]:
    before_state = _read_optional(STATE_PATH)
    command = [binary, *scenario.args]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=scenario.timeout_s,
            env={**os.environ, "NO_COLOR": "1", **extra_env},
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
    after_state = _read_optional(STATE_PATH)

    stdout_preview = stdout.decode("utf-8", errors="replace")[:4000]
    stderr_preview = stderr.decode("utf-8", errors="replace")[:4000]
    pending = _contains_pending_marker(stdout_preview) or _contains_pending_marker(stderr_preview)

    return {
        "name": scenario.name,
        "command": command,
        "env": extra_env,
        "expected_returncodes": list(scenario.expected_returncodes),
        "returncode": returncode,
        "timed_out": timed_out,
        "pending": pending,
        "allow_pending": scenario.allow_pending,
        "ok": (returncode in scenario.expected_returncodes)
        and not timed_out
        and (scenario.allow_pending or not pending),
        "read_only": scenario.read_only,
        "state_changed": before_state != after_state,
        "wall_s": elapsed,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "stdout_preview": stdout_preview,
        "stderr_preview": stderr_preview,
    }


def _contains_pending_marker(text: str) -> bool:
    return "implementation pending" in text or "not implemented yet" in text


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
        "# CLI Contract Snapshot",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Binary: `{payload['binary']}`",
        f"- Env: `{payload.get('env') or {}}`",
        f"- Git branch: `{payload['git'].get('branch', '-')}`",
        f"- Git commit: `{payload['git'].get('commit', '-')}`",
        "",
        "| Scenario | Exit | Expected | Pending | State changed | stdout | stderr |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lines.append(
            "| {name} | {exit} | {expected} | {pending} | {state} | {stdout} | {stderr} |".format(
                name=result["name"],
                exit="timeout" if result["timed_out"] else result["returncode"],
                expected=",".join(str(code) for code in result["expected_returncodes"]),
                pending="yes" if result["pending"] else "no",
                state="yes" if result["state_changed"] else "no",
                stdout=result["stdout_bytes"],
                stderr=result["stderr_bytes"],
            )
        )
    lines.extend(["", "Raw hashes and previews are stored in `raw.json`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture OmniInfer CLI contract snapshots.")
    parser.add_argument("--binary", default="./omniinfer", help="entrypoint binary or script")
    parser.add_argument("--output-dir", type=Path, default=None, help="snapshot output directory")
    parser.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rust-strict", action="store_true", help="set OMNIINFER_RUST_STRICT=1 for the snapshot")
    mode.add_argument("--force-python", action="store_true", help="set OMNIINFER_FORCE_PYTHON=1 for the snapshot")
    args = parser.parse_args(argv)

    selected = SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        selected = [scenario for scenario in SCENARIOS if scenario.name in wanted]

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{_utc_stamp()}-cli-contracts"
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_env: dict[str, str] = {}
    if args.rust_strict:
        extra_env["OMNIINFER_RUST_STRICT"] = "1"
    if args.force_python:
        extra_env["OMNIINFER_FORCE_PYTHON"] = "1"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": args.binary,
        "env": extra_env,
        "git": _git_info(),
        "results": [_run_scenario(args.binary, scenario, extra_env=extra_env) for scenario in selected],
    }
    raw_path = output_dir / "raw.json"
    summary_path = output_dir / "summary.md"
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(summary_path, payload)

    failed = [result for result in payload["results"] if not result["ok"]]
    state_mutations = [result for result in payload["results"] if result["read_only"] and result["state_changed"]]
    print(f"Wrote {summary_path}")
    if failed:
        print("Unexpected exit status:", ", ".join(result["name"] for result in failed), file=sys.stderr)
    if state_mutations:
        print("Read-only scenarios changed state:", ", ".join(result["name"] for result in state_mutations), file=sys.stderr)
    return 1 if failed or state_mutations else 0


if __name__ == "__main__":
    raise SystemExit(main())
