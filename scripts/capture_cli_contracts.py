#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import contextlib
import json
import os
import socket
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
DEFAULT_BINARY = REPO_ROOT / ("omniinfer.cmd" if os.name == "nt" else "omniinfer")


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
    ContractScenario("advisor-inspect-json", ["advisor", "inspect", "{fixture_model}", "--json"], timeout_s=60.0),
    ContractScenario(
        "advisor-fit-json",
        ["advisor", "fit", "{fixture_model}", "--ctx-size", "512", "--json"],
        timeout_s=60.0,
    ),
    ContractScenario(
        "advisor-plan-json",
        ["advisor", "plan", "{fixture_model}", "--ctx-size", "512", "--json"],
        timeout_s=60.0,
    ),
    ContractScenario("advisor-recommend-json", ["advisor", "recommend", "-n", "2", "--json"], timeout_s=60.0),
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


def _state_path(extra_env: dict[str, str]) -> Path:
    state_root = extra_env.get("OMNIINFER_RUST_STATE_ROOT")
    if state_root:
        return Path(state_root) / ".local" / "config" / "state.json"
    return STATE_PATH


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_state_root(state_root: Path) -> int:
    port = _free_loopback_port()
    (state_root / ".local" / "config").mkdir(parents=True, exist_ok=True)
    (state_root / ".local" / "logs").mkdir(parents=True, exist_ok=True)
    (state_root / ".local" / "run").mkdir(parents=True, exist_ok=True)
    (state_root / "config").mkdir(parents=True, exist_ok=True)
    config = {
        "host": "127.0.0.1",
        "port": port,
        "startup_timeout": 10,
        "window_mode": "hidden",
        "default_thinking": "off",
    }
    (state_root / "config" / "omniinfer.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return port


def _shutdown_state_root_service(binary: str, extra_env: dict[str, str]) -> None:
    if "OMNIINFER_RUST_STATE_ROOT" not in extra_env:
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            [binary, "shutdown"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            env={**os.environ, "NO_COLOR": "1", **extra_env},
            check=False,
        )


def _run_scenario(
    binary: str,
    scenario: ContractScenario,
    *,
    extra_env: dict[str, str],
    fixture_model: Path,
) -> dict[str, Any]:
    state_path = _state_path(extra_env)
    before_state = _read_optional(state_path)
    expanded_args = [
        str(fixture_model) if arg == "{fixture_model}" else arg
        for arg in scenario.args
    ]
    command = [binary, *expanded_args]
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
    after_state = _read_optional(state_path)

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
        "state_path": str(state_path),
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
        f"- Fixture model: `{payload['fixture_model']}`",
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
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="entrypoint binary or script")
    parser.add_argument("--output-dir", type=Path, default=None, help="snapshot output directory")
    parser.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS])
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="isolate OmniInfer .local/config/log/run state under this root",
    )
    parser.add_argument("--rust-strict", action="store_true", help="set OMNIINFER_RUST_STRICT=1 for the snapshot")
    args = parser.parse_args(argv)

    selected = SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        selected = [scenario for scenario in SCENARIOS if scenario.name in wanted]

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{_utc_stamp()}-cli-contracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_model = _ensure_fixture_model(output_dir)

    extra_env: dict[str, str] = {}
    if args.rust_strict:
        extra_env["OMNIINFER_RUST_STRICT"] = "1"
    if args.state_root:
        state_root = args.state_root.resolve()
        state_port = _prepare_state_root(state_root)
        extra_env["OMNIINFER_RUST_REPO_ROOT"] = str(REPO_ROOT)
        extra_env["OMNIINFER_RUST_STATE_ROOT"] = str(state_root)
    else:
        state_port = None

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": args.binary,
        "env": extra_env,
        "fixture_model": str(fixture_model),
        "state_path": str(_state_path(extra_env)),
        "state_port": state_port,
        "git": _git_info(),
        "results": [
            _run_scenario(args.binary, scenario, extra_env=extra_env, fixture_model=fixture_model)
            for scenario in selected
        ],
    }
    _shutdown_state_root_service(args.binary, extra_env)
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


def _ensure_fixture_model(output_dir: Path) -> Path:
    fixtures = output_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    model = fixtures / "Qwen3.5-4B-Q4_K_M.gguf"
    if not model.exists():
        model.write_bytes(b"OmniInfer advisor contract fixture\n")
    return model


if __name__ == "__main__":
    raise SystemExit(main())
