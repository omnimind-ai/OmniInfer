#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "test_results"

try:
    import resource
except ModuleNotFoundError:
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Scenario:
    name: str
    command: list[str]
    timeout_s: float = 30.0
    description: str = ""
    expected_returncodes: tuple[int, ...] = (0,)


SCENARIOS = [
    Scenario("help", ["./omniinfer", "--help"], description="top-level help"),
    Scenario(
        "advisor-help",
        ["./omniinfer", "advisor"],
        description="advisor scoped help for missing subcommand",
        expected_returncodes=(2,),
    ),
    Scenario("serve-help", ["./omniinfer", "serve", "--help"], description="serve scoped help"),
    Scenario("status", ["./omniinfer", "status"], description="local gateway status"),
    Scenario("thinking-show", ["./omniinfer", "thinking", "show"], description="default thinking state"),
    Scenario("serve-status", ["./omniinfer", "serve", "status", "--port", "9000"], description="detached serve status"),
    Scenario("backend-list", ["./omniinfer", "backend", "list"], description="backend list"),
    Scenario("model-list", ["./omniinfer", "model", "list"], timeout_s=60.0, description="supported model list"),
    Scenario(
        "chat-no-stream-no-model",
        ["./omniinfer", "chat", "--no-stream", "hello"],
        expected_returncodes=(1,),
        timeout_s=60.0,
        description="non-stream chat without a loaded model",
    ),
    Scenario("advisor-system", ["./omniinfer", "advisor", "system"], timeout_s=60.0, description="hardware and backend probe"),
    Scenario(
        "advisor-system-json",
        ["./omniinfer", "advisor", "system", "--json"],
        timeout_s=60.0,
        description="hardware and backend probe with JSON output",
    ),
]

IMPORT_TRACE_SCENARIOS = {
    "help",
    "advisor-help",
    "advisor-system",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _resource_snapshot() -> dict[str, Any]:
    if resource is None:
        return {
            "children_user_cpu_s": 0.0,
            "children_system_cpu_s": 0.0,
            "children_max_rss_kib": 0,
            "children_minor_faults": 0,
            "children_major_faults": 0,
            "children_voluntary_context_switches": 0,
            "children_involuntary_context_switches": 0,
        }
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "children_user_cpu_s": usage.ru_utime,
        "children_system_cpu_s": usage.ru_stime,
        "children_max_rss_kib": usage.ru_maxrss,
        "children_minor_faults": usage.ru_minflt,
        "children_major_faults": usage.ru_majflt,
        "children_voluntary_context_switches": usage.ru_nvcsw,
        "children_involuntary_context_switches": usage.ru_nivcsw,
    }


def _sample_process_rss_kib(pid: int) -> int | None:
    if os.name == "nt":
        return _sample_process_rss_kib_windows(pid)
    status_path = Path("/proc") / str(pid) / "status"
    try:
        raw = status_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in raw.splitlines():
        if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _sample_process_rss_kib_windows(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    if not text or "No tasks" in text:
        return None
    try:
        row = next(csv.reader([text]))
    except (StopIteration, csv.Error):
        return None
    if len(row) < 5:
        return None
    digits = "".join(ch for ch in row[4] if ch.isdigit())
    return int(digits) if digits else None


def _scenario_for_binary(scenario: Scenario, binary: str) -> Scenario:
    command = [binary, *scenario.command[1:]]
    return Scenario(
        name=scenario.name,
        command=command,
        timeout_s=scenario.timeout_s,
        description=scenario.description,
        expected_returncodes=scenario.expected_returncodes,
    )


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


def _shutdown_state_root_service(binary: str, env: dict[str, str]) -> None:
    if "OMNIINFER_RUST_STATE_ROOT" not in env:
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            [binary, "shutdown"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            env=env,
            check=False,
        )


def _run_once(scenario: Scenario, *, env: dict[str, str]) -> dict[str, Any]:
    before = _resource_snapshot()
    started = time.perf_counter()
    proc = subprocess.Popen(
        scenario.command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        **_popen_process_group_kwargs(),
    )
    peak_lock = threading.Lock()
    sampled_peak: int | None = None

    def sample_until_exit() -> None:
        nonlocal sampled_peak
        while proc.poll() is None:
            value = _sample_process_rss_kib(proc.pid)
            if value is not None:
                with peak_lock:
                    sampled_peak = value if sampled_peak is None else max(sampled_peak, value)
            time.sleep(0.01)

    sampler = threading.Thread(target=sample_until_exit, daemon=True)
    sampler.start()
    try:
        stdout, stderr = proc.communicate(timeout=scenario.timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(proc, force=False)
        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc, force=True)
            stdout, stderr = proc.communicate()
    sampler.join(timeout=1.0)
    ended = time.perf_counter()
    after = _resource_snapshot()

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    return {
        "command": scenario.command,
        "expected_returncodes": scenario.expected_returncodes,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "wall_s": ended - started,
        "user_cpu_s": after["children_user_cpu_s"] - before["children_user_cpu_s"],
        "system_cpu_s": after["children_system_cpu_s"] - before["children_system_cpu_s"],
        "max_rss_kib": max(after["children_max_rss_kib"], sampled_peak or 0),
        "sampled_peak_rss_kib": sampled_peak,
        "minor_faults": after["children_minor_faults"] - before["children_minor_faults"],
        "major_faults": after["children_major_faults"] - before["children_major_faults"],
        "voluntary_context_switches": after["children_voluntary_context_switches"] - before["children_voluntary_context_switches"],
        "involuntary_context_switches": after["children_involuntary_context_switches"] - before["children_involuntary_context_switches"],
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_preview": stdout_text[:2000],
        "stderr_preview": stderr_text[:2000],
        "_stderr_full": stderr_text,
    }


def _popen_process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _terminate_process_group(proc: subprocess.Popen[bytes], *, force: bool) -> None:
    with contextlib.suppress(OSError):
        if os.name == "nt":
            if force:
                proc.kill()
            else:
                proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)


def _python_import_command(command: list[str]) -> list[str] | None:
    script = command[0]
    if not script.endswith(".py"):
        return None
    return [sys.executable, "-X", "importtime", script, *command[1:]]


def _parse_import_time(stderr_text: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in stderr_text.splitlines():
        if not line.startswith("import time:"):
            continue
        parts = line.removeprefix("import time:").split("|", maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            self_us = int(parts[0].strip())
            cumulative_us = int(parts[1].strip())
        except ValueError:
            continue
        module = parts[2].strip()
        if module in {"self [us]", "cumulative [us] | imported package"}:
            continue
        rows.append(
            {
                "module": module,
                "self_us": self_us,
                "cumulative_us": cumulative_us,
            }
        )
    rows.sort(key=lambda row: int(row["cumulative_us"]), reverse=True)
    return {
        "module_count": len(rows),
        "top_cumulative": rows[:20],
        "top_self": sorted(rows, key=lambda row: int(row["self_us"]), reverse=True)[:20],
    }


def _run_import_trace(scenario: Scenario, *, env: dict[str, str]) -> dict[str, Any]:
    command = _python_import_command(scenario.command)
    if command is None:
        return {
            "skipped": True,
            "reason": "import-time tracing only applies to Python entrypoints",
        }
    trace_scenario = Scenario(
        name=f"{scenario.name}-import-trace",
        command=command,
        timeout_s=scenario.timeout_s,
        description=f"import-time trace for {scenario.name}",
        expected_returncodes=scenario.expected_returncodes,
    )
    run = _run_once(trace_scenario, env=env)
    run["import_time"] = _parse_import_time(str(run.get("_stderr_full", "")))
    run.pop("_stderr_full", None)
    return run


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "p90": ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.90)))],
    }


def _summarize_runs(runs: list[dict[str, Any]], expected_returncodes: tuple[int, ...]) -> dict[str, Any]:
    success = [r for r in runs if r["returncode"] in expected_returncodes and not r["timed_out"]]
    return {
        "runs": len(runs),
        "successful_runs": len(success),
        "expected_returncodes": list(expected_returncodes),
        "returncodes": [r["returncode"] for r in runs],
        "wall_s": _summarize([float(r["wall_s"]) for r in success]),
        "cpu_s": _summarize([float(r["user_cpu_s"]) + float(r["system_cpu_s"]) for r in success]),
        "max_rss_mib": _summarize([float(r["max_rss_kib"]) / 1024.0 for r in success]),
        "stdout_bytes": _summarize([float(r["stdout_bytes"]) for r in success]),
        "stderr_bytes": _summarize([float(r["stderr_bytes"]) for r in success]),
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CLI Profiling Baseline",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Host: `{payload['host']['hostname']}` / `{payload['host']['platform']}`",
        f"- Python: `{payload['host']['python']}`",
        f"- Git branch: `{payload['git'].get('branch', '-')}`",
        f"- Git commit: `{payload['git'].get('commit', '-')}`",
        f"- Runs per scenario: `{payload['runs_per_scenario']}`",
        "",
        "| Scenario | Success | Wall median | Wall p90 | CPU median | Peak RSS median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in payload["scenarios"]:
        summary = scenario["summary"]
        wall = summary.get("wall_s", {})
        cpu = summary.get("cpu_s", {})
        rss = summary.get("max_rss_mib", {})
        lines.append(
            "| {name} | {ok}/{runs} | {wall:.3f}s | {p90:.3f}s | {cpu:.3f}s | {rss:.1f} MiB |".format(
                name=scenario["name"],
                ok=summary["successful_runs"],
                runs=summary["runs"],
                wall=wall.get("median", 0.0),
                p90=wall.get("p90", 0.0),
                cpu=cpu.get("median", 0.0),
                rss=rss.get("median", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Import-Time Traces",
            "",
            "| Scenario | Modules | Slowest cumulative imports |",
            "|---|---:|---|",
        ]
    )
    for scenario in payload["scenarios"]:
        trace = scenario.get("import_trace")
        if not trace:
            continue
        import_time = trace.get("import_time", {})
        top = import_time.get("top_cumulative", [])[:5]
        top_text = ", ".join(f"{row['module']}={row['cumulative_us'] / 1000:.1f}ms" for row in top)
        lines.append(f"| {scenario['name']} | {import_time.get('module_count', 0)} | {top_text} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `max_rss_mib` uses `resource.RUSAGE_CHILDREN.ru_maxrss` where available plus direct-child RSS sampling (`/proc/<pid>/status` on Unix, `tasklist` on Windows).",
            "- Results are intended for before/after comparison with the Rust control-plane prototype, not as absolute benchmark claims.",
            "- Raw per-run data is stored in `raw.json` next to this file.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile current Python OmniInfer CLI startup and probe paths.")
    parser.add_argument("--runs", type=int, default=5, help="runs per scenario")
    parser.add_argument("--binary", default="./omniinfer", help="entrypoint binary or script to profile")
    parser.add_argument("--output-dir", type=Path, default=None, help="directory for raw.json and summary.md")
    parser.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS], help="scenario to run; repeatable")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="isolate OmniInfer .local/config/log/run state under this root",
    )
    parser.add_argument(
        "--skip-import-trace",
        action="store_true",
        help="skip representative Python -X importtime traces",
    )
    args = parser.parse_args(argv)

    selected = SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        selected = [s for s in SCENARIOS if s.name in wanted]
    if args.runs <= 0:
        parser.error("--runs must be positive")

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{_utc_stamp()}-python-cli-profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    if args.state_root:
        state_root = args.state_root.resolve()
        state_port = _prepare_state_root(state_root)
        env["OMNIINFER_RUST_REPO_ROOT"] = str(REPO_ROOT)
        env["OMNIINFER_RUST_STATE_ROOT"] = str(state_root)
    else:
        state_port = None
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
        },
        "git": _git_info(),
        "runs_per_scenario": args.runs,
        "binary": args.binary,
        "state_root": str(args.state_root.resolve()) if args.state_root else "",
        "state_port": state_port,
        "scenarios": [],
    }

    for scenario in selected:
        scenario = _scenario_for_binary(scenario, args.binary)
        runs = []
        for _ in range(args.runs):
            runs.append(_run_once(scenario, env=env))
        item: dict[str, Any] = {
            "name": scenario.name,
            "description": scenario.description,
            "command": scenario.command,
            "summary": _summarize_runs(runs, scenario.expected_returncodes),
            "runs": runs,
        }
        if not args.skip_import_trace and scenario.name in IMPORT_TRACE_SCENARIOS:
            item["import_trace"] = _run_import_trace(scenario, env=env)
        payload["scenarios"].append(item)

    _shutdown_state_root_service(args.binary, env)

    raw_path = output_dir / "raw.json"
    summary_path = output_dir / "summary.md"
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(summary_path, payload)

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
