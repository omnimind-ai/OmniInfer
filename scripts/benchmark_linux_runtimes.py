#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CORE = REPO_ROOT / "scripts" / "benchmark_backends.py"
DEFAULT_BACKENDS = ("llama.cpp-linux", "llama.cpp-linux-vulkan", "mnn-linux")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Linux benchmark matrix across OmniInfer backends using prebuilt runtimes."
    )
    parser.add_argument(
        "--cli-command",
        nargs="+",
        default=["./omniinfer"],
        help="CLI command prefix used to control OmniInfer, default: ./omniinfer",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Gateway host, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=9000, help="Gateway port, default: 9000")
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend id to benchmark. Repeat to override the default set.",
    )
    parser.add_argument("--gguf-model", help="GGUF model file used for llama.cpp Linux backends")
    parser.add_argument("--mnn-model", help="MNN model directory or config.json used for mnn-linux")
    parser.add_argument("--mmproj", help="Optional mmproj file used for GGUF multimodal runs")
    parser.add_argument("--image", help="Optional image used for multimodal benchmark requests")
    parser.add_argument(
        "--prompt",
        default="Explain in one short paragraph why local inference backends differ in speed.",
        help="Benchmark prompt text",
    )
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum completion tokens, default: 96")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per backend, default: 1")
    parser.add_argument("--measure-runs", type=int, default=1, help="Measured runs per backend, default: 1")
    parser.add_argument(
        "--reload-each-run",
        action="store_true",
        help="Reload the model before every warmup and measured run",
    )
    parser.add_argument(
        "--keep-json",
        action="store_true",
        help="Keep the raw benchmark JSON under tmp/ instead of using a temporary file",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def format_float(value: Any, decimals: int = 2) -> str:
    if value in (None, "-", ""):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def parse_metric(run: dict[str, Any], result: dict[str, Any]) -> dict[str, str]:
    usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    timings = run.get("timings") if isinstance(run.get("timings"), dict) else {}

    prompt_tps = timings.get("prompt_per_second")
    if prompt_tps in (None, "-"):
        prompt_tokens = usage.get("prompt_tokens")
        prompt_ms = timings.get("prompt_ms")
        try:
            if prompt_tokens is not None and prompt_ms not in (None, 0, 0.0):
                prompt_tps = (float(prompt_tokens) * 1000.0) / float(prompt_ms)
        except (TypeError, ValueError, ZeroDivisionError):
            prompt_tps = None

    decode_tps = timings.get("predicted_per_second")
    if decode_tps in (None, "-"):
        decode_tps = timings.get("decode_tps")

    return {
        "backend": str(result.get("backend", "-")),
        "prompt_tps": format_float(prompt_tps, 2),
        "prompt_ms": format_float(timings.get("prompt_ms"), 1),
        "decode_tps": format_float(decode_tps, 2),
        "prompt_tokens": str(usage.get("prompt_tokens", "-")),
        "completion_tokens": str(usage.get("completion_tokens", "-")),
        "rss_loaded_mb": format_float(result.get("memory_loaded", {}).get("total_rss_mb"), 2),
        "rss_after_mb": format_float(result.get("memory_after_run", {}).get("total_rss_mb"), 2),
    }


def print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        ("Backend", "backend"),
        ("Prefill tps", "prompt_tps"),
        ("Prompt ms", "prompt_ms"),
        ("Decode tps", "decode_tps"),
        ("Prompt tok", "prompt_tokens"),
        ("Output tok", "completion_tokens"),
        ("RSS load MB", "rss_loaded_mb"),
        ("RSS after MB", "rss_after_mb"),
    ]

    widths: list[int] = []
    for header, key in columns:
        widths.append(max(len(header), *(len(row[key]) for row in rows)))

    print("  ".join(header.ljust(width) for width, (header, _) in zip(widths, columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[key].ljust(width) for width, (_, key) in zip(widths, columns)))


def run_benchmark(args: argparse.Namespace, output_json: Path) -> list[dict[str, Any]]:
    backends = args.backends or list(DEFAULT_BACKENDS)
    command = [
        sys.executable,
        str(BENCHMARK_CORE),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measure-runs",
        str(args.measure_runs),
        "--output-json",
        str(output_json),
    ]

    if args.gguf_model:
        command.extend(["--gguf-model", str(resolve_path(args.gguf_model))])
    if args.mnn_model:
        command.extend(["--mnn-model", str(resolve_path(args.mnn_model))])

    command.append("--cli-command")
    command.extend(args.cli_command)
    for backend_id in backends:
        command.extend(["--backend", backend_id])
    if args.mmproj:
        command.extend(["--mmproj", str(resolve_path(args.mmproj))])
    if args.image:
        command.extend(["--image", str(resolve_path(args.image))])
    if args.reload_each_run:
        command.append("--reload-each-run")

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if completed.returncode != 0:
        fail(
            "Benchmark run failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    try:
        return json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Unable to read benchmark JSON: {exc}")
        raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()

    if args.keep_json:
        output_json = REPO_ROOT / "tmp" / "linux-runtime-benchmark.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
    else:
        handle = tempfile.NamedTemporaryFile(prefix="omniinfer-linux-bench.", suffix=".json", delete=False)
        handle.close()
        output_json = Path(handle.name)

    raw_results = run_benchmark(args, output_json)
    rows = [parse_metric(result["runs"][-1] if result.get("runs") else {}, result) for result in raw_results]
    print_table(rows)

    if not args.keep_json:
        output_json.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
