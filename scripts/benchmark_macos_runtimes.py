#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CORE = REPO_ROOT / "scripts" / "benchmark_backends.py"
DEFAULT_BACKENDS = ("llama.cpp-mac", "turboquant-mac", "mlx-mac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pretty three-backend benchmark on macOS using prebuilt OmniInfer runtimes."
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
    parser.add_argument("--gguf-model", required=True, help="GGUF model file used for llama.cpp-mac and turboquant-mac")
    parser.add_argument("--mlx-model", required=True, help="MLX model snapshot directory or Hugging Face cache repo root")
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
        help="Keep the raw benchmark JSON next to the script instead of using a temporary file",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def resolve_mlx_snapshot(path: Path) -> Path:
    if not path.exists():
        fail(f"MLX model path was not found: {path}")

    if (path / "config.json").is_file() and (path / "tokenizer.json").is_file():
        return path

    refs_main = path / "refs" / "main"
    if refs_main.is_file():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_id:
            snapshot_path = path / "snapshots" / snapshot_id
            if snapshot_path.is_dir():
                return snapshot_path.resolve()

    snapshots_dir = path / "snapshots"
    if snapshots_dir.is_dir():
        candidates = sorted([item for item in snapshots_dir.iterdir() if item.is_dir()])
        if candidates:
            return candidates[-1].resolve()

    fail(
        "Unable to resolve an MLX snapshot directory. Pass either a snapshot directory that contains "
        "config.json/tokenizer.json, or a Hugging Face cache repo root with refs/main and snapshots/."
    )
    raise AssertionError("unreachable")


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
    return {
        "backend": str(result.get("backend", "-")),
        "prompt_tps": format_float(prompt_tps, 2),
        "prompt_ms": format_float(timings.get("prompt_ms"), 1),
        "decode_tps": format_float(decode_tps, 2),
        "peak_memory_gb": format_float(timings.get("peak_memory_gb"), 3),
        "prompt_tokens": str(usage.get("prompt_tokens", "-")),
        "completion_tokens": str(usage.get("completion_tokens", "-")),
        "rss_loaded_mb": format_float(result.get("memory_loaded", {}).get("total_rss_mb"), 2),
        "rss_after_mb": format_float(result.get("memory_after_run", {}).get("total_rss_mb"), 2),
    }


def rank_backend(rows: list[dict[str, str]], key: str) -> str | None:
    numeric: list[tuple[float, str]] = []
    for row in rows:
        try:
            numeric.append((float(row[key]), row["backend"]))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    numeric.sort(reverse=True)
    return numeric[0][1]


def print_banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\033[1;36m{line}\n{title}\n{line}\033[0m")


def print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        ("Backend", "backend"),
        ("Prefill tps", "prompt_tps"),
        ("Prompt ms", "prompt_ms"),
        ("Decode tps", "decode_tps"),
        ("MLX peak GB", "peak_memory_gb"),
        ("Prompt tok", "prompt_tokens"),
        ("Output tok", "completion_tokens"),
        ("RSS load MB", "rss_loaded_mb"),
        ("RSS after MB", "rss_after_mb"),
    ]

    widths: list[int] = []
    for header, key in columns:
        widths.append(max(len(header), *(len(row[key]) for row in rows)))

    header_line = "  ".join(header.ljust(width) for width, (header, _) in zip(widths, columns))
    sep_line = "  ".join("-" * width for width in widths)
    print(f"\033[1m{header_line}\033[0m")
    print(sep_line)

    for row in rows:
        backend = row["backend"]
        if backend == "llama.cpp-mac":
            color = "\033[38;5;45m"
        elif backend == "turboquant-mac":
            color = "\033[38;5;214m"
        elif backend == "mlx-mac":
            color = "\033[38;5;112m"
        else:
            color = "\033[0m"
        parts = []
        for width, (_, key) in zip(widths, columns):
            parts.append(row[key].ljust(width))
        print(f"{color}{'  '.join(parts)}\033[0m")


def run_benchmark(args: argparse.Namespace, output_json: Path) -> list[dict[str, Any]]:
    backends = args.backends or list(DEFAULT_BACKENDS)
    command = [
        sys.executable,
        str(BENCHMARK_CORE),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gguf-model",
        str(resolve_path(args.gguf_model)),
        "--mlx-model",
        str(resolve_mlx_snapshot(resolve_path(args.mlx_model))),
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
        output_json = REPO_ROOT / "tmp" / "macos-runtime-benchmark.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
    else:
        handle = tempfile.NamedTemporaryFile(prefix="omniinfer-macos-bench.", suffix=".json", delete=False)
        handle.close()
        output_json = Path(handle.name)

    raw_results = run_benchmark(args, output_json)
    rows = [parse_metric(result["runs"][-1] if result.get("runs") else {}, result) for result in raw_results]

    print_banner("OmniInfer macOS Runtime Benchmark")
    print(f"Prompt: {args.prompt}")
    print(f"GGUF model: {resolve_path(args.gguf_model)}")
    print(f"MLX model:  {resolve_mlx_snapshot(resolve_path(args.mlx_model))}")
    if args.mmproj:
        print(f"MMProj:     {resolve_path(args.mmproj)}")
    if args.image:
        print(f"Image:      {resolve_path(args.image)}")
    print("")
    print_table(rows)

    winner_prefill = rank_backend(rows, "prompt_tps")
    winner_decode = rank_backend(rows, "decode_tps")
    print("")
    print("\033[1mHighlights\033[0m")
    print(f"- Fastest prefill: {winner_prefill or '-'}")
    print(f"- Fastest decode:  {winner_decode or '-'}")

    if args.keep_json:
        print(f"- Raw JSON:        {output_json}")
    else:
        try:
            output_json.unlink()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
