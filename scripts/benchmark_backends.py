#!/usr/bin/env python3

from __future__ import annotations

import argparse
import platform
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_backends_for_host() -> list[str]:
    system_name = platform.system()
    if system_name == "Linux":
        return ["llama.cpp-linux", "llama.cpp-linux-vulkan", "mnn-linux"]
    if system_name == "Darwin":
        return ["llama.cpp-mac", "turboquant-mac", "mlx-mac"]
    return ["llama.cpp-linux", "llama.cpp-linux-vulkan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OmniInfer backends with a consistent streaming request")
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
    parser.add_argument("--gguf-model", help="Model path used for external server backends such as llama.cpp-mac and turboquant-mac")
    parser.add_argument("--mlx-model", help="Model directory used for mlx-mac")
    parser.add_argument("--mnn-model", help="Model directory or config.json used for mnn-linux")
    parser.add_argument("--mmproj", help="Optional mmproj file used for GGUF multimodal backends")
    parser.add_argument("--image", help="Optional image path used for multimodal benchmark requests")
    parser.add_argument(
        "--prompt",
        default="Count from 1 to 100 with spaces only.",
        help="Benchmark prompt, default: 'Count from 1 to 100 with spaces only.'",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum completion tokens, default: 128")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per backend before measuring, default: 1")
    parser.add_argument("--measure-runs", type=int, default=1, help="Measured runs per backend, default: 1")
    parser.add_argument(
        "--reload-each-run",
        action="store_true",
        help="Stop and reload the backend model before every warmup and measured run so prefill stays cold",
    )
    parser.add_argument("--output-json", help="Optional path to write the raw benchmark result JSON")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def run_cli(command_prefix: list[str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*command_prefix, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )


def require_cli_success(result: subprocess.CompletedProcess[str], context: str) -> None:
    if result.returncode != 0:
        fail(f"{context} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def request_stream(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "text/event-stream, application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    final_event: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=3600.0) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and "usage" in event:
                    final_event = event
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        fail(f"stream request failed with status {exc.code}: {raw}")
    except urllib.error.URLError as exc:
        fail(f"stream request failed: {exc}")

    if final_event is None:
        fail("stream request completed without a final usage event")
    return final_event


def pid_for_listening_gateway(port: int) -> int | None:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return int(line)
    return None


def pid_for_backend_process(backend_id: str) -> int | None:
    runtime_roots = [REPO_ROOT / ".local" / "runtime" / system_name for system_name in ("linux", "macos", "windows", "android")]
    runtime_bin: Path | None = None
    for runtime_root in runtime_roots:
        candidate = runtime_root / backend_id / "bin" / "llama-server"
        if candidate.is_file():
            runtime_bin = candidate
            break
    if runtime_bin is None:
        return None
    result = subprocess.run(
        ["pgrep", "-f", str(runtime_bin)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return int(line)
    return None


def rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value:
        return None
    try:
        return round(int(value) / 1024.0, 2)
    except ValueError:
        return None


def total_rss_mb(*values: float | None) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable), 2)


def pick_model_for_backend(args: argparse.Namespace, backend_id: str) -> str:
    if backend_id == "mlx-mac":
        if not args.mlx_model:
            fail("mlx-mac requires --mlx-model")
        return str(Path(args.mlx_model).expanduser().resolve())
    if backend_id == "mnn-linux":
        if not args.mnn_model:
            fail("mnn-linux requires --mnn-model")
        return str(Path(args.mnn_model).expanduser().resolve())
    if not args.gguf_model:
        fail(f"{backend_id} requires --gguf-model")
    return str(Path(args.gguf_model).expanduser().resolve())


def benchmark_backend(args: argparse.Namespace, backend_id: str, base_url: str) -> dict[str, Any]:
    model_path = pick_model_for_backend(args, backend_id)

    mmproj_path = (
        str(Path(args.mmproj).expanduser().resolve())
        if args.mmproj and backend_id not in {"mlx-mac", "mnn-linux"}
        else None
    )

    def load_backend_model() -> None:
        require_cli_success(run_cli(args.cli_command, "select", backend_id), f"select {backend_id}")
        load_args = ["model", "load", "-m", model_path]
        if mmproj_path:
            load_args.extend(["-mm", mmproj_path])
        require_cli_success(run_cli(args.cli_command, *load_args), f"load model for {backend_id}")

    load_backend_model()

    time.sleep(1.0)
    gateway_pid = pid_for_listening_gateway(args.port)
    backend_pid = None if backend_id in {"mlx-mac", "mnn-linux"} else pid_for_backend_process(backend_id)

    memory_loaded = {
        "gateway_rss_mb": rss_mb(gateway_pid),
        "backend_rss_mb": rss_mb(backend_pid),
    }
    memory_loaded["total_rss_mb"] = total_rss_mb(
        memory_loaded["gateway_rss_mb"],
        memory_loaded["backend_rss_mb"],
    )

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        suffix = image_path.suffix.lower()
        mime = "image/png"
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        import base64

        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": args.prompt}]

    payload = {
        "backend": backend_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if mmproj_path:
        payload["mmproj"] = mmproj_path

    for _ in range(max(args.warmup_runs, 0)):
        if args.reload_each_run:
            run_cli(args.cli_command, "backend", "stop")
            load_backend_model()
        request_stream(base_url, payload)

    measured_runs: list[dict[str, Any]] = []
    for _ in range(max(args.measure_runs, 1)):
        if args.reload_each_run:
            run_cli(args.cli_command, "backend", "stop")
            load_backend_model()
        measured_runs.append(request_stream(base_url, payload))

    memory_after = {
        "gateway_rss_mb": rss_mb(gateway_pid),
        "backend_rss_mb": rss_mb(backend_pid),
    }
    memory_after["total_rss_mb"] = total_rss_mb(
        memory_after["gateway_rss_mb"],
        memory_after["backend_rss_mb"],
    )

    return {
        "backend": backend_id,
        "model": model_path,
        "gateway_pid": gateway_pid,
        "backend_pid": backend_pid,
        "memory_loaded": memory_loaded,
        "memory_after_run": memory_after,
        "runs": measured_runs,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("Backend\tPrefill tok/s\tPrompt ms\tDecode tok/s\tPrompt tokens\tCompletion tokens\tRSS loaded MB\tRSS after MB")
    for result in results:
        last_run = result["runs"][-1] if result["runs"] else {}
        usage = last_run.get("usage") if isinstance(last_run.get("usage"), dict) else {}
        timings = last_run.get("timings") if isinstance(last_run.get("timings"), dict) else {}
        prompt_tps = timings.get("prompt_per_second")
        if prompt_tps in (None, "-"):
            prompt_tokens = usage.get("prompt_tokens")
            prompt_ms = timings.get("prompt_ms")
            try:
                if prompt_tokens is not None and prompt_ms not in (None, 0, 0.0):
                    prompt_tps = round((float(prompt_tokens) * 1000.0) / float(prompt_ms), 3)
            except (TypeError, ValueError, ZeroDivisionError):
                prompt_tps = "-"
        decode_tps = timings.get("predicted_per_second")
        if decode_tps in (None, "-"):
            decode_tps = timings.get("decode_tps", "-")
        print(
            "\t".join(
                [
                    result["backend"],
                    str(prompt_tps if prompt_tps is not None else "-"),
                    str(timings.get("prompt_ms", "-")),
                    str(decode_tps if decode_tps is not None else "-"),
                    str(usage.get("prompt_tokens", "-")),
                    str(usage.get("completion_tokens", "-")),
                    str(result["memory_loaded"].get("total_rss_mb", "-")),
                    str(result["memory_after_run"].get("total_rss_mb", "-")),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    backends = args.backends or default_backends_for_host()

    require_cli_success(run_cli(args.cli_command, "shutdown"), "shutdown existing service")

    results = [benchmark_backend(args, backend_id, base_url) for backend_id in backends]
    print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote raw results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
