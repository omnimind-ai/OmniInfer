#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "test_results"
RUST_BINARY = REPO_ROOT / "target" / "debug" / "omniinfer"
DEFAULT_IMAGE = REPO_ROOT / "tests" / "fixtures" / "test1.png"
DEFAULT_STEPFUN_MODEL = Path("/home/zhangguanhuai/models/gguf/stepfun/stepfun-ai_GELab-Zero-4B-preview-bf16.gguf")
DEFAULT_STEPFUN_MMPROJ = Path("/home/zhangguanhuai/models/gguf/stepfun/mmproj-stepfun-ai_GELab-Zero-4B-preview-bf16.gguf")
DEFAULT_QWEN27_MODEL = Path("/home/zhangguanhuai/models/gguf/qwen/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_QWEN27_MMPROJ_BF16 = Path("/home/zhangguanhuai/models/gguf/qwen/qwen3.6-27b/mmproj-BF16.gguf")
DEFAULT_QWEN27_MMPROJ_F16 = Path("/home/zhangguanhuai/models/gguf/qwen/qwen3.6-27b/mmproj-F16.gguf")


@dataclass(frozen=True)
class SmokeCase:
    name: str
    model: Path
    mmproj: Path | None
    prompt: str
    image: Path | None
    ctx_size: int
    timeout_s: float


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_state_root(state_root: Path, port: int) -> None:
    (state_root / ".local" / "config").mkdir(parents=True, exist_ok=True)
    (state_root / ".local" / "logs").mkdir(parents=True, exist_ok=True)
    (state_root / ".local" / "run").mkdir(parents=True, exist_ok=True)
    (state_root / "config").mkdir(parents=True, exist_ok=True)
    config = {
        "host": "127.0.0.1",
        "port": port,
        "startup_timeout": 180,
        "window_mode": "hidden",
        "default_thinking": "off",
    }
    (state_root / "config" / "omniinfer.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def _json_request(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any] | str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8-sig", errors="replace")
            try:
                return response.getcode(), json.loads(raw)
            except json.JSONDecodeError:
                return response.getcode(), raw
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8-sig", errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw


def _image_part(path: Path) -> dict[str, Any]:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _chat_payload(case: SmokeCase) -> dict[str, Any]:
    if case.image:
        content: Any = [
            {"type": "text", "text": case.prompt},
            _image_part(case.image),
        ]
    else:
        content = case.prompt
    return {
        "model": "omniinfer",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }


def _extract_text(payload: dict[str, Any] | str) -> str:
    if not isinstance(payload, dict):
        return str(payload)[:400]
    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )[:400]


def _run_case(case: SmokeCase, *, output_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    case_dir = output_dir / case.name
    state_root = case_dir / "state"
    case_dir.mkdir(parents=True, exist_ok=True)
    port = _free_loopback_port()
    _prepare_state_root(state_root, port)
    case_env = {
        **env,
        "NO_COLOR": "1",
        "OMNIINFER_RUST_REPO_ROOT": str(REPO_ROOT),
        "OMNIINFER_RUST_STATE_ROOT": str(state_root),
    }

    command = [
        str(RUST_BINARY),
        "serve",
        "--detach",
        "--backend",
        "llama.cpp-linux-cuda",
        "--model",
        str(case.model),
        "--ctx-size",
        str(case.ctx_size),
        "--port",
        str(port),
        "--startup-timeout",
        "180",
        "--no-smoke-test",
    ]
    if case.mmproj:
        command.extend(["--mmproj", str(case.mmproj)])

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=case.timeout_s,
        env=case_env,
        check=False,
    )
    serve_elapsed = time.perf_counter() - started
    (case_dir / "serve.stdout.txt").write_bytes(completed.stdout)
    (case_dir / "serve.stderr.txt").write_bytes(completed.stderr)

    result: dict[str, Any] = {
        "name": case.name,
        "model": str(case.model),
        "mmproj": str(case.mmproj) if case.mmproj else "",
        "image": str(case.image) if case.image else "",
        "ctx_size": case.ctx_size,
        "port": port,
        "serve_command": command,
        "serve_returncode": completed.returncode,
        "serve_wall_s": serve_elapsed,
        "ok": False,
    }

    try:
        if completed.returncode != 0:
            result["error"] = "serve failed"
            return result

        chat_started = time.perf_counter()
        status, payload = _json_request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            _chat_payload(case),
            timeout_s=180,
        )
        chat_elapsed = time.perf_counter() - chat_started
        (case_dir / "chat.response.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
            if isinstance(payload, dict)
            else str(payload),
            encoding="utf-8",
        )
        result.update(
            {
                "chat_status": status,
                "chat_wall_s": chat_elapsed,
                "response_text": _extract_text(payload),
                "ok": status == 200 and bool(_extract_text(payload).strip()),
            }
        )
        if isinstance(payload, dict):
            result["usage"] = payload.get("usage", {})
    finally:
        stop = subprocess.run(
            [str(RUST_BINARY), "serve", "stop", "--port", str(port)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=case_env,
            check=False,
        )
        (case_dir / "stop.stdout.txt").write_bytes(stop.stdout)
        (case_dir / "stop.stderr.txt").write_bytes(stop.stderr)
        result["stop_returncode"] = stop.returncode

    return result


def _default_cases(include_qwen27: bool) -> list[SmokeCase]:
    cases = [
        SmokeCase(
            name="stepfun-text-with-mmproj",
            model=DEFAULT_STEPFUN_MODEL,
            mmproj=DEFAULT_STEPFUN_MMPROJ,
            prompt="Reply with the word text-ok.",
            image=None,
            ctx_size=2048,
            timeout_s=300,
        ),
        SmokeCase(
            name="stepfun-image-with-mmproj",
            model=DEFAULT_STEPFUN_MODEL,
            mmproj=DEFAULT_STEPFUN_MMPROJ,
            prompt="Describe the image in one short sentence.",
            image=DEFAULT_IMAGE,
            ctx_size=2048,
            timeout_s=300,
        ),
    ]
    if include_qwen27:
        cases.extend(
            [
                SmokeCase(
                    name="qwen27-image-bf16-mmproj",
                    model=DEFAULT_QWEN27_MODEL,
                    mmproj=DEFAULT_QWEN27_MMPROJ_BF16,
                    prompt="Describe the image in one short sentence.",
                    image=DEFAULT_IMAGE,
                    ctx_size=2048,
                    timeout_s=600,
                ),
                SmokeCase(
                    name="qwen27-image-f16-mmproj",
                    model=DEFAULT_QWEN27_MODEL,
                    mmproj=DEFAULT_QWEN27_MMPROJ_F16,
                    prompt="Describe the image in one short sentence.",
                    image=DEFAULT_IMAGE,
                    ctx_size=2048,
                    timeout_s=600,
                ),
            ]
        )
    return cases


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# VLM mmproj Smoke Matrix",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Binary: `{payload['binary']}`",
        f"- CUDA device env: `{payload['cuda_visible_devices']}`",
        "",
        "| Case | Result | Serve | Chat | Response |",
        "|---|---:|---:|---:|---|",
    ]
    for result in payload["results"]:
        lines.append(
            "| {name} | {ok} | {serve:.1f}s | {chat} | {text} |".format(
                name=result["name"],
                ok="ok" if result.get("ok") else "failed",
                serve=float(result.get("serve_wall_s", 0.0)),
                chat=(
                    f"{float(result['chat_wall_s']):.1f}s"
                    if "chat_wall_s" in result
                    else "-"
                ),
                text=str(result.get("response_text") or result.get("error") or "").replace("|", "\\|")[:160],
            )
        )
    lines.extend(["", "Raw JSON and per-case logs are stored next to this file.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real VLM/mmproj smoke matrix through the Rust serve path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cuda-device", default=os.environ.get("OMNIINFER_CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--include-qwen27", action="store_true", help="also run heavier Qwen3.6 27B BF16/F16 mmproj cases")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{_utc_stamp()}-vlm-mmproj-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMNIINFER_CUDA_VISIBLE_DEVICES"] = args.cuda_device

    results = []
    for case in _default_cases(args.include_qwen27):
        missing = [path for path in [case.model, case.mmproj, case.image] if path and not path.exists()]
        if missing:
            results.append(
                {
                    "name": case.name,
                    "model": str(case.model),
                    "mmproj": str(case.mmproj) if case.mmproj else "",
                    "image": str(case.image) if case.image else "",
                    "ok": False,
                    "error": "missing: " + ", ".join(str(path) for path in missing),
                }
            )
            continue
        print(f"Running {case.name}...")
        results.append(_run_case(case, output_dir=output_dir, env=env))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(RUST_BINARY),
        "cuda_visible_devices": args.cuda_device,
        "results": results,
    }
    (output_dir / "raw.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", payload)
    print(f"Wrote {output_dir / 'summary.md'}")
    return 0 if all(result.get("ok") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
