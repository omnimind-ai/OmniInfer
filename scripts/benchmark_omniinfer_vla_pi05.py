#!/usr/bin/env python3
"""Run the Pi0.5 OmniInfer VLA Runtime latency benchmark."""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("benchmark_omniinfer_vla_zmq.py")
LOG_DIR = Path.home() / "OmniInfer/.local/runtime/linux/omniinfer-vla-linux-cuda/logs"


def discover_addr() -> str:
    pattern = re.compile(r"omniinfer-vla-server: bound to (tcp://[^ ]+)\. ready\.")
    candidates: list[str] = []
    for path in sorted(LOG_DIR.glob("omniinfer-vla-server-*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            matches = pattern.findall(path.read_text(errors="ignore"))
        except OSError:
            continue
        candidates.extend(reversed(matches))
    for addr in candidates:
        host, port = addr.removeprefix("tcp://").rsplit(":", 1)
        try:
            with socket.create_connection((host, int(port)), timeout=0.3):
                return addr
        except OSError:
            continue
    raise SystemExit("No active OmniInfer VLA Runtime endpoint found; pass --addr tcp://127.0.0.1:<port>")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timed", type=int, default=3)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--lang-len", type=int, default=48, help="Prepared-input token count; native mode uses the tokenizer instead.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-action", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--pruning", action="store_true", help="Label the launcher's one-step warm-start configuration.")
    parser.add_argument("--task", default="pick up the cup from table and place it in the bowl carefully")
    args = parser.parse_args()
    addr = args.addr or discover_addr()
    command = [
        sys.executable,
        str(SCRIPT),
        "--addr", addr,
        "--arch", "pi05",
        "--num-images", str(args.num_images),
        "--image-size", "224",
        "--lang-len", str(args.lang_len),
        "--warmup", str(args.warmup),
        "--timed", str(args.timed),
    ]
    if args.output:
        command.extend(["--output", args.output])
    if args.save_action:
        command.append("--save-action")
    if args.native:
        command.extend(["--native", "--task", args.task])
    if args.pruning:
        command.append("--pruning")
    print("Pi0.5 OmniInfer VLA Runtime benchmark", flush=True)
    flow = "1-step warm-start (lossy), t=0.5, alpha=0.5, replan=5" if args.pruning else "10-step fixed noise"
    token_label = "native tokenizer" if args.native else f"{args.lang_len} prepared tokens"
    print(f"request: {args.num_images} images, {token_label}, {flow}; params/vision dtype set by server (README launcher: BF16/BF16)", flush=True)
    print(f"endpoint: {addr}", flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
