#!/usr/bin/env python3
"""Run the fixed 2-view GR00T N1.7 OmniInfer VLA Runtime latency benchmark."""

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
    parser.add_argument("--output", default=None)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--task", default="pick up the object")
    args = parser.parse_args()
    addr = args.addr or discover_addr()
    command = [
        sys.executable,
        str(SCRIPT),
        "--addr", addr,
        "--arch", "gr00t_n17",
        "--num-images", "2",
        "--image-size", "256",
        "--lang-len", "156",
        "--warmup", str(args.warmup),
        "--timed", str(args.timed),
    ]
    if args.output:
        command.extend(["--output", args.output])
    if args.native:
        command.extend(["--native", "--task", args.task])
    print("GR00T N1.7 OmniInfer VLA Runtime benchmark", flush=True)
    print("configuration: BF16 params, CUDA Graph, 2x256 image, 64 image tokens/view, 40x132 fixed noise", flush=True)
    print(f"endpoint: {addr}", flush=True)
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
