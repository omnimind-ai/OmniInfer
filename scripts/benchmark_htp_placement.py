"""Review actual llama-server scheduler evidence; never certify model correctness."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

_START = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*new prompt,.*task\.n_tokens\s*=\s*(\d+)")
_END = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*stop processing:")
_NODE = re.compile(r"node #\s*\d+\s*\(\s*(\w+)\s*\):\s*(.*?)\s*\([^\n]*?\)\s*\[\s*(\S+)\s*\]")


def review_placement(text: str) -> dict:
    """Single-slot diagnostic only. Reservation graphs before requests are ignored.

    A positive placement review is only eligibility for independent validation:
    it proves neither correctness nor coverage of unobserved shapes or phases.
    """
    requests = []
    active = None
    errors = []
    for line_number, line in enumerate(text.splitlines(), 1):
        start = _START.search(line)
        if start:
            if active is not None:
                errors.append(f"overlapping or incomplete request at line {line_number}")
            active = {"slot": start[1], "task": start[2], "prompt_tokens": int(start[3]),
                      "complete": False, "nodes": Counter(), "cpu_ops": Counter(),
                      "cpu_examples": [], "graph_starts": 0}
            requests.append(active)
        if active is None:
            continue
        if "node #" in line:
            node = _NODE.search(line)
            if node is None:
                errors.append(f"unrecognized scheduler node at line {line_number}")
            else:
                op, name, device = node.groups()
                active["nodes"][device] += 1
                if re.search(r"node #\s*0\s*\(", line):
                    active["graph_starts"] += 1
                if device == "CPU":
                    active["cpu_ops"][op] += 1
                    if len(active["cpu_examples"]) < 8:
                        active["cpu_examples"].append({"line": line_number, "op": op, "tensor": name})
        end = _END.search(line)
        if end:
            if (end[1], end[2]) != (active["slot"], active["task"]):
                errors.append(f"request identity mismatch at line {line_number}")
            else:
                active["complete"] = True
            active = None
    if active is not None:
        errors.append("incomplete request at end of log")
    for request in requests:
        if not request["nodes"]:
            errors.append(f"no actual scheduler nodes for task {request['task']}")
        # At least two scheduled graphs are needed to review PP and decode.
        # Shapes and graph reuse still require independent raw review.
        if request["graph_starts"] < 2:
            errors.append(f"insufficient graph coverage for task {request['task']}")
    devices = {device for request in requests for device in request["nodes"]}
    cpu = "CPU" in devices
    unknown = {device for device in devices if device != "CPU" and not re.fullmatch(r"HTP\d+", device)}
    if unknown:
        errors.append("unknown execution devices: " + ", ".join(sorted(unknown)))
    if not requests:
        errors.append("no actual request evidence; reservation graphs are insufficient")
    return {"schema_version": 1, "requests": requests, "errors": errors,
            "placement": "cpu_assisted" if cpu else ("unknown" if errors else "htp_observed"),
            "eligible_for_pure_htp_review": bool(requests) and not cpu and not errors,
            "correctness_verified": False, "performance_publishable": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    report = review_placement(raw.decode("utf-8", errors="strict"))
    report["log_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return 0 if report["eligible_for_pure_htp_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
