"""Prospective environment evidence for external benchmark campaign runners.

This module does not change device settings or turn diagnostic samples into
benchmark results. Callers retain every observation and exclude failed cohorts.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from typing import Any


def android_display_snapshot(shell: Callable[[str], str]) -> dict[str, Any]:
    """Read, timestamp and retain display evidence without issuing input events."""
    started = time.time()
    raw = shell(
        "settings get system screen_off_timeout; "
        "settings get system screen_brightness; "
        "settings get system screen_brightness_mode; dumpsys power"
    )
    lines = raw.splitlines()
    if len(lines) < 3 or not all(re.fullmatch(r"\d+", line.strip()) for line in lines[:3]):
        raise ValueError("missing or invalid Android display settings")
    wakefulness = re.search(r"\bmWakefulness=(\w+)", raw)
    if wakefulness is None:
        raise ValueError("missing Android wakefulness evidence")
    return {
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "screen_off_timeout": int(lines[0].strip()),
        "screen_brightness": int(lines[1].strip()),
        "screen_brightness_mode": int(lines[2].strip()),
        "wakefulness": wakefulness.group(1),
        "raw": raw,
    }


def counter_interval_relation(
    start: float | None, end: float, window_start: float, window_end: float
) -> str:
    """Classify an interval-valued counter, never an instantaneous timestamp."""
    values = (end, window_start, window_end)
    if not all(math.isfinite(value) for value in values) or window_end < window_start:
        raise ValueError("invalid measurement window or counter timestamp")
    if start is None:
        # Even with an unknown beginning, an interval that already ended
        # cannot overlap a future protected window.
        if end <= window_start:
            return "outside"
        return "unknown_interval"
    if not math.isfinite(start) or start >= end:
        raise ValueError("counter interval must have finite, increasing boundaries")
    if end <= window_start or start >= window_end:
        return "outside"
    if start >= window_start and end <= window_end:
        return "inside"
    return "boundary_ambiguous"


def foreign_gpu_activity(
    *, start: float | None, end: float, window_start: float, window_end: float,
    samples: list[Mapping[str, Any]], owned_pids: set[int], threshold: float = 5.0,
) -> dict[str, Any]:
    """Return prospective validity and attribution without dropping ambiguity.

    PID 4 has no special exemption. Unknown/straddling intervals with foreign
    activity fail the gate, but do not prove interference inside the window.
    Missing/invalid counters also fail closed, even when utilization is zero.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid utilization threshold")
    relation = counter_interval_relation(start, end, window_start, window_end)
    conflicts = []
    invalid = []
    for sample in samples:
        try:
            pid = int(sample["pid"])
            value = float(sample["utilization"])
            if int(sample["status"]) != 0 or not math.isfinite(value) or value < 0:
                raise ValueError("invalid counter")
        except (KeyError, ValueError, TypeError, OverflowError):
            invalid.append(dict(sample))
            continue
        if pid not in owned_pids and value > threshold:
            conflicts.append(dict(sample))
    relevant = relation != "outside"
    return {
        "relation": relation,
        "valid": not (relevant and (conflicts or invalid or not samples)),
        "proven_foreign_activity_in_window": relation == "inside" and bool(conflicts),
        "conflicts": conflicts,
        "invalid_samples": invalid,
    }


def display_mismatches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Check declared display state; automatic brightness is expected to vary."""
    keys = ["screen_off_timeout", "screen_brightness_mode", "wakefulness"]
    if str(expected.get("screen_brightness_mode")) == "0":
        keys.append("screen_brightness")
    return [key for key in keys if key not in actual or key not in expected
            or str(actual[key]) != str(expected[key])]


def wait_for_prelaunch(
    observe: Callable[[], Mapping[str, Any]],
    verify_ownership: Callable[[], None],
    record: Callable[[Mapping[str, Any]], None],
    *, max_temperature_c: float = 33.0, timeout_seconds: float = 1800,
    poll_seconds: float = 30, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """After asset hashing, wait boundedly and recheck ownership before launch.

    observe returns a timestamped sample with temperature_c and optional other
    raw evidence. verify_ownership must check both the lease and foreign jobs.
    A timeout is a preparation failure; no inference has started here.
    """
    if (not all(math.isfinite(x) for x in (max_temperature_c, timeout_seconds, poll_seconds))
            or timeout_seconds < 0 or poll_seconds <= 0):
        raise ValueError("invalid prelaunch wait limits")
    deadline = clock() + timeout_seconds
    while True:
        verify_ownership()
        sample = observe()
        record(sample)
        temperature = float(sample["temperature_c"])
        if not math.isfinite(temperature):
            raise ValueError("invalid temperature observation")
        if temperature <= max_temperature_c:
            verify_ownership()
            return sample
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("prelaunch cooldown deadline exceeded; inference not started")
        sleep(min(poll_seconds, remaining))


def verify_display(
    observe: Callable[[], Mapping[str, Any]], expected: Mapping[str, Any],
    record: Callable[[Mapping[str, Any]], None], *, timeout_seconds: float = 5,
    poll_seconds: float = 0.2, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Poll asynchronous wake/sleep transitions without overwriting settings.

    A fixed-setting mismatch fails immediately. Only wakefulness may settle;
    every intermediate sample remains in the caller's append-only journal.
    """
    if (not all(math.isfinite(x) for x in (timeout_seconds, poll_seconds))
            or timeout_seconds < 0 or poll_seconds <= 0):
        raise ValueError("invalid display verification limits")
    deadline = clock() + timeout_seconds
    while True:
        sample = observe()
        record(sample)
        mismatch = display_mismatches(sample, expected)
        if not mismatch:
            return sample
        if mismatch != ["wakefulness"]:
            raise RuntimeError("display configuration drift: " + ", ".join(mismatch))
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("display wakefulness transition did not settle")
        sleep(min(poll_seconds, remaining))
