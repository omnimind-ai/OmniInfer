#!/usr/bin/env python3
"""Live OmniInfer + vla.cpp + LIBERO demonstration dashboard."""

from __future__ import annotations

import argparse
import hmac
import html
import io
import ipaddress
import json
import os
import re
import secrets
import shutil
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


VLA_PROTOCOL = "vla.cpp-zmq-server"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
LIBERO_OBJECT_TASKS = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
SUPPORTED_DEMO_ARCHES = ("smolvla", "pi05")
MODEL_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_RENDER_SIZE = 256
DEFAULT_DISPLAY_RENDER_SIZE = 512
MIN_RENDER_SIZE = 128
MAX_RENDER_SIZE = 1024
DEFAULT_FPS = 30
DISPLAY_JPEG_QUALITY = 92
DISPLAY_JPEG_SUBSAMPLING = 1
DEFAULT_PI05_ACTION_STEPS = 10
# The wrist view contains a fixed lower camera border that is not useful in a
# presentation.  This is applied only after policy inference, in the browser
# frame encoder.
DEFAULT_WRIST_DISPLAY_CROP_RATIO = 0.84
OMNIINFER_URL_REQUIREMENT = (
    "OmniInfer URL must be an HTTP loopback IP with an explicit nonzero port "
    "and no credentials, path, query, or fragment"
)


def default_output_dir() -> str:
    """Keep generated videos out of a shared source checkout by default."""
    state_root = os.environ.get("XDG_STATE_HOME")
    if not state_root:
        state_root = str(Path.home() / ".local" / "state")
    return str(Path(state_root) / "omniinfer" / "vla-libero-demo" / "outputs")


def validate_csrf_token(expected: str, received: str | None) -> None:
    """Reject browser mutations that did not originate from this dashboard page."""
    if not received or not hmac.compare_digest(expected, received):
        raise PermissionError("missing or invalid CSRF token")


def validate_dashboard_host(value: str | None) -> tuple[str, int]:
    """Accept only an explicit loopback Host and browser-visible port."""
    if not value:
        raise PermissionError("missing or invalid Host header")
    try:
        parsed = urllib.parse.urlsplit(f"//{value}")
        host = parsed.hostname
        request_port = parsed.port
    except ValueError as error:
        raise PermissionError("missing or invalid Host header") from error
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or host not in {"127.0.0.1", "localhost", "::1"}
        or request_port is None
    ):
        raise PermissionError("missing or invalid Host header")
    return host, request_port


def validate_dashboard_origin(value: str | None, host: str, port: int) -> None:
    """Require mutating browser requests to originate from the current origin."""
    if not value:
        raise PermissionError("missing or invalid Origin header")
    try:
        parsed = urllib.parse.urlsplit(value)
        origin_port = parsed.port
    except ValueError as error:
        raise PermissionError("missing or invalid Origin header") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != host
        or origin_port != port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise PermissionError("missing or invalid Origin header")


def validate_omniinfer_url(value: Any) -> str:
    """Return a canonical loopback gateway URL that cannot disclose admin keys."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OmniInfer URL must be a non-empty string")
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        address = ipaddress.ip_address(host) if host is not None else None
    except ValueError as error:
        raise ValueError(OMNIINFER_URL_REQUIREMENT) from error
    if (
        parsed.scheme != "http"
        or address is None
        or not address.is_loopback
        or port is None
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(OMNIINFER_URL_REQUIREMENT)
    if getattr(address, "scope_id", None) is not None:
        raise ValueError("OmniInfer URL must not contain an IPv6 scope identifier")
    canonical_host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{canonical_host}:{port}"


def validate_arch_options(
    arch: str, tokenizer: str | None, stats_json: str | None
) -> str | None:
    """Validate architecture-specific options before starting the dashboard."""
    if arch != "pi05":
        if stats_json is not None:
            raise ValueError("--stats-json is only supported with --arch pi05")
        return None
    if stats_json is not None:
        stats_path = Path(stats_json).expanduser()
        if not stats_path.is_file():
            raise ValueError(f"--stats-json must be an existing file; got {stats_json!r}")
        return str(stats_path)
    return None


def validate_libero_object_task_id(value: Any) -> int:
    """Return a task id only when it identifies a supported LIBERO object task."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("task_id must be an integer")
    if not 0 <= value < len(LIBERO_OBJECT_TASKS):
        raise ValueError(
            f"task_id must be between 0 and {len(LIBERO_OBJECT_TASKS) - 1}"
        )
    return value


def validate_render_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("render size must be an integer")
    if not MIN_RENDER_SIZE <= value <= MAX_RENDER_SIZE:
        raise ValueError(
            f"render size must be between {MIN_RENDER_SIZE} and {MAX_RENDER_SIZE}"
        )
    return value


def validate_wrist_display_crop_ratio(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("wrist display crop ratio must be a number")
    ratio = float(value)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("wrist display crop ratio must be greater than 0 and at most 1")
    return ratio


def aggregate_result(successes: int, failures: int) -> str:
    """Describe a completed multi-episode rollout without hiding mixed results."""
    if successes and failures:
        return "partial"
    if successes:
        return "success"
    return "failed"


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for a finite sample."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "last_ms": round(values[-1], 2) if values else None,
        "mean_ms": round(statistics.fmean(values), 2) if values else None,
        "p50_ms": round(percentile(values, 0.50), 2) if values else None,
        "p95_ms": round(percentile(values, 0.95), 2) if values else None,
    }


def validate_vla_runtime(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    """Validate OmniInfer's managed VLA protocol response and return its endpoint."""
    protocol = payload.get("external_server_protocol")
    endpoint = payload.get("client_endpoint")
    backend = payload.get("selected_backend") or payload.get("backend")
    model = payload.get("selected_model") or payload.get("model_path") or payload.get("model")

    if protocol != VLA_PROTOCOL:
        raise ValueError(
            f"OmniInfer runtime protocol is {protocol!r}; expected {VLA_PROTOCOL!r}."
        )
    if not isinstance(backend, str) or not backend.startswith("vla.cpp-"):
        raise ValueError(f"OmniInfer selected a non-VLA backend: {backend!r}.")
    if not isinstance(endpoint, str):
        raise ValueError("OmniInfer did not report a VLA client_endpoint.")

    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname not in LOOPBACK_HOSTS or parsed.port is None:
        raise ValueError(
            "VLA client_endpoint must be a loopback tcp:// endpoint reported by OmniInfer; "
            f"got {endpoint!r}."
        )
    return endpoint, backend, str(model) if model is not None else None


@dataclass(frozen=True)
class DemoConfig:
    omniinfer_url: str = "http://127.0.0.1:9000"
    backend: str = "vla.cpp-linux-cuda"
    model: str | None = None
    mmproj: str | None = None
    launch_args: tuple[str, ...] = ()
    admin_api_key: str | None = None
    protoc: str | None = None
    arch: str = "smolvla"
    tokenizer: str | None = None
    stats_json: str | None = None
    task: str = "libero_object"
    task_id: int = 0
    episodes: int = 1
    seed: int = 42
    fps: int = DEFAULT_FPS
    render_size: int = DEFAULT_RENDER_SIZE
    display_render_size: int = DEFAULT_DISPLAY_RENDER_SIZE
    wrist_display_crop_ratio: float = DEFAULT_WRIST_DISPLAY_CROP_RATIO
    output_dir: str = field(default_factory=default_output_dir)
    view_mode: str = "multi-view"
    n_action_steps: int = 1
    recv_timeout_ms: int = 120_000

    def model_load_payload(self) -> dict[str, Any] | None:
        if self.model is None:
            return None
        payload: dict[str, Any] = {
            "model": self.model,
            "backend": self.backend,
            "strict_capabilities": True,
        }
        if self.mmproj:
            payload["mmproj"] = self.mmproj
        if self.launch_args:
            payload["launch_args"] = list(self.launch_args)
        return payload


@dataclass(frozen=True)
class ModelProfile:
    identifier: str
    label: str
    config: DemoConfig

    def public(self) -> dict[str, str]:
        return {"id": self.identifier, "label": self.label, "arch": self.config.arch}


def _profile_path(value: Any, field_name: str, config_dir: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model profile {field_name!r} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _profile_tokenizer(value: Any, config_dir: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model profile 'tokenizer' must be a non-empty string")
    candidate = Path(value).expanduser()
    local_candidate = candidate if candidate.is_absolute() else config_dir / candidate
    explicitly_local = candidate.is_absolute() or value.startswith(("./", "../", "~/"))
    if local_candidate.exists():
        return str(local_candidate.resolve())
    if explicitly_local:
        raise ValueError(f"model profile tokenizer does not exist: {local_candidate}")
    return value


def load_model_profiles(path: str, base: DemoConfig) -> dict[str, ModelProfile]:
    """Load trusted server-side model choices without exposing paths to browsers."""
    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"--model-profiles file does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"--model-profiles must contain valid JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"models"}:
        raise ValueError("--model-profiles JSON must contain only a 'models' object")
    models = payload["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("--model-profiles 'models' must be a non-empty object")
    if len(models) > 32:
        raise ValueError("--model-profiles supports at most 32 models")

    allowed = {
        "label", "arch", "model", "backend", "mmproj", "server_args",
        "tokenizer", "stats_json", "n_action_steps", "omniinfer_url",
        "use_loaded_runtime",
    }
    profiles: dict[str, ModelProfile] = {}
    for identifier, entry in models.items():
        if not isinstance(identifier, str) or not MODEL_PROFILE_ID_RE.fullmatch(identifier):
            raise ValueError(f"invalid model profile id: {identifier!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"model profile {identifier!r} must be an object")
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                f"model profile {identifier!r} has unknown field(s): {sorted(unknown)}"
            )
        label = entry.get("label")
        arch = entry.get("arch")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"model profile {identifier!r} requires a non-empty label")
        if arch not in SUPPORTED_DEMO_ARCHES:
            raise ValueError(
                f"model profile {identifier!r} arch must be one of {SUPPORTED_DEMO_ARCHES}"
            )
        model_value = entry.get("model")
        use_loaded_runtime = entry.get("use_loaded_runtime", False)
        if not isinstance(use_loaded_runtime, bool):
            raise ValueError(
                f"model profile {identifier!r} use_loaded_runtime must be a boolean"
            )
        if (model_value is None) == (not use_loaded_runtime):
            raise ValueError(
                f"model profile {identifier!r} must specify exactly one of "
                "'model' or use_loaded_runtime=true"
            )
        model = (
            _profile_path(model_value, "model", config_path.parent)
            if model_value is not None
            else None
        )
        if model is not None and not Path(model).is_file():
            raise ValueError(f"model profile {identifier!r} model does not exist: {model}")
        mmproj = entry.get("mmproj")
        if mmproj is not None:
            mmproj = _profile_path(mmproj, "mmproj", config_path.parent)
            if not Path(mmproj).is_file():
                raise ValueError(f"model profile {identifier!r} mmproj does not exist: {mmproj}")
        server_args = entry.get("server_args", [])
        if not isinstance(server_args, list) or not all(
            isinstance(value, str) for value in server_args
        ):
            raise ValueError(f"model profile {identifier!r} server_args must be a string array")
        tokenizer = _profile_tokenizer(entry.get("tokenizer"), config_path.parent)
        stats_json = entry.get("stats_json")
        if stats_json is not None:
            stats_json = _profile_path(stats_json, "stats_json", config_path.parent)
        stats_json = validate_arch_options(arch, tokenizer, stats_json)
        n_action_steps = entry.get(
            "n_action_steps", DEFAULT_PI05_ACTION_STEPS if arch == "pi05" else 1
        )
        if isinstance(n_action_steps, bool) or not isinstance(n_action_steps, int) or n_action_steps < 1:
            raise ValueError(
                f"model profile {identifier!r} n_action_steps must be an integer >= 1"
            )
        backend = entry.get("backend", base.backend)
        if not isinstance(backend, str) or not backend.startswith("vla.cpp-"):
            raise ValueError(f"model profile {identifier!r} backend must be a vla.cpp backend")
        omniinfer_url = validate_omniinfer_url(
            entry.get("omniinfer_url", base.omniinfer_url)
        )
        profiles[identifier] = ModelProfile(
            identifier=identifier,
            label=label.strip(),
            config=replace(
                base,
                omniinfer_url=omniinfer_url,
                backend=backend,
                model=model,
                mmproj=mmproj,
                launch_args=tuple(server_args),
                arch=arch,
                tokenizer=tokenizer,
                stats_json=stats_json,
                n_action_steps=n_action_steps,
            ),
        )
    return profiles


def public_profile_error(error: Exception, config: DemoConfig) -> str:
    """Keep trusted profile paths out of dashboard state and browser responses."""
    message = f"{type(error).__name__}: {error}"
    for value in (config.model, config.mmproj, config.stats_json, config.tokenizer):
        if value and (Path(value).is_absolute() or os.path.sep in value):
            message = message.replace(value, "<model-profile-value>")
    return message


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep authorization headers on the explicitly configured loopback origin."""

    def redirect_request(self, request: Any, *args: Any, **kwargs: Any) -> None:
        return None


class OmniInferAPI:
    def __init__(self, base_url: str, admin_api_key: str | None = None):
        self.base_url = validate_omniinfer_url(base_url)
        self.admin_api_key = admin_api_key
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler()
        )

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with self._opener.open(request, timeout=450) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OmniInfer {path} returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot reach OmniInfer at {self.base_url}: {error}") from error

    def resolve_vla_runtime(self, config: DemoConfig) -> tuple[str, str, str | None]:
        load_payload = config.model_load_payload()
        if load_payload is not None:
            payload = self._request("/omni/model/load", load_payload)
        else:
            payload = self._request("/omni/state")
            if not payload.get("backend_ready"):
                raise RuntimeError(
                    "No managed runtime is ready. Pass --model or load a vla.cpp backend first."
                )
        return validate_vla_runtime(payload)


def configure_protoc(protoc: str | None) -> str:
    """Make the protoc required by vla.cpp's Python client explicit and discoverable."""
    if protoc:
        candidate = Path(protoc).expanduser().resolve()
        if not candidate.is_file() or shutil.which(str(candidate)) is None:
            raise ValueError(f"--protoc is not an executable file: {candidate}")
        os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    resolved = shutil.which("protoc")
    if resolved is None:
        raise RuntimeError(
            "vla.cpp's Python client requires protoc to generate its protobuf stub. "
            "Install protobuf-compiler or pass --protoc <path>."
        )
    return resolved


class DemoState:
    def __init__(
        self,
        config: DemoConfig,
        profiles: dict[str, ModelProfile] | None = None,
        default_profile: str = "command-line",
    ):
        if profiles is None:
            profiles = {
                default_profile: ModelProfile(default_profile, config.arch, config)
            }
        self._profiles = profiles
        selected = profiles[default_profile]
        self._lock = threading.Lock()
        self._frame = b""
        self._frame_times: deque[float] = deque(maxlen=120)
        self._next_run_id = 0
        self._last_frame_at = 0.0
        self._policy_ms: deque[float] = deque(maxlen=500)
        self._prediction_ms: deque[float] = deque(maxlen=500)
        self._env_ms: deque[float] = deque(maxlen=500)
        self._loop_ms: deque[float] = deque(maxlen=500)
        self._events: deque[dict[str, Any]] = deque(maxlen=80)
        self._data: dict[str, Any] = {
            "phase": "idle",
            "result": "idle",
            "message": "Ready to start",
            "task": config.task,
            "task_id": config.task_id,
            "task_description": LIBERO_OBJECT_TASKS[config.task_id],
            "task_options": [
                {"task_id": task_id, "instruction": instruction}
                for task_id, instruction in enumerate(LIBERO_OBJECT_TASKS)
            ],
            "model_profile": default_profile,
            "model_options": [profile.public() for profile in profiles.values()],
            "arch": selected.config.arch,
            "backend": selected.config.backend,
            "model": selected.label,
            "client_endpoint": None,
            "episode": 0,
            "episodes": config.episodes,
            "step": 0,
            "successes": 0,
            "failures": 0,
            "reward": 0.0,
            "action": [0.0] * len(ACTION_LABELS),
            "action_labels": ACTION_LABELS,
            "call_kind": None,
            "action_chunk_step": 0,
            "action_chunk_size": selected.config.n_action_steps,
            "run_id": 0,
            "frame_seq": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def begin(self, task_id: int, profile: ModelProfile | None = None) -> None:
        if profile is None:
            profile = self._profiles[str(self._data["model_profile"])]
        with self._lock:
            self._next_run_id += 1
            self._frame = b""
            self._frame_times.clear()
            self._last_frame_at = 0.0
            self._policy_ms.clear()
            self._prediction_ms.clear()
            self._env_ms.clear()
            self._loop_ms.clear()
            self._events.clear()
            self._data.update(
                phase="starting",
                result="running",
                message="Starting OmniInfer-managed VLA runtime",
                task_id=task_id,
                task_description=LIBERO_OBJECT_TASKS[task_id],
                model_profile=profile.identifier,
                model=profile.label,
                arch=profile.config.arch,
                backend=profile.config.backend,
                episodes=profile.config.episodes,
                client_endpoint=None,
                episode=0,
                step=0,
                successes=0,
                failures=0,
                reward=0.0,
                action=[0.0] * len(ACTION_LABELS),
                call_kind=None,
                action_chunk_step=0,
                action_chunk_size=profile.config.n_action_steps,
                run_id=self._next_run_id,
                frame_seq=0,
                started_at=time.time(),
                finished_at=None,
                error=None,
            )
            self._append_event_locked("info", "Demo run started")

    def update(self, **values: Any) -> None:
        with self._lock:
            self._data.update(values)

    def event(self, level: str, message: str) -> None:
        with self._lock:
            self._append_event_locked(level, message)

    def _append_event_locked(self, level: str, message: str) -> None:
        self._events.append({"time": time.time(), "level": level, "message": message})

    def publish_frame(
        self,
        observation: dict[str, Any] | Callable[[], dict[str, Any]],
        view_mode: str,
        *,
        wrist_display_crop_ratio: float = DEFAULT_WRIST_DISPLAY_CROP_RATIO,
        min_interval_seconds: float = 0.0,
        force: bool = False,
    ) -> bool:
        """Encode and publish only the latest display frame at the requested rate."""
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_frame_at < min_interval_seconds:
                return False
            run_id = int(self._data["run_id"])
        if callable(observation):
            observation = observation()
        frame = encode_frame(observation, view_mode, wrist_display_crop_ratio)
        with self._lock:
            if int(self._data["run_id"]) != run_id:
                return False
            self._frame = frame
            self._last_frame_at = now
            self._frame_times.append(now)
            self._data["frame_seq"] += 1
        return True

    def publish_step(
        self,
        *,
        action: list[float],
        policy_ms: float,
        prediction_sent: bool,
        env_ms: float,
        loop_ms: float,
        reward: float,
        step: int,
        action_chunk_step: int = 0,
    ) -> None:
        with self._lock:
            self._policy_ms.append(policy_ms)
            if prediction_sent:
                self._prediction_ms.append(policy_ms)
            self._env_ms.append(env_ms)
            self._loop_ms.append(loop_ms)
            self._data.update(
                action=[round(float(value), 5) for value in action],
                call_kind="model_prediction" if prediction_sent else "action_queue_replay",
                reward=round(float(reward), 5),
                step=step,
                action_chunk_step=action_chunk_step,
                message="Running LIBERO rollout",
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._data)
            payload["events"] = list(self._events)
            frame_times = list(self._frame_times)
            display_fps = None
            frame_intervals = [
                current - previous
                for previous, current in zip(frame_times, frame_times[1:])
                if current > previous
            ]
            if frame_intervals:
                display_fps = round(1.0 / statistics.median(frame_intervals), 1)
            payload["telemetry"] = {
                "display_fps": display_fps,
                "prediction_count": len(self._prediction_ms),
                "action_chunk_step": payload["action_chunk_step"],
                "action_chunk_size": payload["action_chunk_size"],
            }
            payload["latency"] = {
                "policy": metric_summary(list(self._policy_ms)),
                "prediction": metric_summary(list(self._prediction_ms)),
                "environment": metric_summary(list(self._env_ms)),
                "control_loop": metric_summary(list(self._loop_ms)),
                "history_ms": [round(value, 2) for value in list(self._loop_ms)[-80:]],
            }
            return payload

    def frame(self) -> tuple[bytes, int, int]:
        with self._lock:
            return (
                self._frame,
                int(self._data["run_id"]),
                int(self._data["frame_seq"]),
            )


def encode_frame(
    observation: dict[str, Any],
    view_mode: str,
    wrist_display_crop_ratio: float = DEFAULT_WRIST_DISPLAY_CROP_RATIO,
) -> bytes:
    import numpy as np
    from PIL import Image

    front = np.asarray(observation["pixels"]["image"][::-1, ::-1], dtype=np.uint8)
    images = [front]
    if view_mode == "multi-view" and "image2" in observation["pixels"]:
        wrist = np.asarray(observation["pixels"]["image2"][::-1, ::-1], dtype=np.uint8)
        wrist = wrist[
            : max(1, round(wrist.shape[0] * wrist_display_crop_ratio))
        ]
        wrist = np.asarray(
            Image.fromarray(wrist).resize((front.shape[1], front.shape[0]))
        )
        images.append(wrist)
    composed = np.concatenate(images, axis=1)
    output = io.BytesIO()
    Image.fromarray(composed).save(
        output,
        format="JPEG",
        quality=DISPLAY_JPEG_QUALITY,
        subsampling=DISPLAY_JPEG_SUBSAMPLING,
        optimize=False,
    )
    return output.getvalue()


def display_observation(
    environment: Any,
    observation: dict[str, Any],
    *,
    policy_render_size: int,
    display_render_size: int,
) -> dict[str, Any]:
    """Render a browser-only view without changing the policy observation.

    LIBERO's registered observation size feeds the VLA client.  The dashboard
    can sample the same simulator state at a different size, but must retain
    the original observation for action prediction.
    """
    if display_render_size == policy_render_size:
        return observation
    try:
        libero_environment = environment.unwrapped
        simulator = libero_environment._env.env.sim
        pixels = {
            libero_environment.camera_name_mapping[observation_camera_name]: simulator.render(
                width=display_render_size,
                height=display_render_size,
                camera_name=observation_camera_name.removesuffix("_image"),
            )
            for observation_camera_name in libero_environment.camera_name
        }
    except (AttributeError, KeyError, TypeError) as error:
        raise RuntimeError(
            "LIBERO environment does not support independent dashboard rendering"
        ) from error
    return {**observation, "pixels": pixels}


class _DemoPolicyAdapter:
    """Keep demo-side LIBERO conversion independent from LeRobot's full package."""

    def __init__(self, client: Any):
        self._client = client

    def reset(self) -> None:
        self._client.reset()

    def get_action(self, observation: dict[str, Any]) -> Any:
        return self._client.get_action(self.parse_observation(observation))

    def parse_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _quat_to_axis_angle(quat: Any) -> Any:
        import numpy as np

        quat = np.asarray(quat, dtype=np.float32).reshape(4)
        w = float(np.clip(quat[3], -1.0, 1.0))
        denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
        if denominator <= 1e-10:
            return np.zeros(3, dtype=np.float32)
        return (quat[:3] * (2.0 * np.arccos(w) / denominator)).astype(np.float32)


class _LiberoPolicyAdapter(_DemoPolicyAdapter):
    """Equivalent LIBERO image/state conversion for vla.cpp's direct client."""

    def parse_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        images = observation["pixels"]

        def image(key: str) -> Any:
            # vla.cpp's official LiberoProcessorStep flips both axes after
            # converting HWC uint8 to CHW float32 in [0, 1].
            value = np.asarray(images[key], dtype=np.float32)
            value = value[::-1, ::-1].transpose(2, 0, 1) / 255.0
            return np.ascontiguousarray(value, dtype=np.float32)

        robot_state = observation["robot_state"]
        state = np.concatenate(
            (
                np.asarray(robot_state["eef"]["pos"], dtype=np.float32),
                self._quat_to_axis_angle(robot_state["eef"]["quat"]),
                np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32),
            )
        )
        return {
            "observation.images.image": image("image"),
            "observation.images.image2": image("image2"),
            "observation.state": np.ascontiguousarray(state, dtype=np.float32),
            "task": observation.get("task_description", ""),
        }


def create_policy(config: DemoConfig, endpoint: str) -> tuple[Any, Any]:
    from client.vla_cpp_client import VlaCppClient

    raw_client = VlaCppClient(
        vla_addr=endpoint,
        arch=config.arch,
        tokenizer_name=config.tokenizer,
        recv_timeout_ms=config.recv_timeout_ms,
        n_action_steps=config.n_action_steps,
        stats_json=config.stats_json,
    )
    return raw_client, _LiberoPolicyAdapter(client=raw_client)


class DemoController:
    def __init__(
        self,
        config: DemoConfig,
        state: DemoState,
        repository_root: Path,
        profiles: dict[str, ModelProfile] | None = None,
        default_profile: str = "command-line",
    ):
        self.config = config
        self.profiles = profiles or {
            default_profile: ModelProfile(default_profile, config.arch, config)
        }
        self.default_profile = default_profile
        self.state = state
        self.repository_root = repository_root
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self, task_id: int | None = None, model_profile: str | None = None
    ) -> bool:
        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return False
            if task_id is None:
                task_id = int(self.state.snapshot()["task_id"])
            task_id = validate_libero_object_task_id(task_id)
            if model_profile is None:
                model_profile = str(self.state.snapshot()["model_profile"])
            if model_profile not in self.profiles:
                raise ValueError(f"unknown model_profile: {model_profile!r}")
            profile = self.profiles[model_profile]
            self._stop.clear()
            self.state.begin(task_id, profile)
            self._thread = threading.Thread(
                target=self._run_guarded, args=(task_id, profile), daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._guard:
            running = self._thread is not None and self._thread.is_alive()
            if running:
                self._stop.set()
                self.state.update(message="Stopping after the current simulator step")
            return running

    def _run_guarded(self, task_id: int, profile: ModelProfile) -> None:
        try:
            self._run(task_id, profile)
        except Exception as error:  # dashboard must preserve the failure for inspection
            public_error = public_profile_error(error, profile.config)
            self.state.event("error", public_error)
            self.state.update(
                phase="error",
                result="error",
                message="Demo failed",
                error=public_error,
                finished_at=time.time(),
            )
            traceback.print_exc()
        finally:
            with self._guard:
                self._thread = None

    def _run(self, task_id: int, profile: ModelProfile) -> None:
        run_config = profile.config
        eval_root = self.repository_root / "framework" / "vla.cpp" / "eval"
        if not (eval_root / "client" / "vla_cpp_client.py").is_file():
            raise RuntimeError(
                "framework/vla.cpp is not initialized; run git submodule update --init framework/vla.cpp"
            )
        sys.path.insert(0, str(eval_root))

        api = OmniInferAPI(run_config.omniinfer_url, run_config.admin_api_key)
        endpoint, backend, _model_path = api.resolve_vla_runtime(run_config)
        self.state.update(
            phase="initializing",
            message="Initializing VLA client and LIBERO simulator",
            backend=backend,
            model=profile.label,
            client_endpoint="managed loopback",
        )
        self.state.event("info", f"OmniInfer runtime ready: {backend}")

        import gymnasium as gym
        import sim.libero  # noqa: F401 - registers the LIBERO environments

        raw_client = None
        environment = None
        try:
            protoc = configure_protoc(run_config.protoc)
            self.state.event("info", "Protobuf client ready")
            raw_client, policy = create_policy(run_config, endpoint)
            output_dir = Path(run_config.output_dir).resolve()
            run_dir = output_dir / (
                f"run-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}-task-{task_id}"
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            # The dashboard is presentation-facing: do not expose a machine-local
            # output location in its event stream.
            self.state.event("info", "Writing rollout video")
            environment = gym.make(
                f"{run_config.task}/task_{task_id}",
                seed=run_config.seed,
                video_fps=run_config.fps,
                output_video_dir=run_dir,
                video_view_mode=run_config.view_mode,
                observation_width=run_config.render_size,
                observation_height=run_config.render_size,
            )
            self.state.update(phase="running", message="Running LIBERO rollout")

            successes = 0
            failures = 0
            for episode_index in range(run_config.episodes):
                if self._stop.is_set():
                    break
                policy.reset()
                observation, _ = environment.reset()
                frame_interval_seconds = 1.0 / run_config.fps
                self.state.publish_frame(
                    lambda: display_observation(
                        environment,
                        observation,
                        policy_render_size=run_config.render_size,
                        display_render_size=run_config.display_render_size,
                    ),
                    run_config.view_mode,
                    wrist_display_crop_ratio=run_config.wrist_display_crop_ratio,
                    force=True,
                )
                self.state.update(
                    phase="running",
                    result="running",
                    message="Running LIBERO rollout",
                    episode=episode_index + 1,
                    step=0,
                    task_description=observation.get("task_description", ""),
                    reward=0.0,
                    error=None,
                )
                self.state.event(
                    "info", f"Episode {episode_index + 1}/{run_config.episodes} started"
                )

                actions_until_prediction = 0
                episode_finished = False
                step = 0
                while not self._stop.is_set():
                    prediction_sent = actions_until_prediction == 0
                    loop_start = time.perf_counter()
                    policy_start = loop_start
                    action = policy.get_action(observation)
                    policy_ms = (time.perf_counter() - policy_start) * 1000.0
                    if prediction_sent:
                        actions_until_prediction = run_config.n_action_steps - 1
                    else:
                        actions_until_prediction -= 1
                    action_chunk_step = run_config.n_action_steps - actions_until_prediction

                    env_start = time.perf_counter()
                    try:
                        observation, reward, terminated, truncated, info = environment.step(action)
                    except ValueError as error:
                        if "terminated episode" not in str(error):
                            raise
                        failures += 1
                        self.state.event("error", f"Episode aborted by LIBERO: {error}")
                        self.state.update(
                            result="failed",
                            message="LIBERO aborted the episode",
                            failures=failures,
                            error=str(error),
                        )
                        episode_finished = True
                        break
                    env_ms = (time.perf_counter() - env_start) * 1000.0
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    step += 1
                    self.state.publish_frame(
                        lambda: display_observation(
                            environment,
                            observation,
                            policy_render_size=run_config.render_size,
                            display_render_size=run_config.display_render_size,
                        ),
                        run_config.view_mode,
                        wrist_display_crop_ratio=run_config.wrist_display_crop_ratio,
                        min_interval_seconds=frame_interval_seconds,
                        force=terminated or truncated,
                    )
                    self.state.publish_step(
                        action=list(action[: len(ACTION_LABELS)]),
                        policy_ms=policy_ms,
                        prediction_sent=prediction_sent,
                        env_ms=env_ms,
                        loop_ms=loop_ms,
                        reward=reward,
                        step=step,
                        action_chunk_step=action_chunk_step,
                    )

                    if terminated or truncated:
                        success = bool(info.get("is_success", False))
                        if success:
                            successes += 1
                            result = "success"
                            message = "Task completed successfully"
                        else:
                            failures += 1
                            result = "failed"
                            message = "Episode ended without task success"
                        self.state.update(
                            result=result,
                            message=message,
                            successes=successes,
                            failures=failures,
                        )
                        self.state.event(
                            "success" if success else "warning",
                            f"Episode {episode_index + 1}: {result} after {step} steps",
                        )
                        episode_finished = True
                        break

                if self._stop.is_set():
                    break
                if not episode_finished:
                    failures += 1
                    self.state.update(failures=failures, result="failed")

            if self._stop.is_set():
                self.state.update(
                    phase="stopped",
                    result="stopped",
                    message="Demo stopped by user",
                    finished_at=time.time(),
                )
                self.state.event("warning", "Demo stopped by user")
            else:
                result = aggregate_result(successes, failures)
                self.state.update(
                    phase="completed",
                    result=result,
                    message=(
                        f"Completed {run_config.episodes} episode(s): "
                        f"{successes} success, {failures} failure"
                    ),
                    successes=successes,
                    failures=failures,
                    finished_at=time.time(),
                )
                self.state.event(
                    "info", f"Run complete: {successes} success, {failures} failure"
                )
        finally:
            if environment is not None:
                environment.close()
            if raw_client is not None:
                raw_client.sock.close(0)


class DashboardHandler(BaseHTTPRequestHandler):
    controller: DemoController
    state: DemoState
    index_path: Path
    csrf_token: str

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("request Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 0 or length > 4096:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    @staticmethod
    def _require_task_id(payload: dict[str, Any]) -> int:
        if "task_id" not in payload:
            raise ValueError("request body must include task_id")
        return validate_libero_object_task_id(payload["task_id"])

    @staticmethod
    def _require_model_profile(payload: dict[str, Any]) -> str:
        value = payload.get("model_profile")
        if not isinstance(value, str) or not value:
            raise ValueError("request body must include model_profile")
        return value

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        try:
            validate_dashboard_host(self.headers.get("Host"))
        except PermissionError as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.FORBIDDEN)
            return
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            page = self.index_path.read_text(encoding="utf-8")
            body = page.replace(
                "{{CSRF_TOKEN}}", html.escape(self.csrf_token, quote=True)
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            self._send_json(self.state.snapshot())
            return
        if path == "/api/frame.jpg":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                requested_run = int(query.get("run", ["-1"])[0])
                after_seq = int(query.get("after", ["-1"])[0])
            except ValueError:
                self._send_json(
                    {"error": "frame query parameters must be integers"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            body, run_id, frame_seq = self.state.frame()
            if requested_run != run_id or after_seq >= frame_seq or not body:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("X-OmniInfer-Run-Id", str(run_id))
                self.send_header("X-OmniInfer-Frame-Seq", str(frame_seq))
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-OmniInfer-Run-Id", str(run_id))
            self.send_header("X-OmniInfer-Frame-Seq", str(frame_seq))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        path = urllib.parse.urlsplit(self.path).path
        try:
            host, browser_port = validate_dashboard_host(self.headers.get("Host"))
            validate_csrf_token(
                self.csrf_token, self.headers.get("X-OmniInfer-CSRF-Token")
            )
            validate_dashboard_origin(
                self.headers.get("Origin"), host, browser_port
            )
        except PermissionError as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.FORBIDDEN)
            return
        if path == "/api/start":
            try:
                payload = self._read_json()
                unexpected = set(payload) - {"task_id", "model_profile"}
                if unexpected:
                    raise ValueError(
                        f"request body has unexpected field(s): {sorted(unexpected)}"
                    )
                task_id = self._require_task_id(payload)
                model_profile = self._require_model_profile(payload)
                started = self.controller.start(task_id, model_profile)
            except ValueError as error:
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            status = HTTPStatus.OK if started else HTTPStatus.CONFLICT
            response: dict[str, Any] = {"ok": started, "state": self.state.snapshot()}
            if not started:
                response["error"] = "a demo rollout is already running"
            self._send_json(response, status)
            return
        if path == "/api/stop":
            stopping = self.controller.stop()
            self._send_json({"ok": stopping, "state": self.state.snapshot()})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omniinfer-url", default="http://127.0.0.1:9000")
    parser.add_argument("--backend", default="vla.cpp-linux-cuda")
    parser.add_argument("--model", help="VLA checkpoint path; omit to use the loaded runtime")
    parser.add_argument(
        "--model-profiles",
        help="Trusted server-side JSON file defining models selectable in the dashboard",
    )
    parser.add_argument("--mmproj")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="vla-server launch arg; repeat and use --server-arg=--flag for leading dashes",
    )
    parser.add_argument(
        "--admin-api-key-env",
        default="OMNIINFER_ADMIN_API_KEY",
        help="Environment variable containing an optional OmniInfer admin API key",
    )
    parser.add_argument(
        "--protoc",
        help="Path to protoc when protobuf-compiler is not available on PATH",
    )
    parser.add_argument("--arch", choices=SUPPORTED_DEMO_ARCHES, default="smolvla")
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer Hugging Face id or local checkpoint directory override",
    )
    parser.add_argument(
        "--stats-json",
        help=(
            "PI0.5 LIBERO meta/stats.json; when omitted, the vla.cpp client "
            "uses its official lerobot/libero download path"
        ),
    )
    parser.add_argument("--task", choices=["libero_object"], default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--render-size",
        type=int,
        default=DEFAULT_RENDER_SIZE,
        help=(
            "LIBERO policy camera width and height in pixels "
            f"(default: {DEFAULT_RENDER_SIZE}; range: {MIN_RENDER_SIZE}-{MAX_RENDER_SIZE})"
        ),
    )
    parser.add_argument(
        "--display-render-size",
        type=int,
        default=DEFAULT_DISPLAY_RENDER_SIZE,
        help=(
            "browser-only LIBERO camera width and height in pixels "
            f"(default: {DEFAULT_DISPLAY_RENDER_SIZE}; range: {MIN_RENDER_SIZE}-{MAX_RENDER_SIZE})"
        ),
    )
    parser.add_argument(
        "--wrist-display-crop-ratio",
        type=float,
        default=DEFAULT_WRIST_DISPLAY_CROP_RATIO,
        help="fraction of the wrist image height retained for dashboard display",
    )
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--view-mode", choices=["single-view", "multi-view"], default="multi-view")
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument(
        "--listen-port",
        type=int,
        default=0,
        help="Dashboard port (default: 0, ask the OS for an unused port)",
    )
    parser.add_argument(
        "--auto-start", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.model_profiles and (args.server_arg or any(
        value is not None
        for value in (args.model, args.mmproj, args.tokenizer, args.stats_json)
    )):
        parser.error(
            "--model-profiles cannot be combined with --model, --mmproj, "
            "--server-arg, --tokenizer, or --stats-json"
        )
    try:
        args.omniinfer_url = validate_omniinfer_url(args.omniinfer_url)
        validate_libero_object_task_id(args.task_id)
        if args.model_profiles is None:
            args.stats_json = validate_arch_options(
                args.arch, args.tokenizer, args.stats_json
            )
    except ValueError as error:
        parser.error(str(error))
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    try:
        args.render_size = validate_render_size(args.render_size)
        args.display_render_size = validate_render_size(args.display_render_size)
        args.wrist_display_crop_ratio = validate_wrist_display_crop_ratio(
            args.wrist_display_crop_ratio
        )
    except ValueError as error:
        parser.error(str(error))
    if args.n_action_steps is None:
        args.n_action_steps = DEFAULT_PI05_ACTION_STEPS if args.arch == "pi05" else 1
    if args.n_action_steps < 1:
        parser.error("--n-action-steps must be >= 1")
    if not (0 <= args.listen_port <= 65535):
        parser.error("--listen-port must be between 0 and 65535")
    if args.listen_host not in LOOPBACK_HOSTS:
        parser.error("--listen-host must be a loopback address; use SSH port forwarding for remote access")
    return args


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    config = DemoConfig(
        omniinfer_url=args.omniinfer_url,
        backend=args.backend,
        model=args.model,
        mmproj=args.mmproj,
        launch_args=tuple(args.server_arg),
        admin_api_key=os.environ.get(args.admin_api_key_env),
        protoc=args.protoc,
        arch=args.arch,
        tokenizer=args.tokenizer,
        stats_json=args.stats_json,
        task=args.task,
        task_id=args.task_id,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        render_size=args.render_size,
        display_render_size=args.display_render_size,
        wrist_display_crop_ratio=args.wrist_display_crop_ratio,
        output_dir=args.output_dir,
        view_mode=args.view_mode,
        n_action_steps=args.n_action_steps,
        recv_timeout_ms=args.recv_timeout_ms,
    )
    if args.model_profiles:
        try:
            profiles = load_model_profiles(args.model_profiles, config)
        except ValueError as error:
            raise SystemExit(f"Invalid --model-profiles: {error}") from error
        default_profile = next(iter(profiles))
    else:
        default_profile = "command-line"
        profiles = {
            default_profile: ModelProfile(default_profile, config.arch, config)
        }
    state = DemoState(config, profiles, default_profile)
    controller = DemoController(
        config, state, repository_root, profiles, default_profile
    )
    DashboardHandler.controller = controller
    DashboardHandler.state = state
    DashboardHandler.index_path = Path(__file__).with_name("index.html")
    DashboardHandler.csrf_token = secrets.token_urlsafe(32)

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), DashboardHandler)
    actual_port = int(server.server_address[1])
    print(f"VLA LIBERO demo: http://{args.listen_host}:{actual_port}", flush=True)
    if args.listen_host in LOOPBACK_HOSTS:
        print(
            f"Remote host: ssh -L {actual_port}:127.0.0.1:{actual_port} <host>",
            flush=True,
        )
    if args.auto_start:
        controller.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
