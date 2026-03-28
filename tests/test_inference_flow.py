#!/usr/bin/env python3

from __future__ import annotations

import argparse
import atexit
import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "tests" / "pictures" / "test1.png"


def log(message: str) -> None:
    print(f"[omniinfer-flow] {message}")


def fail(message: str) -> None:
    raise SystemExit(f"[omniinfer-flow] ERROR: {message}")


def detect_system_name() -> str:
    system = platform.system().lower()
    if system.startswith("linux"):
        return "linux"
    if system.startswith("darwin") or system.startswith("mac"):
        return "mac"
    if system.startswith("win"):
        return "windows"
    fail(f"unable to auto-detect supported-model system for {platform.system()}")
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the OmniInfer end-to-end inference flow")
    parser.add_argument("--model", required=True, help="GGUF model path used for model-select and direct inference tests")
    parser.add_argument("--mmproj", help="Optional mmproj path; when provided the script also runs multimodal inference")
    parser.add_argument("--backend", help="Explicit backend to select before loading the model")
    parser.add_argument("--host", default="127.0.0.1", help="Gateway host, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=9000, help="Gateway port, default: 9000")
    parser.add_argument(
        "--system",
        choices=("linux", "mac", "windows"),
        default=detect_system_name(),
        help="Catalog system name, default: auto-detect",
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help=f"Image used for multimodal inference, default: {DEFAULT_IMAGE}")
    parser.add_argument("--start-timeout", type=int, default=120, help="Seconds to wait for the gateway to become healthy")
    parser.add_argument("--reuse-running", action="store_true", help="Reuse an already running gateway on the target host:port")
    parser.add_argument("--keep-server", action="store_true", help="Do not shut down the gateway if this script launched it")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep temp logs and response files even when the script succeeds")
    return parser.parse_args()


class FlowRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = Path(args.model).expanduser().resolve()
        self.mmproj = Path(args.mmproj).expanduser().resolve() if args.mmproj else None
        self.image = Path(args.image).expanduser().resolve()
        self.base_url = f"http://{args.host}:{args.port}"
        self.work_dir = Path(tempfile.mkdtemp(prefix="omniinfer-flow."))
        self.gateway_log = self.work_dir / "gateway.log"
        self.gateway_proc: Optional[subprocess.Popen[str]] = None
        self.gateway_log_handle: Optional[Any] = None
        self.own_gateway = False
        self.success = False
        atexit.register(self.cleanup)

    def cleanup(self) -> None:
        if self.own_gateway and not self.args.keep_server:
            try:
                self.request_json("POST", "/omni/shutdown", allow_error=True)
            except BaseException:
                pass
            if self.gateway_proc is not None:
                try:
                    self.gateway_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.gateway_proc.kill()
                    self.gateway_proc.wait(timeout=5)
        if self.gateway_log_handle is not None:
            self.gateway_log_handle.close()
            self.gateway_log_handle = None

        if self.success and not self.args.keep_artifacts:
            shutil.rmtree(self.work_dir, ignore_errors=True)
        else:
            log(f"Artifacts kept at {self.work_dir}")

    def ensure_inputs(self) -> None:
        if not self.model.is_file():
            fail(f"model file not found: {self.model}")
        if self.mmproj and not self.mmproj.is_file():
            fail(f"mmproj file not found: {self.mmproj}")
        if self.mmproj and not self.image.is_file():
            fail(f"image file not found: {self.image}")

    def request_json(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        output_name: Optional[str] = None,
        allow_error: bool = False,
        timeout: float = 600.0,
    ) -> Tuple[int, Any, bytes]:
        url = f"{self.base_url}{endpoint}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request = urllib.request.Request(url=url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                status = resp.getcode()
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
            if not allow_error:
                self._write_artifact(output_name, raw)
                fail(f"HTTP {method} {endpoint} failed with status {status}: {raw.decode('utf-8', errors='replace')}")
        except urllib.error.URLError as exc:
            fail(f"HTTP {method} {endpoint} failed: {exc}")

        parsed: Any
        try:
            parsed = json.loads(raw.decode("utf-8-sig"))
        except json.JSONDecodeError:
            parsed = None

        self._write_artifact(output_name, raw)
        return status, parsed, raw

    def request_stream(self, endpoint: str, payload: Dict[str, Any], output_name: str) -> str:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Accept": "text/event-stream, application/json",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            self._write_artifact(output_name, raw.encode("utf-8"))
            fail(f"Stream request failed with status {exc.code}: {raw}")
        except urllib.error.URLError as exc:
            fail(f"Stream request failed: {exc}")

        self._write_artifact(output_name, content.encode("utf-8"))
        return content

    def _write_artifact(self, output_name: Optional[str], raw: bytes) -> None:
        if output_name is None:
            return
        (self.work_dir / output_name).write_bytes(raw)

    def assert_true(self, condition: bool, label: str, details: Optional[Any] = None) -> None:
        if condition:
            return
        if details is None:
            fail(f"Assertion failed: {label}")
        fail(f"Assertion failed: {label}: {details}")

    def assert_chat_response(self, payload: Any, label: str) -> None:
        self.assert_true(isinstance(payload, dict), f"{label} is a JSON object", payload)
        self.assert_true(payload.get("object") == "chat.completion", f"{label} object", payload)
        choices = payload.get("choices")
        self.assert_true(isinstance(choices, list) and len(choices) > 0, f"{label} choices", payload)
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        self.assert_true(isinstance(message, dict), f"{label} message", payload)
        self.assert_true(message.get("role") == "assistant", f"{label} assistant role", payload)
        content = message.get("content")
        self.assert_true(isinstance(content, (str, list)), f"{label} content present", payload)

    def wait_for_health(self) -> None:
        deadline = time.time() + float(self.args.start_timeout)
        while time.time() < deadline:
            try:
                status, payload, _ = self.request_json("GET", "/health", output_name=None, allow_error=True, timeout=2.0)
            except SystemExit:
                status, payload = 0, None

            if status == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
                return

            if self.gateway_proc is not None and self.gateway_proc.poll() is not None:
                tail = ""
                if self.gateway_log.is_file():
                    tail = self.gateway_log.read_text(encoding="utf-8", errors="replace")
                fail(f"gateway exited before becoming healthy\n{tail}")
            time.sleep(1.0)

        tail = self.gateway_log.read_text(encoding="utf-8", errors="replace") if self.gateway_log.is_file() else ""
        fail(f"timed out waiting for {self.base_url}/health\n{tail}")

    def start_gateway_if_needed(self) -> None:
        try:
            status, payload, _ = self.request_json("GET", "/health", output_name=None, allow_error=True, timeout=2.0)
        except SystemExit:
            status, payload = 0, None

        if status == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
            if self.args.reuse_running:
                log(f"Reusing existing gateway at {self.base_url}")
                return
            fail(f"an existing gateway is already listening at {self.base_url}; rerun with --reuse-running or choose a different --port")

        log(f"Starting OmniInfer gateway on {self.base_url}")
        self.gateway_log.parent.mkdir(parents=True, exist_ok=True)
        self.gateway_log_handle = self.gateway_log.open("w", encoding="utf-8")
        self.gateway_proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(REPO_ROOT / "omniinfer_gateway.py"),
                "--host",
                self.args.host,
                "--port",
                str(self.args.port),
            ],
            cwd=str(REPO_ROOT),
            stdout=self.gateway_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.own_gateway = True
        self.wait_for_health()

    def run(self) -> None:
        self.ensure_inputs()
        self.start_gateway_if_needed()

        log("GET /health")
        _status, payload, _raw = self.request_json("GET", "/health", output_name="health.json")
        self.assert_true(isinstance(payload, dict) and payload.get("status") == "ok", "health status", payload)

        log("GET /omni/backends")
        _status, payload, _raw = self.request_json("GET", "/omni/backends", output_name="backends.json")
        self.assert_true(isinstance(payload, dict) and payload.get("object") == "list", "backend list object", payload)
        self.assert_true(isinstance(payload.get("data"), list) and len(payload["data"]) > 0, "backend list", payload)

        log("GET /omni/state")
        _status, payload, _raw = self.request_json("GET", "/omni/state", output_name="state-initial.json")
        self.assert_true(isinstance(payload, dict), "state response object", payload)
        self.assert_true(isinstance(payload.get("available_backends"), list) and len(payload["available_backends"]) > 0, "state available_backends", payload)

        log(f"GET /omni/supported-models?system={self.args.system}")
        _status, payload, _raw = self.request_json(
            "GET",
            f"/omni/supported-models?{urllib.parse.urlencode({'system': self.args.system})}",
            output_name="supported-models.json",
        )
        self.assert_true(isinstance(payload, dict) and len(payload) > 0, "supported models response", payload)

        log(f"GET /omni/supported-models/best?system={self.args.system}")
        _status, payload, _raw = self.request_json(
            "GET",
            f"/omni/supported-models/best?{urllib.parse.urlencode({'system': self.args.system})}",
            output_name="supported-models-best.json",
        )
        self.assert_true(isinstance(payload, dict) and len(payload) > 0, "supported models best response", payload)

        log("GET /omni/thinking")
        _status, payload, _raw = self.request_json("GET", "/omni/thinking", output_name="thinking-initial.json")
        self.assert_true(isinstance(payload, dict) and "default_enabled" in payload, "thinking default_enabled", payload)

        log("POST /omni/thinking/select (on)")
        _status, payload, _raw = self.request_json(
            "POST",
            "/omni/thinking/select",
            payload={"enabled": True},
            output_name="thinking-on-response.json",
        )
        self.assert_true(isinstance(payload, dict) and payload.get("ok") is True and payload.get("default_enabled") is True, "thinking enabled", payload)

        log("POST /omni/thinking/select (off)")
        _status, payload, _raw = self.request_json(
            "POST",
            "/omni/thinking/select",
            payload={"enabled": False},
            output_name="thinking-off-response.json",
        )
        self.assert_true(isinstance(payload, dict) and payload.get("ok") is True and payload.get("default_enabled") is False, "thinking disabled", payload)

        if self.args.backend:
            log("POST /omni/backend/select")
            _status, payload, _raw = self.request_json(
                "POST",
                "/omni/backend/select",
                payload={"backend": self.args.backend},
                output_name="backend-select-response.json",
            )
            self.assert_true(isinstance(payload, dict) and payload.get("ok") is True and payload.get("selected_backend") == self.args.backend, "backend select", payload)

        model_select_payload: Dict[str, Any] = {"model": str(self.model)}
        if self.mmproj:
            model_select_payload["mmproj"] = str(self.mmproj)
        if self.args.backend:
            model_select_payload["backend"] = self.args.backend

        log("POST /omni/model/select")
        _status, payload, _raw = self.request_json(
            "POST",
            "/omni/model/select",
            payload=model_select_payload,
            output_name="model-select-response.json",
        )
        self.assert_true(isinstance(payload, dict) and payload.get("ok") is True, "model select ok", payload)
        self.assert_true(isinstance(payload.get("selected_model"), str) and payload.get("selected_model"), "selected model", payload)

        text_payload = {
            "think": False,
            "messages": [{"role": "user", "content": "请用一句中文介绍你自己。"}],
            "temperature": 0.2,
            "max_tokens": 128,
        }
        log("POST /v1/chat/completions (preloaded text)")
        _status, payload, _raw = self.request_json(
            "POST",
            "/v1/chat/completions",
            payload=text_payload,
            output_name="text-chat-response.json",
        )
        self.assert_chat_response(payload, "preloaded text completion")

        if self.mmproj:
            image_b64 = base64.b64encode(self.image.read_bytes()).decode("ascii")
            multimodal_payload = {
                "think": False,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请用中文简要描述这张图片。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 256,
            }
            log("POST /v1/chat/completions (multimodal)")
            _status, payload, _raw = self.request_json(
                "POST",
                "/v1/chat/completions",
                payload=multimodal_payload,
                output_name="multimodal-response.json",
            )
            self.assert_chat_response(payload, "multimodal completion")

        stream_payload = {
            "think": False,
            "messages": [{"role": "user", "content": "请连续输出三句短句，每句单独成行。"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.2,
            "max_tokens": 64,
        }
        log("POST /v1/chat/completions (stream)")
        stream_content = self.request_stream("/v1/chat/completions", stream_payload, output_name="stream.txt")
        self.assert_true("data:" in stream_content, "stream contains SSE frames", stream_content[:500])
        self.assert_true("[DONE]" in stream_content, "stream contains DONE marker", stream_content[-500:])

        log("POST /omni/backend/stop")
        _status, payload, _raw = self.request_json("POST", "/omni/backend/stop", output_name="backend-stop-response.json")
        self.assert_true(isinstance(payload, dict) and payload.get("ok") is True and payload.get("stopped") is True, "backend stop", payload)

        direct_payload: Dict[str, Any] = {
            "model": str(self.model),
            "think": False,
            "messages": [{"role": "user", "content": "请再次用一句中文介绍你自己。"}],
            "temperature": 0.2,
            "max_tokens": 128,
        }
        if self.mmproj:
            direct_payload["mmproj"] = str(self.mmproj)
        if self.args.backend:
            direct_payload["backend"] = self.args.backend

        log("POST /v1/chat/completions (direct model loading)")
        _status, payload, _raw = self.request_json(
            "POST",
            "/v1/chat/completions",
            payload=direct_payload,
            output_name="direct-chat-response.json",
        )
        self.assert_chat_response(payload, "direct model loading completion")

        log("GET /omni/state")
        _status, payload, _raw = self.request_json("GET", "/omni/state", output_name="state-final.json")
        self.assert_true(isinstance(payload, dict) and payload.get("backend_ready") is True, "state backend ready after direct chat", payload)

        self.success = True
        log(f"Inference flow passed. Base URL: {self.base_url}")


def main() -> int:
    runner = FlowRunner(parse_args())
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
