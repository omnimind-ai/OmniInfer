from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def resolve_input_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def display_path_reference(path: str, root_dir: str) -> str:
    target = Path(path).resolve()
    root = Path(root_dir).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def wait_http_ready(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.getcode() in (200, 503):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def pick_available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def is_gguf_model(filename: str) -> bool:
    low = filename.lower()
    return low.endswith(".gguf") and "mmproj" not in low


def maybe_auto_mmproj(models_dir: str, model_path: str) -> str | None:
    model_file = Path(model_path)
    sibling_candidates = [
        model_file.with_name("mmproj-F32.gguf"),
        model_file.with_name("mmproj-f32.gguf"),
        model_file.with_name("mmproj-F16.gguf"),
        model_file.with_name("mmproj-f16.gguf"),
    ]
    for candidate in sibling_candidates:
        if candidate.is_file():
            return str(candidate)

    root = Path(models_dir)
    flat_candidates = [
        root / "mmproj-F32.gguf",
        root / "mmproj-f32.gguf",
        root / "mmproj-F16.gguf",
        root / "mmproj-f16.gguf",
    ]
    for candidate in flat_candidates:
        if candidate.is_file():
            return str(candidate)
    return None


@dataclass
class BackendSpec:
    id: str
    label: str
    runtime_dir: str
    llama_server_path: str
    models_dir: str
    description: str
    capabilities: list[str]
    default_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def binary_exists(self) -> bool:
        return Path(self.llama_server_path).is_file()

    @property
    def runtime_path(self) -> Path:
        return Path(self.runtime_dir)

    def to_api_payload(self, selected: bool, loaded_model: str | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "selected": selected,
            "binary_exists": self.binary_exists,
            "models_dir": self.models_dir,
            "capabilities": self.capabilities,
            "description": self.description,
            "loaded_model": loaded_model if selected else None,
        }


@dataclass
class LoadedRuntime:
    backend_id: str
    model_path: str
    model_ref: str
    mmproj_path: str | None
    mmproj_ref: str | None
    host: str
    port: int
    process: subprocess.Popen[Any]


class RuntimeManager:
    def __init__(
        self,
        repo_root: str,
        app_root: str | None,
        backend_host: str,
        backend_port: int,
        startup_timeout_s: int,
        runtime_root: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        default_backend_id: str = "llama.cpp-CPU",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.app_root = Path(app_root).resolve() if app_root else self.repo_root
        self.lock = threading.Lock()
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.startup_timeout_s = startup_timeout_s
        self.runtime_root = (
            Path(resolve_input_path(runtime_root, self.app_root)).resolve()
            if runtime_root
            else self._discover_runtime_root()
        )
        self.backend_overrides = backend_overrides or {}
        self.backends = self._build_default_backends()
        self.selected_backend_id = (
            default_backend_id if default_backend_id in self.backends else next(iter(self.backends))
        )
        self.loaded_runtime: LoadedRuntime | None = None

    def _discover_runtime_root(self) -> Path:
        portable_root = self.app_root / "runtime"
        if portable_root.is_dir():
            return portable_root.resolve()
        return (self.repo_root / "platform" / "Windows").resolve()

    def _resolve_backend_runtime_dir(self, backend_id: str, default_root: Path) -> Path:
        override = self.backend_overrides.get(backend_id, {}).get("runtime_dir")
        if override:
            return Path(resolve_input_path(str(override), self.app_root)).resolve()
        return default_root.resolve()

    def _build_default_backends(self) -> dict[str, BackendSpec]:
        cpu_root = self._resolve_backend_runtime_dir("llama.cpp-CPU", self.runtime_root / "llama.cpp-CPU")
        gpu_root = self._resolve_backend_runtime_dir("llama.cpp-GPU", self.runtime_root / "llama.cpp-GPU")
        cpu_override = self.backend_overrides.get("llama.cpp-CPU", {})
        gpu_override = self.backend_overrides.get("llama.cpp-GPU", {})

        cpu = BackendSpec(
            id="llama.cpp-CPU",
            label="llama.cpp CPU",
            runtime_dir=str(cpu_root),
            llama_server_path=resolve_input_path(
                os.environ.get(
                    "OMNIINFER_LLAMA_CPP_CPU_SERVER_PATH",
                    str(cpu_override.get("server_path") or (cpu_root / "bin" / "llama-server.exe")),
                ),
                self.app_root,
            ),
            models_dir=resolve_input_path(
                os.environ.get(
                    "OMNIINFER_LLAMA_CPP_CPU_MODELS_DIR",
                    str(cpu_override.get("models_dir") or (cpu_root / "models")),
                ),
                self.app_root,
            ),
            description="llama.cpp CPU backend managed by OmniInfer",
            capabilities=["chat", "vision", "stream", "cpu"],
        )

        gpu_ngl = os.environ.get("OMNIINFER_LLAMA_CPP_GPU_NGL", str(gpu_override.get("ngl", "999")))
        gpu = BackendSpec(
            id="llama.cpp-GPU",
            label="llama.cpp GPU",
            runtime_dir=str(gpu_root),
            llama_server_path=resolve_input_path(
                os.environ.get(
                    "OMNIINFER_LLAMA_CPP_GPU_SERVER_PATH",
                    str(gpu_override.get("server_path") or (gpu_root / "bin" / "llama-server.exe")),
                ),
                self.app_root,
            ),
            models_dir=resolve_input_path(
                os.environ.get(
                    "OMNIINFER_LLAMA_CPP_GPU_MODELS_DIR",
                    str(gpu_override.get("models_dir") or (gpu_root / "models")),
                ),
                self.app_root,
            ),
            description="llama.cpp CUDA backend managed by OmniInfer",
            capabilities=["chat", "vision", "stream", "gpu", "cuda"],
            default_args=["-ngl", gpu_ngl],
        )

        return {backend.id: backend for backend in (cpu, gpu)}

    def _get_backend(self, backend_id: str | None = None) -> BackendSpec:
        target = backend_id or self.selected_backend_id
        if target not in self.backends:
            raise ValueError(f"unsupported backend: {target}")
        return self.backends[target]

    def _is_runtime_running_locked(self) -> bool:
        return bool(self.loaded_runtime and self.loaded_runtime.process.poll() is None)

    def _stop_runtime_locked(self) -> None:
        runtime = self.loaded_runtime
        self.loaded_runtime = None
        if runtime is None:
            return

        proc = runtime.process
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _resolve_model_path(self, backend: BackendSpec, model: str) -> str:
        path = Path(model).expanduser()
        if not path.is_absolute():
            path = Path(backend.models_dir) / path
        return str(path.resolve())

    def _resolve_mmproj_path(self, backend: BackendSpec, mmproj: str | None, model_path: str) -> str | None:
        if mmproj:
            path = Path(mmproj).expanduser()
            if not path.is_absolute():
                path = Path(backend.models_dir) / path
            resolved = str(path.resolve())
            if not Path(resolved).is_file():
                raise FileNotFoundError(f"mmproj file not found: {resolved}")
            return resolved

        return maybe_auto_mmproj(backend.models_dir, model_path)

    def _start_runtime_locked(self, backend: BackendSpec, model_path: str, mmproj_path: str | None) -> LoadedRuntime:
        if not backend.binary_exists:
            raise FileNotFoundError(f"llama-server not found: {backend.llama_server_path}")

        target_port = self.backend_port if self.backend_port > 0 else pick_available_port(self.backend_host)
        cmd = [
            backend.llama_server_path,
            "-m",
            model_path,
            "--host",
            self.backend_host,
            "--port",
            str(target_port),
            "--no-webui",
            *backend.default_args,
        ]
        if mmproj_path:
            cmd.extend(["-mm", mmproj_path])

        env = os.environ.copy()
        env.update(backend.env)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(backend.llama_server_path).resolve().parent),
            env=env,
        )

        if not wait_http_ready(self.backend_host, target_port, self.startup_timeout_s):
            message = "backend did not become ready in time"
            if proc.poll() is not None and proc.stdout:
                lines: list[str] = []
                for _ in range(60):
                    line = proc.stdout.readline()
                    if not line:
                        break
                    lines.append(line.rstrip())
                if lines:
                    message += "; backend output: " + " | ".join(lines[-6:])
            self.loaded_runtime = LoadedRuntime(
                backend_id=backend.id,
                model_path=model_path,
                model_ref=display_path_reference(model_path, backend.models_dir),
                mmproj_path=mmproj_path,
                mmproj_ref=display_path_reference(mmproj_path, backend.models_dir) if mmproj_path else None,
                host=self.backend_host,
                port=target_port,
                process=proc,
            )
            self._stop_runtime_locked()
            raise RuntimeError(message)

        runtime = LoadedRuntime(
            backend_id=backend.id,
            model_path=model_path,
            model_ref=display_path_reference(model_path, backend.models_dir),
            mmproj_path=mmproj_path,
            mmproj_ref=display_path_reference(mmproj_path, backend.models_dir) if mmproj_path else None,
            host=self.backend_host,
            port=target_port,
            process=proc,
        )
        self.loaded_runtime = runtime
        return runtime

    def list_backends(self) -> list[dict[str, Any]]:
        with self.lock:
            loaded_model = self.loaded_runtime.model_ref if self._is_runtime_running_locked() and self.loaded_runtime else None
            return [
                backend.to_api_payload(
                    selected=backend.id == self.selected_backend_id,
                    loaded_model=loaded_model if backend.id == self.selected_backend_id else None,
                )
                for backend in self.backends.values()
            ]

    def select_backend(self, backend_id: str) -> dict[str, Any]:
        with self.lock:
            backend = self._get_backend(backend_id)
            if self.selected_backend_id != backend.id:
                self._stop_runtime_locked()
            self.selected_backend_id = backend.id
            return {
                "ok": True,
                "selected_backend": backend.id,
                "binary_exists": backend.binary_exists,
                "models_dir": backend.models_dir,
            }

    def stop_runtime(self) -> dict[str, Any]:
        with self.lock:
            self._stop_runtime_locked()
            return {"ok": True, "stopped": True, "selected_backend": self.selected_backend_id}

    def list_models(self, backend_id: str | None = None) -> list[dict[str, Any]]:
        targets: list[BackendSpec]
        if backend_id and backend_id != "all":
            targets = [self._get_backend(backend_id)]
        elif backend_id == "all":
            targets = list(self.backends.values())
        else:
            targets = [self._get_backend()]

        entries: list[dict[str, Any]] = []
        for backend in targets:
            root = Path(backend.models_dir)
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.gguf")):
                if not path.is_file() or not is_gguf_model(path.name):
                    continue
                rel = path.relative_to(root).as_posix()
                entries.append(
                    {
                        "id": rel,
                        "backend": backend.id,
                        "path": str(path),
                    }
                )
        return entries

    def ensure_model_loaded(
        self,
        model: str | None,
        mmproj: str | None = None,
        backend_id: str | None = None,
    ) -> LoadedRuntime:
        with self.lock:
            backend = self._get_backend(backend_id)
            self.selected_backend_id = backend.id

            if not model:
                if self._is_runtime_running_locked() and self.loaded_runtime:
                    if self.loaded_runtime.backend_id == backend.id:
                        return self.loaded_runtime
                raise RuntimeError("no backend model loaded; call /omni/model/select first or provide 'model'")

            model_path = self._resolve_model_path(backend, model)
            if not Path(model_path).is_file():
                raise FileNotFoundError(f"model file not found: {model_path}")

            mmproj_path = self._resolve_mmproj_path(backend, mmproj, model_path)

            current = self.loaded_runtime if self._is_runtime_running_locked() else None
            if current:
                current_mmproj = current.mmproj_path or ""
                wanted_mmproj = mmproj_path or ""
                if (
                    current.backend_id == backend.id
                    and Path(current.model_path).resolve() == Path(model_path).resolve()
                    and current_mmproj == wanted_mmproj
                ):
                    return current

            self._stop_runtime_locked()
            return self._start_runtime_locked(backend, model_path, mmproj_path)

    def select_model(self, model: str, mmproj: str | None = None, backend_id: str | None = None) -> dict[str, Any]:
        runtime = self.ensure_model_loaded(model=model, mmproj=mmproj, backend_id=backend_id)
        return {
            "ok": True,
            "selected_backend": runtime.backend_id,
            "selected_model": runtime.model_ref,
            "selected_mmproj": runtime.mmproj_ref,
        }

    def current_proxy_target(self) -> tuple[str, int] | None:
        with self.lock:
            if self._is_runtime_running_locked() and self.loaded_runtime:
                return self.loaded_runtime.host, self.loaded_runtime.port
            return None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            runtime = self.loaded_runtime if self._is_runtime_running_locked() else None
            return {
                "backend": self.selected_backend_id,
                "model": runtime.model_ref if runtime else None,
                "mmproj": runtime.mmproj_ref if runtime else None,
                "backend_ready": bool(runtime),
            }
