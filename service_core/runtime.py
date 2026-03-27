from __future__ import annotations

import copy
import ctypes
import json
import os
import platform
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

SYSTEM_MODEL_LIST_URLS: dict[str, str] = {
    "windows": "https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/windows/model_list.json",
    "mac": "https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/mac/model_list.json",
}

BACKEND_PRIORITY: dict[str, int] = {
    "llama.cpp-cuda": 0,
    "llama.cpp-cpu": 1,
}


def resolve_input_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def display_path_reference(path: str, root_dir: str | None) -> str:
    target = Path(path).resolve()
    if not root_dir:
        return str(target)
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


def get_available_memory_bytes() -> int:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("sullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)

    page_size = os.sysconf("SC_PAGE_SIZE")
    avail_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * avail_pages)


def get_available_cuda_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    free_mib_values: list[int] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            free_mib_values.append(int(value))
        except ValueError:
            continue

    if not free_mib_values:
        return None

    # Current runtime uses one CUDA backend at a time; pick the GPU with the most free VRAM.
    return max(free_mib_values) * 1024 * 1024


def bytes_to_gib(value: int) -> float:
    return round(float(value) / float(1024 ** 3), 2)


def parse_size_gib(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def current_system_name() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system.startswith("darwin") or system.startswith("mac"):
        return "mac"
    raise ValueError(f"unsupported host system: {platform.system()}")


def is_gguf_model(filename: str) -> bool:
    low = filename.lower()
    return low.endswith(".gguf") and "mmproj" not in low


def maybe_auto_mmproj(models_dir: str | None, model_path: str) -> str | None:
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

    if not models_dir:
        return None

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
    models_dir: str | None
    catalog_url: str | None
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
        backend_window_mode: str = "visible",
        runtime_root: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        default_backend_id: str = "llama.cpp-cpu",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.app_root = Path(app_root).resolve() if app_root else self.repo_root
        self.lock = threading.Lock()
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.startup_timeout_s = startup_timeout_s
        self.backend_window_mode = backend_window_mode
        if runtime_root:
            requested_runtime_root = Path(resolve_input_path(runtime_root, self.app_root)).resolve()
            self.runtime_root = (
                requested_runtime_root if requested_runtime_root.is_dir() else self._discover_runtime_root()
            )
        else:
            self.runtime_root = self._discover_runtime_root()
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

    def _resolve_backend_models_dir(
        self,
        backend_id: str,
        override: dict[str, Any],
        env_var: str,
        default_root: Path,
    ) -> str | None:
        env_value = os.environ.get(env_var)
        if env_value:
            return resolve_input_path(env_value, self.app_root)

        if "models_dir" in override:
            override_value = override.get("models_dir")
            if override_value in (None, ""):
                return None
            return resolve_input_path(str(override_value), self.app_root)

        return resolve_input_path(str(default_root), self.app_root)

    def _build_default_backends(self) -> dict[str, BackendSpec]:
        cpu_root = self._resolve_backend_runtime_dir("llama.cpp-cpu", self.runtime_root / "llama.cpp-cpu")
        gpu_root = self._resolve_backend_runtime_dir("llama.cpp-cuda", self.runtime_root / "llama.cpp-cuda")
        cpu_override = self.backend_overrides.get("llama.cpp-cpu", {})
        gpu_override = self.backend_overrides.get("llama.cpp-cuda", {})

        cpu = BackendSpec(
            id="llama.cpp-cpu",
            label="llama.cpp cpu",
            runtime_dir=str(cpu_root),
            llama_server_path=resolve_input_path(
                os.environ.get(
                    "OMNIINFER_LLAMA_CPP_CPU_SERVER_PATH",
                    str(cpu_override.get("server_path") or (cpu_root / "bin" / "llama-server.exe")),
                ),
                self.app_root,
            ),
            models_dir=self._resolve_backend_models_dir(
                "llama.cpp-cpu",
                cpu_override,
                "OMNIINFER_LLAMA_CPP_CPU_MODELS_DIR",
                cpu_root / "models",
            ),
            catalog_url=None,
            description="llama.cpp CPU backend managed by OmniInfer",
            capabilities=["chat", "vision", "stream", "cpu"],
        )

        gpu_ngl = os.environ.get("OMNIINFER_LLAMA_CPP_CUDA_NGL", str(gpu_override.get("ngl", "999")))
        gpu = BackendSpec(
            id="llama.cpp-cuda",
            label="llama.cpp CUDA",
            runtime_dir=str(gpu_root),
            llama_server_path=resolve_input_path(
                os.environ.get(
                    "OMNIINFER_LLAMA_CPP_CUDA_SERVER_PATH",
                    str(gpu_override.get("server_path") or (gpu_root / "bin" / "llama-server.exe")),
                ),
                self.app_root,
            ),
            models_dir=self._resolve_backend_models_dir(
                "llama.cpp-cuda",
                gpu_override,
                "OMNIINFER_LLAMA_CPP_CUDA_MODELS_DIR",
                gpu_root / "models",
            ),
            catalog_url=str(gpu_override.get("catalog_url")) if gpu_override.get("catalog_url") else None,
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
            if not backend.models_dir:
                raise ValueError("relative model path requires a configured models_dir or an absolute model path")
            path = Path(backend.models_dir) / path
        return str(path.resolve())

    def _resolve_models_scan_root(self, backend: BackendSpec, models_path: str | None) -> Path:
        if models_path:
            return Path(resolve_input_path(models_path, self.app_root)).resolve()
        if not backend.models_dir:
            raise ValueError("models_path is required because this backend has no default models_dir configured")
        return Path(backend.models_dir).resolve()

    def _fetch_system_catalog(self, system_name: str) -> dict[str, Any]:
        system_key = (system_name or "").strip().lower()
        if system_key not in SYSTEM_MODEL_LIST_URLS:
            raise ValueError("field 'system' must be one of: windows, mac")
        req = urllib.request.Request(
            SYSTEM_MODEL_LIST_URLS[system_key],
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid model catalog payload for system: {system_key}")
        return payload

    def _available_memory_gib_for_backend(self, backend_name: str) -> float:
        if backend_name == "llama.cpp-cuda":
            cuda_free_bytes = get_available_cuda_memory_bytes()
            if cuda_free_bytes is not None:
                return bytes_to_gib(cuda_free_bytes)
            return 0.0
        return bytes_to_gib(get_available_memory_bytes())

    def _safety_margin_gib_for_backend(self, backend_name: str) -> float:
        if backend_name == "llama.cpp-cuda":
            return 0.5
        return 1.0

    def _annotate_supported_models(self, payload: Any, available_memory_gib: float, safety_margin_gib: float) -> Any:
        if isinstance(payload, dict):
            quantizations = payload.get("quantization")
            if isinstance(quantizations, dict):
                vision = payload.get("vision")
                vision_size_gib = parse_size_gib(vision.get("size")) if isinstance(vision, dict) else 0.0
                annotated: dict[str, Any] = {}
                for key, value in payload.items():
                    if key != "quantization":
                        annotated[key] = self._annotate_supported_models(value, available_memory_gib, safety_margin_gib)
                        continue

                    quantization_payload: dict[str, Any] = {}
                    for quant_name, quant_info in value.items():
                        if not isinstance(quant_info, dict):
                            quantization_payload[quant_name] = quant_info
                            continue
                        required_memory_gib = round(parse_size_gib(quant_info.get("size")) + vision_size_gib, 2)
                        quantization_payload[quant_name] = {
                            **quant_info,
                            "required_memory_gib": required_memory_gib,
                            "suitable": available_memory_gib >= round(required_memory_gib + safety_margin_gib, 2),
                        }
                    annotated[key] = quantization_payload
                return annotated

            return {
                key: self._annotate_supported_models(value, available_memory_gib, safety_margin_gib)
                for key, value in payload.items()
            }

        if isinstance(payload, list):
            return [self._annotate_supported_models(item, available_memory_gib, safety_margin_gib) for item in payload]

        return payload

    def _annotated_system_catalog(self, system_name: str) -> dict[str, Any]:
        catalog = self._fetch_system_catalog(system_name)
        annotated: dict[str, Any] = {}
        for backend_name, backend_payload in catalog.items():
            annotated[backend_name] = self._annotate_supported_models(
                backend_payload,
                self._available_memory_gib_for_backend(backend_name),
                self._safety_margin_gib_for_backend(backend_name),
            )
        return annotated

    def _candidate_rank(self, candidate: dict[str, Any]) -> tuple[int, int]:
        return (
            0 if candidate.get("suitable") else 1,
            BACKEND_PRIORITY.get(str(candidate.get("backend")), 999),
        )

    def _merge_best_supported_models(self, annotated_catalog: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        quantization_candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

        for backend_name, backend_payload in annotated_catalog.items():
            if not isinstance(backend_payload, dict):
                continue
            for family_name, family_models in backend_payload.items():
                if not isinstance(family_models, dict):
                    continue
                target_family = merged.setdefault(family_name, {})
                for model_name, model_info in family_models.items():
                    if not isinstance(model_info, dict):
                        continue
                    target_model = target_family.setdefault(model_name, {})
                    for key, value in model_info.items():
                        if key == "quantization":
                            continue
                        if key not in target_model:
                            target_model[key] = copy.deepcopy(value)

                    quantizations = model_info.get("quantization")
                    if not isinstance(quantizations, dict):
                        continue
                    target_quantizations = target_model.setdefault("quantization", {})
                    for quant_name, quant_info in quantizations.items():
                        if not isinstance(quant_info, dict):
                            continue
                        candidate = {
                            "backend": backend_name,
                            "payload": copy.deepcopy(quant_info),
                            "required_memory_gib": quant_info.get("required_memory_gib"),
                            "suitable": bool(quant_info.get("suitable")),
                        }
                        candidate_key = (family_name, model_name, quant_name)
                        quantization_candidates.setdefault(candidate_key, []).append(candidate)
                        target_quantizations.setdefault(quant_name, copy.deepcopy(quant_info))

        for (family_name, model_name, quant_name), candidates in quantization_candidates.items():
            target_quant = merged[family_name][model_name]["quantization"][quant_name]
            suitable_candidates = [candidate for candidate in candidates if candidate["suitable"]]
            if suitable_candidates:
                best_candidate = min(suitable_candidates, key=self._candidate_rank)
                target_quant.clear()
                target_quant.update(best_candidate["payload"])
                target_quant["backend"] = str(best_candidate["backend"])
                continue

            best_candidate = min(candidates, key=self._candidate_rank)
            target_quant.clear()
            target_quant.update(best_candidate["payload"])
            target_quant["backend"] = ""

        return merged

    def _flatten_catalog_candidates(self, annotated_catalog: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for backend_name, backend_payload in annotated_catalog.items():
            if backend_name not in self.backends or not isinstance(backend_payload, dict):
                continue
            for family_name, family_models in backend_payload.items():
                if not isinstance(family_models, dict):
                    continue
                for model_name, model_info in family_models.items():
                    if not isinstance(model_info, dict):
                        continue
                    quantizations = model_info.get("quantization")
                    if not isinstance(quantizations, dict):
                        continue
                    vision = model_info.get("vision")
                    mmproj_download = vision.get("download") if isinstance(vision, dict) else None
                    mmproj_basename = Path(str(mmproj_download)).name.lower() if mmproj_download else None
                    for quant_name, quant_info in quantizations.items():
                        if not isinstance(quant_info, dict):
                            continue
                        download = quant_info.get("download")
                        if not download:
                            continue
                        candidates.append(
                            {
                                "backend": backend_name,
                                "family": family_name,
                                "model_name": model_name,
                                "quantization": quant_name,
                                "model_basename": Path(str(download)).name.lower(),
                                "mmproj_basename": mmproj_basename,
                                "required_memory_gib": quant_info.get("required_memory_gib"),
                                "suitable": bool(quant_info.get("suitable")),
                            }
                        )
        return candidates

    def _auto_select_backend_for_model(self, model_path: str, mmproj_path: str | None) -> str:
        system_name = current_system_name()
        annotated_catalog = self._annotated_system_catalog(system_name)
        model_basename = Path(model_path).name.lower()
        mmproj_basename = Path(mmproj_path).name.lower() if mmproj_path else None

        candidates = [
            candidate
            for candidate in self._flatten_catalog_candidates(annotated_catalog)
            if candidate["model_basename"] == model_basename
            and (mmproj_basename is None or candidate["mmproj_basename"] in (None, mmproj_basename))
        ]
        if not candidates:
            raise ValueError(
                f"model is not present in the current {system_name} supported-model catalog: {Path(model_path).name}"
            )

        best_candidate = min(candidates, key=self._candidate_rank)
        if not best_candidate["suitable"]:
            raise RuntimeError(
                "no suitable backend was found for this model on the current device; "
                f"best candidate is {best_candidate['backend']} and requires about "
                f"{best_candidate['required_memory_gib']} GiB of memory"
            )
        return str(best_candidate["backend"])

    def _resolve_mmproj_path(self, backend: BackendSpec, mmproj: str | None, model_path: str) -> str | None:
        if mmproj:
            path = Path(mmproj).expanduser()
            if not path.is_absolute():
                if not backend.models_dir:
                    raise ValueError("relative mmproj path requires a configured models_dir or an absolute mmproj path")
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

        creationflags = 0
        if os.name == "nt" and self.backend_window_mode == "hidden":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(backend.llama_server_path).resolve().parent),
            env=env,
            creationflags=creationflags,
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

    def list_supported_models(self, system_name: str) -> dict[str, Any]:
        return self._annotated_system_catalog(system_name)

    def list_supported_models_best(self, system_name: str) -> dict[str, Any]:
        return self._merge_best_supported_models(self._annotated_system_catalog(system_name))

    def ensure_model_loaded(
        self,
        model: str | None,
        mmproj: str | None = None,
        backend_id: str | None = None,
    ) -> LoadedRuntime:
        with self.lock:
            if not model:
                if self._is_runtime_running_locked() and self.loaded_runtime:
                    if backend_id and self.loaded_runtime.backend_id != backend_id:
                        raise RuntimeError("a different backend is currently loaded; reload the model to switch")
                    return self.loaded_runtime
                raise RuntimeError("no backend model loaded; call /omni/model/select first or provide 'model'")

            preferred_backend = self._get_backend(backend_id) if backend_id else self._get_backend()
            model_path = self._resolve_model_path(preferred_backend, model)
            if not Path(model_path).is_file():
                raise FileNotFoundError(f"model file not found: {model_path}")

            mmproj_path = self._resolve_mmproj_path(preferred_backend, mmproj, model_path)
            resolved_backend_id = backend_id or self._auto_select_backend_for_model(model_path, mmproj_path)
            backend = self._get_backend(resolved_backend_id)
            self.selected_backend_id = backend.id

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
