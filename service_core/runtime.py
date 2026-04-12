from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any

logger = logging.getLogger("runtime")

from service_core.backends import BackendSpec
from service_core.drivers import EmbeddedBackendDriver, get_embedded_backend_driver
from service_core.model_catalog import SupportedModelCatalog
from service_core.platforms import (
    HostPlatform,
    current_host_platform,
    discover_llama_cpp_model_artifacts,
    display_path_reference,
    maybe_auto_mmproj,
    parse_optional_int,
    pick_available_port,
    wait_http_ready,
)


def _assign_to_kill_on_close_job(pid: int) -> Any | None:
    """Assign a process to a Job Object with KILL_ON_JOB_CLOSE on Windows.

    When the last handle to the Job is closed (i.e. this process exits or
    crashes), the OS automatically terminates all processes in the Job.
    Returns the job handle on success, None on non-Windows or failure.
    """
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation_PerProcessUserTimeLimit", ctypes.c_int64),
                ("BasicLimitInformation_PerJobUserTimeLimit", ctypes.c_int64),
                ("BasicLimitInformation_LimitFlags", ctypes.c_uint32),
                ("BasicLimitInformation_MinimumWorkingSetSize", ctypes.c_size_t),
                ("BasicLimitInformation_MaximumWorkingSetSize", ctypes.c_size_t),
                ("BasicLimitInformation_ActiveProcessLimit", ctypes.c_uint32),
                ("BasicLimitInformation_Affinity", ctypes.c_size_t),
                ("BasicLimitInformation_PriorityClass", ctypes.c_uint32),
                ("BasicLimitInformation_SchedulingClass", ctypes.c_uint32),
                ("IoInfo_ReadOperationCount", ctypes.c_uint64),
                ("IoInfo_WriteOperationCount", ctypes.c_uint64),
                ("IoInfo_OtherOperationCount", ctypes.c_uint64),
                ("IoInfo_ReadTransferCount", ctypes.c_uint64),
                ("IoInfo_WriteTransferCount", ctypes.c_uint64),
                ("IoInfo_OtherTransferCount", ctypes.c_uint64),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation_LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(job)
            return None

        h_process = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h_process:
            kernel32.CloseHandle(job)
            return None

        ok = kernel32.AssignProcessToJobObject(job, h_process)
        kernel32.CloseHandle(h_process)
        if not ok:
            kernel32.CloseHandle(job)
            return None

        return job
    except Exception:
        return None


@dataclass
class LoadedRuntime:
    backend_id: str
    model_path: str
    model_ref: str
    mmproj_path: str | None
    mmproj_ref: str | None
    ctx_size: int | None
    runtime_mode: str
    host: str | None
    port: int | None
    process: subprocess.Popen[Any] | None
    launch_args: list[str]
    request_defaults: dict[str, Any]
    log_path: str | None = None
    log_handle: TextIOWrapper | None = None
    embedded_driver: EmbeddedBackendDriver | None = None
    embedded_state: Any = None


@dataclass(frozen=True)
class ExternalRuntimeLaunch:
    cmd: list[str]
    port: int
    ctx_size: int | None
    log_file_name: str


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
        self.platform: HostPlatform = current_host_platform()
        self.lock = threading.Lock()
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.startup_timeout_s = startup_timeout_s
        self.backend_window_mode = backend_window_mode
        self.runtime_root = self.platform.discover_runtime_root(
            repo_root=self.repo_root,
            app_root=self.app_root,
            requested_runtime_root=runtime_root,
        )
        self.backend_overrides = backend_overrides or {}
        self.backends = self.platform.build_backends(
            app_root=self.app_root,
            runtime_root=self.runtime_root,
            backend_overrides=self.backend_overrides,
        )
        self.catalog = SupportedModelCatalog(self.platform, self.backends.keys())
        resolved_default_backend = self._normalize_backend_id(default_backend_id)
        self.selected_backend_id = (
            resolved_default_backend if resolved_default_backend in self.backends else next(iter(self.backends))
        )
        self.loaded_runtime: LoadedRuntime | None = None
        logger.info("RuntimeManager initialized: platform=%s runtime_root=%s", self.platform.system_name, self.runtime_root)
        logger.info("Backends discovered: %s", ", ".join(self.backends.keys()))
        logger.info("Default backend: %s", self.selected_backend_id)

    def _normalize_backend_id(self, backend_id: str | None) -> str | None:
        if not backend_id:
            return None
        return self.platform.resolve_catalog_backend_id(backend_id)

    def _extract_server_arg_value(self, args: list[str], flags: tuple[str, ...]) -> str | None:
        value: str | None = None
        i = 0
        while i < len(args):
            token = args[i]
            if token in flags:
                if i + 1 < len(args):
                    value = args[i + 1]
                    i += 2
                    continue
                break
            i += 1
        return value

    def _with_server_arg(self, args: list[str], flags: tuple[str, ...], value: Any | None) -> list[str]:
        updated: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token in flags:
                i += 2 if i + 1 < len(args) else 1
                continue
            updated.append(token)
            i += 1
        if value is not None:
            updated.extend([flags[0], str(value)])
        return updated

    def _without_server_arg(self, args: list[str], flags: tuple[str, ...]) -> list[str]:
        updated: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token in flags:
                i += 2 if i + 1 < len(args) else 1
                continue
            updated.append(token)
            i += 1
        return updated

    def _validate_launch_args(self, args: list[str]) -> None:
        reserved_flags = {
            "-m",
            "--model",
            "-mm",
            "--mmproj",
            "--host",
            "--port",
            "--no-webui",
        }
        i = 0
        while i < len(args):
            token = str(args[i])
            flag_name = token.split("=", 1)[0]
            if flag_name in reserved_flags:
                raise ValueError(
                    f"launch arg {flag_name!r} is managed by OmniInfer and must not be set in backend config"
                )
            i += 1

    def _get_backend(self, backend_id: str | None = None) -> BackendSpec:
        target = self._normalize_backend_id(backend_id) or self.selected_backend_id
        if target not in self.backends:
            raise ValueError(f"unsupported backend: {target}")
        return self.backends[target]

    def _is_runtime_running_locked(self) -> bool:
        if not self.loaded_runtime:
            return False
        if self.loaded_runtime.runtime_mode == "embedded":
            return True
        return bool(self.loaded_runtime.process and self.loaded_runtime.process.poll() is None)

    def _stop_runtime_locked(self) -> None:
        runtime = self.loaded_runtime
        self.loaded_runtime = None
        if runtime is None:
            return

        logger.info("Stopping runtime: backend=%s mode=%s", runtime.backend_id, runtime.runtime_mode)

        if runtime.runtime_mode == "embedded":
            if runtime.embedded_driver is not None:
                try:
                    runtime.embedded_driver.unload_model(runtime.embedded_state)
                except Exception:
                    logger.warning("Error unloading embedded model", exc_info=True)
            logger.info("Embedded runtime stopped")
            return

        proc = runtime.process
        if proc is not None and proc.poll() is None:
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

        job = getattr(self, "_backend_job_handle", None)
        if job is not None:
            try:
                ctypes.windll.kernel32.CloseHandle(job)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._backend_job_handle = None

        logger.info("External runtime stopped (backend=%s)", runtime.backend_id)
        if runtime.log_handle:
            try:
                runtime.log_handle.flush()
                runtime.log_handle.close()
            except OSError:
                pass

    def _resolve_model_path(self, backend: BackendSpec, model: str) -> tuple[str, str | None]:
        path = Path(model).expanduser()
        if not path.is_absolute():
            if not backend.models_dir:
                raise ValueError("relative model path requires a configured models_dir or an absolute model path")
            path = Path(backend.models_dir) / path
        resolved = path.resolve()
        if backend.model_artifact == "file" and resolved.is_dir():
            return discover_llama_cpp_model_artifacts(resolved)
        return str(resolved), None

    def _ensure_supported_model_artifact(self, backend: BackendSpec, model_path: str) -> None:
        target = Path(model_path)
        if backend.model_artifact == "directory":
            if not target.is_dir():
                logger.error("Model directory not found: %s", model_path)
                raise FileNotFoundError(f"model directory not found: {model_path}")
            return
        if backend.model_artifact == "file":
            if not target.is_file():
                logger.error("Model file not found: %s", model_path)
                raise FileNotFoundError(f"model file not found: {model_path}")
            return
        if not target.exists():
            logger.error("Model path not found: %s", model_path)
            raise FileNotFoundError(f"model path not found: {model_path}")

    def _resolve_mmproj_path(
        self,
        backend: BackendSpec,
        mmproj: str | None,
        model_path: str,
        auto_mmproj_path: str | None = None,
    ) -> str | None:
        if not backend.supports_mmproj:
            if mmproj:
                raise ValueError(f"{backend.id} does not support mmproj files")
            return None

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

        if auto_mmproj_path:
            return auto_mmproj_path

        return maybe_auto_mmproj(backend.models_dir, model_path)

    def _start_runtime_locked(
        self,
        backend: BackendSpec,
        model_path: str,
        mmproj_path: str | None,
        ctx_size: int | None = None,
        launch_args: list[str] | None = None,
        request_defaults: dict[str, Any] | None = None,
    ) -> LoadedRuntime:
        if backend.runtime_mode == "embedded":
            return self._start_embedded_runtime_locked(
                backend,
                model_path,
                mmproj_path,
                ctx_size=ctx_size,
                request_defaults=request_defaults,
            )
        return self._start_external_runtime_locked(
            backend,
            model_path,
            mmproj_path,
            ctx_size=ctx_size,
            launch_args=launch_args,
            request_defaults=request_defaults,
        )

    def _start_embedded_runtime_locked(
        self,
        backend: BackendSpec,
        model_path: str,
        mmproj_path: str | None,
        ctx_size: int | None = None,
        request_defaults: dict[str, Any] | None = None,
    ) -> LoadedRuntime:
        if not backend.binary_exists:
            logger.error("Embedded backend %s not available: required Python packages missing", backend.id)
            raise RuntimeError(
                f"{backend.id} is not available in the current Python environment; "
                "install its required Python packages before loading a model"
            )

        logger.info("Loading embedded model via %s: %s", backend.id, model_path)
        driver = get_embedded_backend_driver(backend.id)
        model_ref = display_path_reference(model_path, backend.models_dir)
        state = driver.load_model(
            model_path=model_path,
            model_ref=model_ref,
            mmproj_path=mmproj_path,
            ctx_size=ctx_size,
            load_options=dict(request_defaults or {}),
        )
        runtime = LoadedRuntime(
            backend_id=backend.id,
            model_path=model_path,
            model_ref=model_ref,
            mmproj_path=mmproj_path,
            mmproj_ref=display_path_reference(mmproj_path, backend.models_dir) if mmproj_path else None,
            ctx_size=ctx_size,
            runtime_mode="embedded",
            host=None,
            port=None,
            process=None,
            launch_args=[],
            request_defaults=dict(request_defaults or {}),
            embedded_driver=driver,
            embedded_state=state,
        )
        logger.info("Embedded model loaded: %s (backend=%s)", model_ref, backend.id)
        self.loaded_runtime = runtime
        return runtime

    def _start_external_runtime_locked(
        self,
        backend: BackendSpec,
        model_path: str,
        mmproj_path: str | None,
        ctx_size: int | None = None,
        launch_args: list[str] | None = None,
        request_defaults: dict[str, Any] | None = None,
    ) -> LoadedRuntime:
        if not backend.binary_exists or not backend.launcher_path:
            logger.error("Backend launcher not found: %s (path=%s)", backend.id, backend.launcher_path or "(unset)")
            raise FileNotFoundError(f"backend launcher not found: {backend.launcher_path or '(unset)'}")

        logger.info("Starting external backend %s", backend.id)
        effective_launch_args = list(backend.default_args if launch_args is None else launch_args)
        self._validate_launch_args(effective_launch_args)
        launch = self._prepare_external_runtime_launch(
            backend,
            model_path,
            mmproj_path,
            ctx_size,
            launch_args=effective_launch_args,
        )

        env = os.environ.copy()
        env.update(backend.env)
        env = self.platform.prepare_runtime_env(env, backend)

        logs_dir = backend.runtime_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / launch.log_file_name
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)

        creationflags = 0
        startupinfo = None
        if os.name == "nt" and self.backend_window_mode == "hidden":
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
            startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)

        logger.info("Command: %s", " ".join(launch.cmd))
        logger.info("Log file: %s", log_path)
        logger.debug("Environment overrides: %s", backend.env)

        proc = subprocess.Popen(
            launch.cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(backend.launcher_path).resolve().parent),
            env=env,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )

        self._backend_job_handle = _assign_to_kill_on_close_job(proc.pid)

        logger.info("Backend %s started (PID %d), waiting for health check on port %d...", backend.id, proc.pid, launch.port)
        _health_start = time.perf_counter()
        if not wait_http_ready(self.backend_host, launch.port, self.startup_timeout_s):
            message = "backend did not become ready in time"
            if proc.poll() is not None:
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    lines = []
                if lines:
                    message += "; backend output: " + " | ".join(lines[-6:])
            self.loaded_runtime = LoadedRuntime(
                backend_id=backend.id,
                model_path=model_path,
                model_ref=display_path_reference(model_path, backend.models_dir),
                mmproj_path=mmproj_path,
                mmproj_ref=display_path_reference(mmproj_path, backend.models_dir) if mmproj_path else None,
                ctx_size=launch.ctx_size,
                runtime_mode="external_server",
                host=self.backend_host,
                port=launch.port,
                process=proc,
                launch_args=list(effective_launch_args),
                request_defaults=dict(request_defaults or {}),
                log_path=str(log_path),
                log_handle=log_handle,
            )
            logger.error("Backend %s did not become ready within %ds", backend.id, self.startup_timeout_s)
            self._stop_runtime_locked()
            raise RuntimeError(message)

        logger.info("Backend health check: passed in %.1fs", time.perf_counter() - _health_start)

        runtime = LoadedRuntime(
            backend_id=backend.id,
            model_path=model_path,
            model_ref=display_path_reference(model_path, backend.models_dir),
            mmproj_path=mmproj_path,
            mmproj_ref=display_path_reference(mmproj_path, backend.models_dir) if mmproj_path else None,
            ctx_size=launch.ctx_size,
            runtime_mode="external_server",
            host=self.backend_host,
            port=launch.port,
            process=proc,
            launch_args=list(effective_launch_args),
            request_defaults=dict(request_defaults or {}),
            log_path=str(log_path),
            log_handle=log_handle,
        )
        self.loaded_runtime = runtime
        return runtime

    def _prepare_external_runtime_launch(
        self,
        backend: BackendSpec,
        model_path: str,
        mmproj_path: str | None,
        ctx_size: int | None = None,
        launch_args: list[str] | None = None,
    ) -> ExternalRuntimeLaunch:
        target_port = self.backend_port if self.backend_port > 0 else pick_available_port(self.backend_host)
        server_args = list(backend.default_args if launch_args is None else launch_args)
        if ctx_size is not None:
            server_args = self._with_server_arg(server_args, ("-c", "--ctx-size"), ctx_size)
        effective_ctx_size = parse_optional_int(
            self._extract_server_arg_value(server_args, ("-c", "--ctx-size"))
        )

        protocol = backend.external_server_protocol or "llama.cpp-server"
        if protocol != "llama.cpp-server":
            raise RuntimeError(f"unsupported external runtime protocol for {backend.id}: {protocol}")

        if not backend.launcher_path:
            raise FileNotFoundError(f"backend launcher not found: {backend.id}")

        cmd = [
            backend.launcher_path,
            "-m",
            model_path,
            "--host",
            self.backend_host,
            "--port",
            str(target_port),
            "--no-webui",
            *server_args,
        ]
        if mmproj_path:
            cmd.extend(["-mm", mmproj_path])

        return ExternalRuntimeLaunch(
            cmd=cmd,
            port=target_port,
            ctx_size=effective_ctx_size,
            log_file_name=backend.log_file_name or "runtime.log",
        )

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
                logger.info("Switching backend: %s -> %s", self.selected_backend_id, backend.id)
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
        return self.catalog.list_supported_models(system_name)

    def list_supported_models_best(self, system_name: str) -> dict[str, Any]:
        return self.catalog.list_supported_models_best(system_name)

    def ensure_model_loaded(
        self,
        model: str | None,
        mmproj: str | None = None,
        backend_id: str | None = None,
        ctx_size: int | None = None,
        launch_args: list[str] | None = None,
        request_defaults: dict[str, Any] | None = None,
    ) -> LoadedRuntime:
        with self.lock:
            requested_backend_id = self._normalize_backend_id(backend_id)
            if not model:
                if self._is_runtime_running_locked() and self.loaded_runtime:
                    if requested_backend_id and self.loaded_runtime.backend_id != requested_backend_id:
                        raise RuntimeError("a different backend is currently loaded; reload the model to switch")
                    desired_launch_args = list(self.loaded_runtime.launch_args if launch_args is None else launch_args)
                    desired_request_defaults = dict(
                        self.loaded_runtime.request_defaults if request_defaults is None else request_defaults
                    )
                    compare_current_launch_args = list(self.loaded_runtime.launch_args)
                    compare_desired_launch_args = list(desired_launch_args)
                    if ctx_size is None:
                        compare_current_launch_args = self._without_server_arg(
                            compare_current_launch_args,
                            ("-c", "--ctx-size"),
                        )
                        compare_desired_launch_args = self._without_server_arg(
                            compare_desired_launch_args,
                            ("-c", "--ctx-size"),
                        )
                    if (
                        (ctx_size is None or self.loaded_runtime.ctx_size == ctx_size)
                        and compare_desired_launch_args == compare_current_launch_args
                    ):
                        self.loaded_runtime.request_defaults = desired_request_defaults
                        return self.loaded_runtime
                    current_runtime = self.loaded_runtime
                    backend = self._get_backend(current_runtime.backend_id)
                    if not backend.supports_ctx_size:
                        raise ValueError(f"{backend.id} does not support ctx_size overrides")
                    self.selected_backend_id = backend.id
                    self._stop_runtime_locked()
                    return self._start_runtime_locked(
                        backend,
                        current_runtime.model_path,
                        current_runtime.mmproj_path,
                        ctx_size=ctx_size,
                        launch_args=desired_launch_args,
                        request_defaults=desired_request_defaults,
                    )
                logger.warning("No model loaded and no model specified in request")
                raise RuntimeError("no backend model loaded; call /omni/model/select first or provide 'model'")

            logger.info("Loading model: model=%s backend=%s ctx_size=%s", model, backend_id or "(auto)", ctx_size)
            preferred_backend = self._get_backend(requested_backend_id) if requested_backend_id else self._get_backend()
            if ctx_size is not None and not preferred_backend.supports_ctx_size:
                raise ValueError(f"{preferred_backend.id} does not support ctx_size overrides")
            model_path, auto_mmproj_path = self._resolve_model_path(preferred_backend, model)
            self._ensure_supported_model_artifact(preferred_backend, model_path)

            # Log model file size and available memory for OOM diagnosis
            try:
                model_size = Path(model_path).stat().st_size
                logger.info("Model file: %s (%.2f GiB)", model_path, model_size / (1024**3))
            except OSError:
                logger.info("Model path: %s (size unknown)", model_path)
            try:
                from service_core.platforms.common import bytes_to_gib, get_available_memory_bytes
                logger.info("Available memory before load: %.2f GiB", bytes_to_gib(get_available_memory_bytes()))
            except Exception:
                pass


            mmproj_path = self._resolve_mmproj_path(
                preferred_backend,
                mmproj,
                model_path,
                auto_mmproj_path=auto_mmproj_path,
            )
            if requested_backend_id:
                resolved_backend_id = requested_backend_id
            elif preferred_backend.runtime_mode == "embedded":
                resolved_backend_id = preferred_backend.id
            else:
                resolved_backend_id = self.catalog.auto_select_backend_for_model(model_path, mmproj_path)
            backend = self._get_backend(resolved_backend_id)
            if ctx_size is not None and not backend.supports_ctx_size:
                raise ValueError(f"{backend.id} does not support ctx_size overrides")
            self.selected_backend_id = backend.id
            effective_launch_args = list(backend.default_args if launch_args is None else launch_args)
            effective_request_defaults = dict(request_defaults or {})

            current = self.loaded_runtime if self._is_runtime_running_locked() else None
            if current:
                current_mmproj = current.mmproj_path or ""
                wanted_mmproj = mmproj_path or ""
                compare_current_launch_args = list(current.launch_args)
                compare_wanted_launch_args = list(effective_launch_args)
                if ctx_size is None:
                    compare_current_launch_args = self._without_server_arg(
                        compare_current_launch_args,
                        ("-c", "--ctx-size"),
                    )
                    compare_wanted_launch_args = self._without_server_arg(
                        compare_wanted_launch_args,
                        ("-c", "--ctx-size"),
                    )
                if (
                    current.backend_id == backend.id
                    and Path(current.model_path).resolve() == Path(model_path).resolve()
                    and current_mmproj == wanted_mmproj
                    and (ctx_size is None or current.ctx_size == ctx_size)
                    and compare_current_launch_args == compare_wanted_launch_args
                ):
                    current.request_defaults = effective_request_defaults
                    return current

            self._stop_runtime_locked()
            return self._start_runtime_locked(
                backend,
                model_path,
                mmproj_path,
                ctx_size=ctx_size,
                launch_args=effective_launch_args,
                request_defaults=effective_request_defaults,
            )

    def select_model(
        self,
        model: str,
        mmproj: str | None = None,
        backend_id: str | None = None,
        ctx_size: int | None = None,
        launch_args: list[str] | None = None,
        request_defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = self.ensure_model_loaded(
            model=model,
            mmproj=mmproj,
            backend_id=backend_id,
            ctx_size=ctx_size,
            launch_args=launch_args,
            request_defaults=request_defaults,
        )
        return {
            "ok": True,
            "selected_backend": runtime.backend_id,
            "selected_model": runtime.model_ref,
            "selected_mmproj": runtime.mmproj_ref,
            "selected_ctx_size": runtime.ctx_size,
        }

    def current_proxy_target(self) -> tuple[str, int] | None:
        with self.lock:
            if (
                self._is_runtime_running_locked()
                and self.loaded_runtime
                and self.loaded_runtime.runtime_mode == "external_server"
                and self.loaded_runtime.host
                and self.loaded_runtime.port is not None
            ):
                return self.loaded_runtime.host, self.loaded_runtime.port
            return None

    def current_runtime_mode(self) -> str | None:
        with self.lock:
            if self._is_runtime_running_locked() and self.loaded_runtime:
                return self.loaded_runtime.runtime_mode
            return None

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            runtime = self.loaded_runtime if self._is_runtime_running_locked() else None
            if runtime is None or runtime.runtime_mode != "embedded" or runtime.embedded_driver is None:
                raise RuntimeError("selected backend is not ready for embedded inference")
            return runtime.embedded_driver.chat_completion(runtime.embedded_state, payload)

    def stream_chat_completion(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        with self.lock:
            runtime = self.loaded_runtime if self._is_runtime_running_locked() else None
            if runtime is None or runtime.runtime_mode != "embedded" or runtime.embedded_driver is None:
                raise RuntimeError("selected backend is not ready for embedded inference")
            return list(runtime.embedded_driver.stream_chat_completion(runtime.embedded_state, payload))

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            runtime = self.loaded_runtime if self._is_runtime_running_locked() else None
            return {
                "backend": self.selected_backend_id,
                "model": runtime.model_ref if runtime else None,
                "mmproj": runtime.mmproj_ref if runtime else None,
                "ctx_size": runtime.ctx_size if runtime else None,
                "request_defaults": dict(runtime.request_defaults) if runtime else {},
                "backend_ready": bool(runtime),
            }
