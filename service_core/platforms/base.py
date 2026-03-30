from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from service_core.backends import BackendSpec, BackendTemplate
from service_core.platforms.common import (
    bytes_to_gib,
    get_available_cuda_memory_bytes,
    get_available_memory_bytes,
    get_available_rocm_memory_bytes,
    parse_extra_args,
    parse_optional_int,
    prepend_env_path,
    resolve_input_path,
)


class HostPlatform(ABC):
    @property
    @abstractmethod
    def system_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def runtime_folder_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_backend_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def backend_templates(self) -> tuple[BackendTemplate, ...]:
        raise NotImplementedError

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        return {}

    @property
    def gpu_backend_ids(self) -> frozenset[str]:
        return frozenset()

    def default_config_backends(self) -> dict[str, dict[str, Any]]:
        defaults: dict[str, dict[str, Any]] = {}
        for template in self.backend_templates:
            config: dict[str, Any] = {}
            if template.default_ngl is not None:
                config["ngl"] = template.default_ngl
            defaults[template.id] = config
        return defaults

    def discover_runtime_root(self, repo_root: Path, app_root: Path, requested_runtime_root: str | None = None) -> Path:
        if requested_runtime_root:
            requested_path = Path(resolve_input_path(requested_runtime_root, app_root)).resolve()
            if requested_path.is_dir():
                return requested_path

        portable_root = app_root / "runtime"
        if portable_root.is_dir():
            return portable_root.resolve()

        canonical_local_root = (repo_root / ".local" / "runtime" / self.runtime_folder_name).resolve()
        if canonical_local_root.is_dir():
            return canonical_local_root

        return canonical_local_root

    def resolve_catalog_backend_id(self, backend_id: str) -> str:
        return self.catalog_backend_aliases.get(backend_id, backend_id)

    def available_memory_gib_for_backend(self, backend_name: str) -> float:
        if backend_name == "llama.cpp-cuda":
            cuda_free_bytes = get_available_cuda_memory_bytes()
            if cuda_free_bytes is not None:
                return bytes_to_gib(cuda_free_bytes)
            return 0.0
        if backend_name == "llama.cpp-linux-rocm":
            rocm_free_bytes = get_available_rocm_memory_bytes()
            if rocm_free_bytes is not None:
                return bytes_to_gib(rocm_free_bytes)
            return 0.0
        return bytes_to_gib(get_available_memory_bytes())

    def safety_margin_gib_for_backend(self, backend_name: str) -> float:
        if backend_name in self.gpu_backend_ids:
            return 0.5
        return 1.0

    def prepare_runtime_env(self, env: dict[str, str], backend: BackendSpec) -> dict[str, str]:
        if self.system_name != "windows" and backend.launcher_path:
            prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(backend.launcher_path).resolve().parent))
        return env

    def build_backends(
        self,
        *,
        app_root: Path,
        runtime_root: Path,
        backend_overrides: dict[str, dict[str, Any]] | None,
    ) -> dict[str, BackendSpec]:
        overrides = backend_overrides or {}
        backends: dict[str, BackendSpec] = {}
        for template in self.backend_templates:
            override = overrides.get(template.id, {})
            default_runtime_dir = self._default_runtime_dir(runtime_root, template)
            runtime_dir = self._resolve_backend_runtime_dir(app_root, override, default_runtime_dir)
            models_dir = self._resolve_backend_models_dir(
                app_root=app_root,
                override=override,
                env_var=f"{template.env_prefix}_MODELS_DIR",
                default_root=runtime_dir / "models",
            )
            launcher_path: str | None = None
            if template.launcher_name:
                launcher_default = runtime_dir / "bin" / template.launcher_name
                launcher_path = resolve_input_path(
                    os.environ.get(
                        f"{template.env_prefix}_LAUNCHER_PATH",
                        os.environ.get(
                            f"{template.env_prefix}_SERVER_PATH",
                            str(override.get("launcher_path") or override.get("server_path") or launcher_default),
                        ),
                    ),
                    app_root,
                )
            backends[template.id] = BackendSpec(
                id=template.id,
                label=template.label,
                runtime_dir=str(runtime_dir),
                launcher_path=launcher_path,
                models_dir=models_dir,
                catalog_url=str(override.get("catalog_url")) if override.get("catalog_url") else None,
                description=template.description,
                capabilities=list(template.capabilities),
                default_args=self._backend_server_args(
                    override=override,
                    env_prefix=template.env_prefix,
                    default_ngl=template.default_ngl,
                ),
                runtime_mode=template.runtime_mode,
                model_artifact=template.model_artifact,
                supports_mmproj=template.supports_mmproj,
                supports_ctx_size=template.supports_ctx_size,
                python_modules=template.python_modules,
            )
        return backends

    def _default_runtime_dir(self, runtime_root: Path, template: BackendTemplate) -> Path:
        primary = runtime_root / template.runtime_dir_name
        if primary.exists():
            return primary.resolve()
        for fallback in template.fallback_runtime_dir_names:
            candidate = runtime_root / fallback
            if candidate.exists():
                return candidate.resolve()
        return primary.resolve()

    def _resolve_backend_runtime_dir(self, app_root: Path, override: dict[str, Any], default_root: Path) -> Path:
        runtime_override = override.get("runtime_dir")
        if runtime_override:
            return Path(resolve_input_path(str(runtime_override), app_root)).resolve()
        return default_root.resolve()

    def _resolve_backend_models_dir(
        self,
        *,
        app_root: Path,
        override: dict[str, Any],
        env_var: str,
        default_root: Path,
    ) -> str | None:
        env_value = os.environ.get(env_var)
        if env_value:
            return resolve_input_path(env_value, app_root)
        if "models_dir" in override:
            override_value = override.get("models_dir")
            if override_value in (None, ""):
                return None
            return resolve_input_path(str(override_value), app_root)
        return resolve_input_path(str(default_root), app_root)

    def _backend_server_args(
        self,
        *,
        override: dict[str, Any],
        env_prefix: str,
        default_ngl: str | None,
    ) -> list[str]:
        args: list[str] = []
        ngl_value = (
            os.environ.get(f"{env_prefix}_NGL", str(override.get("ngl", default_ngl)))
            if default_ngl is not None
            else None
        )
        if ngl_value not in (None, ""):
            args.extend(["-ngl", str(ngl_value)])

        ctx_size = parse_optional_int(os.environ.get(f"{env_prefix}_CTX_SIZE", override.get("ctx_size")))
        if ctx_size is not None and ctx_size > 0:
            args.extend(["-c", str(ctx_size)])

        parallel = parse_optional_int(os.environ.get(f"{env_prefix}_PARALLEL", override.get("parallel")))
        if parallel is not None and parallel > 0:
            args.extend(["-np", str(parallel)])

        cache_ram = parse_optional_int(os.environ.get(f"{env_prefix}_CACHE_RAM", override.get("cache_ram")))
        if cache_ram is not None:
            args.extend(["-cram", str(cache_ram)])

        args.extend(parse_extra_args(override.get("extra_args")))
        return args
