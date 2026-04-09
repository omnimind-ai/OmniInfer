from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BACKEND_PRIORITY: dict[str, int] = {
    "llama.cpp-mac": 0,
    "llama.cpp-mac-intel": 1,
    "turboquant-mac": 0,
    "mlx-mac": 0,
    "llama.cpp-cuda": 0,
    "llama.cpp-vulkan": 0,
    "llama.cpp-sycl": 0,
    "llama.cpp-hip": 0,
    "llama.cpp-linux-rocm": 0,
    "llama.cpp-linux-vulkan": 0,
    "llama.cpp-linux-omni": 0,
    "llama.cpp-linux-openvino": 0,
    "llama.cpp-linux": 1,
    "llama.cpp-linux-s390x": 1,
    "llama.cpp-cpu": 1,
    "llama.cpp-windows-arm64": 1,
    "llama.cpp-ios": 0,
}


@dataclass(frozen=True)
class BackendTemplate:
    id: str
    label: str
    family: str
    runtime_dir_name: str
    launcher_name: str | None
    description: str
    capabilities: tuple[str, ...]
    env_prefix: str
    default_ngl: str | None = None
    default_extra_args: tuple[str, ...] = ()
    fallback_runtime_dir_names: tuple[str, ...] = ()
    runtime_mode: str = "external_server"
    model_artifact: str = "file"
    supports_mmproj: bool = True
    supports_ctx_size: bool = True
    python_modules: tuple[str, ...] = ()
    external_server_protocol: str | None = "llama.cpp-server"
    log_file_name: str = "runtime.log"


@dataclass
class BackendSpec:
    id: str
    label: str
    family: str
    runtime_dir: str
    launcher_path: str | None
    models_dir: str | None
    catalog_url: str | None
    description: str
    capabilities: list[str]
    default_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    runtime_mode: str = "external_server"
    model_artifact: str = "file"
    supports_mmproj: bool = True
    supports_ctx_size: bool = True
    python_modules: tuple[str, ...] = ()
    external_server_protocol: str | None = "llama.cpp-server"
    log_file_name: str = "runtime.log"

    @property
    def binary_exists(self) -> bool:
        if self.runtime_mode == "embedded":
            for module_name in self.python_modules:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    return False
            return True
        if not self.launcher_path:
            return False
        return Path(self.launcher_path).is_file()

    @property
    def runtime_path(self) -> Path:
        return Path(self.runtime_dir)

    def to_api_payload(self, selected: bool, loaded_model: str | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "selected": selected,
            "binary_exists": self.binary_exists,
            "models_dir": self.models_dir,
            "capabilities": self.capabilities,
            "description": self.description,
            "loaded_model": loaded_model if selected else None,
        }
