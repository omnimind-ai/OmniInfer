from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BACKEND_PRIORITY: dict[str, int] = {
    "llama.cpp-mac": 0,
    "llama.cpp-cuda": 0,
    "llama.cpp-vulkan": 0,
    "llama.cpp-linux-rocm": 0,
    "llama.cpp-linux": 1,
    "llama.cpp-cpu": 1,
}


@dataclass(frozen=True)
class BackendTemplate:
    id: str
    label: str
    runtime_dir_name: str
    server_binary_name: str
    description: str
    capabilities: tuple[str, ...]
    env_prefix: str
    default_ngl: str | None = None
    fallback_runtime_dir_names: tuple[str, ...] = ()


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
