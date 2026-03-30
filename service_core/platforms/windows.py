from __future__ import annotations

from service_core.backends import WINDOWS_LLAMA_CPP_TEMPLATES
from service_core.platforms.base import HostPlatform


class WindowsPlatform(HostPlatform):
    @property
    def system_name(self) -> str:
        return "windows"

    @property
    def runtime_folder_name(self) -> str:
        return "windows"

    @property
    def legacy_runtime_folder_names(self) -> tuple[str, ...]:
        return ("Windows",)

    @property
    def default_backend_id(self) -> str:
        return "llama.cpp-cpu"

    @property
    def backend_templates(self):
        return WINDOWS_LLAMA_CPP_TEMPLATES

    @property
    def gpu_backend_ids(self) -> frozenset[str]:
        return frozenset({"llama.cpp-cuda", "llama.cpp-vulkan"})
