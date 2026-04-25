from __future__ import annotations

import platform

from service_core.backends import WINDOWS_IK_LLAMA_CPP_TEMPLATES, WINDOWS_LLAMA_CPP_TEMPLATES
from service_core.platforms.base import HostPlatform


class WindowsPlatform(HostPlatform):
    @property
    def system_name(self) -> str:
        return "windows"

    @property
    def runtime_folder_name(self) -> str:
        return "windows"

    @property
    def default_backend_id(self) -> str:
        if platform.machine().lower() in {"arm64", "aarch64"}:
            return "llama.cpp-windows-arm64"
        return "llama.cpp-cpu"

    @property
    def backend_templates(self):
        return WINDOWS_LLAMA_CPP_TEMPLATES + WINDOWS_IK_LLAMA_CPP_TEMPLATES

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        if platform.machine().lower() in {"arm64", "aarch64"}:
            aliases["llama.cpp-cpu"] = "llama.cpp-windows-arm64"
        return aliases

    @property
    def gpu_backend_ids(self) -> frozenset[str]:
        return frozenset({"llama.cpp-cuda", "llama.cpp-vulkan", "llama.cpp-sycl", "llama.cpp-hip", "ik_llama.cpp-cuda"})
