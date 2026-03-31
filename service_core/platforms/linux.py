from __future__ import annotations

import platform

from service_core.backends import LINUX_LLAMA_CPP_TEMPLATES
from service_core.platforms.base import HostPlatform


class LinuxPlatform(HostPlatform):
    @property
    def system_name(self) -> str:
        return "linux"

    @property
    def runtime_folder_name(self) -> str:
        return "linux"

    @property
    def default_backend_id(self) -> str:
        if platform.machine().lower() == "s390x":
            return "llama.cpp-linux-s390x"
        return "llama.cpp-linux"

    @property
    def backend_templates(self):
        return LINUX_LLAMA_CPP_TEMPLATES

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        aliases = {
            "llama.cpp-vulkan": "llama.cpp-linux-vulkan",
            "llama.cpp-openvino": "llama.cpp-linux-openvino",
        }
        if platform.machine().lower() == "s390x":
            aliases["llama.cpp-linux"] = "llama.cpp-linux-s390x"
        return aliases

    @property
    def gpu_backend_ids(self) -> frozenset[str]:
        return frozenset({"llama.cpp-linux-rocm", "llama.cpp-linux-vulkan"})
