from __future__ import annotations

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
    def legacy_runtime_folder_names(self) -> tuple[str, ...]:
        return ("Linux",)

    @property
    def default_backend_id(self) -> str:
        return "llama.cpp-linux"

    @property
    def backend_templates(self):
        return LINUX_LLAMA_CPP_TEMPLATES

    @property
    def gpu_backend_ids(self) -> frozenset[str]:
        return frozenset({"llama.cpp-linux-rocm"})
