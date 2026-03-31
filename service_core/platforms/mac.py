from __future__ import annotations

import platform

from service_core.backends import MAC_LLAMA_CPP_TEMPLATES, MAC_MLX_TEMPLATES, MAC_TURBOQUANT_TEMPLATES
from service_core.platforms.base import HostPlatform


class MacPlatform(HostPlatform):
    @property
    def system_name(self) -> str:
        return "mac"

    @property
    def runtime_folder_name(self) -> str:
        return "macos"

    @property
    def default_backend_id(self) -> str:
        if platform.machine().lower() in {"x86_64", "amd64"}:
            return "llama.cpp-mac-intel"
        return "llama.cpp-mac"

    @property
    def backend_templates(self):
        return MAC_LLAMA_CPP_TEMPLATES + MAC_TURBOQUANT_TEMPLATES + MAC_MLX_TEMPLATES

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        target = "llama.cpp-mac-intel" if platform.machine().lower() in {"x86_64", "amd64"} else "llama.cpp-mac"
        return {"llama.cpp-cpu": target}
