from __future__ import annotations

from service_core.backends import MAC_LLAMA_CPP_TEMPLATES
from service_core.platforms.base import HostPlatform


class MacPlatform(HostPlatform):
    @property
    def system_name(self) -> str:
        return "mac"

    @property
    def runtime_folder_name(self) -> str:
        return "Mac"

    @property
    def default_backend_id(self) -> str:
        return "llama.cpp-mac"

    @property
    def backend_templates(self):
        return MAC_LLAMA_CPP_TEMPLATES

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        return {"llama.cpp-cpu": "llama.cpp-mac"}
