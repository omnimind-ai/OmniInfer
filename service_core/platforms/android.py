from __future__ import annotations

from service_core.backends import ANDROID_LLAMA_CPP_TEMPLATES
from service_core.platforms.base import HostPlatform


class AndroidPlatform(HostPlatform):
    """Android host scaffold.

    Android differs from the desktop hosts in packaging, process lifetime,
    and runtime path conventions. We still model it as a first-class platform
    so future Android-specific logic has a clear home instead of leaking into
    the shared runtime flow.
    """

    @property
    def system_name(self) -> str:
        return "android"

    @property
    def runtime_folder_name(self) -> str:
        return "android"

    @property
    def default_backend_id(self) -> str:
        return "llama.cpp-android"

    @property
    def backend_templates(self):
        return ANDROID_LLAMA_CPP_TEMPLATES

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        return {"llama.cpp-cpu": "llama.cpp-android"}
