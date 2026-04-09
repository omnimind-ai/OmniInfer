from __future__ import annotations

from service_core.backends import IOS_LLAMA_CPP_TEMPLATES
from service_core.platforms.base import HostPlatform


class IOSPlatform(HostPlatform):
    """iOS host scaffold.

    iOS runs an in-process HTTP server and embedded inference backend
    rather than an external subprocess.  Like Android, it uses a native
    bridge (Swift/C instead of JNI) to call into the C++ inference
    backends directly.
    """

    @property
    def system_name(self) -> str:
        return "ios"

    @property
    def runtime_folder_name(self) -> str:
        return "ios"

    @property
    def default_backend_id(self) -> str:
        return "llama.cpp-ios"

    @property
    def backend_templates(self):
        return IOS_LLAMA_CPP_TEMPLATES

    @property
    def catalog_backend_aliases(self) -> dict[str, str]:
        return {"llama.cpp-cpu": "llama.cpp-ios"}
