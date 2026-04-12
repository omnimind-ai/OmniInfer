from __future__ import annotations

from service_core.platforms.android import AndroidPlatform
from service_core.platforms.base import HostPlatform
from service_core.platforms.common import current_system_name
from service_core.platforms.ios import IOSPlatform
from service_core.platforms.linux import LinuxPlatform
from service_core.platforms.mac import MacPlatform
from service_core.platforms.windows import WindowsPlatform


_PLATFORM_REGISTRY: dict[str, HostPlatform] = {
    "android": AndroidPlatform(),
    "ios": IOSPlatform(),
    "windows": WindowsPlatform(),
    "mac": MacPlatform(),
    "linux": LinuxPlatform(),
}


def get_host_platform(system_name: str | None = None) -> HostPlatform:
    key = (system_name or current_system_name()).strip().lower()
    if key not in _PLATFORM_REGISTRY:
        raise ValueError(f"unsupported host system: {system_name}")
    return _PLATFORM_REGISTRY[key]


def current_host_platform() -> HostPlatform:
    return get_host_platform()


def default_backend_for_current_host() -> str:
    return current_host_platform().default_backend_id
