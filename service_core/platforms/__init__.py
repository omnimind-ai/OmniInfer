from service_core.platforms.android import AndroidPlatform
from service_core.platforms.base import HostPlatform
from service_core.platforms.common import (
    SYSTEM_MODEL_LIST_URLS,
    current_system_name,
    display_path_reference,
    maybe_auto_mmproj,
    parse_optional_int,
    parse_size_gib,
    pick_available_port,
    resolve_input_path,
    wait_http_ready,
)
from service_core.platforms.linux import LinuxPlatform
from service_core.platforms.mac import MacPlatform
from service_core.platforms.registry import (
    current_host_platform,
    default_backend_for_current_host,
    get_host_platform,
)
from service_core.platforms.windows import WindowsPlatform

__all__ = [
    "HostPlatform",
    "AndroidPlatform",
    "LinuxPlatform",
    "MacPlatform",
    "SYSTEM_MODEL_LIST_URLS",
    "WindowsPlatform",
    "current_host_platform",
    "current_system_name",
    "default_backend_for_current_host",
    "display_path_reference",
    "get_host_platform",
    "maybe_auto_mmproj",
    "parse_optional_int",
    "parse_size_gib",
    "pick_available_port",
    "resolve_input_path",
    "wait_http_ready",
]
