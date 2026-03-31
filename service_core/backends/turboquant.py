from __future__ import annotations

from service_core.backends.base import BackendTemplate


MAC_TURBOQUANT_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="turboquant-mac",
        label="TurboQuant Metal",
        runtime_dir_name="turboquant-mac",
        launcher_name="llama-server",
        description="TurboQuant llama.cpp-compatible Metal backend managed by OmniInfer on macOS",
        capabilities=("chat", "vision", "stream", "metal", "apple", "shared-memory", "turboquant"),
        env_prefix="OMNIINFER_TURBOQUANT_MAC",
        default_ngl="999",
        default_extra_args=("-fa", "on", "--cache-type-k", "turbo4", "--cache-type-v", "turbo4"),
        external_server_protocol="llama.cpp-server",
        log_file_name="turboquant-server.log",
    ),
)
