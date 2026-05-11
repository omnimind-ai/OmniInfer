from __future__ import annotations

from service_core.drivers.base import EmbeddedBackendDriver


def get_embedded_backend_driver(backend_id: str) -> EmbeddedBackendDriver:
    if backend_id == "mlx-mac":
        from service_core.drivers.mlx import MlxMacDriver

        return MlxMacDriver()
    if backend_id == "mnn-linux":
        from service_core.drivers.mnn import MnnLinuxDriver

        return MnnLinuxDriver()
    raise ValueError(f"unsupported embedded backend: {backend_id}")


__all__ = ["EmbeddedBackendDriver", "get_embedded_backend_driver"]
