from __future__ import annotations

from service_core.drivers.base import EmbeddedBackendDriver
from service_core.drivers.mlx import MlxMacDriver


def get_embedded_backend_driver(backend_id: str) -> EmbeddedBackendDriver:
    if backend_id == "mlx-mac":
        return MlxMacDriver()
    raise ValueError(f"unsupported embedded backend: {backend_id}")


__all__ = ["EmbeddedBackendDriver", "get_embedded_backend_driver", "MlxMacDriver"]
