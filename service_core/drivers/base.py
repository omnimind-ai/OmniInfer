from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class EmbeddedBackendDriver(ABC):
    @abstractmethod
    def load_model(
        self,
        *,
        model_path: str,
        model_ref: str,
        mmproj_path: str | None,
        ctx_size: int | None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self, state: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def chat_completion(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def stream_chat_completion(self, state: Any, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        raise NotImplementedError
