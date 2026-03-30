from __future__ import annotations

import importlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from service_core.drivers.base import EmbeddedBackendDriver


@dataclass
class MlxLoadedModel:
    model: Any
    tokenizer: Any
    model_path: str
    model_ref: str


class MlxMacDriver(EmbeddedBackendDriver):
    def load_model(
        self,
        *,
        model_path: str,
        model_ref: str,
        mmproj_path: str | None,
        ctx_size: int | None,
    ) -> MlxLoadedModel:
        if mmproj_path:
            raise ValueError("mlx-mac does not support mmproj files")
        if ctx_size is not None:
            raise ValueError("mlx-mac does not support ctx_size overrides")

        load = self._import_attr("mlx_lm", "load")
        model, tokenizer = load(model_path)
        return MlxLoadedModel(model=model, tokenizer=tokenizer, model_path=model_path, model_ref=model_ref)

    def unload_model(self, state: Any) -> None:
        del state

    def chat_completion(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(state.tokenizer, payload)
        prompt_tokens = self._count_tokens(state.tokenizer, prompt)
        generate = self._import_attr("mlx_lm", "generate")
        kwargs = {
            "prompt": prompt,
            "max_tokens": self._max_tokens(payload),
        }
        sampler = self._build_sampler(payload)
        if sampler is not None:
            kwargs["sampler"] = sampler
        try:
            text = generate(state.model, state.tokenizer, verbose=False, **kwargs)
        except TypeError:
            text = generate(state.model, state.tokenizer, **kwargs)

        completion_text = str(text)
        completion_tokens = self._count_tokens(state.tokenizer, completion_text)
        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or state.model_ref)
        usage = self._usage(prompt_tokens, completion_tokens)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    def stream_chat_completion(self, state: Any, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        prompt = self._build_prompt(state.tokenizer, payload)
        prompt_tokens = self._count_tokens(state.tokenizer, prompt)
        stream_generate = self._import_attr("mlx_lm", "stream_generate")
        kwargs = {
            "prompt": prompt,
            "max_tokens": self._max_tokens(payload),
        }
        sampler = self._build_sampler(payload)
        if sampler is not None:
            kwargs["sampler"] = sampler

        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or state.model_ref)
        text_parts: list[str] = []

        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        for item in stream_generate(state.model, state.tokenizer, **kwargs):
            chunk_text = str(getattr(item, "text", "") or "")
            if not chunk_text:
                continue
            text_parts.append(chunk_text)
            yield {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
            }

        completion_text = "".join(text_parts)
        completion_tokens = self._count_tokens(state.tokenizer, completion_text)
        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": self._usage(prompt_tokens, completion_tokens),
        }

    def _import_attr(self, module_name: str, attr_name: str) -> Any:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                "mlx-mac requires the 'mlx' and 'mlx-lm' Python packages. "
                "Install them in the Python environment that launches OmniInfer."
            ) from exc
        return getattr(module, attr_name)

    def _build_prompt(self, tokenizer: Any, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("field 'messages' must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("all chat messages must be JSON objects")
            content = message.get("content")
            if isinstance(content, list):
                raise ValueError("mlx-mac Phase 1 only supports text chat messages")

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                prompt = apply_chat_template(messages, add_generation_prompt=True)
            if isinstance(prompt, str):
                return prompt
            decode = getattr(tokenizer, "decode", None)
            if callable(decode):
                return str(decode(prompt))

        lines: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _build_sampler(self, payload: dict[str, Any]) -> Any | None:
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        try:
            sample_utils = importlib.import_module("mlx_lm.sample_utils")
        except ImportError:
            return None
        make_sampler = getattr(sample_utils, "make_sampler", None)
        if not callable(make_sampler):
            return None

        attempts = [
            {"temp": temperature, "top_p": top_p},
            {"temperature": temperature, "top_p": top_p},
            {"temp": temperature},
            {"temperature": temperature},
        ]
        for kwargs in attempts:
            filtered = {key: value for key, value in kwargs.items() if value is not None}
            if not filtered:
                continue
            try:
                return make_sampler(**filtered)
            except TypeError:
                continue
        return None

    def _count_tokens(self, tokenizer: Any, text: str) -> int | None:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return None
        try:
            tokens = encode(text)
        except Exception:
            return None
        try:
            return len(tokens)
        except TypeError:
            return None

    def _max_tokens(self, payload: dict[str, Any]) -> int:
        raw = payload.get("max_tokens")
        try:
            value = int(raw) if raw is not None else 128
        except (TypeError, ValueError):
            value = 128
        return value if value > 0 else 128

    def _usage(self, prompt_tokens: int | None, completion_tokens: int | None) -> dict[str, Any]:
        prompt_value = prompt_tokens or 0
        completion_value = completion_tokens or 0
        return {
            "prompt_tokens": prompt_value,
            "completion_tokens": completion_value,
            "total_tokens": prompt_value + completion_value,
        }
