from __future__ import annotations

import importlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from service_core.drivers.base import EmbeddedBackendDriver


@dataclass
class MlxLoadedModel:
    model: Any
    processor: Any
    tokenizer: Any
    model_path: str
    model_ref: str
    engine: str
    supports_vision: bool


@dataclass(frozen=True)
class MlxGenerationResult:
    text: str
    elapsed_s: float
    first_token_s: float | None = None
    final_item: Any = None


class MlxMacDriver(EmbeddedBackendDriver):
    def load_model(
        self,
        *,
        model_path: str,
        model_ref: str,
        mmproj_path: str | None,
        ctx_size: int | None,
        load_options: dict[str, Any] | None = None,
    ) -> MlxLoadedModel:
        del load_options
        if mmproj_path:
            raise ValueError("mlx-mac does not support mmproj files")
        if ctx_size is not None:
            raise ValueError("mlx-mac does not support ctx_size overrides")

        supports_vision = self._model_supports_vision(model_path)
        if supports_vision:
            load = self._import_attr("mlx_vlm", "load")
            model, processor = load(model_path)
            tokenizer = self._resolve_tokenizer(processor)
            return MlxLoadedModel(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                model_path=model_path,
                model_ref=model_ref,
                engine="mlx_vlm",
                supports_vision=True,
            )

        load = self._import_attr("mlx_lm", "load")
        model, tokenizer = load(model_path)
        return MlxLoadedModel(
            model=model,
            processor=tokenizer,
            tokenizer=tokenizer,
            model_path=model_path,
            model_ref=model_ref,
            engine="mlx_lm",
            supports_vision=False,
        )

    def unload_model(self, state: Any) -> None:
        del state

    def chat_completion(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if getattr(state, "engine", "mlx_lm") == "mlx_vlm":
            return self._vlm_chat_completion(state, payload)
        return self._lm_chat_completion(state, payload)

    def _lm_chat_completion(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_text_prompt(state.tokenizer, payload)
        prompt_tokens = self._count_tokens(state.tokenizer, prompt)
        kwargs = self._lm_generate_kwargs(prompt, payload)
        completion = self._generate_text_response("mlx_lm", state.model, state.tokenizer, kwargs, prefer_stream=False)
        completion_text = completion.text
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
            "timings": self._build_timings(
                first_token_s=completion.first_token_s,
                elapsed_s=completion.elapsed_s,
                prompt_tps=self._result_float_metric(completion.final_item, "prompt_tps"),
                generation_tps=self._result_float_metric(completion.final_item, "generation_tps"),
                peak_memory_gb=self._result_float_metric(completion.final_item, "peak_memory"),
                completion_tokens=completion_tokens,
            ),
        }

    def _vlm_chat_completion(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        prompt, images = self._build_multimodal_prompt(state, payload)
        kwargs = self._vlm_generate_kwargs(prompt, images, payload)
        completion = self._generate_text_response("mlx_vlm", state.model, state.processor, kwargs, prefer_stream=False)

        completion_text = completion.text
        prompt_tokens = self._result_metric(completion.final_item, "prompt_tokens") or self._count_tokens(
            state.tokenizer,
            prompt,
        )
        counted_completion_tokens = self._count_tokens(
            state.tokenizer,
            completion_text,
        )
        metric_completion_tokens = self._result_metric(completion.final_item, "generation_tokens")
        completion_tokens = self._prefer_larger_token_count(counted_completion_tokens, metric_completion_tokens)
        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or state.model_ref)

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
            "usage": self._usage(prompt_tokens, completion_tokens),
            "timings": self._build_timings(
                first_token_s=completion.first_token_s,
                elapsed_s=completion.elapsed_s,
                prompt_tps=self._result_float_metric(completion.final_item, "prompt_tps"),
                generation_tps=self._result_float_metric(completion.final_item, "generation_tps"),
                peak_memory_gb=self._result_float_metric(completion.final_item, "peak_memory"),
                completion_tokens=completion_tokens,
            ),
        }

    def stream_chat_completion(self, state: Any, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        if getattr(state, "engine", "mlx_lm") == "mlx_vlm":
            yield from self._vlm_stream_chat_completion(state, payload)
            return
        yield from self._lm_stream_chat_completion(state, payload)

    def _lm_stream_chat_completion(self, state: Any, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        prompt = self._build_text_prompt(state.tokenizer, payload)
        prompt_tokens = self._count_tokens(state.tokenizer, prompt)
        stream_generate = self._import_attr("mlx_lm", "stream_generate")
        kwargs = self._lm_generate_kwargs(prompt, payload)

        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or state.model_ref)
        text_parts: list[str] = []
        started = time.perf_counter()
        first_token_s: float | None = None
        final_item: Any = None

        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        for item in stream_generate(state.model, state.tokenizer, **kwargs):
            final_item = item
            chunk_text = str(getattr(item, "text", "") or "")
            if not chunk_text:
                continue
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
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
        elapsed_s = time.perf_counter() - started
        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": self._usage(prompt_tokens, completion_tokens),
            "timings": self._build_timings(
                first_token_s=first_token_s,
                elapsed_s=elapsed_s,
                prompt_tps=self._result_float_metric(final_item, "prompt_tps"),
                generation_tps=self._result_float_metric(final_item, "generation_tps"),
                peak_memory_gb=self._result_float_metric(final_item, "peak_memory"),
                completion_tokens=completion_tokens,
            ),
        }

    def _vlm_stream_chat_completion(self, state: Any, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        prompt, images = self._build_multimodal_prompt(state, payload)
        stream_generate = self._import_attr("mlx_vlm", "stream_generate")
        kwargs = self._vlm_generate_kwargs(prompt, images, payload)

        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or state.model_ref)
        text_parts: list[str] = []
        final_item: Any = None
        started = time.perf_counter()
        first_token_s: float | None = None

        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        for item in stream_generate(state.model, state.processor, **kwargs):
            final_item = item
            chunk_text = str(getattr(item, "text", "") or "")
            if not chunk_text:
                continue
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            text_parts.append(chunk_text)
            yield {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
            }

        completion_text = "".join(text_parts)
        prompt_tokens = self._result_metric(final_item, "prompt_tokens") or self._count_tokens(state.tokenizer, prompt)
        counted_completion_tokens = self._count_tokens(state.tokenizer, completion_text)
        metric_completion_tokens = self._result_metric(final_item, "generation_tokens")
        completion_tokens = self._prefer_larger_token_count(counted_completion_tokens, metric_completion_tokens)
        elapsed_s = time.perf_counter() - started
        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": self._usage(prompt_tokens, completion_tokens),
            "timings": self._build_timings(
                first_token_s=first_token_s,
                elapsed_s=elapsed_s,
                prompt_tps=self._result_float_metric(final_item, "prompt_tps"),
                generation_tps=self._result_float_metric(final_item, "generation_tps"),
                peak_memory_gb=self._result_float_metric(final_item, "peak_memory"),
                completion_tokens=completion_tokens,
            ),
        }

    def _generate_text_response(
        self,
        module_name: str,
        model: Any,
        processor: Any,
        kwargs: dict[str, Any],
        *,
        prefer_stream: bool,
    ) -> MlxGenerationResult:
        stream_generate = self._optional_import_attr(module_name, "stream_generate")
        if prefer_stream and callable(stream_generate):
            started = time.perf_counter()
            first_token_s: float | None = None
            text_parts: list[str] = []
            final_item: Any = None
            for item in stream_generate(model, processor, **kwargs):
                final_item = item
                chunk_text = str(getattr(item, "text", "") or "")
                if not chunk_text:
                    continue
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
                text_parts.append(chunk_text)
            return MlxGenerationResult(
                text="".join(text_parts),
                elapsed_s=time.perf_counter() - started,
                first_token_s=first_token_s,
                final_item=final_item,
            )

        generate = self._import_attr(module_name, "generate")
        started = time.perf_counter()
        try:
            text = generate(model, processor, verbose=False, **kwargs)
        except TypeError:
            text = generate(model, processor, **kwargs)
        elapsed_s = time.perf_counter() - started
        if hasattr(text, "text"):
            return MlxGenerationResult(
                text=str(getattr(text, "text", "") or ""),
                elapsed_s=elapsed_s,
                final_item=text,
            )
        return MlxGenerationResult(text=str(text), elapsed_s=elapsed_s)

    def _import_attr(self, module_name: str, attr_name: str) -> Any:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                "mlx-mac requires the 'mlx', 'mlx-lm', and 'mlx-vlm' Python packages. "
                "Vision-capable models also require 'torch' and 'torchvision'. "
                "Install them in the Python environment that launches OmniInfer."
            ) from exc
        return getattr(module, attr_name)

    def _optional_import_attr(self, module_name: str, attr_name: str) -> Any | None:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                "mlx-mac requires the 'mlx', 'mlx-lm', and 'mlx-vlm' Python packages. "
                "Vision-capable models also require 'torch' and 'torchvision'. "
                "Install them in the Python environment that launches OmniInfer."
            ) from exc
        return getattr(module, attr_name, None)

    def _normalize_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("field 'messages' must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("all chat messages must be JSON objects")
        return messages

    def _build_text_prompt(self, tokenizer: Any, payload: dict[str, Any]) -> str:
        raw_messages = self._normalize_messages(payload)
        messages: list[dict[str, Any]] = []
        for raw_message in raw_messages:
            message = dict(raw_message)
            content = message.get("content")
            if isinstance(content, list):
                if self._extract_images_from_content(content):
                    raise ValueError("the currently loaded mlx text model does not support image inputs")
                message["content"] = self._extract_text_from_content(content)
            messages.append(message)

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
            content = self._extract_text_from_content(message.get("content"))
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _build_multimodal_prompt(self, state: MlxLoadedModel, payload: dict[str, Any]) -> tuple[str, list[str]]:
        messages = self._normalize_messages(payload)
        images = self._extract_images_from_messages(messages)
        apply_chat_template = self._import_attr("mlx_vlm", "apply_chat_template")
        prompt = apply_chat_template(
            state.processor,
            state.model.config,
            messages,
            add_generation_prompt=True,
            num_images=len(images),
        )
        if isinstance(prompt, str):
            return prompt, images
        decode = getattr(state.tokenizer, "decode", None)
        if callable(decode):
            return str(decode(prompt)), images
        return str(prompt), images

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[str]:
        images: list[str] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images.extend(self._extract_images_from_content(content))
        return images

    def _extract_images_from_content(self, content: Any) -> list[str]:
        if not isinstance(content, list):
            return []
        images: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("multimodal message parts must be JSON objects")
            part_type = str(item.get("type") or "")
            if part_type == "text":
                continue
            if part_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if not image_url:
                    raise ValueError("image_url parts must include a non-empty url")
                images.append(str(image_url))
                continue
            raise ValueError(f"unsupported mlx-mac message content type: {part_type or '<missing>'}")
        return images

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("multimodal message parts must be JSON objects")
            part_type = str(item.get("type") or "")
            if part_type == "text":
                parts.append(str(item.get("text") or ""))
            elif part_type == "image_url":
                continue
            else:
                raise ValueError(f"unsupported mlx-mac message content type: {part_type or '<missing>'}")
        return "\n".join(part for part in parts if part)

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

    def _lm_generate_kwargs(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            "prompt": prompt,
            "max_tokens": self._max_tokens(payload),
        }
        kwargs.update(self._mlx_generation_overrides(payload))
        sampler = self._build_sampler(payload)
        if sampler is not None:
            kwargs["sampler"] = sampler
        return kwargs

    def _vlm_generate_kwargs(self, prompt: str, images: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": self._max_tokens(payload),
        }
        kwargs.update(self._mlx_generation_overrides(payload))
        image_value: str | list[str] | None = None
        if len(images) == 1:
            image_value = images[0]
        elif images:
            image_value = images
        if image_value is not None:
            kwargs["image"] = image_value

        temperature = self._float_payload(payload.get("temperature"))
        if temperature is not None:
            kwargs["temperature"] = temperature
        top_p = self._float_payload(payload.get("top_p"))
        if top_p is not None:
            kwargs["top_p"] = top_p
        return kwargs

    def _mlx_generation_overrides(self, payload: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        max_kv_size = self._positive_int_payload(payload.get("max_kv_size"))
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size

        prefill_step_size = self._positive_int_payload(payload.get("prefill_step_size"))
        if prefill_step_size is not None:
            kwargs["prefill_step_size"] = prefill_step_size

        kv_bits = self._positive_int_payload(payload.get("kv_bits"))
        if kv_bits is not None:
            kwargs["kv_bits"] = kv_bits

        kv_group_size = self._positive_int_payload(payload.get("kv_group_size"))
        if kv_group_size is not None:
            kwargs["kv_group_size"] = kv_group_size

        quantized_kv_start = self._non_negative_int_payload(payload.get("quantized_kv_start"))
        if quantized_kv_start is not None:
            kwargs["quantized_kv_start"] = quantized_kv_start

        return kwargs

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

    def _float_payload(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _positive_int_payload(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _non_negative_int_payload(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _resolve_tokenizer(self, processor: Any) -> Any:
        tokenizer = getattr(processor, "tokenizer", None)
        return tokenizer if tokenizer is not None else processor

    def _model_supports_vision(self, model_path: str) -> bool:
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if config.get("vision_config") is not None:
            return True
        model_type = str(config.get("model_type") or "").lower()
        if any(token in model_type for token in ("vl", "vision", "omni")):
            return True
        architectures = config.get("architectures") or []
        for architecture in architectures:
            arch_text = str(architecture).lower()
            if any(token in arch_text for token in ("vl", "vision", "omni")):
                return True
        return False

    def _result_metric(self, result: Any, field_name: str) -> int | None:
        value = getattr(result, field_name, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _result_float_metric(self, result: Any, field_name: str) -> float | None:
        value = getattr(result, field_name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _usage(self, prompt_tokens: int | None, completion_tokens: int | None) -> dict[str, Any]:
        prompt_value = prompt_tokens or 0
        completion_value = completion_tokens or 0
        return {
            "prompt_tokens": prompt_value,
            "completion_tokens": completion_value,
            "total_tokens": prompt_value + completion_value,
        }

    def _prefer_larger_token_count(self, primary: int | None, fallback: int | None) -> int | None:
        if primary is None:
            return fallback
        if fallback is None:
            return primary
        return max(primary, fallback)

    def _build_timings(
        self,
        *,
        first_token_s: float | None,
        elapsed_s: float,
        prompt_tps: float | None,
        generation_tps: float | None,
        peak_memory_gb: float | None,
        completion_tokens: int | None,
    ) -> dict[str, Any]:
        first_token_value = first_token_s if first_token_s is not None else elapsed_s
        predicted_s = max(elapsed_s - first_token_value, 0.0)
        timings: dict[str, Any] = {
            "prompt_ms": round(first_token_value * 1000, 3),
            "predicted_ms": round(predicted_s * 1000, 3),
        }
        if prompt_tps is not None and prompt_tps > 0:
            timings["prompt_per_second"] = round(prompt_tps, 3)
        if generation_tps is not None and generation_tps > 0:
            timings["predicted_per_second"] = round(generation_tps, 3)
        elif completion_tokens and predicted_s > 0:
            timings["predicted_per_second"] = round(completion_tokens / predicted_s, 3)
        if peak_memory_gb is not None and peak_memory_gb > 0:
            timings["peak_memory_gb"] = round(peak_memory_gb, 3)
        return timings
