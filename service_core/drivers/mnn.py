from __future__ import annotations

import base64
import binascii
import struct
import io
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

logger = logging.getLogger("driver.mnn")

from service_core.drivers.base import EmbeddedBackendDriver


@dataclass
class MnnLoadedModel:
    model: Any
    cv: Any
    config_path: str
    model_ref: str
    supports_vision: bool
    config: dict[str, Any]
    temp_files: list[str] = field(default_factory=list)


class MnnLinuxDriver(EmbeddedBackendDriver):
    def load_model(
        self,
        *,
        model_path: str,
        model_ref: str,
        mmproj_path: str | None,
        ctx_size: int | None,
        load_options: dict[str, Any] | None = None,
    ) -> MnnLoadedModel:
        if mmproj_path:
            raise ValueError("mnn-linux does not support mmproj files")
        if ctx_size is not None:
            raise ValueError("mnn-linux does not support ctx_size overrides")

        logger.info("MNN loading model: %s", model_path)
        llm_module = self._import_module("MNN.llm")
        cv_module = self._import_module("MNN.cv")
        config_path = self._resolve_config_path(model_path)
        config = self._load_config(load_options)
        model = llm_module.create(config_path)
        model.set_config(config)
        model.load()
        supports_vision = self._config_supports_vision(config_path)
        logger.info("MNN model loaded: %s (vision=%s)", model_ref, supports_vision)
        return MnnLoadedModel(
            model=model,
            cv=cv_module,
            config_path=config_path,
            model_ref=model_ref,
            supports_vision=supports_vision,
            config=config,
        )

    def unload_model(self, state: Any) -> None:
        logger.info("MNN model unloaded")
        for path in getattr(state, "temp_files", []):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        del state

    def chat_completion(self, state: Any, payload: dict[str, Any]) -> dict[str, Any]:
        prompt, prompt_tokens = self._build_prompt(state, payload)
        max_tokens = self._max_tokens(payload)
        self._configure_generation(state, max_tokens)
        started = time.perf_counter()
        text = str(state.model.response(prompt, False) or "")
        elapsed_s = time.perf_counter() - started
        context = state.model.context
        completion_tokens = self._completion_tokens(context, text)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model") or state.model_ref),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": self._usage(prompt_tokens, completion_tokens),
            "timings": self._timings(context, elapsed_s, completion_tokens),
        }

    def stream_chat_completion(self, state: Any, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        prompt, prompt_tokens = self._build_prompt(state, payload)
        max_tokens = self._max_tokens(payload)
        self._configure_generation(state, max_tokens)
        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = str(payload.get("model") or state.model_ref)
        started = time.perf_counter()

        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }

        text = str(state.model.response(prompt, False) or "")
        first_token_s: float | None = None
        if text:
            first_token_s = time.perf_counter() - started
            yield {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }

        elapsed_s = time.perf_counter() - started
        context = state.model.context
        completion_tokens = self._completion_tokens(context, text)
        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": self._usage(prompt_tokens, completion_tokens),
            "timings": self._timings(
                context,
                elapsed_s,
                completion_tokens,
                first_token_s=first_token_s,
            ),
        }

    def _build_prompt(self, state: MnnLoadedModel, payload: dict[str, Any]) -> tuple[Any, int]:
        messages = self._normalize_messages(payload)
        prompt_messages: list[dict[str, str]] = []
        images: list[dict[str, Any]] = []
        temp_files: list[str] = []
        image_index = 0

        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type") or "")
                    if part_type == "text":
                        text_parts.append(str(part.get("text") or ""))
                        continue
                    if part_type == "image_url":
                        if not state.supports_vision:
                            raise ValueError("mnn-linux text model does not support image inputs")
                        image_url = part.get("image_url")
                        url = image_url.get("url") if isinstance(image_url, dict) else None
                        if not isinstance(url, str) or not url.strip():
                            raise ValueError("image_url.url is required for multimodal prompts")
                        text_parts.append(f"<img>image_{image_index}</img>")
                        images.append(self._load_image_part(state, url, temp_files))
                        image_index += 1
                prompt_messages.append({"role": role, "content": "".join(text_parts)})
                continue
            prompt_messages.append({"role": role, "content": str(content or "")})

        prompt_text = state.model.apply_chat_template(prompt_messages)
        prompt_tokens = len(state.model.tokenizer_encode(prompt_text))
        if not images:
            return prompt_text, prompt_tokens

        state.temp_files.extend(temp_files)
        return {"text": prompt_text, "images": images}, prompt_tokens

    def _load_image_part(
        self,
        state: MnnLoadedModel,
        url: str,
        temp_files: list[str],
    ) -> dict[str, Any]:
        image_path = self._resolve_image_path(url, temp_files)
        image = state.cv.imread(image_path)
        height, width = self._infer_image_size(image)
        return {"data": image, "height": height, "width": width}

    def _resolve_image_path(self, url: str, temp_files: list[str]) -> str:
        if url.startswith("data:"):
            return self._write_data_url(url, temp_files)
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return url
        if parsed.scheme == "file":
            return Path(parsed.path).resolve().as_posix()
        return str(Path(url).expanduser().resolve())

    def _write_data_url(self, url: str, temp_files: list[str]) -> str:
        _header, encoded = url.split(",", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid data URL image payload") from exc
        image_type = _detect_image_type(raw) or "png"
        with tempfile.NamedTemporaryFile(prefix="omniinfer-mnn-", suffix=f".{image_type}", delete=False) as handle:
            handle.write(raw)
            temp_files.append(handle.name)
            return handle.name

    def _infer_image_size(self, image: Any) -> tuple[int, int]:
        shape = getattr(image, "shape", None)
        if callable(shape):
            shape = shape()
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        if hasattr(image, "read"):
            try:
                from PIL import Image  # type: ignore

                with Image.open(image) as pil_image:
                    return int(pil_image.height), int(pil_image.width)
            except Exception:
                pass
        return 0, 0

    def _normalize_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("field 'messages' must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("each message must be an object")
            role = str(item.get("role") or "").strip()
            if not role:
                raise ValueError("each message must include a role")
            normalized.append({"role": role, "content": item.get("content", "")})
        return normalized

    def _config_supports_vision(self, config_path: str) -> bool:
        try:
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("is_visual"):
            return True
        config_file = Path(config_path)
        if (config_file.parent / "visual.mnn").is_file():
            return True
        return isinstance(payload.get("mllm"), dict)

    def _resolve_config_path(self, model_path: str) -> str:
        path = Path(model_path).expanduser().resolve()
        if path.is_dir():
            candidate = path / "config.json"
            if candidate.is_file():
                return str(candidate)
            raise FileNotFoundError(f"mnn model config not found: {candidate}")
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"mnn model path not found: {model_path}")

    def _load_config(self, load_options: dict[str, Any] | None) -> dict[str, Any]:
        thread_num = int(os.environ.get("OMNIINFER_MNN_LINUX_THREADS") or max(os.cpu_count() or 1, 1))
        backend_type = str(os.environ.get("OMNIINFER_MNN_LINUX_BACKEND") or "cpu")
        config: dict[str, Any] = {
            "backend_type": backend_type,
            "thread_num": thread_num,
        }
        if isinstance(load_options, dict):
            for key in ("backend_type", "thread_num", "precision", "memory", "power", "max_new_tokens"):
                if key in load_options and load_options[key] not in (None, ""):
                    config[key] = load_options[key]
        return config

    def _max_tokens(self, payload: dict[str, Any]) -> int:
        raw_value = payload.get("max_tokens", 128)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return 128
        return value if value > 0 else 128

    def _configure_generation(self, state: MnnLoadedModel, max_tokens: int) -> None:
        next_config = dict(state.config)
        next_config["max_new_tokens"] = max_tokens
        state.model.set_config(next_config)
        state.config = next_config

    def _completion_tokens(self, context: Any, text: str) -> int:
        gen_seq_len = int(getattr(context, "gen_seq_len", 0) or 0)
        if gen_seq_len > 0:
            return gen_seq_len
        return max(len(text.split()), 0)

    def _usage(self, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _timings(
        self,
        context: Any,
        elapsed_s: float,
        completion_tokens: int,
        *,
        first_token_s: float | None = None,
    ) -> dict[str, float]:
        prompt_ms = round(float(getattr(context, "prefill_us", 0) or 0) / 1000.0, 3)
        decode_ms = round(float(getattr(context, "decode_us", 0) or 0) / 1000.0, 3)
        vision_ms = round(float(getattr(context, "vision_us", 0) or 0) / 1000.0, 3)
        total_ms = round(elapsed_s * 1000.0, 3)
        timings: dict[str, float] = {
            "prompt_ms": prompt_ms,
            "decode_ms": decode_ms,
            "predicted_ms": decode_ms,
            "total_ms": total_ms,
        }
        if vision_ms > 0:
            timings["vision_ms"] = vision_ms
        if first_token_s is not None:
            timings["first_token_ms"] = round(first_token_s * 1000.0, 3)
        if completion_tokens > 0 and decode_ms > 0:
            decode_tps = round(completion_tokens / (decode_ms / 1000.0), 3)
            timings["decode_tps"] = decode_tps
            timings["predicted_per_second"] = decode_tps
        prompt_tokens = int(getattr(context, "prompt_len", 0) or 0)
        if prompt_tokens > 0 and prompt_ms > 0:
            prefill_tps = round(prompt_tokens / (prompt_ms / 1000.0), 3)
            timings["prefill_tps"] = prefill_tps
            timings["prompt_per_second"] = prefill_tps
        return timings

    def _import_module(self, module_name: str) -> Any:
        import importlib

        return importlib.import_module(module_name)


def _detect_image_type(data: bytes) -> str | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] in (b"BM",):
        return "bmp"
    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "tiff"
    return None
