#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.drivers.mlx import MlxMacDriver


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def encode(self, text):
        return text.split()


class FakeProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer


class FakeModel:
    def __init__(self, model_type: str) -> None:
        self.config = types.SimpleNamespace(model_type=model_type)


class FakeStreamItem:
    def __init__(
        self,
        text: str,
        *,
        prompt_tokens: int | None = None,
        generation_tokens: int | None = None,
        prompt_tps: float | None = None,
        generation_tps: float | None = None,
        peak_memory: float | None = None,
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.prompt_tps = prompt_tps
        self.generation_tps = generation_tps
        self.peak_memory = peak_memory


class MlxDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = MlxMacDriver()

    def _fake_modules(self) -> dict[str, types.ModuleType]:
        tokenizer = FakeTokenizer()
        processor = FakeProcessor(tokenizer)

        mlx_module = types.ModuleType("mlx")

        mlx_lm_module = types.ModuleType("mlx_lm")

        def lm_load(model_path):
            return FakeModel("llama"), tokenizer

        def lm_generate(model, tokenizer_obj, prompt, max_tokens, sampler=None, verbose=False, **kwargs):
            self.assertEqual(model.config.model_type, "llama")
            self.assertIs(tokenizer_obj, tokenizer)
            self.assertIn("user: hello", prompt)
            self.assertEqual(max_tokens, 32)
            self.assertEqual(sampler, {"temp": 0.5, "top_p": 0.9})
            self.assertEqual(kwargs.get("prefill_step_size"), 512)
            self.assertEqual(kwargs.get("kv_bits"), 4)
            self.assertEqual(kwargs.get("kv_group_size"), 32)
            self.assertEqual(kwargs.get("quantized_kv_start"), 256)
            self.assertFalse(verbose)
            return "hello world"

        def lm_stream_generate(model, tokenizer_obj, prompt, max_tokens, sampler=None, **kwargs):
            self.assertEqual(model.config.model_type, "llama")
            self.assertIs(tokenizer_obj, tokenizer)
            self.assertIn("user: hello", prompt)
            self.assertEqual(max_tokens, 32)
            self.assertEqual(sampler, {"temp": 0.5, "top_p": 0.9})
            self.assertEqual(kwargs.get("prefill_step_size"), 512)
            self.assertEqual(kwargs.get("kv_bits"), 4)
            self.assertEqual(kwargs.get("kv_group_size"), 32)
            self.assertEqual(kwargs.get("quantized_kv_start"), 256)
            yield FakeStreamItem("hello ", prompt_tps=123.4, generation_tps=45.6, peak_memory=3.21)
            yield FakeStreamItem("world", prompt_tps=123.4, generation_tps=45.6, peak_memory=3.21)

        mlx_lm_module.load = lm_load
        mlx_lm_module.generate = lm_generate
        mlx_lm_module.stream_generate = lm_stream_generate

        sample_utils_module = types.ModuleType("mlx_lm.sample_utils")

        def make_sampler(temp=None, top_p=None, **_kwargs):
            return {"temp": temp, "top_p": top_p}

        sample_utils_module.make_sampler = make_sampler

        mlx_vlm_module = types.ModuleType("mlx_vlm")

        def vlm_load(model_path):
            return FakeModel("qwen2_vl"), processor

        def vlm_apply_chat_template(processor_obj, config, messages, add_generation_prompt=True, num_images=0):
            self.assertIs(processor_obj, processor)
            self.assertEqual(config.model_type, "qwen2_vl")
            self.assertTrue(add_generation_prompt)
            self.assertEqual(num_images, 1)
            self.assertEqual(messages[0]["role"], "user")
            return "vision prompt"

        def vlm_generate(model, processor_obj, prompt, image=None, max_tokens=128, verbose=False, **kwargs):
            self.assertEqual(model.config.model_type, "qwen2_vl")
            self.assertIs(processor_obj, processor)
            self.assertEqual(prompt, "vision prompt")
            self.assertEqual(image, "data:image/png;base64,AAA")
            self.assertEqual(max_tokens, 24)
            self.assertEqual(kwargs.get("temperature"), 0.3)
            self.assertEqual(kwargs.get("top_p"), 0.8)
            self.assertEqual(kwargs.get("prefill_step_size"), 1024)
            self.assertEqual(kwargs.get("kv_bits"), 8)
            self.assertEqual(kwargs.get("kv_group_size"), 64)
            self.assertEqual(kwargs.get("quantized_kv_start"), 0)
            self.assertFalse(verbose)
            return types.SimpleNamespace(text="vision response", prompt_tokens=11, generation_tokens=4, peak_memory=4.56)

        def vlm_stream_generate(model, processor_obj, prompt, image=None, max_tokens=128, **kwargs):
            self.assertEqual(model.config.model_type, "qwen2_vl")
            self.assertIs(processor_obj, processor)
            self.assertEqual(prompt, "vision prompt")
            self.assertEqual(image, "data:image/png;base64,AAA")
            self.assertEqual(max_tokens, 24)
            self.assertEqual(kwargs.get("temperature"), 0.3)
            self.assertEqual(kwargs.get("top_p"), 0.8)
            self.assertEqual(kwargs.get("prefill_step_size"), 1024)
            self.assertEqual(kwargs.get("kv_bits"), 8)
            self.assertEqual(kwargs.get("kv_group_size"), 64)
            self.assertEqual(kwargs.get("quantized_kv_start"), 0)
            yield FakeStreamItem("vision ", prompt_tokens=11, generation_tokens=1, prompt_tps=222.2, generation_tps=33.3, peak_memory=4.56)
            yield FakeStreamItem("response", prompt_tokens=11, generation_tokens=2, prompt_tps=222.2, generation_tps=33.3, peak_memory=4.56)

        mlx_vlm_module.load = vlm_load
        mlx_vlm_module.generate = vlm_generate
        mlx_vlm_module.stream_generate = vlm_stream_generate
        mlx_vlm_module.apply_chat_template = vlm_apply_chat_template

        return {
            "mlx": mlx_module,
            "mlx_lm": mlx_lm_module,
            "mlx_lm.sample_utils": sample_utils_module,
            "mlx_vlm": mlx_vlm_module,
        }

    def _write_model_dir(self, *, vision: bool) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = {"architectures": ["FakeModel"], "model_type": "llama"}
        if vision:
            config["vision_config"] = {"enabled": True}
            config["architectures"] = ["Qwen2VLForConditionalGeneration"]
            config["model_type"] = "qwen2_vl"
        Path(tmpdir.name, "config.json").write_text(json.dumps(config), encoding="utf-8")
        return tmpdir.name

    def test_text_chat_completion_and_streaming(self) -> None:
        payload = {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 32,
            "prefill_step_size": 512,
            "kv_bits": 4,
            "kv_group_size": 32,
            "quantized_kv_start": 256,
        }
        model_dir = self._write_model_dir(vision=False)

        with patch.dict(sys.modules, self._fake_modules(), clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(state, dict(payload))
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(state.engine, "mlx_lm")
        self.assertFalse(state.supports_vision)
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(response["usage"]["prompt_tokens"], 3)
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertIn("timings", response)
        self.assertIn("prompt_ms", response["timings"])
        self.assertEqual(events[-1]["timings"]["peak_memory_gb"], 3.21)
        self.assertEqual(events[-1]["timings"]["prompt_per_second"], 123.4)
        self.assertEqual(events[-1]["timings"]["predicted_per_second"], 45.6)
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello ")
        self.assertEqual(events[2]["choices"][0]["delta"]["content"], "world")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("timings", events[-1])

    def test_vision_chat_completion_and_streaming(self) -> None:
        payload = {
            "model": "vision-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    ],
                }
            ],
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 24,
            "prefill_step_size": 1024,
            "kv_bits": 8,
            "kv_group_size": 64,
            "quantized_kv_start": 0,
        }
        model_dir = self._write_model_dir(vision=True)

        with patch.dict(sys.modules, self._fake_modules(), clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="vision-demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(state, dict(payload))
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(state.engine, "mlx_vlm")
        self.assertTrue(state.supports_vision)
        self.assertEqual(response["choices"][0]["message"]["content"], "vision response")
        self.assertEqual(response["usage"]["prompt_tokens"], 11)
        self.assertEqual(response["usage"]["completion_tokens"], 4)
        self.assertIn("timings", response)
        self.assertEqual(response["timings"]["peak_memory_gb"], 4.56)
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "vision ")
        self.assertEqual(events[2]["choices"][0]["delta"]["content"], "response")
        self.assertEqual(events[-1]["usage"]["prompt_tokens"], 11)
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("timings", events[-1])
        self.assertEqual(events[-1]["timings"]["peak_memory_gb"], 4.56)
        self.assertEqual(events[-1]["timings"]["prompt_per_second"], 222.2)
        self.assertEqual(events[-1]["timings"]["predicted_per_second"], 33.3)

    def test_text_model_rejects_image_inputs(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    ],
                }
            ]
        }
        model_dir = self._write_model_dir(vision=False)

        with patch.dict(sys.modules, self._fake_modules(), clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            with self.assertRaises(ValueError):
                self.driver.chat_completion(state, payload)
