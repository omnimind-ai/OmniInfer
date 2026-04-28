#!/usr/bin/env python3
"""Embedded backend drivers (MLX, MNN): model loading, chat completion, streaming."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.drivers.mlx import MlxMacDriver
from service_core.drivers.mnn import MnnLinuxDriver

_FAKE_IMAGE_URL = "data:image/png;base64," + base64.b64encode(b"fake-png-bytes").decode("ascii")


# ═══════════════════════════════════════════════════════════════════════════════
# MLX fakes
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def encode(self, text):
        return text.split()


class _FakeProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer


class _FakeModel:
    def __init__(self, model_type: str) -> None:
        self.config = types.SimpleNamespace(model_type=model_type)


class _FakeStreamItem:
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


class _MlxCalls:
    """Records arguments passed to fake MLX generate functions."""

    def __init__(self) -> None:
        self.generate: list[dict] = []
        self.stream_generate: list[dict] = []
        self.vlm_generate: list[dict] = []
        self.vlm_stream_generate: list[dict] = []
        self.vlm_apply_chat_template: list[dict] = []


def _mlx_fake_modules() -> tuple[dict[str, types.ModuleType], _MlxCalls]:
    calls = _MlxCalls()
    tokenizer = _FakeTokenizer()
    processor = _FakeProcessor(tokenizer)

    mlx_module = types.ModuleType("mlx")
    mlx_lm_module = types.ModuleType("mlx_lm")

    def lm_load(model_path):
        return _FakeModel("llama"), tokenizer

    def lm_generate(model, tokenizer_obj, prompt, max_tokens, sampler=None, verbose=False, **kwargs):
        calls.generate.append({"prompt": prompt, "max_tokens": max_tokens, "sampler": sampler, "verbose": verbose, **kwargs})
        return "hello world"

    def lm_stream_generate(model, tokenizer_obj, prompt, max_tokens, sampler=None, **kwargs):
        calls.stream_generate.append({"prompt": prompt, "max_tokens": max_tokens, "sampler": sampler, **kwargs})
        yield _FakeStreamItem("hello ", prompt_tps=123.4, generation_tps=45.6, peak_memory=3.21)
        yield _FakeStreamItem("world", prompt_tps=123.4, generation_tps=45.6, peak_memory=3.21)

    mlx_lm_module.load = lm_load
    mlx_lm_module.generate = lm_generate
    mlx_lm_module.stream_generate = lm_stream_generate

    sample_utils_module = types.ModuleType("mlx_lm.sample_utils")
    sample_utils_module.make_sampler = lambda temp=None, top_p=None, **_kw: {"temp": temp, "top_p": top_p}

    mlx_vlm_module = types.ModuleType("mlx_vlm")

    def vlm_load(model_path):
        return _FakeModel("qwen2_vl"), processor

    def vlm_apply_chat_template(processor_obj, config, messages, add_generation_prompt=True, num_images=0):
        calls.vlm_apply_chat_template.append({"num_images": num_images, "add_generation_prompt": add_generation_prompt})
        return "vision prompt"

    def vlm_generate(model, processor_obj, prompt, image=None, max_tokens=128, verbose=False, **kwargs):
        calls.vlm_generate.append({"prompt": prompt, "image": image, "max_tokens": max_tokens, "verbose": verbose, **kwargs})
        return types.SimpleNamespace(text="vision response", prompt_tokens=11, generation_tokens=4, peak_memory=4.56)

    def vlm_stream_generate(model, processor_obj, prompt, image=None, max_tokens=128, **kwargs):
        calls.vlm_stream_generate.append({"prompt": prompt, "image": image, "max_tokens": max_tokens, **kwargs})
        yield _FakeStreamItem("vision ", prompt_tokens=11, generation_tokens=1, prompt_tps=222.2, generation_tps=33.3, peak_memory=4.56)
        yield _FakeStreamItem("response", prompt_tokens=11, generation_tokens=2, prompt_tps=222.2, generation_tps=33.3, peak_memory=4.56)

    mlx_vlm_module.load = vlm_load
    mlx_vlm_module.generate = vlm_generate
    mlx_vlm_module.stream_generate = vlm_stream_generate
    mlx_vlm_module.apply_chat_template = vlm_apply_chat_template

    modules = {
        "mlx": mlx_module,
        "mlx_lm": mlx_lm_module,
        "mlx_lm.sample_utils": sample_utils_module,
        "mlx_vlm": mlx_vlm_module,
    }
    return modules, calls


# ═══════════════════════════════════════════════════════════════════════════════
# MNN fakes
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeContext:
    def __init__(self) -> None:
        self.prompt_len = 6
        self.gen_seq_len = 3
        self.prefill_us = 100_000
        self.decode_us = 60_000
        self.vision_us = 40_000


class _FakeLlm:
    def __init__(self) -> None:
        self.loaded = False
        self.config = None
        self.context = _FakeContext()
        self.last_prompt = None

    def set_config(self, config):
        self.config = dict(config)
        return True

    def load(self):
        self.loaded = True

    def apply_chat_template(self, messages):
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def tokenizer_encode(self, prompt):
        if isinstance(prompt, dict):
            return [101, 102, 103, 104, 105, 106]
        return str(prompt).split()

    def response(self, prompt, stream=False, max_new_tokens=-1):
        self.last_prompt = prompt
        if isinstance(prompt, dict):
            self.context.prompt_len = 8
            self.context.gen_seq_len = 2
            return "vision answer"
        self.context.prompt_len = 6
        self.context.gen_seq_len = 3
        return "hello world again"


class _FakeCvModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("MNN.cv")
        self.loaded_paths: list[str] = []

    def imread(self, path):
        self.loaded_paths.append(path)
        return types.SimpleNamespace(shape=(420, 420, 3), path=path)


class _MnnCalls:
    """Records arguments passed to fake MNN factory."""

    def __init__(self) -> None:
        self.create_paths: list[str] = []


def _mnn_fake_modules(fake_cv: _FakeCvModule) -> tuple[dict[str, types.ModuleType], _MnnCalls]:
    calls = _MnnCalls()
    llm_module = types.ModuleType("MNN.llm")

    def create(config_path):
        calls.create_paths.append(str(config_path))
        return _FakeLlm()

    llm_module.create = create
    mnn_module = types.ModuleType("MNN")
    mnn_module.llm = llm_module
    mnn_module.cv = fake_cv
    modules = {
        "MNN": mnn_module,
        "MNN.llm": llm_module,
        "MNN.cv": fake_cv,
    }
    return modules, calls


# ═══════════════════════════════════════════════════════════════════════════════
# MLX driver tests
# ═══════════════════════════════════════════════════════════════════════════════


class MlxDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = MlxMacDriver()

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

    # --- text model: chat completion ---

    def test_text_chat_completion(self) -> None:
        payload = {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.5, "top_p": 0.9, "max_tokens": 32,
            "prefill_step_size": 512, "kv_bits": 4, "kv_group_size": 32, "quantized_kv_start": 256,
        }
        model_dir = self._write_model_dir(vision=False)
        modules, calls = _mlx_fake_modules()

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(state, dict(payload))

        self.assertEqual(state.engine, "mlx_lm")
        self.assertFalse(state.supports_vision)

        gen = calls.generate[0]
        self.assertIn("user: hello", gen["prompt"])
        self.assertEqual(gen["max_tokens"], 32)
        self.assertEqual(gen["sampler"], {"temp": 0.5, "top_p": 0.9})
        self.assertEqual(gen["prefill_step_size"], 512)
        self.assertEqual(gen["kv_bits"], 4)

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(response["usage"]["prompt_tokens"], 3)
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertIn("timings", response)

    def test_text_streaming(self) -> None:
        payload = {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.5, "top_p": 0.9, "max_tokens": 32,
            "prefill_step_size": 512, "kv_bits": 4, "kv_group_size": 32, "quantized_kv_start": 256,
        }
        model_dir = self._write_model_dir(vision=False)
        modules, _ = _mlx_fake_modules()

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello ")
        self.assertEqual(events[2]["choices"][0]["delta"]["content"], "world")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[-1]["timings"]["peak_memory_gb"], 3.21)
        self.assertEqual(events[-1]["timings"]["prompt_per_second"], 123.4)
        self.assertEqual(events[-1]["timings"]["predicted_per_second"], 45.6)

    # --- vision model: chat completion ---

    def test_vision_chat_completion(self) -> None:
        payload = {
            "model": "vision-model",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]}],
            "temperature": 0.3, "top_p": 0.8, "max_tokens": 24,
            "prefill_step_size": 1024, "kv_bits": 8, "kv_group_size": 64, "quantized_kv_start": 0,
        }
        model_dir = self._write_model_dir(vision=True)
        modules, calls = _mlx_fake_modules()

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="vision-demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(state, dict(payload))

        self.assertEqual(state.engine, "mlx_vlm")
        self.assertTrue(state.supports_vision)

        gen = calls.vlm_generate[0]
        self.assertEqual(gen["prompt"], "vision prompt")
        self.assertEqual(gen["image"], "data:image/png;base64,AAA")
        self.assertEqual(gen["max_tokens"], 24)
        self.assertEqual(gen["temperature"], 0.3)
        self.assertEqual(gen["prefill_step_size"], 1024)
        self.assertEqual(gen["kv_bits"], 8)
        self.assertEqual(calls.vlm_apply_chat_template[0]["num_images"], 1)

        self.assertEqual(response["choices"][0]["message"]["content"], "vision response")
        self.assertEqual(response["usage"]["prompt_tokens"], 11)
        self.assertEqual(response["usage"]["completion_tokens"], 4)
        self.assertEqual(response["timings"]["peak_memory_gb"], 4.56)

    def test_vision_streaming(self) -> None:
        payload = {
            "model": "vision-model",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]}],
            "temperature": 0.3, "top_p": 0.8, "max_tokens": 24,
            "prefill_step_size": 1024, "kv_bits": 8, "kv_group_size": 64, "quantized_kv_start": 0,
        }
        model_dir = self._write_model_dir(vision=True)
        modules, _ = _mlx_fake_modules()

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="vision-demo", mmproj_path=None, ctx_size=None)
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "vision ")
        self.assertEqual(events[2]["choices"][0]["delta"]["content"], "response")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[-1]["timings"]["prompt_per_second"], 222.2)
        self.assertEqual(events[-1]["timings"]["predicted_per_second"], 33.3)

    # --- error cases ---

    def test_text_model_rejects_image_inputs(self) -> None:
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]}]
        }
        model_dir = self._write_model_dir(vision=False)
        modules, _ = _mlx_fake_modules()

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            with self.assertRaises(ValueError):
                self.driver.chat_completion(state, payload)

    def test_load_rejects_mmproj(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        modules, _ = _mlx_fake_modules()
        with patch.dict(sys.modules, modules, clear=False):
            with self.assertRaises(ValueError):
                self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path="/tmp/mmproj.gguf", ctx_size=None)

    def test_load_rejects_ctx_size(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        modules, _ = _mlx_fake_modules()
        with patch.dict(sys.modules, modules, clear=False):
            with self.assertRaises(ValueError):
                self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=4096)

    def test_unload_model(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        modules, _ = _mlx_fake_modules()
        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            self.driver.unload_model(state)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# MNN driver tests
# ═══════════════════════════════════════════════════════════════════════════════


class MnnDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = MnnLinuxDriver()

    def _write_model_dir(self, *, vision: bool, visual_asset: bool = False) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = {"is_visual": vision}
        Path(tmpdir.name, "config.json").write_text(json.dumps(config), encoding="utf-8")
        if visual_asset:
            Path(tmpdir.name, "visual.mnn").write_text("", encoding="utf-8")
        return tmpdir.name

    # --- text model ---

    def test_text_chat_completion(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        fake_cv = _FakeCvModule()
        payload = {"model": "demo-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
        modules, calls = _mnn_fake_modules(fake_cv)

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None, load_options={"thread_num": 12})
            response = self.driver.chat_completion(state, dict(payload))

        self.assertTrue(state.model.loaded)
        self.assertEqual(state.model.config["thread_num"], 12)
        self.assertEqual(state.model.config["max_new_tokens"], 16)
        self.assertTrue(calls.create_paths[0].endswith("config.json"))
        self.assertEqual(response["choices"][0]["message"]["content"], "hello world again")
        self.assertEqual(response["usage"]["prompt_tokens"], 3)
        self.assertEqual(response["usage"]["completion_tokens"], 3)
        self.assertIn("predicted_per_second", response["timings"])
        self.assertFalse(fake_cv.loaded_paths)

    def test_text_streaming(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        fake_cv = _FakeCvModule()
        payload = {"model": "demo-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
        modules, _ = _mnn_fake_modules(fake_cv)

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello world again")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("predicted_per_second", events[-1]["timings"])

    # --- multimodal model ---

    def test_multimodal_chat_completion(self) -> None:
        model_dir = self._write_model_dir(vision=True)
        fake_cv = _FakeCvModule()
        image_url = _FAKE_IMAGE_URL
        payload = {"model": "vision-model", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}], "max_tokens": 24}
        modules, _ = _mnn_fake_modules(fake_cv)

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="vision-demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(state, dict(payload))
            self.driver.unload_model(state)

        self.assertEqual(response["choices"][0]["message"]["content"], "vision answer")
        self.assertEqual(state.model.config["max_new_tokens"], 24)
        self.assertEqual(response["usage"]["prompt_tokens"], 3)
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertIn("predicted_per_second", response["timings"])
        self.assertEqual(len(fake_cv.loaded_paths), 1)

    def test_multimodal_streaming(self) -> None:
        model_dir = self._write_model_dir(vision=True)
        fake_cv = _FakeCvModule()
        payload = {"model": "vision-model", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": _FAKE_IMAGE_URL}},
        ]}], "max_tokens": 24}
        modules, _ = _mnn_fake_modules(fake_cv)

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="vision-demo", mmproj_path=None, ctx_size=None)
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "vision answer")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_text_model_rejects_image_inputs(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        fake_cv = _FakeCvModule()
        image_url = _FAKE_IMAGE_URL
        modules, _ = _mnn_fake_modules(fake_cv)

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="demo", mmproj_path=None, ctx_size=None)
            with self.assertRaises(ValueError):
                self.driver.chat_completion(
                    state,
                    {"messages": [{"role": "user", "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]}]},
                )

    def test_visual_asset_enables_image_inputs(self) -> None:
        model_dir = self._write_model_dir(vision=False, visual_asset=True)
        fake_cv = _FakeCvModule()
        image_url = _FAKE_IMAGE_URL
        modules, _ = _mnn_fake_modules(fake_cv)

        with patch.dict(sys.modules, modules, clear=False):
            state = self.driver.load_model(model_path=model_dir, model_ref="asset-demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(
                state,
                {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}], "max_tokens": 12},
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "vision answer")
        self.assertEqual(len(fake_cv.loaded_paths), 1)


if __name__ == "__main__":
    unittest.main()
