#!/usr/bin/env python3

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from service_core.drivers.mlx import MlxMacDriver


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def encode(self, text):
        return text.split()


class FakeStreamItem:
    def __init__(self, text: str) -> None:
        self.text = text


class MlxDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = MlxMacDriver()

    def _fake_modules(self) -> dict[str, types.ModuleType]:
        mlx_module = types.ModuleType("mlx")

        mlx_lm_module = types.ModuleType("mlx_lm")
        tokenizer = FakeTokenizer()

        def load(model_path):
            return object(), tokenizer

        def generate(model, tokenizer_obj, prompt, max_tokens, sampler=None, verbose=False):
            self.assertIsNotNone(model)
            self.assertIs(tokenizer_obj, tokenizer)
            self.assertIn("user: hello", prompt)
            self.assertEqual(max_tokens, 32)
            self.assertEqual(sampler, {"temp": 0.5, "top_p": 0.9})
            self.assertFalse(verbose)
            return "hello world"

        def stream_generate(model, tokenizer_obj, prompt, max_tokens, sampler=None):
            self.assertIsNotNone(model)
            self.assertIs(tokenizer_obj, tokenizer)
            self.assertIn("user: hello", prompt)
            self.assertEqual(max_tokens, 32)
            self.assertEqual(sampler, {"temp": 0.5, "top_p": 0.9})
            yield FakeStreamItem("hello ")
            yield FakeStreamItem("world")

        mlx_lm_module.load = load
        mlx_lm_module.generate = generate
        mlx_lm_module.stream_generate = stream_generate

        sample_utils_module = types.ModuleType("mlx_lm.sample_utils")

        def make_sampler(temp=None, top_p=None, **_kwargs):
            return {"temp": temp, "top_p": top_p}

        sample_utils_module.make_sampler = make_sampler

        return {
            "mlx": mlx_module,
            "mlx_lm": mlx_lm_module,
            "mlx_lm.sample_utils": sample_utils_module,
        }

    def test_chat_completion_and_streaming(self) -> None:
        payload = {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 32,
        }

        with patch.dict(sys.modules, self._fake_modules(), clear=False):
            state = self.driver.load_model(model_path="/tmp/demo", model_ref="demo", mmproj_path=None, ctx_size=None)
            response = self.driver.chat_completion(state, dict(payload))
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(response["usage"]["prompt_tokens"], 3)
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello ")
        self.assertEqual(events[2]["choices"][0]["delta"]["content"], "world")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_rejects_multimodal_messages_in_phase_one(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "describe"}],
                }
            ]
        }

        with patch.dict(sys.modules, self._fake_modules(), clear=False):
            state = self.driver.load_model(model_path="/tmp/demo", model_ref="demo", mmproj_path=None, ctx_size=None)
            with self.assertRaises(ValueError):
                self.driver.chat_completion(state, payload)
