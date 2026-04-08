#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from service_core.drivers.mnn import MnnLinuxDriver


class FakeContext:
    def __init__(self) -> None:
        self.prompt_len = 6
        self.gen_seq_len = 3
        self.prefill_us = 100_000
        self.decode_us = 60_000
        self.vision_us = 40_000


class FakeLlm:
    def __init__(self) -> None:
        self.loaded = False
        self.config = None
        self.context = FakeContext()
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


class FakeCvModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("MNN.cv")
        self.loaded_paths: list[str] = []

    def imread(self, path):
        self.loaded_paths.append(path)
        return types.SimpleNamespace(shape=(420, 420, 3), path=path)


class MnnDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = MnnLinuxDriver()

    def _fake_modules(self, fake_cv: FakeCvModule) -> dict[str, types.ModuleType]:
        llm_module = types.ModuleType("MNN.llm")

        def create(config_path):
            self.assertTrue(str(config_path).endswith("config.json"))
            return FakeLlm()

        llm_module.create = create
        mnn_module = types.ModuleType("MNN")
        mnn_module.llm = llm_module
        mnn_module.cv = fake_cv
        return {
            "MNN": mnn_module,
            "MNN.llm": llm_module,
            "MNN.cv": fake_cv,
        }

    def _write_model_dir(self, *, vision: bool, visual_asset: bool = False) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = {"is_visual": vision}
        Path(tmpdir.name, "config.json").write_text(json.dumps(config), encoding="utf-8")
        if visual_asset:
            Path(tmpdir.name, "visual.mnn").write_text("", encoding="utf-8")
        return tmpdir.name

    def test_text_chat_completion_and_streaming(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        fake_cv = FakeCvModule()
        payload = {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        }

        with patch.dict(sys.modules, self._fake_modules(fake_cv), clear=False):
            state = self.driver.load_model(
                model_path=model_dir,
                model_ref="demo",
                mmproj_path=None,
                ctx_size=None,
                load_options={"thread_num": 12},
            )
            response = self.driver.chat_completion(state, dict(payload))
            events = list(self.driver.stream_chat_completion(state, dict(payload)))

        self.assertTrue(state.model.loaded)
        self.assertEqual(state.model.config["thread_num"], 12)
        self.assertEqual(state.model.config["max_new_tokens"], 16)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello world again")
        self.assertEqual(response["usage"]["prompt_tokens"], 6)
        self.assertEqual(response["usage"]["completion_tokens"], 3)
        self.assertIn("predicted_per_second", response["timings"])
        self.assertIn("prompt_per_second", response["timings"])
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello world again")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("predicted_per_second", events[-1]["timings"])
        self.assertFalse(fake_cv.loaded_paths)

    def test_multimodal_chat_completion_and_streaming(self) -> None:
        model_dir = self._write_model_dir(vision=True)
        fake_cv = FakeCvModule()
        image_url = "data:image/png;base64," + base64.b64encode(b"fake-png-bytes").decode("ascii")
        payload = {
            "model": "vision-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 24,
        }

        with patch.dict(sys.modules, self._fake_modules(fake_cv), clear=False):
            state = self.driver.load_model(
                model_path=model_dir,
                model_ref="vision-demo",
                mmproj_path=None,
                ctx_size=None,
            )
            response = self.driver.chat_completion(state, dict(payload))
            events = list(self.driver.stream_chat_completion(state, dict(payload)))
            self.driver.unload_model(state)

        self.assertEqual(response["choices"][0]["message"]["content"], "vision answer")
        self.assertEqual(state.model.config["max_new_tokens"], 24)
        self.assertEqual(response["usage"]["prompt_tokens"], 8)
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertIn("predicted_per_second", response["timings"])
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "vision answer")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(len(fake_cv.loaded_paths), 2)

    def test_text_model_rejects_image_inputs(self) -> None:
        model_dir = self._write_model_dir(vision=False)
        fake_cv = FakeCvModule()

        with patch.dict(sys.modules, self._fake_modules(fake_cv), clear=False):
            state = self.driver.load_model(
                model_path=model_dir,
                model_ref="demo",
                mmproj_path=None,
                ctx_size=None,
            )
            with self.assertRaises(ValueError):
                self.driver.chat_completion(
                    state,
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "describe"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,"
                                            + base64.b64encode(b"fake-png-bytes").decode("ascii")
                                        },
                                    },
                                ],
                            }
                        ]
                    },
                )

    def test_visual_asset_enables_image_inputs(self) -> None:
        model_dir = self._write_model_dir(vision=False, visual_asset=True)
        fake_cv = FakeCvModule()
        image_url = "data:image/png;base64," + base64.b64encode(b"fake-png-bytes").decode("ascii")

        with patch.dict(sys.modules, self._fake_modules(fake_cv), clear=False):
            state = self.driver.load_model(
                model_path=model_dir,
                model_ref="asset-demo",
                mmproj_path=None,
                ctx_size=None,
            )
            response = self.driver.chat_completion(
                state,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "max_tokens": 12,
                },
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "vision answer")
        self.assertEqual(len(fake_cv.loaded_paths), 1)
