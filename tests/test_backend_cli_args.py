#!/usr/bin/env python3

from __future__ import annotations

import unittest

from service_core.backend_cli_args import (
    parse_backend_chat_extra_args,
    parse_backend_load_extra_args,
)
from service_core.backends.base import BackendSpec


def make_backend(backend_id: str, family: str) -> BackendSpec:
    return BackendSpec(
        id=backend_id,
        label=backend_id,
        family=family,
        runtime_dir=".",
        launcher_path=None,
        models_dir=None,
        catalog_url=None,
        description="",
        capabilities=[],
    )


class BackendCliArgParserTests(unittest.TestCase):
    def test_llama_cpp_load_args_passthrough_with_ctx_size(self) -> None:
        backend = make_backend("llama.cpp-vulkan", "llama.cpp")

        parsed = parse_backend_load_extra_args(
            backend,
            ["-ngl", "99", "-t", "8", "--threads-batch", "4", "-c", "4096"],
        )

        self.assertEqual(parsed.ctx_size, 4096)
        self.assertEqual(parsed.launch_args, ["-ngl", "99", "-t", "8", "--threads-batch", "4"])

    def test_turboquant_chat_args_use_llama_compatible_parser(self) -> None:
        backend = make_backend("turboquant-mac", "turboquant")

        parsed = parse_backend_chat_extra_args(
            backend,
            ["-n", "128", "--top-k", "40", "--stop", "<END>", "--stop", "</END>", "-e"],
        )

        self.assertEqual(parsed.request_overrides["max_tokens"], 128)
        self.assertEqual(parsed.request_overrides["top_k"], 40)
        self.assertEqual(parsed.request_overrides["ignore_eos"], True)
        self.assertEqual(parsed.request_overrides["stop"], ["<END>", "</END>"])

    def test_mlx_chat_args_accept_image_again(self) -> None:
        backend = make_backend("mlx-mac", "mlx-lm")

        parsed = parse_backend_chat_extra_args(
            backend,
            ["--image", "demo.png", "--temperature", "0.3", "--top-p", "0.8", "--stop", "<END>"],
        )

        self.assertEqual(parsed.image, "demo.png")
        self.assertEqual(parsed.request_overrides["temperature"], 0.3)
        self.assertEqual(parsed.request_overrides["top_p"], 0.8)
        self.assertEqual(parsed.request_overrides["stop"], ["<END>"])


if __name__ == "__main__":
    unittest.main()
