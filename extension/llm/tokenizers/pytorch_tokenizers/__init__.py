# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PyTorch Tokenizers - Fast tokenizers for PyTorch

This package provides Python bindings for fast C++ tokenizer implementations
including HuggingFace, TikToken, Llama2C, and SentencePiece tokenizers.
"""
# @lint-ignore-every LICENSELINT

from typing import Optional

from .hf_tokenizer import HuggingFaceTokenizer
from .llama2c import Llama2cTokenizer
from .tiktoken import TiktokenTokenizer

__version__ = "0.1.0"

try:
    from .pytorch_tokenizers_cpp import (  # @manual=//pytorch/tokenizers:pytorch_tokenizers_cpp
        Error,
        HFTokenizer as CppHFTokenizer,
        Llama2cTokenizer as CppLlama2cTokenizer,
        SPTokenizer as CppSPTokenizer,
        Tiktoken as CppTiktoken,
        TokenIndex,
        Tokenizer,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import C++ tokenizer bindings: {e}. "
        "Make sure the package was built correctly with pybind11."
    ) from e


def get_tokenizer(tokenizer_path: str, tokenizer_config_path: Optional[str] = None):
    if tokenizer_path.endswith(".json"):
        tokenizer = HuggingFaceTokenizer(tokenizer_path, tokenizer_config_path)
    else:
        try:
            tokenizer = Llama2cTokenizer(model_path=str(tokenizer_path))
        except Exception:
            print("Using Tiktokenizer")
            tokenizer = TiktokenTokenizer(model_path=str(tokenizer_path))
    return tokenizer


__all__ = [
    "CppHFTokenizer",
    "CppLlama2cTokenizer",
    "CppSPTokenizer",
    "CppTiktoken",
    "Error",
    "HFTokenizer",
    "Llama2cTokenizer",
    "TiktokenTokenizer",
    "TokenIndex",
    "Tokenizer",
]
