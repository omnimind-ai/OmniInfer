load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.python_library(
        name = "tokenizers",
        srcs = [
            "__init__.py",
            "constants.py",
            "hf_tokenizer.py",
            "llama2c.py",
            "tiktoken.py",
        ],
        base_module = "pytorch_tokenizers",
        visibility = ["PUBLIC"],
        _is_external_target = True,
        deps = [
            "fbsource//third-party/pypi/sentencepiece:sentencepiece",
            "fbsource//third-party/pypi/tiktoken:tiktoken",
            "fbsource//third-party/pypi/tokenizers:tokenizers",
            "//pytorch/tokenizers:pytorch_tokenizers_cpp",  # @manual
        ],
    )
