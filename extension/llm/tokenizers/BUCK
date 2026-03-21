load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_executorch_supported_platforms", "runtime")
load("@fbsource//xplat/executorch/third-party:glob_defs.bzl", "subdir_glob")

oncall("executorch")

PLATFORMS = get_executorch_supported_platforms()

runtime.cxx_library(
    name = "headers",
    header_namespace = "",
    exported_headers = subdir_glob([
        ("include", "pytorch/tokenizers/*.h"),
    ]),
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    exported_deps = [
        "fbsource//third-party/re2:re2",
    ],
    exported_external_deps = [
        "sentencepiece",
    ],
)

runtime.cxx_library(
    name = "regex",
    srcs = [
        "src/re2_regex.cpp",
        "src/regex.cpp",
    ],
    header_namespace = "",
    platforms = PLATFORMS,
    visibility = ["//pytorch/tokenizers/..."],
    exported_deps = [
        "fbsource//third-party/re2:re2",
        ":headers",
    ],
)

runtime.cxx_library(
    name = "regex_lookahead",
    srcs = [
        "src/pcre2_regex.cpp",
        "src/regex_lookahead.cpp",
        "src/std_regex.cpp",
    ],
    header_namespace = "",
    compiler_flags = [
        "-Wno-global-constructors",
        "-Wno-missing-prototypes",
    ],
    # Making sure this library is not being stripped by linker.
    # @lint-ignore BUCKLINT: Avoid link_whole=True
    link_whole = True,
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    exported_deps = [
        ":headers",
        ":regex",
    ],
    exported_external_deps = [
        "pcre2",
    ],
)

runtime.cxx_library(
    name = "bpe_tokenizer_base",
    srcs = [
        "src/bpe_tokenizer_base.cpp",
    ],
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
    ],
    exported_deps = [
        "fbsource//third-party/re2:re2",
        ":headers",
    ],
)

runtime.cxx_library(
    name = "sentencepiece",
    srcs = [
        "src/sentencepiece.cpp",
    ],
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    deps = [
        ":regex",
    ],
    exported_deps = [
        ":headers",
    ],
    exported_external_deps = [
        "sentencepiece",
        "abseil-cpp",
    ],
)

runtime.cxx_library(
    name = "tiktoken",
    srcs = [
        "src/tiktoken.cpp",
    ],
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    deps = [
        ":regex",
    ],
    exported_deps = [
        "fbsource//third-party/re2:re2",
        ":bpe_tokenizer_base",
        ":headers",
    ],
)

runtime.cxx_library(
    name = "hf_tokenizer",
    srcs = [
        "src/hf_tokenizer.cpp",
        "src/normalizer.cpp",
        "src/post_processor.cpp",
        "src/pre_tokenizer.cpp",
        "src/token_decoder.cpp",
    ],
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    deps = [
        "fbsource//third-party/nlohmann-json:nlohmann-json",
        ":regex",
    ],
    exported_deps = [
        "fbsource//third-party/re2:re2",
        ":bpe_tokenizer_base",
        ":headers",
        "//pytorch/tokenizers/third-party:unicode",
    ],
    exported_external_deps = [
        "nlohmann_json",
    ],
)

runtime.cxx_library(
    name = "llama2c_tokenizer",
    srcs = [
        "src/llama2c_tokenizer.cpp",
    ],
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    exported_deps = [
        ":headers",
    ],
)

runtime.cxx_library(
    name = "tekken",
    srcs = [
        "src/tekken.cpp",
    ],
    platforms = PLATFORMS,
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    deps = [
        ":regex",
    ],
    exported_deps = [
        "fbsource//third-party/re2:re2",
        ":bpe_tokenizer_base",
        ":headers",
    ],
    exported_external_deps = [
        "nlohmann_json",
    ],
)

runtime.cxx_python_extension(
    name = "pytorch_tokenizers_cpp",
    srcs = [
        "src/python_bindings.cpp",
    ],
    base_module = "pytorch_tokenizers",
    visibility = [
        "//pytorch/tokenizers/...",
        "@EXECUTORCH_CLIENTS",
    ],
    deps = [
        ":hf_tokenizer",
        ":llama2c_tokenizer",
        ":sentencepiece",
        ":tekken",
        ":tiktoken",
    ],
    external_deps = [
        "pybind11",
    ],
)
