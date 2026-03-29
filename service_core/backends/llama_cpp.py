from __future__ import annotations

from service_core.backends.base import BackendTemplate


WINDOWS_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-cpu",
        label="llama.cpp cpu",
        runtime_dir_name="llama.cpp-cpu",
        server_binary_name="llama-server.exe",
        description="llama.cpp CPU backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu"),
        env_prefix="OMNIINFER_LLAMA_CPP_CPU",
    ),
    BackendTemplate(
        id="llama.cpp-cuda",
        label="llama.cpp CUDA",
        runtime_dir_name="llama.cpp-cuda",
        server_binary_name="llama-server.exe",
        description="llama.cpp CUDA backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "cuda"),
        env_prefix="OMNIINFER_LLAMA_CPP_CUDA",
        default_ngl="999",
    ),
)


MAC_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-mac",
        label="llama.cpp Metal",
        runtime_dir_name="llama.cpp-mac",
        server_binary_name="llama-server",
        description="llama.cpp Metal backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "metal", "apple", "shared-memory"),
        env_prefix="OMNIINFER_LLAMA_CPP_MAC",
        default_ngl="999",
    ),
)


LINUX_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-linux",
        label="llama.cpp Linux",
        runtime_dir_name="llama.cpp-linux",
        server_binary_name="llama-server",
        description="llama.cpp Linux backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu", "linux"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-linux-rocm",
        label="llama.cpp Linux ROCm",
        runtime_dir_name="llama.cpp-linux-rocm",
        fallback_runtime_dir_names=("llama.cpp-linux-ROCm",),
        server_binary_name="llama-server",
        description="llama.cpp Linux ROCm backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "rocm", "linux"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX_ROCM",
        default_ngl="999",
    ),
)


ANDROID_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-android",
        label="llama.cpp Android",
        runtime_dir_name="llama.cpp-android",
        server_binary_name="llama-server",
        description="llama.cpp Android backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "android", "mobile"),
        env_prefix="OMNIINFER_LLAMA_CPP_ANDROID",
        default_ngl="999",
    ),
)
