from __future__ import annotations

from service_core.backends.base import BackendTemplate


LINUX_IK_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="ik_llama.cpp-linux",
        label="ik_llama.cpp Linux",
        family="llama.cpp",
        runtime_dir_name="ik_llama.cpp-linux",
        launcher_name="llama-server",
        description="ik_llama.cpp Linux CPU backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu", "linux"),
        env_prefix="OMNIINFER_IK_LLAMA_CPP_LINUX",
    ),
    BackendTemplate(
        id="ik_llama.cpp-linux-cuda",
        label="ik_llama.cpp Linux CUDA",
        family="llama.cpp",
        runtime_dir_name="ik_llama.cpp-linux-cuda",
        launcher_name="llama-server",
        description="ik_llama.cpp Linux CUDA backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "cuda", "linux"),
        env_prefix="OMNIINFER_IK_LLAMA_CPP_LINUX_CUDA",
        default_ngl="999",
    ),
)


WINDOWS_IK_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="ik_llama.cpp-cpu",
        label="ik_llama.cpp CPU",
        family="llama.cpp",
        runtime_dir_name="ik_llama.cpp-cpu",
        launcher_name="llama-server.exe",
        description="ik_llama.cpp CPU backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu"),
        env_prefix="OMNIINFER_IK_LLAMA_CPP_CPU",
    ),
    BackendTemplate(
        id="ik_llama.cpp-cuda",
        label="ik_llama.cpp CUDA",
        family="llama.cpp",
        runtime_dir_name="ik_llama.cpp-cuda",
        launcher_name="llama-server.exe",
        description="ik_llama.cpp CUDA backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "cuda"),
        env_prefix="OMNIINFER_IK_LLAMA_CPP_CUDA",
        default_ngl="999",
    ),
)
