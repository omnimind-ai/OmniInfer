from __future__ import annotations

from service_core.backends.base import BackendTemplate


WINDOWS_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-cpu",
        label="llama.cpp cpu",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-cpu",
        launcher_name="llama-server.exe",
        description="llama.cpp CPU backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu"),
        env_prefix="OMNIINFER_LLAMA_CPP_CPU",
    ),
    BackendTemplate(
        id="llama.cpp-cuda",
        label="llama.cpp CUDA",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-cuda",
        launcher_name="llama-server.exe",
        description="llama.cpp CUDA backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "cuda"),
        env_prefix="OMNIINFER_LLAMA_CPP_CUDA",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-vulkan",
        label="llama.cpp Vulkan",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-vulkan",
        launcher_name="llama-server.exe",
        description="llama.cpp Vulkan backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "vulkan"),
        env_prefix="OMNIINFER_LLAMA_CPP_VULKAN",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-windows-arm64",
        label="llama.cpp Windows arm64",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-windows-arm64",
        launcher_name="llama-server.exe",
        description="llama.cpp Windows arm64 CPU backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu", "windows", "arm64"),
        env_prefix="OMNIINFER_LLAMA_CPP_WINDOWS_ARM64",
    ),
    BackendTemplate(
        id="llama.cpp-sycl",
        label="llama.cpp SYCL",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-sycl",
        launcher_name="llama-server.exe",
        description="llama.cpp Windows SYCL backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "sycl", "intel", "windows"),
        env_prefix="OMNIINFER_LLAMA_CPP_SYCL",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-hip",
        label="llama.cpp HIP",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-hip",
        launcher_name="llama-server.exe",
        description="llama.cpp Windows HIP backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "hip", "amd", "windows"),
        env_prefix="OMNIINFER_LLAMA_CPP_HIP",
        default_ngl="999",
    ),
)


MAC_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-mac",
        label="llama.cpp Metal",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-mac",
        launcher_name="llama-server",
        description="llama.cpp Metal backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "metal", "apple", "shared-memory"),
        env_prefix="OMNIINFER_LLAMA_CPP_MAC",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-mac-intel",
        label="llama.cpp macOS Intel",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-mac-intel",
        launcher_name="llama-server",
        description="llama.cpp macOS Intel x64 backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu", "macos", "x64", "intel"),
        env_prefix="OMNIINFER_LLAMA_CPP_MAC_INTEL",
    ),
)


LINUX_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-linux",
        label="llama.cpp Linux",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-linux",
        launcher_name="llama-server",
        description="llama.cpp Linux backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu", "linux"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-linux-rocm",
        label="llama.cpp Linux ROCm",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-linux-rocm",
        fallback_runtime_dir_names=("llama.cpp-linux-ROCm",),
        launcher_name="llama-server",
        description="llama.cpp Linux ROCm backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "rocm", "linux"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX_ROCM",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-linux-vulkan",
        label="llama.cpp Linux Vulkan",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-linux-vulkan",
        launcher_name="llama-server",
        description="llama.cpp Linux Vulkan backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "gpu", "vulkan", "linux"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX_VULKAN",
        default_ngl="999",
    ),
    BackendTemplate(
        id="llama.cpp-linux-s390x",
        label="llama.cpp Linux s390x",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-linux-s390x",
        launcher_name="llama-server",
        description="llama.cpp Linux s390x CPU backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "cpu", "linux", "s390x"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX_S390X",
    ),
    BackendTemplate(
        id="llama.cpp-linux-openvino",
        label="llama.cpp Linux OpenVINO",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-linux-openvino",
        launcher_name="llama-server",
        description="llama.cpp Linux OpenVINO backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "linux", "openvino", "intel"),
        env_prefix="OMNIINFER_LLAMA_CPP_LINUX_OPENVINO",
    ),
)


ANDROID_LLAMA_CPP_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="llama.cpp-android",
        label="llama.cpp Android",
        family="llama.cpp",
        runtime_dir_name="llama.cpp-android",
        launcher_name="llama-server",
        description="llama.cpp Android backend managed by OmniInfer",
        capabilities=("chat", "vision", "stream", "android", "mobile"),
        env_prefix="OMNIINFER_LLAMA_CPP_ANDROID",
        default_ngl="999",
    ),
)
