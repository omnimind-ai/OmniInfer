from __future__ import annotations

from service_core.backends.base import BackendTemplate


LINUX_VLLM_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="vllm-linux-cuda",
        label="vLLM Linux CUDA",
        family="vllm",
        runtime_dir_name="vllm-linux-cuda",
        launcher_name="vllm",
        description="vLLM OpenAI-compatible server backend managed by OmniInfer on Linux CUDA",
        capabilities=("chat", "stream", "gpu", "cuda", "linux", "openai-compatible"),
        env_prefix="OMNIINFER_VLLM_LINUX_CUDA",
        model_artifact="reference",
        supports_mmproj=False,
        external_server_protocol="vllm-openai-server",
        log_file_name="vllm-server.log",
    ),
)
