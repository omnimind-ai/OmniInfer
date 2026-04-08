from __future__ import annotations

from service_core.backends.base import BackendTemplate


LINUX_MNN_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="mnn-linux",
        label="MNN Linux",
        family="mnn",
        runtime_dir_name="mnn-linux",
        launcher_name=None,
        description="Embedded MNN LLM/VLM backend managed directly by OmniInfer on Linux",
        capabilities=("chat", "vision", "stream", "cpu", "linux", "embedded", "mnn"),
        env_prefix="OMNIINFER_MNN_LINUX",
        runtime_mode="embedded",
        model_artifact="path",
        supports_mmproj=False,
        supports_ctx_size=False,
        python_modules=("MNN", "MNN.llm", "MNN.cv"),
        external_server_protocol=None,
    ),
)
