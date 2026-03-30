from __future__ import annotations

from service_core.backends.base import BackendTemplate


MAC_MLX_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="mlx-mac",
        label="MLX LM",
        runtime_dir_name="mlx-mac",
        launcher_name=None,
        description="Embedded MLX LM backend managed directly by OmniInfer on macOS",
        capabilities=("chat", "stream", "metal", "apple", "shared-memory", "embedded"),
        env_prefix="OMNIINFER_MLX_MAC",
        runtime_mode="embedded",
        model_artifact="directory",
        supports_mmproj=False,
        supports_ctx_size=False,
        python_modules=("mlx", "mlx_lm"),
    ),
)
