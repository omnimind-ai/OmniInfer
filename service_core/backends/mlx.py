from __future__ import annotations

from service_core.backends.base import BackendTemplate


MAC_MLX_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="mlx-mac",
        label="MLX LM",
        family="mlx-lm",
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


IOS_MLX_TEMPLATES: tuple[BackendTemplate, ...] = (
    BackendTemplate(
        id="mlx-ios",
        label="MLX iOS",
        family="mlx-lm",
        runtime_dir_name="mlx-ios",
        launcher_name=None,
        description="Embedded MLX LM backend via mlx-swift on iOS",
        capabilities=("chat", "stream", "metal", "apple", "mobile", "ios", "embedded"),
        env_prefix="OMNIINFER_MLX_IOS",
        runtime_mode="embedded",
        model_artifact="directory",
        supports_mmproj=False,
        supports_ctx_size=False,
        external_server_protocol=None,
    ),
)
