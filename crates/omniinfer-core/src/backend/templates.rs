use super::registry::{BackendTemplate, HostInfo, HostSystem};

pub(super) fn backend_templates(host: HostInfo) -> &'static [BackendTemplate] {
    match host.system {
        HostSystem::Linux => LINUX_TEMPLATES,
        HostSystem::Windows => WINDOWS_TEMPLATES,
        HostSystem::Mac => MAC_TEMPLATES,
        HostSystem::Android => ANDROID_TEMPLATES,
        HostSystem::Ios => IOS_TEMPLATES,
    }
}

const fn template(
    id: &'static str,
    label: &'static str,
    family: &'static str,
    runtime_dir_name: &'static str,
    launcher_name: Option<&'static str>,
    description: &'static str,
    capabilities: &'static [&'static str],
    env_prefix: &'static str,
) -> BackendTemplate {
    BackendTemplate {
        id,
        label,
        family,
        runtime_dir_name,
        launcher_name,
        description,
        capabilities,
        env_prefix,
        default_ngl: None,
        default_extra_args: &[],
        fallback_runtime_dir_names: &[],
        runtime_mode: "external_server",
        model_artifact: "file",
        supports_mmproj: true,
        supports_ctx_size: true,
        python_modules: &[],
        external_server_protocol: Some("llama.cpp-server"),
        log_file_name: "runtime.log",
    }
}

const LINUX_TEMPLATES: &[BackendTemplate] = &[
    template(
        "llama.cpp-linux",
        "llama.cpp Linux",
        "llama.cpp",
        "llama.cpp-linux",
        Some("llama-server"),
        "llama.cpp Linux CPU backend managed by OmniInfer",
        &["chat", "vision", "stream", "cpu", "linux"],
        "OMNIINFER_LLAMA_CPP_LINUX",
    ),
    template(
        "llama.cpp-linux-cuda",
        "llama.cpp Linux CUDA",
        "llama.cpp",
        "llama.cpp-linux-cuda",
        Some("llama-server"),
        "llama.cpp Linux CUDA backend managed by OmniInfer",
        &["chat", "vision", "stream", "gpu", "cuda", "linux"],
        "OMNIINFER_LLAMA_CPP_LINUX_CUDA",
    ),
    BackendTemplate {
        default_ngl: Some("999"),
        fallback_runtime_dir_names: &["llama.cpp-linux-ROCm"],
        ..template(
            "llama.cpp-linux-rocm",
            "llama.cpp Linux ROCm",
            "llama.cpp",
            "llama.cpp-linux-rocm",
            Some("llama-server"),
            "llama.cpp Linux ROCm backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "rocm", "linux"],
            "OMNIINFER_LLAMA_CPP_LINUX_ROCM",
        )
    },
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-linux-vulkan",
            "llama.cpp Linux Vulkan",
            "llama.cpp",
            "llama.cpp-linux-vulkan",
            Some("llama-server"),
            "llama.cpp Linux Vulkan backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "vulkan", "linux"],
            "OMNIINFER_LLAMA_CPP_LINUX_VULKAN",
        )
    },
    BackendTemplate {
        model_artifact: "diffusion-model",
        supports_mmproj: false,
        supports_ctx_size: false,
        external_server_protocol: Some("stable-diffusion.cpp-server"),
        log_file_name: "stable-diffusion-server.log",
        ..template(
            "stable-diffusion.cpp-linux-vulkan",
            "stable-diffusion.cpp Linux Vulkan",
            "stable-diffusion.cpp",
            "stable-diffusion.cpp-linux-vulkan",
            Some("sd-server"),
            "stable-diffusion.cpp image/video generation server managed by OmniInfer on Linux Vulkan",
            &[
                "image-generation",
                "video-generation",
                "native-audio",
                "gpu",
                "vulkan",
                "linux",
                "async-jobs",
            ],
            "OMNIINFER_STABLE_DIFFUSION_CPP_LINUX_VULKAN",
        )
    },
    template(
        "llama.cpp-linux-s390x",
        "llama.cpp Linux s390x",
        "llama.cpp",
        "llama.cpp-linux-s390x",
        Some("llama-server"),
        "llama.cpp Linux s390x CPU backend managed by OmniInfer",
        &["chat", "vision", "stream", "cpu", "linux", "s390x"],
        "OMNIINFER_LLAMA_CPP_LINUX_S390X",
    ),
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "omniinfer-native-linux",
            "OmniInfer Native Linux (EAGLE3)",
            "llama.cpp",
            "omniinfer-native-linux",
            Some("llama-server"),
            "OmniInfer Native Linux backend with EAGLE3 speculative decoding",
            &[
                "chat", "vision", "stream", "gpu", "vulkan", "linux", "eagle3",
            ],
            "OMNIINFER_NATIVE_LINUX",
        )
    },
    template(
        "llama.cpp-linux-openvino",
        "llama.cpp Linux OpenVINO",
        "llama.cpp",
        "llama.cpp-linux-openvino",
        Some("llama-server"),
        "llama.cpp Linux OpenVINO backend managed by OmniInfer",
        &["chat", "vision", "stream", "linux", "openvino", "intel"],
        "OMNIINFER_LLAMA_CPP_LINUX_OPENVINO",
    ),
    BackendTemplate {
        default_extra_args: &["--jinja"],
        ..template(
            "ik_llama.cpp-linux",
            "ik_llama.cpp Linux",
            "llama.cpp",
            "ik_llama.cpp-linux",
            Some("llama-server"),
            "ik_llama.cpp Linux CPU backend managed by OmniInfer",
            &["chat", "vision", "stream", "cpu", "linux"],
            "OMNIINFER_IK_LLAMA_CPP_LINUX",
        )
    },
    BackendTemplate {
        default_ngl: Some("999"),
        default_extra_args: &["--jinja"],
        ..template(
            "ik_llama.cpp-linux-cuda",
            "ik_llama.cpp Linux CUDA",
            "llama.cpp",
            "ik_llama.cpp-linux-cuda",
            Some("llama-server"),
            "ik_llama.cpp Linux CUDA backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "cuda", "linux"],
            "OMNIINFER_IK_LLAMA_CPP_LINUX_CUDA",
        )
    },
    BackendTemplate {
        runtime_mode: "embedded",
        model_artifact: "path",
        supports_mmproj: false,
        supports_ctx_size: false,
        python_modules: &["MNN", "MNN.llm", "MNN.cv"],
        external_server_protocol: None,
        ..template(
            "mnn-linux",
            "MNN Linux",
            "mnn",
            "mnn-linux",
            None,
            "Embedded MNN LLM/VLM backend managed directly by OmniInfer on Linux",
            &[
                "chat", "vision", "stream", "cpu", "linux", "embedded", "mnn",
            ],
            "OMNIINFER_MNN_LINUX",
        )
    },
    BackendTemplate {
        model_artifact: "reference",
        supports_mmproj: false,
        external_server_protocol: Some("vllm-openai-server"),
        log_file_name: "vllm-server.log",
        ..template(
            "vllm-linux-cuda",
            "vLLM Linux CUDA",
            "vllm",
            "vllm-linux-cuda",
            Some("vllm"),
            "vLLM OpenAI-compatible server backend managed by OmniInfer on Linux CUDA",
            &[
                "chat",
                "stream",
                "gpu",
                "cuda",
                "linux",
                "openai-compatible",
            ],
            "OMNIINFER_VLLM_LINUX_CUDA",
        )
    },
    BackendTemplate {
        model_artifact: "reference",
        supports_mmproj: false,
        external_server_protocol: Some("freetoken-openai-server"),
        log_file_name: "freetoken-server.log",
        ..template(
            "freetoken-linux-cuda",
            "FreeToken Linux CUDA",
            "freetoken",
            "freetoken-linux-cuda",
            Some("ft"),
            "FreeToken edge-native MoE server managed by OmniInfer on Linux CUDA",
            &[
                "chat",
                "stream",
                "gpu",
                "cuda",
                "cuda13",
                "linux",
                "x64",
                "openai-compatible",
                "anthropic-compatible",
                "moe",
            ],
            "OMNIINFER_FREETOKEN_LINUX_CUDA",
        )
    },
    BackendTemplate {
        model_artifact: "vla-artifact",
        supports_ctx_size: false,
        external_server_protocol: Some("vla.cpp-zmq-server"),
        log_file_name: "vla-server.log",
        ..template(
            "vla.cpp-linux",
            "vla.cpp Linux",
            "vla.cpp",
            "vla.cpp-linux",
            Some("vla-server"),
            "vla.cpp ZeroMQ/protobuf VLA action server managed by OmniInfer on Linux CPU",
            &[
                "vision", "action", "robotics", "cpu", "linux", "zeromq", "protobuf",
            ],
            "OMNIINFER_VLA_CPP_LINUX",
        )
    },
    BackendTemplate {
        model_artifact: "vla-artifact",
        supports_ctx_size: false,
        external_server_protocol: Some("vla.cpp-zmq-server"),
        log_file_name: "vla-server.log",
        ..template(
            "vla.cpp-linux-cuda",
            "vla.cpp Linux CUDA",
            "vla.cpp",
            "vla.cpp-linux-cuda",
            Some("vla-server"),
            "vla.cpp ZeroMQ/protobuf VLA action server managed by OmniInfer on Linux CUDA",
            &[
                "vision", "action", "robotics", "gpu", "cuda", "linux", "zeromq", "protobuf",
            ],
            "OMNIINFER_VLA_CPP_LINUX_CUDA",
        )
    },
];

const WINDOWS_TEMPLATES: &[BackendTemplate] = &[
    template(
        "llama.cpp-cpu",
        "llama.cpp cpu",
        "llama.cpp",
        "llama.cpp-cpu",
        Some("llama-server.exe"),
        "llama.cpp CPU backend managed by OmniInfer",
        &["chat", "vision", "stream", "cpu"],
        "OMNIINFER_LLAMA_CPP_CPU",
    ),
    template(
        "llama.cpp-cuda",
        "llama.cpp CUDA",
        "llama.cpp",
        "llama.cpp-cuda",
        Some("llama-server.exe"),
        "llama.cpp CUDA backend managed by OmniInfer",
        &["chat", "vision", "stream", "gpu", "cuda"],
        "OMNIINFER_LLAMA_CPP_CUDA",
    ),
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-vulkan",
            "llama.cpp Vulkan",
            "llama.cpp",
            "llama.cpp-vulkan",
            Some("llama-server.exe"),
            "llama.cpp Vulkan backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "vulkan"],
            "OMNIINFER_LLAMA_CPP_VULKAN",
        )
    },
    BackendTemplate {
        model_artifact: "diffusion-model",
        supports_mmproj: false,
        supports_ctx_size: false,
        external_server_protocol: Some("stable-diffusion.cpp-server"),
        log_file_name: "stable-diffusion-server.log",
        ..template(
            "stable-diffusion.cpp-vulkan",
            "stable-diffusion.cpp Vulkan",
            "stable-diffusion.cpp",
            "stable-diffusion.cpp-vulkan",
            Some("sd-server.exe"),
            "stable-diffusion.cpp image/video generation server managed by OmniInfer on Windows Vulkan",
            &[
                "image-generation",
                "video-generation",
                "native-audio",
                "gpu",
                "vulkan",
                "windows",
                "async-jobs",
            ],
            "OMNIINFER_STABLE_DIFFUSION_CPP_VULKAN",
        )
    },
    template(
        "llama.cpp-windows-arm64",
        "llama.cpp Windows arm64",
        "llama.cpp",
        "llama.cpp-windows-arm64",
        Some("llama-server.exe"),
        "llama.cpp Windows arm64 CPU backend managed by OmniInfer",
        &["chat", "vision", "stream", "cpu", "windows", "arm64"],
        "OMNIINFER_LLAMA_CPP_WINDOWS_ARM64",
    ),
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-sycl",
            "llama.cpp SYCL",
            "llama.cpp",
            "llama.cpp-sycl",
            Some("llama-server.exe"),
            "llama.cpp Windows SYCL backend managed by OmniInfer",
            &[
                "chat", "vision", "stream", "gpu", "sycl", "intel", "windows",
            ],
            "OMNIINFER_LLAMA_CPP_SYCL",
        )
    },
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-hip",
            "llama.cpp HIP",
            "llama.cpp",
            "llama.cpp-hip",
            Some("llama-server.exe"),
            "llama.cpp Windows HIP backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "hip", "amd", "windows"],
            "OMNIINFER_LLAMA_CPP_HIP",
        )
    },
    BackendTemplate {
        default_extra_args: &["--jinja"],
        ..template(
            "ik_llama.cpp-cpu",
            "ik_llama.cpp CPU",
            "llama.cpp",
            "ik_llama.cpp-cpu",
            Some("llama-server.exe"),
            "ik_llama.cpp CPU backend managed by OmniInfer",
            &["chat", "vision", "stream", "cpu"],
            "OMNIINFER_IK_LLAMA_CPP_CPU",
        )
    },
    BackendTemplate {
        default_ngl: Some("999"),
        default_extra_args: &["--jinja"],
        ..template(
            "ik_llama.cpp-cuda",
            "ik_llama.cpp CUDA",
            "llama.cpp",
            "ik_llama.cpp-cuda",
            Some("llama-server.exe"),
            "ik_llama.cpp CUDA backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "cuda"],
            "OMNIINFER_IK_LLAMA_CPP_CUDA",
        )
    },
    BackendTemplate {
        model_artifact: "reference",
        supports_mmproj: false,
        external_server_protocol: Some("vllm-wsl2-openai-server"),
        log_file_name: "vllm-wsl2-server.log",
        ..template(
            "vllm-wsl2-cuda",
            "vLLM WSL2 CUDA",
            "vllm",
            "vllm-wsl2-cuda",
            Some("vllm-wsl2.json"),
            "Official vLLM Linux CUDA runtime managed by OmniInfer through WSL2",
            &[
                "chat",
                "stream",
                "gpu",
                "cuda",
                "windows",
                "wsl2",
                "openai-compatible",
            ],
            "OMNIINFER_VLLM_WSL2_CUDA",
        )
    },
    BackendTemplate {
        model_artifact: "reference",
        supports_mmproj: false,
        external_server_protocol: Some("vllm-wsl2-openai-server"),
        log_file_name: "vllm-wsl2-rocm-server.log",
        ..template(
            "vllm-wsl2-rocm",
            "vLLM WSL2 ROCm",
            "vllm",
            "vllm-wsl2-rocm",
            Some("vllm-wsl2.json"),
            "Official vLLM Linux ROCm runtime for Ryzen AI managed by OmniInfer through WSL2",
            &[
                "chat",
                "stream",
                "gpu",
                "rocm",
                "amd",
                "windows",
                "wsl2",
                "openai-compatible",
            ],
            "OMNIINFER_VLLM_WSL2_ROCM",
        )
    },
];

const MAC_TEMPLATES: &[BackendTemplate] = &[
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-mac",
            "llama.cpp Metal",
            "llama.cpp",
            "llama.cpp-mac",
            Some("llama-server"),
            "llama.cpp Metal backend managed by OmniInfer",
            &[
                "chat",
                "vision",
                "stream",
                "metal",
                "apple",
                "arm64",
                "shared-memory",
            ],
            "OMNIINFER_LLAMA_CPP_MAC",
        )
    },
    template(
        "llama.cpp-mac-intel",
        "llama.cpp macOS Intel",
        "llama.cpp",
        "llama.cpp-mac-intel",
        Some("llama-server"),
        "llama.cpp macOS Intel x64 backend managed by OmniInfer",
        &["chat", "vision", "stream", "cpu", "macos", "x64", "intel"],
        "OMNIINFER_LLAMA_CPP_MAC_INTEL",
    ),
    BackendTemplate {
        default_ngl: Some("999"),
        default_extra_args: &[
            "-fa",
            "on",
            "--cache-type-k",
            "turbo4",
            "--cache-type-v",
            "turbo4",
        ],
        log_file_name: "turboquant-server.log",
        ..template(
            "turboquant-mac",
            "TurboQuant Metal",
            "turboquant",
            "turboquant-mac",
            Some("llama-server"),
            "TurboQuant llama.cpp-compatible Metal backend managed by OmniInfer on macOS",
            &[
                "chat",
                "vision",
                "stream",
                "metal",
                "apple",
                "shared-memory",
                "turboquant",
            ],
            "OMNIINFER_TURBOQUANT_MAC",
        )
    },
    BackendTemplate {
        runtime_mode: "embedded",
        model_artifact: "directory",
        supports_mmproj: false,
        supports_ctx_size: false,
        python_modules: &["mlx", "mlx_lm", "mlx_vlm", "torch", "torchvision"],
        external_server_protocol: None,
        ..template(
            "mlx-mac",
            "MLX LM/VLM",
            "mlx-lm",
            "mlx-mac",
            None,
            "Embedded MLX LM/VLM backend managed directly by OmniInfer on macOS",
            &[
                "chat",
                "vision",
                "stream",
                "metal",
                "apple",
                "shared-memory",
                "embedded",
            ],
            "OMNIINFER_MLX_MAC",
        )
    },
];

const ANDROID_TEMPLATES: &[BackendTemplate] = &[BackendTemplate {
    default_ngl: Some("999"),
    ..template(
        "llama.cpp-android",
        "llama.cpp Android",
        "llama.cpp",
        "llama.cpp-android",
        Some("llama-server"),
        "llama.cpp Android backend managed by OmniInfer",
        &["chat", "vision", "stream", "android", "mobile"],
        "OMNIINFER_LLAMA_CPP_ANDROID",
    )
}];

const IOS_TEMPLATES: &[BackendTemplate] = &[
    BackendTemplate {
        default_ngl: Some("999"),
        runtime_mode: "embedded",
        external_server_protocol: None,
        ..template(
            "llama.cpp-ios",
            "llama.cpp iOS",
            "llama.cpp",
            "llama.cpp-ios",
            None,
            "llama.cpp iOS Metal backend managed by OmniInfer",
            &[
                "chat", "vision", "stream", "metal", "apple", "mobile", "ios",
            ],
            "OMNIINFER_LLAMA_CPP_IOS",
        )
    },
    BackendTemplate {
        runtime_mode: "embedded",
        model_artifact: "directory",
        supports_mmproj: false,
        supports_ctx_size: false,
        external_server_protocol: None,
        ..template(
            "mlx-ios",
            "MLX iOS",
            "mlx-lm",
            "mlx-ios",
            None,
            "Embedded MLX LM backend via mlx-swift on iOS",
            &[
                "chat", "stream", "metal", "apple", "mobile", "ios", "embedded",
            ],
            "OMNIINFER_MLX_IOS",
        )
    },
];
