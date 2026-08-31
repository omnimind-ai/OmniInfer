use super::*;

#[test]
fn amd_gpu_output_detection_accepts_vendor_names() {
    assert!(output_mentions_amd_gpu(
        b"DriverDesc    REG_SZ    AMD Radeon(TM) 8060S Graphics"
    ));
    assert!(output_mentions_amd_gpu(
        b"Name\nAdvanced Micro Devices Radeon Pro"
    ));
}

#[test]
fn amd_gpu_output_detection_rejects_non_gpu_architecture_text() {
    assert!(!output_mentions_amd_gpu(
        b"DriverDesc    REG_SZ    NVIDIA GeForce RTX 4090"
    ));
    assert!(!output_mentions_amd_gpu(b"Architecture    REG_SZ    AMD64"));
}

#[test]
fn nvidia_driver_branch_parser_handles_release_versions() {
    assert_eq!(parse_nvidia_driver_branch("595.84"), Some(595));
    assert_eq!(parse_nvidia_driver_branch(" 580.105.08 "), Some(580));
    assert_eq!(parse_nvidia_driver_branch("unknown"), None);
}

#[test]
fn linux_registry_includes_primary_backends() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Linux,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    assert!(registry.get("llama.cpp-linux-cuda").is_some());
    assert!(registry.get("vllm-linux-cuda").is_some());
    assert!(registry.get("freetoken-linux-cuda").is_some());
    assert!(registry.get("mnn-linux").is_some());
    let diffusion = registry
        .get("stable-diffusion.cpp-linux-vulkan")
        .expect("Linux diffusion backend");
    assert_eq!(diffusion.family, "stable-diffusion.cpp");
    assert_eq!(diffusion.model_artifact, "diffusion-model");
    assert!(!diffusion.supports_mmproj);
    assert!(!diffusion.supports_ctx_size);
    assert_eq!(
        diffusion.external_server_protocol.as_deref(),
        Some("stable-diffusion.cpp-server")
    );
}

#[test]
fn generic_recommendation_excludes_action_only_backends() {
    let rows = vec![
        json!({
            "id": "vla.cpp-linux-cuda",
            "binary_exists": true,
            "hardware_compatible": true,
            "priority": 0,
            "capabilities": ["vision", "action", "robotics"],
        }),
        json!({
            "id": "llama.cpp-linux",
            "binary_exists": true,
            "hardware_compatible": true,
            "priority": 1,
            "capabilities": ["chat"],
        }),
    ];

    assert_eq!(
        recommended_backend(&rows).as_deref(),
        Some("llama.cpp-linux")
    );
    assert_eq!(recommended_backend(&rows[..1]), None);
}

#[test]
fn official_llama_backend_has_safe_cache_defaults() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Linux,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let backend = registry.get("llama.cpp-linux-cuda").unwrap();
    assert_eq!(
        backend.default_args,
        vec![
            "--slot-prompt-similarity",
            "0",
            "--cache-idle-slots",
            "--cache-ram",
            "8192"
        ]
    );

    let ik = registry.get("ik_llama.cpp-linux-cuda").unwrap();
    assert_eq!(ik.default_args, vec!["--jinja", "-ngl", "999"]);

    let windows = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Windows,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let windows_cuda = windows.get("llama.cpp-cuda").unwrap();
    assert!(!windows_cuda.default_args.iter().any(|arg| arg == "-ngl"));
}

#[test]
fn vllm_uses_reference_artifact_without_mmproj() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Linux,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let backend = registry.get("vllm-linux-cuda").unwrap();
    assert_eq!(backend.model_artifact, "reference");
    assert!(!backend.supports_mmproj);
    assert_eq!(
        backend.external_server_protocol.as_deref(),
        Some("vllm-openai-server")
    );
}

#[test]
fn freetoken_uses_reference_artifact_and_cuda13_contract() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Linux,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let backend = registry.get("freetoken-linux-cuda").unwrap();
    assert_eq!(backend.family, "freetoken");
    assert_eq!(backend.model_artifact, "reference");
    assert!(!backend.supports_mmproj);
    assert_eq!(
        backend.external_server_protocol.as_deref(),
        Some("freetoken-openai-server")
    );
    for capability in [
        "gpu",
        "cuda",
        "cuda13",
        "linux",
        "x64",
        "openai-compatible",
        "anthropic-compatible",
        "moe",
    ] {
        assert!(
            backend.capabilities.iter().any(|value| value == capability),
            "missing capability {capability}"
        );
    }
    assert_eq!(backend_priority("freetoken-linux-cuda"), 3);
}

#[test]
fn windows_registry_exposes_managed_wsl2_vllm_backend() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Windows,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let backend = registry.get("vllm-wsl2-cuda").unwrap();
    assert_eq!(backend.family, "vllm");
    assert_eq!(backend.model_artifact, "reference");
    assert!(!backend.supports_mmproj);
    assert_eq!(
        backend.external_server_protocol.as_deref(),
        Some("vllm-wsl2-openai-server")
    );
    for capability in ["gpu", "cuda", "windows", "wsl2", "openai-compatible"] {
        assert!(
            backend.capabilities.iter().any(|value| value == capability),
            "missing capability {capability}"
        );
    }
    assert!(
        gpu_backend_ids(registry.host).contains(&backend.id.as_str()),
        "managed WSL2 vLLM must participate in Windows GPU detection"
    );
    let rocm = registry.get("vllm-wsl2-rocm").unwrap();
    assert_eq!(rocm.family, "vllm");
    assert_eq!(rocm.model_artifact, "reference");
    assert!(!rocm.supports_mmproj);
    for capability in ["gpu", "rocm", "amd", "windows", "wsl2", "openai-compatible"] {
        assert!(
            rocm.capabilities.iter().any(|value| value == capability),
            "missing capability {capability}"
        );
    }
    assert!(
        gpu_backend_ids(registry.host).contains(&rocm.id.as_str()),
        "managed WSL2 ROCm vLLM must participate in Windows GPU detection"
    );
}

#[test]
fn windows_registry_exposes_vulkan_diffusion_backend() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Windows,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let backend = registry
        .get("stable-diffusion.cpp-vulkan")
        .expect("Windows diffusion backend");
    assert_eq!(backend.family, "stable-diffusion.cpp");
    assert_eq!(backend.model_artifact, "diffusion-model");
    assert!(!backend.supports_mmproj);
    assert!(!backend.supports_ctx_size);
    assert_eq!(
        backend.external_server_protocol.as_deref(),
        Some("stable-diffusion.cpp-server")
    );
    for capability in [
        "image-generation",
        "video-generation",
        "native-audio",
        "vulkan",
        "async-jobs",
    ] {
        assert!(
            backend.capabilities.iter().any(|value| value == capability),
            "missing capability {capability}"
        );
    }
    assert!(gpu_backend_ids(registry.host).contains(&backend.id.as_str()));
}

#[test]
fn overrides_and_env_are_applied() {
    let overrides = json!({
        "llama.cpp-linux-cuda": {
            "ngl": "12",
            "ctx_size": 4096,
            "parallel": 2,
            "cache_ram": 0,
            "extra_args": "--flash-attn on",
            "models_dir": "models/custom"
        }
    });
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Linux,
            machine: "x86_64",
        },
        "runtime",
        &overrides,
    );
    let backend = registry.get("llama.cpp-linux-cuda").unwrap();
    assert_eq!(
        backend.default_args,
        vec![
            "--slot-prompt-similarity",
            "0",
            "--cache-idle-slots",
            "-ngl",
            "12",
            "-c",
            "4096",
            "-np",
            "2",
            "--cache-ram",
            "0",
            "--flash-attn",
            "on"
        ]
    );
    assert!(
        backend
            .models_dir
            .as_deref()
            .unwrap()
            .ends_with("models/custom")
    );
}

#[test]
fn payload_marks_runtime_metadata() {
    let registry = BackendRegistry::build(
        HostInfo {
            system: HostSystem::Linux,
            machine: "x86_64",
        },
        "runtime",
        &Value::Null,
    );
    let payload = registry
        .get("llama.cpp-linux-cuda")
        .unwrap()
        .to_api_payload(false, None, Some("compatible"), Some(0));
    assert_eq!(payload["id"], "llama.cpp-linux-cuda");
    assert_eq!(payload["compatibility"], "compatible");
    assert_eq!(payload["hardware_compatible"], true);
    assert_eq!(payload["priority"], 0);
}

#[test]
fn validated_llama_backends_rank_before_experimental_ik_backends() {
    assert!(backend_priority("llama.cpp-cuda") < backend_priority("ik_llama.cpp-cuda"));
    assert!(backend_priority("llama.cpp-cuda") < backend_priority("vllm-wsl2-cuda"));
    assert!(backend_priority("llama.cpp-linux-cuda") < backend_priority("ik_llama.cpp-linux-cuda"));
    assert!(backend_priority("llama.cpp-cpu") < backend_priority("ik_llama.cpp-cpu"));
}

#[test]
fn mac_arm64_only_accepts_arm_llama_backend() {
    for machine in ["aarch64", "arm64"] {
        let registry = BackendRegistry::build(
            HostInfo {
                system: HostSystem::Mac,
                machine,
            },
            "runtime",
            &Value::Null,
        );
        assert!(is_hardware_compatible(
            registry.host,
            registry.get("llama.cpp-mac").unwrap()
        ));
        assert!(!is_hardware_compatible(
            registry.host,
            registry.get("llama.cpp-mac-intel").unwrap()
        ));
    }
}

#[test]
fn mac_x86_64_only_accepts_intel_llama_backend() {
    for machine in ["x86_64", "amd64"] {
        let registry = BackendRegistry::build(
            HostInfo {
                system: HostSystem::Mac,
                machine,
            },
            "runtime",
            &Value::Null,
        );
        assert!(!is_hardware_compatible(
            registry.host,
            registry.get("llama.cpp-mac").unwrap()
        ));
        assert!(is_hardware_compatible(
            registry.host,
            registry.get("llama.cpp-mac-intel").unwrap()
        ));
    }
}
