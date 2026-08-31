use super::*;

fn test_budget(bytes: u64) -> ResourceBudget {
    ResourceBudget::from_domains(BTreeMap::from([(MemoryDomain::Host, bytes)])).unwrap()
}

#[test]
fn detects_llama_context_args() {
    assert!(launch_args_have_ctx_size(
        "llama.cpp",
        &["-c".to_string(), "8192".to_string()]
    ));
    assert!(launch_args_have_ctx_size(
        "llama.cpp",
        &["--ctx-size=4096".to_string()]
    ));
    assert!(!launch_args_have_ctx_size(
        "llama.cpp",
        &["-ngl".to_string(), "999".to_string()]
    ));
}

#[test]
fn detects_vllm_context_args() {
    assert!(launch_args_have_ctx_size(
        "vllm",
        &["--max-model-len=65536".to_string()]
    ));
    assert!(!launch_args_have_ctx_size(
        "vllm",
        &["--gpu-memory-utilization".to_string(), "0.9".to_string()]
    ));
}

#[test]
fn detects_freetoken_context_args() {
    assert!(launch_args_have_ctx_size(
        "freetoken",
        &["--max-seq-len-override=8192".to_string()]
    ));
    assert!(!launch_args_have_ctx_size(
        "freetoken",
        &["--memory-ratio".to_string(), "0.8".to_string()]
    ));
}

#[test]
fn no_mmproj_disables_explicit_and_discovered_projector_selection() {
    assert_eq!(
        select_mmproj_path(
            true,
            Some("explicit.gguf".to_string()),
            Some("discovered.gguf".to_string()),
            Some("automatic.gguf".to_string()),
        ),
        None
    );
    assert_eq!(
        select_mmproj_path(
            false,
            None,
            Some("discovered.gguf".to_string()),
            Some("automatic.gguf".to_string()),
        ),
        Some("discovered.gguf".to_string())
    );
    assert_eq!(
        select_mmproj_path(false, None, None, Some("automatic.gguf".to_string())),
        Some("automatic.gguf".to_string())
    );
}

#[test]
fn no_mmproj_gateway_payload_is_strictly_boolean() {
    assert!(no_mmproj_from_payload(&json!({"no_mmproj": true})).unwrap());
    assert!(!no_mmproj_from_payload(&json!({})).unwrap());
    assert!(no_mmproj_from_payload(&json!({"no_mmproj": "true"})).is_err());
}

#[test]
fn restore_selection_exposes_persisted_no_mmproj_and_legacy_default() {
    let mut payload = json!({});
    let selected = omniinfer_core::local_state::SelectedModel {
        model: "/models/model.gguf".to_string(),
        mmproj: None,
        no_mmproj: true,
        ctx_size: Some(4096),
        request_defaults: Map::new(),
    };
    let state = omniinfer_core::local_state::LocalState {
        selected_backend: Some("llama.cpp-linux".to_string()),
        selected_model: Some(selected),
        ..Default::default()
    };
    annotate_restore_state(&mut payload, &state, &BTreeMap::new());
    assert_eq!(payload["restore_selection"]["no_mmproj"], true);

    let mut legacy_payload = json!({});
    let legacy_state = omniinfer_core::local_state::LocalState {
        selected_model: Some(omniinfer_core::local_state::SelectedModel {
            model: "/models/legacy.gguf".to_string(),
            mmproj: Some("/models/mmproj.gguf".to_string()),
            no_mmproj: false,
            ctx_size: None,
            request_defaults: Map::new(),
        }),
        ..Default::default()
    };
    annotate_restore_state(&mut legacy_payload, &legacy_state, &BTreeMap::new());
    assert_eq!(legacy_payload["restore_selection"]["no_mmproj"], false);
}

#[test]
fn failed_load_transaction_rolls_back_reservation() {
    let mut manager = RustRuntimeManager {
        resource_ledger: Some(ResourceLedger::new(
            ResourceCapacity::new(1, BTreeMap::from([(MemoryDomain::Host, 1024)])).unwrap(),
        )),
        ..Default::default()
    };
    let reservation = manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .reserve("failed-load", test_budget(768))
        .unwrap();

    let result: Result<()> = manager.with_reservation(reservation, |_| {
        Err(anyhow::anyhow!("simulated readiness timeout"))
    });

    assert!(result.is_err());
    let snapshot = manager.resource_ledger.as_ref().unwrap().snapshot();
    assert!(snapshot.reserved.is_empty());
    assert!(snapshot.committed.is_empty());
}

#[test]
fn multi_gpu_components_are_split_into_non_overlapping_domains() {
    let domains = vec![
        MemoryDomain::Cuda("0".to_string()),
        MemoryDomain::Cuda("1".to_string()),
    ];
    let components = distribute_component("weights", 101, &domains).unwrap();
    let budget = ResourceBudget::from_components(components).unwrap();

    assert_eq!(budget.domains()[&MemoryDomain::Cuda("0".to_string())], 51);
    assert_eq!(budget.domains()[&MemoryDomain::Cuda("1".to_string())], 50);
    assert!(
        !budget
            .domains()
            .contains_key(&MemoryDomain::Cuda("0,1".to_string()))
    );
}

#[test]
fn uncertain_multi_gpu_mapping_reserves_full_budget_per_device() {
    let domains = vec![
        MemoryDomain::Cuda("0".to_string()),
        MemoryDomain::Cuda("1".to_string()),
    ];
    let components = assign_component("weights", 101, &domains, true).unwrap();
    let budget = ResourceBudget::from_components(components).unwrap();

    assert_eq!(budget.domains()[&MemoryDomain::Cuda("0".to_string())], 101);
    assert_eq!(budget.domains()[&MemoryDomain::Cuda("1".to_string())], 101);
}

#[test]
fn explicit_budget_cannot_understate_local_estimate() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-resource-budget-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("model.gguf");
    fs::write(&model, vec![0_u8; 1024]).unwrap();
    let backend_id = if cfg!(target_os = "linux") {
        "llama.cpp-linux"
    } else if cfg!(target_os = "macos") {
        "llama.cpp-mac-intel"
    } else {
        "llama.cpp-cpu"
    };
    let registry = BackendRegistry::load_current();
    let backend = registry
        .get(backend_id)
        .expect("test platform should expose a CPU external backend");

    let result = build_runtime_resource_budget(
        &json!({"resource_budget_bytes": 1024}),
        backend,
        model.to_str().unwrap(),
        None,
        512,
        &[],
        None,
        false,
    );

    assert!(result.is_err());
    fs::remove_dir_all(root).ok();
}

#[test]
fn visual_projector_does_not_change_model_context_components() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-resource-formula-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("model.gguf");
    let projector = root.join("mmproj.gguf");
    fs::File::create(&model).unwrap().set_len(5 * GIB).unwrap();
    fs::File::create(&projector).unwrap().set_len(GIB).unwrap();
    let backend = BackendRegistry::load_current()
        .get("llama.cpp-linux-cuda")
        .cloned()
        .unwrap_or_else(|| backend_registry::BackendSpec {
            id: "llama.cpp-linux-cuda".to_string(),
            label: "test".to_string(),
            family: "llama.cpp".to_string(),
            runtime_dir: root.display().to_string(),
            launcher_path: None,
            models_dir: None,
            catalog_url: None,
            description: "test".to_string(),
            capabilities: vec!["cuda".to_string()],
            default_args: Vec::new(),
            runtime_mode: "external_server".to_string(),
            model_artifact: "gguf".to_string(),
            supports_mmproj: true,
            supports_ctx_size: true,
            python_modules: Vec::new(),
            external_server_protocol: Some("llama_cpp".to_string()),
            log_file_name: "test.log".to_string(),
        });
    let text = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        2048,
        &[],
        Some("0"),
        false,
    )
    .unwrap();
    let visual = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        Some(projector.to_str().unwrap()),
        2048,
        &[],
        Some("0"),
        false,
    )
    .unwrap();
    for name in ["kv_cache", "activation"] {
        let text_bytes = text
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        let visual_bytes = visual
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        assert_eq!(text_bytes, visual_bytes, "component {name}");
    }
    assert!(
        visual
            .components()
            .iter()
            .any(|component| component.name == "mmproj")
    );
    for name in ["framework_overhead", "allocator_slack"] {
        let text_bytes = text
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        let visual_bytes = visual
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap()
            .bytes;
        assert!(visual_bytes > text_bytes, "component {name}");
    }
    assert_eq!(
        visual.domains().keys().collect::<Vec<_>>(),
        text.domains().keys().collect::<Vec<_>>()
    );
    for domain in text.domains().keys() {
        assert!(
            visual.domains()[domain] > text.domains()[domain],
            "domain {}",
            domain.key()
        );
    }
    fs::remove_dir_all(root).ok();
}

#[test]
fn diffusion_budget_splits_h3_components_by_effective_placement() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-diffusion-budget-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("minimax_h3_fl2va_pruned-Q4_K.gguf");
    let llm_old = root.join("old-llm.gguf");
    let llm = root.join("qwen3vl_32b_minimax_h3-Q4_K_M.gguf");
    let vae = root.join("minimax_h3_video_vae_fp16.safetensors");
    let audio_vae = root.join("minimax_h3_audio_vae_fp32.safetensors");
    for (path, bytes) in [
        (&model, 10 * MIB),
        (&llm_old, 99 * MIB),
        (&llm, 11 * MIB),
        (&vae, 5 * MIB),
        (&audio_vae, 2 * MIB),
    ] {
        fs::File::create(path).unwrap().set_len(bytes).unwrap();
    }
    let backend = backend_registry::BackendSpec {
        id: "stable-diffusion.cpp-linux-vulkan".to_string(),
        label: "test".to_string(),
        family: "stable-diffusion.cpp".to_string(),
        runtime_dir: root.display().to_string(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: vec!["vulkan".to_string()],
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "diffusion-model".to_string(),
        supports_mmproj: false,
        supports_ctx_size: false,
        python_modules: Vec::new(),
        external_server_protocol: Some("stable-diffusion.cpp-server".to_string()),
        log_file_name: "stable-diffusion-server.log".to_string(),
    };
    let launch_args = [
        format!("--llm={}", llm_old.display()),
        "--qwen2vl".to_string(),
        llm.display().to_string(),
        "--vae".to_string(),
        vae.display().to_string(),
        "--audio-vae".to_string(),
        audio_vae.display().to_string(),
        "--backend".to_string(),
        "te=cpu".to_string(),
    ];
    let budget = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        DEFAULT_LOAD_CONTEXT_SIZE,
        &launch_args,
        None,
        false,
    )
    .unwrap();
    for (name, bytes, domain) in [
        (
            "diffusion_weights",
            10 * MIB,
            MemoryDomain::Vulkan("0".to_string()),
        ),
        ("text_encoder_weights", 11 * MIB, MemoryDomain::Host),
        (
            "video_vae_weights",
            5 * MIB,
            MemoryDomain::Vulkan("0".to_string()),
        ),
        (
            "audio_vae_weights",
            2 * MIB,
            MemoryDomain::Vulkan("0".to_string()),
        ),
    ] {
        let component = budget
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap();
        assert_eq!(component.bytes, bytes, "component {name}");
        assert_eq!(component.domain, domain, "component {name}");
    }
    assert!(budget.components().iter().any(|component| {
        component.name == "runtime_workspace" && component.domain == MemoryDomain::Host
    }));
    assert!(budget.components().iter().any(|component| {
        component.name == "runtime_workspace"
            && component.domain == MemoryDomain::Vulkan("0".to_string())
    }));

    let mut offloaded_args = launch_args.to_vec();
    offloaded_args.push("--offload-to-cpu".to_string());
    let offloaded = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        DEFAULT_LOAD_CONTEXT_SIZE,
        &offloaded_args,
        None,
        false,
    )
    .unwrap();
    for component in offloaded
        .components()
        .iter()
        .filter(|component| component.name.ends_with("_weights"))
    {
        assert_eq!(component.domain, MemoryDomain::Host);
    }
    assert_eq!(
        offloaded
            .components()
            .iter()
            .filter(|component| component.name.ends_with("_runtime_staging"))
            .map(|component| {
                assert_eq!(component.domain, MemoryDomain::Vulkan("0".to_string()));
                component.bytes
            })
            .sum::<u64>(),
        17 * MIB
    );

    let mut explicit_params_args = offloaded_args.clone();
    explicit_params_args.extend(["--params-backend".to_string(), "te=vulkan0".to_string()]);
    let explicit_params = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        DEFAULT_LOAD_CONTEXT_SIZE,
        &explicit_params_args,
        None,
        false,
    )
    .unwrap();
    assert_eq!(
        explicit_params
            .components()
            .iter()
            .find(|component| component.name == "text_encoder_weights")
            .unwrap()
            .domain,
        MemoryDomain::Vulkan("0".to_string())
    );

    let mut explicit_runtime_args = launch_args.to_vec();
    explicit_runtime_args.extend([
        "--clip-on-cpu".to_string(),
        "--backend".to_string(),
        "te=vulkan0".to_string(),
    ]);
    let explicit_runtime = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        DEFAULT_LOAD_CONTEXT_SIZE,
        &explicit_runtime_args,
        None,
        false,
    )
    .unwrap();
    assert_eq!(
        explicit_runtime
            .components()
            .iter()
            .find(|component| component.name == "text_encoder_weights")
            .unwrap()
            .domain,
        MemoryDomain::Vulkan("0".to_string())
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn diffusion_budget_rejects_dynamic_or_unknown_placement() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-diffusion-placement-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("model.gguf");
    fs::File::create(&model).unwrap().set_len(GIB).unwrap();
    let backend = backend_registry::BackendSpec {
        id: "stable-diffusion.cpp-vulkan".to_string(),
        label: "test".to_string(),
        family: "stable-diffusion.cpp".to_string(),
        runtime_dir: root.display().to_string(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: vec!["vulkan".to_string()],
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "diffusion-model".to_string(),
        supports_mmproj: false,
        supports_ctx_size: false,
        python_modules: Vec::new(),
        external_server_protocol: Some("stable-diffusion.cpp-server".to_string()),
        log_file_name: "stable-diffusion-server.log".to_string(),
    };
    for args in [
        vec!["--auto-fit".to_string()],
        vec!["--params-backend=diffusion=disk".to_string()],
        vec!["--backend=diffusion=vulkan0&vulkan1".to_string()],
        vec!["--max-vram=8".to_string()],
        vec!["--rpc-servers=127.0.0.1:50052".to_string()],
        vec!["--embd-dir".to_string(), root.display().to_string()],
        vec!["--lora-model-dir".to_string(), root.display().to_string()],
        vec!["--type".to_string(), "f32".to_string()],
        vec!["--tensor-type-rules".to_string(), "model=f32".to_string()],
        vec!["--backend".to_string()],
    ] {
        assert!(
            build_runtime_resource_budget(
                &json!({}),
                &backend,
                model.to_str().unwrap(),
                None,
                DEFAULT_LOAD_CONTEXT_SIZE,
                &args,
                None,
                false,
            )
            .is_err(),
            "args {args:?}"
        );
    }
    fs::remove_dir_all(root).ok();
}

#[test]
fn diffusion_budget_tracks_auxiliary_module_assignments() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-diffusion-detector-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("model.gguf");
    let detector = root.join("detector.gguf");
    let llm_vision = root.join("llm-vision.gguf");
    let connectors = root.join("connectors.safetensors");
    fs::File::create(&model).unwrap().set_len(GIB).unwrap();
    fs::File::create(&detector)
        .unwrap()
        .set_len(128 * MIB)
        .unwrap();
    fs::File::create(&llm_vision)
        .unwrap()
        .set_len(96 * MIB)
        .unwrap();
    fs::File::create(&connectors)
        .unwrap()
        .set_len(64 * MIB)
        .unwrap();
    let backend = backend_registry::BackendSpec {
        id: "stable-diffusion.cpp-vulkan".to_string(),
        label: "test".to_string(),
        family: "stable-diffusion.cpp".to_string(),
        runtime_dir: root.display().to_string(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: vec!["vulkan".to_string()],
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "diffusion-model".to_string(),
        supports_mmproj: false,
        supports_ctx_size: false,
        python_modules: Vec::new(),
        external_server_protocol: Some("stable-diffusion.cpp-server".to_string()),
        log_file_name: "stable-diffusion-server.log".to_string(),
    };
    let budget = build_runtime_resource_budget(
        &json!({}),
        &backend,
        model.to_str().unwrap(),
        None,
        DEFAULT_LOAD_CONTEXT_SIZE,
        &[
            "--ad-model".to_string(),
            detector.display().to_string(),
            "--llm_vision".to_string(),
            llm_vision.display().to_string(),
            "--embeddings-connectors".to_string(),
            connectors.display().to_string(),
            "--backend".to_string(),
            "te=cpu,detector=cpu,clipvision=vulkan0".to_string(),
        ],
        None,
        false,
    )
    .unwrap();
    let detector_component = budget
        .components()
        .iter()
        .find(|component| component.name == "detector_weights")
        .unwrap();
    assert_eq!(detector_component.bytes, 128 * MIB);
    assert_eq!(detector_component.domain, MemoryDomain::Host);
    for (name, bytes) in [
        ("llm_vision_weights", 96 * MIB),
        ("embedding_connector_weights", 64 * MIB),
    ] {
        let component = budget
            .components()
            .iter()
            .find(|component| component.name == name)
            .unwrap();
        assert_eq!(component.bytes, bytes);
        assert_eq!(component.domain, MemoryDomain::Host);
    }
    fs::remove_dir_all(root).ok();
}

#[test]
fn vulkan_capacity_probe_returns_requested_domain() {
    let available = vulkan_available_bytes(&["0".to_string()]).unwrap();
    assert!(available[&MemoryDomain::Vulkan("0".to_string())] > 0);
}

#[test]
fn freetoken_reserves_host_model_and_elastic_cuda_pool() {
    let root = std::env::temp_dir().join(format!(
        "omniinfer-freetoken-budget-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&root).unwrap();
    let model = root.join("model.gguf");
    fs::File::create(&model).unwrap().set_len(GIB).unwrap();
    let backend = backend_registry::BackendSpec {
        id: "freetoken-linux-cuda".to_string(),
        label: "test".to_string(),
        family: "freetoken".to_string(),
        runtime_dir: root.display().to_string(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: vec!["cuda".to_string()],
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "reference".to_string(),
        supports_mmproj: false,
        supports_ctx_size: true,
        python_modules: Vec::new(),
        external_server_protocol: Some("freetoken-openai-server".to_string()),
        log_file_name: "freetoken.log".to_string(),
    };
    let budget = build_runtime_resource_budget(
        &json!({"launch_args": ["--memory-ratio", "0.5"]}),
        &backend,
        model.to_str().unwrap(),
        None,
        8192,
        &["--memory-ratio".to_string(), "0.5".to_string()],
        Some("0"),
        false,
    )
    .unwrap();
    assert_eq!(budget.domains()[&MemoryDomain::Host], GIB + 384 * MIB);
    assert_eq!(
        budget.domains()[&MemoryDomain::Cuda("0".to_string())],
        512 * GIB
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn freetoken_remote_reference_requires_host_budget() {
    let backend = backend_registry::BackendSpec {
        id: "freetoken-linux-cuda".to_string(),
        label: "test".to_string(),
        family: "freetoken".to_string(),
        runtime_dir: "runtime".to_string(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: vec!["cuda".to_string()],
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "reference".to_string(),
        supports_mmproj: false,
        supports_ctx_size: true,
        python_modules: Vec::new(),
        external_server_protocol: Some("freetoken-openai-server".to_string()),
        log_file_name: "freetoken.log".to_string(),
    };
    let error = build_runtime_resource_budget(
        &json!({}),
        &backend,
        "Qwen/Qwen3.6-35B-A3B",
        None,
        8192,
        &[],
        Some("0"),
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("host-memory reservation"));
}

fn speculative_test_backend(id: &str, family: &str, cuda: bool) -> backend_registry::BackendSpec {
    backend_registry::BackendSpec {
        id: id.to_string(),
        label: "test".to_string(),
        family: family.to_string(),
        runtime_dir: String::new(),
        launcher_path: None,
        models_dir: None,
        catalog_url: None,
        description: "test".to_string(),
        capabilities: cuda.then(|| "cuda".to_string()).into_iter().collect(),
        default_args: Vec::new(),
        runtime_mode: "external_server".to_string(),
        model_artifact: "gguf".to_string(),
        supports_mmproj: true,
        supports_ctx_size: true,
        python_modules: Vec::new(),
        external_server_protocol: Some("llama_cpp".to_string()),
        log_file_name: "test.log".to_string(),
    }
}

fn speculative_snapshot(
    capacity: u64,
    reserved: u64,
    committed: u64,
) -> omniinfer_core::resource_ledger::ResourceLedgerSnapshot {
    omniinfer_core::resource_ledger::ResourceLedgerSnapshot {
        capacity_snapshot_id: 1,
        capacities: BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), capacity)]),
        reserved: BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), reserved)])
            .into_iter()
            .filter(|(_, bytes)| *bytes > 0)
            .collect(),
        committed: BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), committed)])
            .into_iter()
            .filter(|(_, bytes)| *bytes > 0)
            .collect(),
    }
}

fn speculative_budget(estimated: u64, slack: u64) -> ResourceBudget {
    ResourceBudget::from_components(vec![
        BudgetComponent {
            name: "model".to_string(),
            domain: MemoryDomain::Cuda("0".to_string()),
            bytes: estimated - slack,
        },
        BudgetComponent {
            name: "allocator_slack".to_string(),
            domain: MemoryDomain::Cuda("0".to_string()),
            bytes: slack,
        },
    ])
    .unwrap()
}

#[test]
fn speculative_cuda_admission_enforces_narrow_boundaries() {
    let backend = speculative_test_backend("llama.cpp-linux-cuda", "llama.cpp", true);
    let budget = speculative_budget(1_000, 100);
    let accepted = [json!({}), json!({"mmproj": "/projector.gguf"})];
    for payload in accepted {
        let decision = speculative_reservation(
            &backend,
            &payload,
            &budget,
            false,
            Some(speculative_snapshot(900, 0, 0)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(decision.available, 900);
        assert_eq!(decision.shortfall, 100);
        assert_eq!(decision.waived_slack, decision.shortfall);
        assert_eq!(
            decision.budget.domains()[&MemoryDomain::Cuda("0".to_string())],
            900
        );
    }

    let less_than_slack = speculative_reservation(
        &backend,
        &json!({}),
        &budget,
        false,
        Some(speculative_snapshot(950, 0, 0)),
    )
    .unwrap()
    .unwrap();
    assert_eq!(less_than_slack.shortfall, 50);
    assert_eq!(less_than_slack.waived_slack, 50);

    for (available, payload, replicate) in [
        (899, json!({}), false),
        (1_000 - 100 - 1, json!({}), false),
        (900, json!({"resource_budget_bytes": 1}), false),
        (900, json!({}), true),
    ] {
        assert!(
            speculative_reservation(
                &backend,
                &payload,
                &budget,
                replicate,
                Some(speculative_snapshot(available, 0, 0)),
            )
            .unwrap()
            .is_none()
        );
    }

    let oversized_slack = speculative_budget(
        3 * SPECULATIVE_ALLOCATOR_SLACK_LIMIT,
        2 * SPECULATIVE_ALLOCATOR_SLACK_LIMIT,
    );
    assert!(
        speculative_reservation(
            &backend,
            &json!({}),
            &oversized_slack,
            false,
            Some(speculative_snapshot(
                2 * SPECULATIVE_ALLOCATOR_SLACK_LIMIT - 1,
                0,
                0
            )),
        )
        .unwrap()
        .is_none()
    );

    for (candidate, is_cuda, id, family, reserved, committed) in [
        (Some(900), false, "llama.cpp-linux-cuda", "llama.cpp", 0, 0),
        (Some(900), true, "other-cuda", "other", 0, 0),
        (Some(900), true, "llama.cpp-linux-cuda", "llama.cpp", 1, 0),
        (Some(900), true, "llama.cpp-linux-cuda", "llama.cpp", 0, 1),
    ] {
        let backend = speculative_test_backend(id, family, is_cuda);
        assert!(
            speculative_reservation(
                &backend,
                &json!({}),
                &budget,
                false,
                Some(speculative_snapshot(
                    candidate.unwrap(),
                    reserved,
                    committed
                )),
            )
            .unwrap()
            .is_none()
        );
    }

    let multi = ResourceBudget::from_components(vec![
        BudgetComponent {
            name: "model".to_string(),
            domain: MemoryDomain::Cuda("0".to_string()),
            bytes: 900,
        },
        BudgetComponent {
            name: "model".to_string(),
            domain: MemoryDomain::Cuda("1".to_string()),
            bytes: 900,
        },
    ])
    .unwrap();
    assert!(
        speculative_reservation(
            &backend,
            &json!({}),
            &multi,
            false,
            Some(speculative_snapshot(900, 0, 0)),
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn speculative_reservation_is_exclusive_and_rolls_back() {
    let backend = speculative_test_backend("llama.cpp-linux-cuda", "llama.cpp", true);
    let estimated = 1_000;
    let available = 900;
    let budget = speculative_budget(estimated, 100);
    let decision = speculative_reservation(
        &backend,
        &json!({}),
        &budget,
        false,
        Some(speculative_snapshot(available, 0, 0)),
    )
    .unwrap()
    .unwrap();
    let capacity = ResourceCapacity::new(
        1,
        BTreeMap::from([(MemoryDomain::Cuda("0".to_string()), available)]),
    )
    .unwrap();
    let mut ledger = ResourceLedger::new(capacity);
    let reservation = ledger
        .reserve("speculative", decision.budget.clone())
        .unwrap();
    assert!(ledger.reserve("second", decision.budget.clone()).is_err());
    assert!(ledger.rollback(reservation));
    assert!(ledger.reserve("second", decision.budget).is_ok());
}

#[test]
fn speculative_admission_payload_is_deterministic_and_separate_from_cuda_warning() {
    assert_eq!(speculative_admission_payload(None), Value::Null);
    let admission = SpeculativeAdmission {
        device: "0".to_string(),
        estimated: 1_000,
        exclusive: 900,
        shortfall: 100,
        waived_allocator_slack: 100,
    };
    assert_eq!(
        speculative_admission_payload(Some(&admission)),
        json!({
            "speculative": true,
            "device": "0",
            "estimated_cuda_bytes": 1_000,
            "exclusive_reservation_bytes": 900,
            "shortfall_bytes": 100,
            "waived_allocator_slack_bytes": 100,
        })
    );
    let smaller_shortfall = SpeculativeAdmission {
        waived_allocator_slack: 37,
        shortfall: 37,
        ..admission
    };
    assert_eq!(
        speculative_admission_payload(Some(&smaller_shortfall))["waived_allocator_slack_bytes"],
        37
    );
}

#[test]
fn speculative_domain_exclusivity_survives_refresh_and_releases_by_owner() {
    let capacity = ResourceCapacity::new(
        1,
        BTreeMap::from([
            (MemoryDomain::Cuda("0".to_string()), 1024 * GIB),
            (MemoryDomain::Cuda("1".to_string()), 1024 * GIB),
        ]),
    )
    .unwrap();
    let mut ledger = ResourceLedger::new(capacity);
    let owner_reservation = ledger
        .reserve(
            "speculative-owner",
            ResourceBudget::from_components(vec![BudgetComponent {
                name: "owner".to_string(),
                domain: MemoryDomain::Cuda("0".to_string()),
                bytes: 1,
            }])
            .unwrap(),
        )
        .unwrap();
    let owner_allocation = ledger.commit(owner_reservation).unwrap();
    let mut manager = RustRuntimeManager {
        resource_ledger: Some(ledger),
        speculative_domains: BTreeMap::from([(
            MemoryDomain::Cuda("0".to_string()),
            owner_allocation,
        )]),
        next_capacity_snapshot: 2,
        ..Default::default()
    };
    let cuda0 = ResourceBudget::from_components(vec![BudgetComponent {
        name: "follow_on".to_string(),
        domain: MemoryDomain::Cuda("0".to_string()),
        bytes: 1,
    }])
    .unwrap();
    let error = manager
        .reserve_runtime_resources("same-device", &cuda0, Some("0"), &[])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exclusively held by a speculative runtime"),
        "{error:#}"
    );
    // Model-key promotion/reuse cannot clear an owner: cleanup is allocation-identity based.
    let wrong_reservation = manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .reserve(
            "reused-old-key",
            ResourceBudget::from_components(vec![BudgetComponent {
                name: "other".to_string(),
                domain: MemoryDomain::Cuda("1".to_string()),
                bytes: 1,
            }])
            .unwrap(),
        )
        .unwrap();
    let wrong_allocation = manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .commit(wrong_reservation)
        .unwrap();
    manager.clear_speculative_owner(wrong_allocation);
    assert!(
        manager
            .speculative_domains
            .contains_key(&MemoryDomain::Cuda("0".to_string()))
    );
    manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .release(wrong_allocation);

    let cuda1 = ResourceBudget::from_components(vec![BudgetComponent {
        name: "other-device".to_string(),
        domain: MemoryDomain::Cuda("1".to_string()),
        bytes: 1,
    }])
    .unwrap();
    let other_reservation = manager
        .reserve_runtime_resources("other-device", &cuda1, Some("1"), &[])
        .unwrap();
    manager
        .resource_ledger
        .as_mut()
        .unwrap()
        .rollback(other_reservation);

    manager.clear_speculative_owner(owner_allocation);
    let released_reservation = manager
        .reserve_runtime_resources("after-release", &cuda0, Some("0"), &[])
        .unwrap();
    assert!(
        manager
            .resource_ledger
            .as_mut()
            .unwrap()
            .rollback(released_reservation)
    );
}

#[test]
fn recognizes_only_supported_vla_checkpoint_extensions() {
    assert!(is_vla_checkpoint_path(
        PathBuf::from("model.gguf").as_path()
    ));
    assert!(is_vla_checkpoint_path(
        PathBuf::from("model.SAFETENSORS").as_path()
    ));
    assert!(!is_vla_checkpoint_path(
        PathBuf::from("model.bin").as_path()
    ));
    assert!(!is_vla_checkpoint_path(PathBuf::from("model").as_path()));
}

#[test]
fn official_llama_launch_args_extend_defaults_with_user_overrides_last() {
    let defaults = vec![
        "--slot-prompt-similarity".to_string(),
        "0".to_string(),
        "--cache-idle-slots".to_string(),
        "-ngl".to_string(),
        "999".to_string(),
        "--cache-ram".to_string(),
        "8192".to_string(),
    ];
    let requested = vec![
        "-np".to_string(),
        "5".to_string(),
        "--cache-ram".to_string(),
        "32768".to_string(),
    ];

    assert_eq!(
        merged_launch_args(
            "llama.cpp-linux-cuda",
            "llama.cpp",
            &defaults,
            Some(&requested)
        ),
        vec![
            "--slot-prompt-similarity",
            "0",
            "--cache-idle-slots",
            "-ngl",
            "999",
            "--cache-ram",
            "8192",
            "-np",
            "5",
            "--cache-ram",
            "32768"
        ]
    );
    assert_eq!(
        merged_launch_args("llama.cpp-linux-cuda", "llama.cpp", &defaults, None),
        defaults
    );

    let automatic = vec!["--gpu-layers=auto".to_string()];
    assert_eq!(
        merged_launch_args(
            "llama.cpp-linux-cuda",
            "llama.cpp",
            &defaults,
            Some(&automatic)
        ),
        vec![
            "--slot-prompt-similarity",
            "0",
            "--cache-idle-slots",
            "--cache-ram",
            "8192"
        ]
    );

    let partial = vec!["-ngl".to_string(), "12".to_string()];
    let merged = merged_launch_args(
        "llama.cpp-linux-cuda",
        "llama.cpp",
        &defaults,
        Some(&partial),
    );
    assert_eq!(gpu_layers_value(&merged), Some("12"));
}

#[test]
fn official_cuda_policy_covers_linux_and_windows_modes() {
    for id in ["llama.cpp-linux-cuda", "llama.cpp-cuda"] {
        let backend = speculative_test_backend(id, "llama.cpp", true);
        assert_eq!(
            llama_cpp_cuda_placement_policy(&backend, &[]).unwrap(),
            Some(LlamaCppCudaPlacementPolicy::Auto)
        );
        assert_eq!(
            llama_cpp_cuda_placement_policy(&backend, &["-ngl".to_string(), "24".to_string()])
                .unwrap(),
            Some(LlamaCppCudaPlacementPolicy::ExplicitPartial(24))
        );
        assert_eq!(
            llama_cpp_cuda_placement_policy(&backend, &["--gpu-layers=999".to_string()]).unwrap(),
            Some(LlamaCppCudaPlacementPolicy::ExplicitFull)
        );
        assert!(
            llama_cpp_cuda_placement_policy(&backend, &["-ngl".to_string()]).is_err(),
            "a dangling GPU-layer flag must fail before launch"
        );
    }
}

#[test]
fn partial_offload_manages_trace_verbosity_and_rejects_disabled_logs() {
    let automatic = managed_placement_evidence_args(
        &["--jinja".to_string()],
        Some(LlamaCppCudaPlacementPolicy::Auto),
    )
    .unwrap();
    assert!(automatic.ends_with(&["-lv".to_string(), "4".to_string()]));
    assert_eq!(
        managed_placement_evidence_args(&automatic, Some(LlamaCppCudaPlacementPolicy::Auto))
            .unwrap(),
        automatic,
        "managed launch arguments must remain idempotent"
    );
    let error = managed_placement_evidence_args(
        &["--log-disable".to_string()],
        Some(LlamaCppCudaPlacementPolicy::ExplicitPartial(12)),
    )
    .unwrap_err();
    assert!(error.to_string().contains("remove --log-disable"));

    let explicit_full = vec!["--log-disable".to_string()];
    assert_eq!(
        managed_placement_evidence_args(
            &explicit_full,
            Some(LlamaCppCudaPlacementPolicy::ExplicitFull)
        )
        .unwrap(),
        explicit_full
    );
}

#[test]
fn partial_offload_provisional_budget_guards_host_and_cuda() {
    let cuda = MemoryDomain::Cuda("0".to_string());
    let estimated = ResourceBudget::from_domains(BTreeMap::from([(cuda.clone(), 1_000)])).unwrap();
    let snapshot = omniinfer_core::resource_ledger::ResourceLedgerSnapshot {
        capacity_snapshot_id: 1,
        capacities: BTreeMap::from([(MemoryDomain::Host, 2_000), (cuda.clone(), 600)]),
        reserved: BTreeMap::new(),
        committed: BTreeMap::new(),
    };
    let provisional = provisional_partial_offload_budget(&estimated, &snapshot).unwrap();
    assert_eq!(provisional.domains()[&MemoryDomain::Host], 1_000);
    assert_eq!(provisional.domains()[&cuda], 600);
}

#[test]
fn parses_partial_llama_cpp_buffers_into_reconciled_domains() {
    let placement = parse_llama_cpp_runtime_placement_text(
        "\
load_tensors: offloaded 20/41 layers to GPU
load_tensors: CPU_Mapped model buffer size = 6000.00 MiB
load_tensors: CUDA0 model buffer size = 12000.00 MiB
llama_kv_cache: CPU KV buffer size = 64.00 MiB
llama_kv_cache: CUDA0 KV buffer size = 256.00 MiB
sched_reserve: CPU compute buffer size = 128.00 MiB
sched_reserve: CUDA0 compute buffer size = 512.00 MiB
",
        "3",
        LlamaCppCudaPlacementPolicy::Auto,
    )
    .unwrap();
    assert_eq!(placement.mode, "partial");
    assert_eq!(placement.offloaded_layers, Some(20));
    assert_eq!(placement.total_layers, Some(41));
    assert!(placement.reported_bytes[&MemoryDomain::Host] > 6 * GIB);
    assert!(placement.reported_bytes[&MemoryDomain::Cuda("3".to_string())] > 12 * GIB);
    assert!(
        placement.reconciled_budget.domains()[&MemoryDomain::Host]
            > placement.reported_bytes[&MemoryDomain::Host]
    );
}

#[test]
fn host_model_buffer_takes_precedence_over_all_layers_offloaded() {
    let placement = parse_llama_cpp_runtime_placement_text(
        "load_tensors: offloaded 41/41 layers to GPU\n\
         load_tensors: CPU_Mapped model buffer size = 20699.72 MiB\n\
         load_tensors: CUDA0 model buffer size = 9340.14 MiB\n\
         llama_kv_cache: CUDA0 KV buffer size = 80.00 MiB\n",
        "5",
        LlamaCppCudaPlacementPolicy::Auto,
    )
    .unwrap();
    assert_eq!(placement.mode, "partial");
    assert_eq!(placement.offloaded_layers, Some(41));
    assert_eq!(placement.total_layers, Some(41));
}

#[test]
fn host_scratch_buffer_does_not_make_cuda_model_partial() {
    let placement = parse_llama_cpp_runtime_placement_text(
        "load_tensors: offloaded 41/41 layers to GPU\n\
         load_tensors: CUDA0 model buffer size = 9340.14 MiB\n\
         sched_reserve: CPU compute buffer size = 24.93 MiB\n\
         sched_reserve: CUDA0 compute buffer size = 497.00 MiB\n",
        "5",
        LlamaCppCudaPlacementPolicy::Auto,
    )
    .unwrap();
    assert_eq!(placement.mode, "full");
}

#[test]
fn placement_parser_preserves_cuda_visible_device_order() {
    let placement = parse_llama_cpp_runtime_placement_text(
        "load_tensors: offloaded 2/4 layers to GPU\n\
         load_tensors: CPU_Mapped model buffer size = 8.00 MiB\n\
         load_tensors: CUDA0 model buffer size = 16.00 MiB\n\
         load_tensors: CUDA1 model buffer size = 4.00 MiB\n",
        "3,1",
        LlamaCppCudaPlacementPolicy::Auto,
    )
    .unwrap();
    assert_eq!(
        placement.reported_bytes[&MemoryDomain::Cuda("3".to_string())],
        16 * MIB
    );
    assert_eq!(
        placement.reported_bytes[&MemoryDomain::Cuda("1".to_string())],
        4 * MIB
    );
}

#[test]
fn automatic_placement_without_buffer_evidence_fails_closed() {
    let error = parse_llama_cpp_runtime_placement_text(
        "load_tensors: offloaded 2/4 layers to GPU\n",
        "0",
        LlamaCppCudaPlacementPolicy::Auto,
    )
    .unwrap_err();
    assert!(error.to_string().contains("did not report CPU/CUDA buffer"));
}

#[test]
fn placement_parser_sums_persistent_buffers_and_uses_peak_scratch_buffers() {
    let placement = parse_llama_cpp_runtime_placement_text(
        "load_tensors: offloaded 2/4 layers to GPU\n\
         load_tensors: CUDA0 model buffer size = 16.00 MiB\n\
         load_tensors: CUDA0 model buffer size = 4.00 MiB\n\
         sched_reserve: CUDA0 compute buffer size = 8.00 MiB\n\
         sched_reserve: CUDA0 compute buffer size = 12.00 MiB\n\
         llama_context: CUDA0  output buffer size = 2.00 MiB\n",
        "0",
        LlamaCppCudaPlacementPolicy::Auto,
    )
    .unwrap();
    assert_eq!(
        placement.reported_bytes[&MemoryDomain::Cuda("0".to_string())],
        34 * MIB
    );
}

#[test]
fn non_official_llama_launch_args_keep_replacement_semantics() {
    let defaults = vec!["--jinja".to_string(), "-ngl".to_string(), "999".to_string()];
    let requested = vec!["-ngl".to_string(), "12".to_string()];

    assert_eq!(
        merged_launch_args(
            "ik_llama.cpp-linux-cuda",
            "llama.cpp",
            &defaults,
            Some(&requested)
        ),
        requested
    );
}

#[test]
fn wsl_rocm_cold_start_retry_requires_a_safe_total_budget() {
    assert_eq!(
        wsl_rocm_cold_start_retry_timeout("vllm-wsl2-rocm", Duration::from_secs(420)),
        Some(Duration::from_secs(120))
    );
    assert_eq!(
        wsl_rocm_cold_start_retry_timeout("vllm-wsl2-rocm", Duration::from_secs(359)),
        None
    );
    assert_eq!(
        wsl_rocm_cold_start_retry_timeout("vllm-wsl2-cuda", Duration::from_secs(420)),
        None
    );
}

#[test]
fn ready_timeout_retries_once_with_the_remaining_budget() {
    let total_timeout = Duration::from_secs(300);
    let mut attempts = Vec::new();
    let cancelled = AtomicBool::new(false);
    let result = retry_after_ready_timeout(
        total_timeout,
        Duration::from_secs(120),
        Duration::ZERO,
        &cancelled,
        |timeout| {
            attempts.push(timeout);
            if attempts.len() == 1 {
                Err(RuntimeProcessError::ReadyTimeout)
            } else {
                Ok("ready")
            }
        },
    )
    .unwrap();

    assert_eq!(result, "ready");
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0], Duration::from_secs(120));
    assert!(attempts[1] <= total_timeout);
    assert!(attempts[1] >= Duration::from_secs(299));
}

#[test]
fn cold_start_retry_does_not_mask_early_exit() {
    let mut attempts = 0;
    let cancelled = AtomicBool::new(false);
    let error = retry_after_ready_timeout(
        Duration::from_secs(300),
        Duration::from_secs(120),
        Duration::ZERO,
        &cancelled,
        |_| {
            attempts += 1;
            Err::<(), _>(RuntimeProcessError::EarlyExit)
        },
    )
    .unwrap_err();

    assert!(matches!(error, RuntimeProcessError::EarlyExit));
    assert_eq!(attempts, 1);
}

#[test]
fn ready_timeout_does_not_retry_without_post_cooldown_budget() {
    let mut attempts = 0;
    let cancelled = AtomicBool::new(false);
    let error = retry_after_ready_timeout(
        Duration::from_millis(1),
        Duration::ZERO,
        Duration::from_millis(1),
        &cancelled,
        |_| {
            attempts += 1;
            Err::<(), _>(RuntimeProcessError::ReadyTimeout)
        },
    )
    .unwrap_err();

    assert!(matches!(error, RuntimeProcessError::ReadyTimeout));
    assert_eq!(attempts, 1);
}
