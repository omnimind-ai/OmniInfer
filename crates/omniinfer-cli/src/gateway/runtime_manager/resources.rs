use super::*;

pub(super) fn build_runtime_resource_budget(
    payload: &Value,
    backend: &backend_registry::BackendSpec,
    model: &str,
    mmproj: Option<&str>,
    ctx_size: u32,
    launch_args: &[String],
    cuda_visible_devices: Option<&str>,
    replicate_across_domains: bool,
) -> Result<ResourceBudget> {
    let domains = if cfg!(target_os = "macos")
        || backend
            .capabilities
            .iter()
            .any(|value| value == "shared-memory")
    {
        vec![MemoryDomain::Unified("system".to_string())]
    } else if backend.capabilities.iter().any(|value| value == "cuda") {
        let devices = parse_cuda_devices(cuda_visible_devices.ok_or_else(|| {
            anyhow::anyhow!("CUDA resource budgeting requires a selected device")
        })?);
        if devices.is_empty() {
            anyhow::bail!("CUDA resource budgeting requires a selected device");
        }
        devices.into_iter().map(MemoryDomain::Cuda).collect()
    } else {
        vec![MemoryDomain::Host]
    };
    let explicit_total = payload
        .get("resource_budget_bytes")
        .and_then(Value::as_u64)
        .filter(|bytes| *bytes > 0);
    let weights = artifact_size_bytes(&PathBuf::from(model))?;
    let projector = mmproj
        .map(|path| artifact_size_bytes(&PathBuf::from(path)))
        .transpose()?
        .flatten()
        .unwrap_or(0);
    if backend.family == "freetoken" {
        return build_freetoken_resource_budget(
            payload,
            weights,
            explicit_total,
            cuda_visible_devices,
        );
    }
    if backend.family == "stable-diffusion.cpp" {
        let weights = weights.ok_or_else(|| {
            anyhow::anyhow!("stable-diffusion.cpp model artifact size is unknown: {model}")
        })?;
        return build_stable_diffusion_resource_budget(
            model,
            weights.max(1),
            launch_args,
            explicit_total,
        );
    }
    let Some(weights) = weights else {
        let total = explicit_total.ok_or_else(|| {
            anyhow::anyhow!(
                "model artifact size is unknown; provide a non-zero resource_budget_bytes value"
            )
        })?;
        return Ok(ResourceBudget::from_components(assign_component(
            "client_provided_total",
            total,
            &domains,
            replicate_across_domains,
        )?)?);
    };
    let weights = weights.max(1);
    let base = weights
        .checked_add(projector)
        .ok_or_else(|| anyhow::anyhow!("model artifact size overflow"))?;
    // Projector bytes affect framework/slack, not model KV or activation sizing.
    let parameter_proxy = weights.saturating_mul(2).max(GIB);
    let ctx = u64::from(ctx_size.max(1));
    let kv_cache = checked_scaled(parameter_proxy, 3, 100)?
        .checked_mul(ctx)
        .and_then(|bytes| bytes.checked_div(u64::from(DEFAULT_LOAD_CONTEXT_SIZE)))
        .unwrap_or(u64::MAX)
        .max(256 * MIB);
    let activation_ctx = ctx.min(u64::from(DEFAULT_LOAD_CONTEXT_SIZE) * 4);
    let activation = checked_scaled(parameter_proxy, 1, 100)?
        .checked_mul(activation_ctx)
        .and_then(|bytes| bytes.checked_div(u64::from(DEFAULT_LOAD_CONTEXT_SIZE)))
        .unwrap_or(u64::MAX)
        .max(128 * MIB);
    let framework = checked_scaled(base, 8, 100)?.max(384 * MIB);
    let allocator_slack = checked_scaled(base, 4, 100)?.max(160 * MIB);
    let mut components = Vec::new();
    for (name, bytes) in [
        ("weights", weights),
        ("kv_cache", kv_cache),
        ("activation", activation),
        ("framework_overhead", framework),
        ("allocator_slack", allocator_slack),
    ] {
        components.extend(assign_component(
            name,
            bytes,
            &domains,
            replicate_across_domains,
        )?);
    }
    if projector > 0 {
        components.extend(assign_component(
            "mmproj",
            projector,
            &domains,
            replicate_across_domains,
        )?);
    }
    let estimated = ResourceBudget::from_components(components)?;
    if let Some(explicit_total) = explicit_total {
        let estimated_minimum = if replicate_across_domains {
            estimated.domains().values().copied().max().unwrap_or(0)
        } else {
            estimated
                .domains()
                .values()
                .try_fold(0_u64, |total, bytes| total.checked_add(*bytes))
                .ok_or_else(|| anyhow::anyhow!("resource budget overflow"))?
        };
        if explicit_total < estimated_minimum {
            anyhow::bail!(
                "resource_budget_bytes is below the estimated minimum of {estimated_minimum} bytes"
            );
        }
        return Ok(ResourceBudget::from_components(assign_component(
            "client_provided_total",
            explicit_total,
            &domains,
            replicate_across_domains,
        )?)?);
    }
    Ok(estimated)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DiffusionModule {
    Diffusion,
    TextEncoder,
    ClipVision,
    Vae,
    ControlNet,
    PhotoMaker,
    Upscaler,
    Detector,
}

#[derive(Debug)]
struct DiffusionArtifact {
    component_name: &'static str,
    module: DiffusionModule,
    bytes: u64,
}

#[derive(Debug, Default)]
struct DiffusionBackendAssignment {
    default: Option<String>,
    modules: BTreeMap<DiffusionModule, String>,
}

impl DiffusionBackendAssignment {
    fn resolve(&self, module: DiffusionModule) -> Option<&str> {
        self.modules
            .get(&module)
            .map(String::as_str)
            .or(self.default.as_deref())
    }
}

fn build_stable_diffusion_resource_budget(
    model: &str,
    model_bytes: u64,
    launch_args: &[String],
    explicit_total: Option<u64>,
) -> Result<ResourceBudget> {
    if launch_flag_enabled(launch_args, "--auto-fit") {
        anyhow::bail!(
            "stable-diffusion.cpp --auto-fit placement cannot be proven before launch; use explicit --backend and --params-backend assignments"
        );
    }
    for flag in [
        "--rpc-servers",
        "--embd-dir",
        "--lora-model-dir",
        "--hires-upscalers-dir",
    ] {
        if launch_arg_value(launch_args, flag)?.is_some() {
            anyhow::bail!(
                "stable-diffusion.cpp {flag} can change runtime resources dynamically and is not supported by safe admission"
            );
        }
    }
    if launch_arg_value(launch_args, "--max-vram")?.is_some_and(|value| value.trim() != "0") {
        anyhow::bail!(
            "stable-diffusion.cpp --max-vram graph placement is not supported by pre-launch resource admission"
        );
    }
    for flag in ["--type", "--tensor-type-rules"] {
        if launch_arg_value(launch_args, flag)?.is_some() {
            anyhow::bail!(
                "stable-diffusion.cpp {flag} can change in-memory weight sizes and is not supported by safe admission"
            );
        }
    }
    let mut artifacts = vec![DiffusionArtifact {
        component_name: "diffusion_weights",
        module: DiffusionModule::Diffusion,
        bytes: model_bytes,
    }];
    for (flags, component_name, module) in [
        (
            &["--clip_l"][..],
            "clip_l_weights",
            DiffusionModule::TextEncoder,
        ),
        (
            &["--clip_g"][..],
            "clip_g_weights",
            DiffusionModule::TextEncoder,
        ),
        (
            &["--t5xxl"][..],
            "t5xxl_weights",
            DiffusionModule::TextEncoder,
        ),
        (
            &["--llm", "--qwen2vl"][..],
            "text_encoder_weights",
            DiffusionModule::TextEncoder,
        ),
        (
            &["--clip_vision"][..],
            "clip_vision_weights",
            DiffusionModule::ClipVision,
        ),
        (
            &["--llm_vision", "--qwen2vl_vision"][..],
            "llm_vision_weights",
            DiffusionModule::TextEncoder,
        ),
        (
            &["--high-noise-diffusion-model"][..],
            "high_noise_diffusion_weights",
            DiffusionModule::Diffusion,
        ),
        (
            &["--uncond-diffusion-model"][..],
            "unconditional_diffusion_weights",
            DiffusionModule::Diffusion,
        ),
        (
            &["--embeddings-connectors"][..],
            "embedding_connector_weights",
            DiffusionModule::TextEncoder,
        ),
        (&["--vae"][..], "video_vae_weights", DiffusionModule::Vae),
        (
            &["--audio-vae"][..],
            "audio_vae_weights",
            DiffusionModule::Vae,
        ),
        (
            &["--taesd", "--tae"][..],
            "tiny_autoencoder_weights",
            DiffusionModule::Vae,
        ),
        (
            &["--control-net"][..],
            "control_net_weights",
            DiffusionModule::ControlNet,
        ),
        (
            &["--ip-adapter"][..],
            "ip_adapter_weights",
            DiffusionModule::Diffusion,
        ),
        (
            &["--motion-module"][..],
            "motion_module_weights",
            DiffusionModule::Diffusion,
        ),
        (
            &["--photo-maker"][..],
            "photo_maker_weights",
            DiffusionModule::PhotoMaker,
        ),
        (
            &["--pulid-weights"][..],
            "pulid_weights",
            DiffusionModule::PhotoMaker,
        ),
        (
            &["--upscale-model"][..],
            "upscaler_weights",
            DiffusionModule::Upscaler,
        ),
        (
            &["--ad-model"][..],
            "detector_weights",
            DiffusionModule::Detector,
        ),
    ] {
        let Some((flag, path)) = launch_arg_value_any(launch_args, flags)? else {
            continue;
        };
        let bytes = artifact_size_bytes(&PathBuf::from(path))?.ok_or_else(|| {
            anyhow::anyhow!("stable-diffusion.cpp component size is unknown for {flag}: {path}")
        })?;
        artifacts.push(DiffusionArtifact {
            component_name,
            module,
            bytes: bytes.max(1),
        });
    }

    let mut runtime_assignment = DiffusionBackendAssignment::default();
    if launch_flag_enabled(launch_args, "--clip-on-cpu") {
        runtime_assignment
            .modules
            .insert(DiffusionModule::TextEncoder, "cpu".to_string());
    }
    if launch_flag_enabled(launch_args, "--vae-on-cpu") {
        runtime_assignment
            .modules
            .insert(DiffusionModule::Vae, "cpu".to_string());
    }
    if launch_flag_enabled(launch_args, "--control-net-cpu") {
        runtime_assignment
            .modules
            .insert(DiffusionModule::ControlNet, "cpu".to_string());
    }
    let explicit_runtime = parse_diffusion_backend_assignment(
        launch_arg_value(launch_args, "--backend")?.unwrap_or(""),
    )?;
    if explicit_runtime.default.is_some() {
        runtime_assignment.default = explicit_runtime.default;
    }
    runtime_assignment.modules.extend(explicit_runtime.modules);
    let mut params_assignment = if launch_flag_enabled(launch_args, "--offload-to-cpu") {
        DiffusionBackendAssignment {
            default: Some("cpu".to_string()),
            modules: BTreeMap::new(),
        }
    } else {
        DiffusionBackendAssignment::default()
    };
    let explicit_params = parse_diffusion_backend_assignment(
        launch_arg_value(launch_args, "--params-backend")?.unwrap_or(""),
    )?;
    if explicit_params.default.is_some() {
        params_assignment.default = explicit_params.default;
    }
    params_assignment.modules.extend(explicit_params.modules);

    let mut components = Vec::new();
    let mut domain_bases = BTreeMap::<MemoryDomain, u64>::new();
    for artifact in &artifacts {
        let runtime_domain = diffusion_memory_domain(
            runtime_assignment
                .resolve(artifact.module)
                .unwrap_or("default"),
        )?;
        let params_domain = match params_assignment.resolve(artifact.module) {
            Some(value) => diffusion_memory_domain(value)?,
            None => runtime_domain.clone(),
        };
        let params_total = domain_bases.entry(params_domain.clone()).or_insert(0);
        *params_total = params_total
            .checked_add(artifact.bytes)
            .ok_or_else(|| anyhow::anyhow!("diffusion parameter budget overflow"))?;
        components.push(BudgetComponent {
            name: artifact.component_name.to_string(),
            domain: params_domain.clone(),
            bytes: artifact.bytes,
        });
        if runtime_domain != params_domain {
            let runtime_total = domain_bases.entry(runtime_domain.clone()).or_insert(0);
            *runtime_total = runtime_total
                .checked_add(artifact.bytes)
                .ok_or_else(|| anyhow::anyhow!("diffusion runtime budget overflow"))?;
            components.push(BudgetComponent {
                name: format!("{}_runtime_staging", artifact.component_name),
                domain: runtime_domain,
                bytes: artifact.bytes,
            });
        }
    }

    for (domain, base) in domain_bases {
        for (name, bytes) in [
            (
                "runtime_workspace",
                checked_scaled(base, 8, 100)?.max(256 * MIB),
            ),
            (
                "framework_overhead",
                checked_scaled(base, 8, 100)?.max(384 * MIB),
            ),
            (
                "allocator_slack",
                checked_scaled(base, 4, 100)?.max(160 * MIB),
            ),
        ] {
            components.push(BudgetComponent {
                name: name.to_string(),
                domain: domain.clone(),
                bytes,
            });
        }
    }

    let mut budget = ResourceBudget::from_components(components)?;
    if let Some(explicit_total) = explicit_total {
        let estimated_total = sum_domain_bytes(budget.domains())?;
        if explicit_total < estimated_total {
            anyhow::bail!(
                "resource_budget_bytes is below the estimated stable-diffusion.cpp minimum of {estimated_total} bytes for {model}"
            );
        }
        let extra = explicit_total - estimated_total;
        if extra > 0 {
            let mut components = budget.components().to_vec();
            let mut remaining = extra;
            let domains = budget.domains().iter().collect::<Vec<_>>();
            for (index, (domain, bytes)) in domains.iter().enumerate() {
                let share = if index + 1 == domains.len() {
                    remaining
                } else {
                    proportional_share(extra, **bytes, estimated_total)?.min(remaining)
                };
                if share > 0 {
                    components.push(BudgetComponent {
                        name: "client_provided_slack".to_string(),
                        domain: (*domain).clone(),
                        bytes: share,
                    });
                    remaining -= share;
                }
            }
            budget = ResourceBudget::from_components(components)?;
        }
    }
    Ok(budget)
}

fn sum_domain_bytes(domains: &BTreeMap<MemoryDomain, u64>) -> Result<u64> {
    domains.values().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("resource budget overflow"))
    })
}

fn proportional_share(value: u64, numerator: u64, denominator: u64) -> Result<u64> {
    if denominator == 0 {
        anyhow::bail!("resource budget denominator must be non-zero");
    }
    u64::try_from(u128::from(value) * u128::from(numerator) / u128::from(denominator))
        .map_err(|_| anyhow::anyhow!("resource budget overflow"))
}

fn launch_arg_value<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>> {
    Ok(launch_arg_value_any(args, &[flag])?.map(|(_, value)| value))
}

fn launch_arg_value_any<'a>(
    args: &'a [String],
    flags: &[&str],
) -> Result<Option<(String, &'a str)>> {
    let mut selected = None;
    let mut index = 0;
    while index < args.len() {
        let token = args[index].as_str();
        let inline = flags.iter().find_map(|flag| {
            token
                .strip_prefix(&format!("{flag}="))
                .map(|value| (*flag, value))
        });
        if let Some((flag, value)) = inline {
            if value.is_empty() {
                anyhow::bail!("stable-diffusion.cpp {flag} requires a value");
            }
            selected = Some((flag.to_string(), value));
        } else if let Some(flag) = flags.iter().copied().find(|flag| token == *flag) {
            let value = args
                .get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| anyhow::anyhow!("stable-diffusion.cpp {flag} requires a value"))?;
            if value.starts_with('-') {
                anyhow::bail!("stable-diffusion.cpp {flag} requires a value");
            }
            selected = Some((flag.to_string(), value));
            index += 1;
        }
        index += 1;
    }
    Ok(selected)
}

fn launch_flag_enabled(args: &[String], flag: &str) -> bool {
    args.iter().any(|token| {
        token == flag
            || token
                .strip_prefix(&format!("{flag}="))
                .is_some_and(|value| value.eq_ignore_ascii_case("true") || value == "1")
    })
}

fn parse_diffusion_backend_assignment(spec: &str) -> Result<DiffusionBackendAssignment> {
    let mut assignment = DiffusionBackendAssignment::default();
    for raw_part in spec.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            continue;
        }
        let Some((raw_key, raw_value)) = part.split_once('=') else {
            assignment.default = Some(part.to_string());
            continue;
        };
        let key = raw_key.trim().to_ascii_lowercase().replace(['-', '_'], "");
        let value = raw_value.trim();
        if key.is_empty() || value.is_empty() {
            anyhow::bail!("invalid stable-diffusion.cpp backend assignment: {part}");
        }
        if matches!(key.as_str(), "all" | "default" | "*") {
            assignment.default = Some(value.to_string());
            continue;
        }
        let module = match key.as_str() {
            "diffusion" | "model" | "unet" | "dit" => DiffusionModule::Diffusion,
            "te" | "clip" | "text" | "textencoder" | "textencoders" | "conditioner" | "cond"
            | "llm" | "t5" | "t5xxl" => DiffusionModule::TextEncoder,
            "clipvision" | "vision" => DiffusionModule::ClipVision,
            "vae" | "firststage" | "autoencoder" | "tae" => DiffusionModule::Vae,
            "controlnet" | "control" => DiffusionModule::ControlNet,
            "photomaker" | "photomakerid" | "pmid" | "photo" => DiffusionModule::PhotoMaker,
            "upscaler" | "esrgan" | "hires" => DiffusionModule::Upscaler,
            "detector" | "adetailer" | "yolo" => DiffusionModule::Detector,
            _ => anyhow::bail!(
                "stable-diffusion.cpp resource budgeting does not support module assignment: {raw_key}"
            ),
        };
        assignment.modules.insert(module, value.to_string());
    }
    Ok(assignment)
}

fn diffusion_memory_domain(value: &str) -> Result<MemoryDomain> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "cpu" {
        return Ok(MemoryDomain::Host);
    }
    if matches!(normalized.as_str(), "" | "auto" | "default" | "gpu") {
        return Ok(MemoryDomain::Vulkan("0".to_string()));
    }
    if let Some(index) = normalized.strip_prefix("vulkan")
        && !index.is_empty()
        && index.chars().all(|character| character.is_ascii_digit())
    {
        return Ok(MemoryDomain::Vulkan(index.to_string()));
    }
    anyhow::bail!(
        "stable-diffusion.cpp resource budgeting cannot map backend device '{value}'; use cpu or vulkan<index>"
    )
}

fn build_freetoken_resource_budget(
    payload: &Value,
    weights: Option<u64>,
    explicit_host_bytes: Option<u64>,
    cuda_visible_devices: Option<&str>,
) -> Result<ResourceBudget> {
    let host_total = match (weights, explicit_host_bytes) {
        (Some(weights), explicit) => {
            let framework = checked_scaled(weights, 8, 100)?.max(384 * MIB);
            let minimum = weights
                .checked_add(framework)
                .ok_or_else(|| anyhow::anyhow!("FreeToken host resource budget overflow"))?;
            if let Some(explicit) = explicit
                && explicit < minimum
            {
                anyhow::bail!(
                    "resource_budget_bytes is below the estimated FreeToken host minimum of {minimum} bytes"
                );
            }
            explicit.unwrap_or(minimum)
        }
        (None, Some(explicit)) => explicit,
        (None, None) => {
            anyhow::bail!(
                "FreeToken model size is unknown; provide a non-zero resource_budget_bytes host-memory reservation"
            );
        }
    };
    let devices = cuda_visible_devices.ok_or_else(|| {
        anyhow::anyhow!("FreeToken CUDA resource budgeting requires a selected device")
    })?;
    let available = cuda_available_bytes(devices)?;
    let ratio = freetoken_memory_ratio_millionths(payload)?;
    let mut components = vec![BudgetComponent {
        name: "host_model_and_runtime".to_string(),
        domain: MemoryDomain::Host,
        bytes: host_total,
    }];
    for (domain, bytes) in available {
        components.push(BudgetComponent {
            name: "elastic_cuda_pool".to_string(),
            domain,
            bytes: checked_scaled(bytes, ratio, 1_000_000)?.max(1),
        });
    }
    Ok(ResourceBudget::from_components(components)?)
}

fn freetoken_memory_ratio_millionths(payload: &Value) -> Result<u64> {
    let args = payload
        .get("launch_args")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let mut value = None;
    let mut index = 0;
    while index < args.len() {
        let token = args[index];
        if let Some(inline) = token.strip_prefix("--memory-ratio=") {
            value = Some(inline);
        } else if token == "--memory-ratio" {
            value = args.get(index + 1).copied();
            index += 1;
        }
        index += 1;
    }
    let ratio = value.unwrap_or("0.9").parse::<f64>().map_err(|_| {
        anyhow::anyhow!("FreeToken --memory-ratio must be a number between 0 and 1")
    })?;
    if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
        anyhow::bail!("FreeToken --memory-ratio must be a number between 0 and 1");
    }
    Ok((ratio * 1_000_000.0).round() as u64)
}

pub(super) fn artifact_size_bytes(path: &PathBuf) -> Result<Option<u64>> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(None);
    };
    if metadata.is_file() {
        return Ok(Some(metadata.len()));
    }
    if !metadata.is_dir() {
        return Ok(None);
    }
    let mut total = 0_u64;
    let mut pending = vec![path.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                total = total
                    .checked_add(entry.metadata()?.len())
                    .ok_or_else(|| anyhow::anyhow!("model artifact size overflow"))?;
            }
        }
    }
    Ok((total > 0).then_some(total))
}

pub(super) fn checked_scaled(value: u64, numerator: u64, denominator: u64) -> Result<u64> {
    value
        .checked_mul(numerator)
        .and_then(|scaled| scaled.checked_div(denominator))
        .ok_or_else(|| anyhow::anyhow!("resource budget overflow"))
}

pub(super) fn assign_component(
    name: &str,
    bytes: u64,
    domains: &[MemoryDomain],
    replicate_across_domains: bool,
) -> Result<Vec<BudgetComponent>> {
    if replicate_across_domains {
        if domains.is_empty() || bytes == 0 {
            anyhow::bail!("resource component requires non-zero bytes and at least one domain");
        }
        return Ok(domains
            .iter()
            .map(|domain| BudgetComponent {
                name: name.to_string(),
                domain: domain.clone(),
                bytes,
            })
            .collect());
    }
    distribute_component(name, bytes, domains)
}

pub(super) fn distribute_component(
    name: &str,
    bytes: u64,
    domains: &[MemoryDomain],
) -> Result<Vec<BudgetComponent>> {
    let count = u64::try_from(domains.len())?;
    if count == 0 || bytes == 0 {
        anyhow::bail!("resource component requires non-zero bytes and at least one domain");
    }
    let base = bytes / count;
    let remainder = bytes % count;
    domains
        .iter()
        .enumerate()
        .map(|(index, domain)| {
            let bytes = base + u64::from(u64::try_from(index)? < remainder);
            Ok(BudgetComponent {
                name: name.to_string(),
                domain: domain.clone(),
                bytes,
            })
        })
        .collect()
}

pub(super) fn parse_cuda_devices(visible_devices: &str) -> Vec<String> {
    let mut devices = visible_devices
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    devices.sort();
    devices.dedup();
    devices
}

pub(super) fn detect_available_resources(
    cuda_visible_devices: Option<&str>,
    vulkan_devices: &[String],
) -> Result<BTreeMap<MemoryDomain, u64>> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available_memory = system.available_memory();
    if available_memory == 0 {
        anyhow::bail!("available system memory could not be detected");
    }
    let mut domains = BTreeMap::from([
        (MemoryDomain::Host, available_memory),
        (
            MemoryDomain::Unified("system".to_string()),
            available_memory,
        ),
    ]);
    if let Some(devices) = cuda_visible_devices {
        domains.extend(cuda_available_bytes(devices)?);
    }
    if !vulkan_devices.is_empty() {
        domains.extend(vulkan_available_bytes(vulkan_devices)?);
    }
    Ok(domains)
}

#[cfg(test)]
pub(super) fn vulkan_available_bytes(
    requested_devices: &[String],
) -> Result<BTreeMap<MemoryDomain, u64>> {
    if std::env::var_os("OMNIINFER_TEST_REAL_VULKAN_CAPACITY").is_some() {
        return query_vulkan_available_bytes(requested_devices);
    }
    const TEST_VULKAN_CAPACITY: u64 = 1024 * GIB;
    if requested_devices.is_empty() {
        anyhow::bail!("Vulkan device selection is empty");
    }
    Ok(requested_devices
        .iter()
        .cloned()
        .map(|device| (MemoryDomain::Vulkan(device), TEST_VULKAN_CAPACITY))
        .collect())
}

#[cfg(not(test))]
pub(super) fn vulkan_available_bytes(
    requested_devices: &[String],
) -> Result<BTreeMap<MemoryDomain, u64>> {
    query_vulkan_available_bytes(requested_devices)
}

fn query_vulkan_available_bytes(
    requested_devices: &[String],
) -> Result<BTreeMap<MemoryDomain, u64>> {
    use ash::vk;
    use std::ffi::CStr;

    if requested_devices.is_empty() {
        anyhow::bail!("Vulkan device selection is empty");
    }
    let entry = unsafe { ash::Entry::load() }
        .map_err(|error| anyhow::anyhow!("failed to load the Vulkan loader: {error}"))?;
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 1, 0),
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|error| anyhow::anyhow!("failed to create a Vulkan capacity probe: {error}"))?;
    let result = (|| -> Result<BTreeMap<MemoryDomain, u64>> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|error| anyhow::anyhow!("failed to enumerate Vulkan devices: {error}"))?;
        let mut available = BTreeMap::new();
        for requested in requested_devices {
            let index = requested.parse::<usize>().map_err(|_| {
                anyhow::anyhow!("invalid Vulkan device index in resource budget: {requested}")
            })?;
            let device = *physical_devices.get(index).ok_or_else(|| {
                anyhow::anyhow!(
                    "selected Vulkan device was not reported by the loader: {requested}"
                )
            })?;
            let extensions = unsafe { instance.enumerate_device_extension_properties(device) }
                .map_err(|error| {
                    anyhow::anyhow!("failed to query Vulkan device {requested} extensions: {error}")
                })?;
            let supports_budget = extensions.iter().any(|extension| {
                let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                name == vk::EXT_MEMORY_BUDGET_NAME
            });
            if !supports_budget {
                anyhow::bail!(
                    "Vulkan device {requested} does not expose VK_EXT_memory_budget; safe admission is unavailable"
                );
            }
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let memory_properties = {
                let mut memory =
                    vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
                unsafe { instance.get_physical_device_memory_properties2(device, &mut memory) };
                memory.memory_properties
            };
            let heap_count = usize::try_from(memory_properties.memory_heap_count)?;
            let mut free = 0_u64;
            for index in 0..heap_count {
                let heap = memory_properties.memory_heaps[index];
                if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    let heap_free = budget.heap_budget[index]
                        .saturating_sub(budget.heap_usage[index])
                        .min(heap.size);
                    free = free
                        .checked_add(heap_free)
                        .ok_or_else(|| anyhow::anyhow!("Vulkan capacity overflow"))?;
                }
            }
            if free == 0 {
                anyhow::bail!(
                    "Vulkan device {requested} reported zero free memory through VK_EXT_memory_budget"
                );
            }
            available.insert(MemoryDomain::Vulkan(requested.clone()), free);
        }
        Ok(available)
    })();
    unsafe { instance.destroy_instance(None) };
    result
}

#[cfg(test)]
pub(super) fn detect_cuda_device_ids() -> Result<Vec<String>> {
    Ok(vec!["0".to_string()])
}

#[cfg(not(test))]
pub(super) fn detect_cuda_device_ids() -> Result<Vec<String>> {
    let output = std::process::Command::new(nvidia_smi_executable())
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()?;
    if !output.status.success() {
        anyhow::bail!("nvidia-smi device query failed");
    }
    let devices = parse_cuda_devices(&String::from_utf8_lossy(&output.stdout).replace('\n', ","));
    if devices.is_empty() {
        anyhow::bail!("nvidia-smi did not report any CUDA devices");
    }
    Ok(devices)
}

#[cfg(test)]
pub(super) fn cuda_available_bytes(visible_devices: &str) -> Result<BTreeMap<MemoryDomain, u64>> {
    const TEST_CUDA_CAPACITY: u64 = 1024 * GIB;
    let requested = parse_cuda_devices(visible_devices);
    if requested.is_empty() {
        anyhow::bail!("CUDA device selection is empty");
    }
    Ok(requested
        .into_iter()
        .map(|device| (MemoryDomain::Cuda(device), TEST_CUDA_CAPACITY))
        .collect())
}

#[cfg(not(test))]
pub(super) fn cuda_available_bytes(visible_devices: &str) -> Result<BTreeMap<MemoryDomain, u64>> {
    let requested = parse_cuda_devices(visible_devices);
    if requested.is_empty() {
        anyhow::bail!("CUDA device selection is empty");
    }
    let output = std::process::Command::new(nvidia_smi_executable())
        .args([
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        anyhow::bail!("nvidia-smi memory query failed");
    }
    let rows = String::from_utf8_lossy(&output.stdout);
    let mut available = BTreeMap::new();
    for requested_device in &requested {
        let memory_mib = rows.lines().find_map(|line| {
            let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
            (parts.len() >= 3 && (parts[0] == *requested_device || parts[1] == *requested_device))
                .then(|| parts[2].parse::<u64>().ok())
                .flatten()
        });
        let memory_mib = memory_mib.ok_or_else(|| {
            anyhow::anyhow!(
                "selected CUDA device was not reported by nvidia-smi: {requested_device}"
            )
        })?;
        available.insert(
            MemoryDomain::Cuda(requested_device.clone()),
            memory_mib
                .checked_mul(MIB)
                .ok_or_else(|| anyhow::anyhow!("CUDA capacity overflow"))?,
        );
    }
    Ok(available)
}

#[cfg(not(test))]
fn nvidia_smi_executable() -> std::ffi::OsString {
    std::env::var_os("OMNIINFER_VLLM_NVIDIA_SMI").unwrap_or_else(|| "nvidia-smi".into())
}

pub(super) fn merge_domain_totals(
    left: &BTreeMap<MemoryDomain, u64>,
    right: &BTreeMap<MemoryDomain, u64>,
) -> Result<BTreeMap<MemoryDomain, u64>> {
    let mut merged = left.clone();
    for (domain, bytes) in right {
        let total = merged.entry(domain.clone()).or_insert(0);
        *total = total
            .checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("resource usage overflow"))?;
    }
    Ok(merged)
}

pub(super) fn domain_bytes_payload(domains: &BTreeMap<MemoryDomain, u64>) -> Value {
    Value::Object(
        domains
            .iter()
            .map(|(domain, bytes)| (domain.key(), json!(bytes)))
            .collect(),
    )
}

pub(super) fn resource_budget_payload(budget: &ResourceBudget) -> Value {
    json!({
        "domains_bytes": domain_bytes_payload(budget.domains()),
        "components": budget.components().iter().map(|component| json!({
            "name": component.name,
            "domain": component.domain.key(),
            "bytes": component.bytes,
        })).collect::<Vec<_>>(),
    })
}
