use super::*;

pub(super) fn build_runtime_resource_budget(
    payload: &Value,
    backend: &backend_registry::BackendSpec,
    model: &str,
    mmproj: Option<&str>,
    ctx_size: u32,
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
    let diffusion_components = if backend.family == "stable-diffusion.cpp" {
        stable_diffusion_component_bytes(payload)?
    } else {
        0
    };
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
        .and_then(|bytes| bytes.checked_add(diffusion_components))
        .ok_or_else(|| anyhow::anyhow!("model artifact size overflow"))?;
    // Projector bytes affect framework/slack, not model KV or activation sizing.
    let parameter_proxy = weights
        .saturating_add(diffusion_components)
        .saturating_mul(2)
        .max(GIB);
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
    if diffusion_components > 0 {
        components.extend(assign_component(
            "diffusion_components",
            diffusion_components,
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

fn stable_diffusion_component_bytes(payload: &Value) -> Result<u64> {
    let args = payload
        .get("launch_args")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let mut total = 0_u64;
    for flag in ["--llm", "--vae", "--audio-vae"] {
        let mut selected = None;
        let mut index = 0;
        while index < args.len() {
            let token = args[index];
            if let Some(value) = token.strip_prefix(&format!("{flag}=")) {
                selected = Some(value);
            } else if token == flag {
                selected = args.get(index + 1).copied();
                index += 1;
            }
            index += 1;
        }
        let Some(path) = selected else {
            continue;
        };
        let bytes = artifact_size_bytes(&PathBuf::from(path))?.ok_or_else(|| {
            anyhow::anyhow!("stable-diffusion.cpp component size is unknown for {flag}: {path}")
        })?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("diffusion component size overflow"))?;
    }
    Ok(total)
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
    Ok(domains)
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
