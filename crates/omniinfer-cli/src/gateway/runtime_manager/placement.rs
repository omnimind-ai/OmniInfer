use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::*;

const MAX_PLACEMENT_LOG_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LlamaCppPlacementPolicy {
    Auto,
    ExplicitPartial(u32),
    ExplicitFull,
}

impl LlamaCppPlacementPolicy {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ExplicitPartial(_) => "explicit_partial",
            Self::ExplicitFull => "explicit_full",
        }
    }

    pub(super) fn permits_partial_offload(self) -> bool {
        !matches!(self, Self::ExplicitFull)
    }

    pub(super) fn requested_gpu_layers(self) -> Option<u32> {
        match self {
            Self::ExplicitPartial(layers) => Some(layers),
            _ => None,
        }
    }
}

pub(super) fn managed_placement_evidence_args(
    backend_id: &str,
    launch_args: &[String],
    policy: Option<LlamaCppPlacementPolicy>,
) -> Result<Vec<String>> {
    let Some(policy) = policy else {
        return Ok(launch_args.to_vec());
    };
    if launch_args
        .iter()
        .any(|arg| arg.split_once('=').map(|(flag, _)| flag).unwrap_or(arg) == "--log-disable")
    {
        anyhow::bail!(
            "{} llama.cpp placement requires startup logging; remove --log-disable",
            policy.as_str()
        );
    }
    // ik_llama.cpp already emits the buffer placement evidence we need at its
    // default INFO level, and does not implement the official llama.cpp -lv
    // verbosity flag.
    if backend_id.starts_with("ik_llama.cpp") {
        return Ok(launch_args.to_vec());
    }
    if launch_args.ends_with(&["-lv".to_string(), "4".to_string()]) {
        return Ok(launch_args.to_vec());
    }
    let mut managed = launch_args.to_vec();
    managed.extend(["-lv".to_string(), "4".to_string()]);
    Ok(managed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimePlacement {
    pub(super) policy: LlamaCppPlacementPolicy,
    pub(super) mode: String,
    pub(super) offloaded_layers: Option<u32>,
    pub(super) total_layers: Option<u32>,
    pub(super) reported_bytes: BTreeMap<MemoryDomain, u64>,
    pub(super) reconciled_budget: ResourceBudget,
}

fn ik_llama_cpu_moe_layers(launch_args: &[String]) -> Result<Option<u32>> {
    let mut layers = None;
    let mut index = 0;
    while index < launch_args.len() {
        let flag = launch_args[index].as_str();
        match flag {
            "-cmoe" | "--cpu-moe" => layers = Some(999),
            "-ncmoe" | "--n-cpu-moe" => {
                let value = launch_args.get(index + 1).ok_or_else(|| {
                    anyhow::anyhow!("{flag} requires a non-negative integer value")
                })?;
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| anyhow::anyhow!("{flag} value must be a non-negative integer"))?;
                layers = Some(parsed);
                index += 1;
            }
            _ => {}
        }
        index += 1;
    }
    Ok(layers)
}

pub(super) fn llama_cpp_placement_policy(
    backend: &backend_registry::BackendSpec,
    launch_args: &[String],
) -> Result<Option<LlamaCppPlacementPolicy>> {
    if backend.family != "llama.cpp"
        || !(backend.id.starts_with("llama.cpp-") || backend.id.starts_with("ik_llama.cpp-"))
        || !backend
            .capabilities
            .iter()
            .any(|cap| cap == "cuda" || cap == "vulkan")
    {
        return Ok(None);
    }
    // ik_llama.cpp's CPU-MoE options intentionally place the expert tensors in
    // host memory even when -ngl 999 is present in its backend defaults.
    // Treat that combination as automatic partial offload so admission can
    // reserve host plus CUDA ceilings and reconcile them from startup logs.
    if backend.id.starts_with("ik_llama.cpp") {
        let cpu_moe_layers = ik_llama_cpu_moe_layers(launch_args)?.unwrap_or(0);
        // Native --fit can move experts to the CPU independently of ncmoe,
        // including with the backend's default -ngl 999.
        if cpu_moe_layers > 0 || launch_args.iter().any(|arg| arg == "--fit") {
            return Ok(Some(LlamaCppPlacementPolicy::Auto));
        }
    }
    let Some(value) = gpu_layers_value(launch_args) else {
        if launch_args
            .iter()
            .any(|arg| matches!(arg.as_str(), "-ngl" | "--gpu-layers"))
        {
            anyhow::bail!("llama.cpp GPU-layer argument is missing its value");
        }
        return Ok(Some(LlamaCppPlacementPolicy::Auto));
    };
    if value.eq_ignore_ascii_case("auto") {
        return Ok(Some(LlamaCppPlacementPolicy::Auto));
    }
    if matches!(value.to_ascii_lowercase().as_str(), "all" | "max") {
        return Ok(Some(LlamaCppPlacementPolicy::ExplicitFull));
    }
    let layers = value.parse::<u32>().map_err(|_| {
        anyhow::anyhow!("llama.cpp GPU layers must be 'auto', 'all', or a non-negative integer")
    })?;
    Ok(Some(if layers >= 999 {
        LlamaCppPlacementPolicy::ExplicitFull
    } else {
        LlamaCppPlacementPolicy::ExplicitPartial(layers)
    }))
}

pub(super) fn provisional_llama_cpp_placement_budget(
    estimated: &ResourceBudget,
    snapshot: &omniinfer_core::resource_ledger::ResourceLedgerSnapshot,
) -> Result<ResourceBudget> {
    let gpus = estimated
        .domains()
        .iter()
        .filter(|(domain, _)| matches!(domain, MemoryDomain::Cuda(_) | MemoryDomain::Vulkan(_)))
        .collect::<Vec<_>>();
    if gpus.is_empty() {
        anyhow::bail!("llama.cpp placement reconciliation requires a selected GPU device");
    }
    let estimated_total = gpus.iter().try_fold(0_u64, |total, (_, bytes)| {
        total
            .checked_add(**bytes)
            .ok_or_else(|| anyhow::anyhow!("llama.cpp placement budget overflow"))
    })?;
    let available = snapshot.available()?;
    let host_available = available.get(&MemoryDomain::Host).copied().unwrap_or(0);
    if host_available == 0 {
        anyhow::bail!("llama.cpp placement requires available host memory");
    }
    let host_ceiling = if gpus
        .iter()
        .any(|(domain, _)| matches!(domain, MemoryDomain::Vulkan(_)))
    {
        estimated_total.min(host_available)
    } else {
        estimated_total
    };
    let mut components = vec![BudgetComponent {
        name: "llama_cpp_host_ceiling".to_string(),
        domain: MemoryDomain::Host,
        bytes: host_ceiling,
    }];
    for (gpu_domain, _) in gpus {
        let gpu_available = available.get(gpu_domain).copied().unwrap_or(0);
        if gpu_available == 0 {
            anyhow::bail!(
                "llama.cpp placement reconciliation requires available memory on {}",
                gpu_domain.key(),
            );
        }
        components.push(BudgetComponent {
            name: "llama_cpp_device_ceiling".to_string(),
            domain: gpu_domain.clone(),
            bytes: estimated_total.min(gpu_available),
        });
    }
    ResourceBudget::from_components(components).map_err(Into::into)
}

pub(super) fn parse_llama_cpp_runtime_placement(
    log_path: &Path,
    start_offset: u64,
    cuda_visible_devices: &str,
    vulkan_devices: &BTreeMap<String, String>,
    policy: LlamaCppPlacementPolicy,
) -> Result<RuntimePlacement> {
    let mut file = fs::File::open(log_path)?;
    let end = file.metadata()?.len();
    let length = end.saturating_sub(start_offset);
    if length > MAX_PLACEMENT_LOG_BYTES {
        anyhow::bail!(
            "llama.cpp startup log exceeded the {} byte placement parsing limit",
            MAX_PLACEMENT_LOG_BYTES
        );
    }
    file.seek(SeekFrom::Start(start_offset))?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    parse_llama_cpp_runtime_placement_text(&text, cuda_visible_devices, vulkan_devices, policy)
}

pub(super) fn parse_llama_cpp_runtime_placement_text(
    text: &str,
    cuda_visible_devices: &str,
    vulkan_devices: &BTreeMap<String, String>,
    policy: LlamaCppPlacementPolicy,
) -> Result<RuntimePlacement> {
    let devices = ordered_cuda_devices(cuda_visible_devices);
    let mut buffers = BTreeMap::<(MemoryDomain, &'static str, String), u64>::new();
    let mut layers = None;
    for line in text.lines() {
        if let Some(parsed) = parse_offloaded_layers(line) {
            layers = Some(parsed);
        }
        // ik_llama.cpp reports persistent model buffers as
        // llm_load_tensors: CUDA_Host/CUDA0 buffer size = ... rather than
        // including the word model used by official llama.cpp.
        if line.contains("llm_load_tensors:") && !line.contains(" model buffer size") {
            let marker = " buffer size";
            if let Some(marker_index) = line.find(marker) {
                let Some(label) = line[..marker_index].split_whitespace().last() else {
                    continue;
                };
                let Some(domain) = buffer_domain(label, &devices, vulkan_devices)? else {
                    continue;
                };
                let Some(bytes) = parse_buffer_bytes(&line[marker_index + marker.len()..])? else {
                    continue;
                };
                let key = (domain, "model", label.to_string());
                let current = buffers.entry(key).or_insert(0);
                *current = current
                    .checked_add(bytes)
                    .ok_or_else(|| anyhow::anyhow!("llama.cpp placement byte count overflow"))?;
            }
        }
        for (marker, category) in [
            (" model buffer size", "model"),
            (" KV buffer size", "kv"),
            (" compute buffer size", "compute"),
            (" output buffer size", "output"),
        ] {
            let Some(marker_index) = line.find(marker) else {
                continue;
            };
            let Some(label) = line[..marker_index].split_whitespace().last() else {
                continue;
            };
            let Some(domain) = buffer_domain(label, &devices, vulkan_devices)? else {
                continue;
            };
            let Some(bytes) = parse_buffer_bytes(&line[marker_index + marker.len()..])? else {
                continue;
            };
            let key = (domain, category, label.to_string());
            let current = buffers.entry(key).or_insert(0);
            if matches!(category, "compute" | "output") {
                *current = (*current).max(bytes);
            } else {
                *current = current
                    .checked_add(bytes)
                    .ok_or_else(|| anyhow::anyhow!("llama.cpp placement byte count overflow"))?;
            }
        }
    }
    if buffers.is_empty() {
        anyhow::bail!("llama.cpp startup log did not report CPU/GPU buffer placement");
    }
    let mut categorized = BTreeMap::<(MemoryDomain, &'static str), u64>::new();
    for ((domain, category, _), bytes) in buffers {
        let total = categorized.entry((domain, category)).or_insert(0);
        *total = total
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("llama.cpp placement byte count overflow"))?;
    }
    let host_model_bytes = categorized
        .get(&(MemoryDomain::Host, "model"))
        .copied()
        .unwrap_or(0);
    let gpu_model_bytes = categorized
        .iter()
        .filter(|((domain, category), _)| {
            matches!(domain, MemoryDomain::Cuda(_) | MemoryDomain::Vulkan(_))
                && *category == "model"
        })
        .try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| anyhow::anyhow!("llama.cpp placement byte count overflow"))
        })?;
    let mut reported = BTreeMap::<MemoryDomain, u64>::new();
    let mut components = Vec::new();
    for ((domain, category), bytes) in categorized {
        let total = reported.entry(domain.clone()).or_insert(0);
        *total = total
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("llama.cpp placement byte count overflow"))?;
        components.push(BudgetComponent {
            name: format!("reported_{category}_buffers"),
            domain,
            bytes,
        });
    }
    for (domain, bytes) in &reported {
        components.push(BudgetComponent {
            name: "runtime_overhead".to_string(),
            domain: domain.clone(),
            bytes: if matches!(domain, MemoryDomain::Host) {
                384 * MIB
            } else {
                128 * MIB
            },
        });
        components.push(BudgetComponent {
            name: "reconciliation_slack".to_string(),
            domain: domain.clone(),
            bytes: checked_scaled(*bytes, 4, 100)?.max(160 * MIB),
        });
    }
    let has_host = reported.contains_key(&MemoryDomain::Host);
    let has_gpu = reported
        .keys()
        .any(|domain| matches!(domain, MemoryDomain::Cuda(_) | MemoryDomain::Vulkan(_)));
    let (offloaded_layers, total_layers) = layers.unzip();
    let all_layers_offloaded = matches!(
        (offloaded_layers, total_layers),
        (Some(offloaded), Some(total)) if total > 0 && offloaded == total
    );
    let incidental_host_mapping_limit = (gpu_model_bytes / 20).max(512 * MIB);
    let host_model_is_material = host_model_bytes > incidental_host_mapping_limit;
    let mode = match (host_model_bytes > 0, gpu_model_bytes > 0, has_host, has_gpu) {
        (true, true, _, _) if all_layers_offloaded && !host_model_is_material => "full",
        (true, true, _, _) => "partial",
        (true, false, _, true) => "partial",
        (false, true, _, _) => "full",
        (false, false, _, true) => "full",
        (_, _, true, false) => "cpu",
        _ => "unknown",
    }
    .to_string();
    if policy.permits_partial_offload() && mode == "unknown" {
        anyhow::bail!("llama.cpp startup log reported an indeterminate placement");
    }
    if matches!(policy, LlamaCppPlacementPolicy::ExplicitFull)
        && (mode != "full" || !all_layers_offloaded || gpu_model_bytes == 0)
    {
        anyhow::bail!(
            "llama.cpp did not satisfy the requested full GPU offload (observed mode: {mode})"
        );
    }
    Ok(RuntimePlacement {
        policy,
        mode,
        offloaded_layers,
        total_layers,
        reported_bytes: reported,
        reconciled_budget: ResourceBudget::from_components(components)?,
    })
}

fn parse_offloaded_layers(line: &str) -> Option<(u32, u32)> {
    let value = line.split("offloaded ").nth(1)?.split_whitespace().next()?;
    let (offloaded, total) = value.split_once('/')?;
    Some((offloaded.parse().ok()?, total.parse().ok()?))
}

fn ordered_cuda_devices(visible_devices: &str) -> Vec<String> {
    let mut devices = Vec::new();
    for device in visible_devices.split(',').map(str::trim) {
        if !device.is_empty() && !devices.iter().any(|current| current == device) {
            devices.push(device.to_string());
        }
    }
    devices
}

fn buffer_domain(
    label: &str,
    devices: &[String],
    vulkan_devices: &BTreeMap<String, String>,
) -> Result<Option<MemoryDomain>> {
    let upper = label.to_ascii_uppercase();
    if upper == "CUDA_SPLIT" {
        anyhow::bail!(
            "CUDA_Split model placement lacks per-device memory evidence; use ik_llama.cpp --split-mode layer or --split-mode none"
        );
    }
    if upper.starts_with("CPU") || upper == "CUDA_HOST" || upper == "VULKAN_HOST" {
        return Ok(Some(MemoryDomain::Host));
    }
    if let Some(index) = upper.strip_prefix("VULKAN") {
        let physical = vulkan_devices
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("unreserved Vulkan buffer device: {label}"))?;
        return Ok(Some(MemoryDomain::Vulkan(physical.clone())));
    }
    if let Some(index) = upper.strip_prefix("CUDA") {
        let logical = index.parse::<usize>()?;
        let physical = devices
            .get(logical)
            .ok_or_else(|| anyhow::anyhow!("unreserved CUDA buffer device: {label}"))?;
        return Ok(Some(MemoryDomain::Cuda(physical.clone())));
    }
    Ok(None)
}

fn parse_buffer_bytes(rest: &str) -> Result<Option<u64>> {
    let Some((_, value)) = rest.split_once('=') else {
        return Ok(None);
    };
    let mut fields = value.split_whitespace();
    let Some(number) = fields.next() else {
        return Ok(None);
    };
    let Some(unit) = fields.next() else {
        return Ok(None);
    };
    let number = number
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("invalid llama.cpp buffer size: {number}"))?;
    if !number.is_finite() || number < 0.0 {
        anyhow::bail!("invalid llama.cpp buffer size: {number}");
    }
    let scale = match unit.trim_matches(',') {
        "KiB" => 1024_f64,
        "MiB" => MIB as f64,
        "GiB" => GIB as f64,
        _ => return Ok(None),
    };
    let bytes = number * scale;
    if bytes > u64::MAX as f64 {
        anyhow::bail!("llama.cpp buffer size overflow");
    }
    Ok(Some(bytes.round() as u64))
}
