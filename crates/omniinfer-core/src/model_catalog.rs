use std::collections::BTreeMap;
use std::process::Command;

use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::backend_registry::{
    BackendRegistry, BackendScope, HostInfo, HostSystem, backend_priority,
};

const LINUX_CATALOG: &str = include_str!("../model_catalogs/linux.json");
const MAC_CATALOG: &str = include_str!("../model_catalogs/mac.json");
const WINDOWS_CATALOG: &str = include_str!("../model_catalogs/windows.json");

#[derive(Debug, Error, PartialEq)]
pub enum ModelCatalogError {
    #[error("field 'system' must be one of: windows, mac, linux")]
    InvalidSystem,
    #[error("invalid bundled model catalog for system: {0}")]
    InvalidCatalog(String),
}

pub fn list_supported_models(system_name: &str) -> Result<Value, ModelCatalogError> {
    let system = parse_system_name(system_name)?;
    let mut catalog = bundled_catalog(system)?;
    let memory = MemoryContext::detect();
    annotate_catalog_root(&mut catalog, system, &memory);
    Ok(catalog)
}

pub fn list_supported_models_best(system_name: &str) -> Result<Value, ModelCatalogError> {
    let system = parse_system_name(system_name)?;
    let annotated = list_supported_models(system_name)?;
    let installed_backends = BackendRegistry::build(system_host(system), "", &Value::Null)
        .rows(BackendScope::Installed)
        .into_iter()
        .filter_map(|row| row.get("id").and_then(Value::as_str).map(str::to_string))
        .collect::<Vec<_>>();
    Ok(merge_best_supported_models(
        system,
        annotated,
        &installed_backends,
    ))
}

fn parse_system_name(system_name: &str) -> Result<HostSystem, ModelCatalogError> {
    match system_name.trim().to_ascii_lowercase().as_str() {
        "linux" => Ok(HostSystem::Linux),
        "mac" | "macos" | "darwin" => Ok(HostSystem::Mac),
        "windows" | "win" => Ok(HostSystem::Windows),
        _ => Err(ModelCatalogError::InvalidSystem),
    }
}

fn bundled_catalog(system: HostSystem) -> Result<Value, ModelCatalogError> {
    let (name, raw) = match system {
        HostSystem::Linux => ("linux", LINUX_CATALOG),
        HostSystem::Mac => ("mac", MAC_CATALOG),
        HostSystem::Windows => ("windows", WINDOWS_CATALOG),
        _ => return Err(ModelCatalogError::InvalidSystem),
    };
    serde_json::from_str(raw).map_err(|_| ModelCatalogError::InvalidCatalog(name.to_string()))
}

fn annotate_catalog_root(value: &mut Value, system: HostSystem, memory: &MemoryContext) {
    let Some(backends) = value.as_object_mut() else {
        return;
    };
    for (catalog_backend, backend_payload) in backends {
        annotate_catalog(backend_payload, system, catalog_backend, memory);
    }
}

fn annotate_catalog(
    value: &mut Value,
    system: HostSystem,
    catalog_backend: &str,
    memory: &MemoryContext,
) {
    match value {
        Value::Object(map) => {
            if map.get("quantization").and_then(Value::as_object).is_some() {
                annotate_model_quantizations(map, system, catalog_backend, memory);
                return;
            }
            for child in map.values_mut() {
                annotate_catalog(child, system, catalog_backend, memory);
            }
        }
        Value::Array(items) => {
            for child in items {
                annotate_catalog(child, system, catalog_backend, memory);
            }
        }
        _ => {}
    }
}

fn annotate_model_quantizations(
    model: &mut Map<String, Value>,
    system: HostSystem,
    catalog_backend: &str,
    memory: &MemoryContext,
) {
    let vision_size_gib = model
        .get("vision")
        .and_then(|vision| vision.get("size"))
        .map(parse_size_gib)
        .unwrap_or(0.0);
    let Some(quantizations) = model.get_mut("quantization").and_then(Value::as_object_mut) else {
        return;
    };
    for quant_info in quantizations.values_mut() {
        let Some(quant_map) = quant_info.as_object_mut() else {
            continue;
        };
        let required =
            round_gib(quant_map.get("size").map(parse_size_gib).unwrap_or(0.0) + vision_size_gib);
        quant_map.insert("required_memory_gib".to_string(), json!(required));
        let available = available_memory_for_catalog_backend(system, catalog_backend, memory);
        let margin = safety_margin_gib(system, catalog_backend);
        quant_map.insert(
            "suitable".to_string(),
            json!(available >= round_gib(required + margin)),
        );
    }
}

fn merge_best_supported_models(
    system: HostSystem,
    annotated_catalog: Value,
    installed_backend_ids: &[String],
) -> Value {
    let installed = installed_backend_ids
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let Some(backends) = annotated_catalog.as_object() else {
        return json!({});
    };
    let mut merged = Map::new();
    let mut quantization_candidates: BTreeMap<(String, String, String), Vec<QuantCandidate>> =
        BTreeMap::new();

    for (catalog_backend, backend_payload) in backends {
        let runtime_backend = resolve_catalog_backend_id(system, catalog_backend);
        if !installed.contains(runtime_backend.as_str()) {
            continue;
        }
        let Some(families) = backend_payload.as_object() else {
            continue;
        };
        for (family_name, family_models) in families {
            let Some(models) = family_models.as_object() else {
                continue;
            };
            let target_family = object_entry(&mut merged, family_name);
            for (model_name, model_info) in models {
                let Some(model_map) = model_info.as_object() else {
                    continue;
                };
                let target_model = object_entry(target_family, model_name);
                for (key, value) in model_map {
                    if key != "quantization" && !target_model.contains_key(key) {
                        target_model.insert(key.clone(), value.clone());
                    }
                }
                let Some(quantizations) = model_map.get("quantization").and_then(Value::as_object)
                else {
                    continue;
                };
                let target_quantizations = object_entry(target_model, "quantization");
                for (quant_name, quant_info) in quantizations {
                    let Some(quant_map) = quant_info.as_object() else {
                        continue;
                    };
                    target_quantizations
                        .entry(quant_name.clone())
                        .or_insert_with(|| Value::Object(quant_map.clone()));
                    quantization_candidates
                        .entry((family_name.clone(), model_name.clone(), quant_name.clone()))
                        .or_default()
                        .push(QuantCandidate {
                            backend: runtime_backend.clone(),
                            payload: Value::Object(quant_map.clone()),
                            suitable: quant_map
                                .get("suitable")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                        });
                }
            }
        }
    }

    for ((family_name, model_name, quant_name), candidates) in quantization_candidates {
        let Some(best) = best_candidate(&candidates) else {
            continue;
        };
        let Some(target_quant) = merged
            .get_mut(&family_name)
            .and_then(Value::as_object_mut)
            .and_then(|family| family.get_mut(&model_name))
            .and_then(Value::as_object_mut)
            .and_then(|model| model.get_mut("quantization"))
            .and_then(Value::as_object_mut)
            .and_then(|quantizations| quantizations.get_mut(&quant_name))
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        let replacement = best.payload.as_object().cloned().unwrap_or_else(Map::new);
        target_quant.clear();
        target_quant.extend(replacement);
        target_quant.insert(
            "backend".to_string(),
            Value::String(if best.suitable {
                best.backend.clone()
            } else {
                String::new()
            }),
        );
    }

    Value::Object(merged)
}

#[derive(Debug, Clone)]
struct QuantCandidate {
    backend: String,
    payload: Value,
    suitable: bool,
}

fn best_candidate(candidates: &[QuantCandidate]) -> Option<&QuantCandidate> {
    let suitable = candidates
        .iter()
        .filter(|candidate| candidate.suitable)
        .collect::<Vec<_>>();
    if !suitable.is_empty() {
        return suitable
            .into_iter()
            .min_by_key(|candidate| backend_priority(&candidate.backend));
    }
    candidates
        .iter()
        .min_by_key(|candidate| backend_priority(&candidate.backend))
}

fn object_entry<'a>(map: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    map.entry(key.to_string())
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .expect("object entry should be an object")
}

fn resolve_catalog_backend_id(system: HostSystem, backend_id: &str) -> String {
    match (system, backend_id) {
        (HostSystem::Linux, "llama.cpp-cuda") => "llama.cpp-linux-cuda".to_string(),
        (HostSystem::Linux, "llama.cpp-vulkan") => "llama.cpp-linux-vulkan".to_string(),
        (HostSystem::Linux, "llama.cpp-openvino") => "llama.cpp-linux-openvino".to_string(),
        (HostSystem::Linux, "llama.cpp-linux") if std::env::consts::ARCH == "s390x" => {
            "llama.cpp-linux-s390x".to_string()
        }
        (HostSystem::Mac, "llama.cpp-cpu")
            if matches!(std::env::consts::ARCH, "x86_64" | "amd64") =>
        {
            "llama.cpp-mac-intel".to_string()
        }
        (HostSystem::Mac, "llama.cpp-cpu") => "llama.cpp-mac".to_string(),
        (HostSystem::Windows, "llama.cpp-cpu")
            if matches!(std::env::consts::ARCH, "aarch64" | "arm64") =>
        {
            "llama.cpp-windows-arm64".to_string()
        }
        _ => backend_id.to_string(),
    }
}

fn system_host(system: HostSystem) -> HostInfo {
    HostInfo {
        system,
        machine: std::env::consts::ARCH,
    }
}

#[derive(Debug, Clone, Copy)]
struct MemoryContext {
    ram_available_gib: f64,
    cuda_available_gib: f64,
}

impl MemoryContext {
    fn detect() -> Self {
        Self {
            ram_available_gib: available_ram_gib().unwrap_or(0.0),
            cuda_available_gib: available_cuda_gib().unwrap_or(0.0),
        }
    }
}

fn available_memory_for_catalog_backend(
    system: HostSystem,
    catalog_backend: &str,
    memory: &MemoryContext,
) -> f64 {
    let runtime_backend = resolve_catalog_backend_id(system, catalog_backend);
    if is_gpu_backend(&runtime_backend) && runtime_backend.contains("cuda") {
        return memory.cuda_available_gib;
    }
    memory.ram_available_gib
}

fn safety_margin_gib(system: HostSystem, catalog_backend: &str) -> f64 {
    let runtime_backend = resolve_catalog_backend_id(system, catalog_backend);
    if is_gpu_backend(&runtime_backend) {
        0.5
    } else {
        1.0
    }
}

fn is_gpu_backend(backend_id: &str) -> bool {
    matches!(
        backend_id,
        "llama.cpp-linux-cuda"
            | "llama.cpp-linux-rocm"
            | "llama.cpp-linux-vulkan"
            | "omniinfer-native-linux"
            | "ik_llama.cpp-linux-cuda"
            | "vllm-linux-cuda"
            | "llama.cpp-cuda"
            | "llama.cpp-vulkan"
            | "llama.cpp-sycl"
            | "llama.cpp-hip"
            | "ik_llama.cpp-cuda"
    )
}

fn available_ram_gib() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("MemAvailable:") else {
                continue;
            };
            let kb = rest.split_whitespace().next()?.parse::<f64>().ok()?;
            return Some(round_gib(kb / 1024.0 / 1024.0));
        }
    }
    None
}

fn available_cuda_gib() -> Option<f64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .map(|mib| round_gib(mib / 1024.0))
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn parse_size_gib(value: &Value) -> f64 {
    match value {
        Value::Number(number) => number.as_f64().unwrap_or(0.0),
        Value::String(text) => text.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn round_gib(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_bundled_catalog_with_memory_annotations() {
        let catalog = list_supported_models("linux").unwrap();
        let quant = catalog
            .get("llama.cpp-linux")
            .and_then(|value| value.get("Qwen2.5"))
            .and_then(|value| value.get("Qwen2.5-0.5B-Instruct"))
            .and_then(|value| value.get("quantization"))
            .and_then(|value| value.get("Q4_K_M"))
            .unwrap();
        assert_eq!(quant["required_memory_gib"], json!(0.49));
        assert!(quant.get("suitable").and_then(Value::as_bool).is_some());
    }

    #[test]
    fn rejects_invalid_system() {
        assert!(matches!(
            list_supported_models("android").unwrap_err(),
            ModelCatalogError::InvalidSystem
        ));
    }

    #[test]
    fn merges_best_catalog_for_installed_backends() {
        let best = list_supported_models_best("linux").unwrap();
        assert!(best.is_object());
    }
}
