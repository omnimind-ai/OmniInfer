use std::collections::BTreeMap;
use std::process::Command;

use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::backend_registry::{
    BackendRegistry, BackendScope, HostInfo, HostSystem, backend_priority,
};

const LINUX_CATALOG: &str = include_str!("../../model_catalogs/linux.json");
const MAC_CATALOG: &str = include_str!("../../model_catalogs/mac.json");
const WINDOWS_CATALOG: &str = include_str!("../../model_catalogs/windows.json");
const DOWNLOAD_ASSETS: &str = include_str!("../../model_catalogs/download_assets.json");

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
    let download_assets = bundled_download_assets()?;
    let memory = MemoryContext::detect();
    annotate_catalog_root(&mut catalog, system, &memory, &download_assets);
    Ok(catalog)
}

pub fn list_supported_models_best(system_name: &str) -> Result<Value, ModelCatalogError> {
    let system = parse_system_name(system_name)?;
    let annotated = list_supported_models(system_name)?;
    let installed_backends = BackendRegistry::build(system_host(system), "", &Value::Null)
        .rows(BackendScope::Installed)
        .into_iter()
        .filter(|row| row.get("hardware_compatible").and_then(Value::as_bool) == Some(true))
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

fn bundled_download_assets() -> Result<Map<String, Value>, ModelCatalogError> {
    let payload: Value = serde_json::from_str(DOWNLOAD_ASSETS)
        .map_err(|_| ModelCatalogError::InvalidCatalog("download-assets".to_string()))?;
    if payload.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err(ModelCatalogError::InvalidCatalog(
            "download-assets".to_string(),
        ));
    }
    payload
        .get("assets")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| ModelCatalogError::InvalidCatalog("download-assets".to_string()))
}

fn annotate_catalog_root(
    value: &mut Value,
    system: HostSystem,
    memory: &MemoryContext,
    download_assets: &Map<String, Value>,
) {
    let Some(backends) = value.as_object_mut() else {
        return;
    };
    for (catalog_backend, backend_payload) in backends {
        annotate_catalog(
            backend_payload,
            system,
            catalog_backend,
            memory,
            download_assets,
        );
    }
}

fn annotate_catalog(
    value: &mut Value,
    system: HostSystem,
    catalog_backend: &str,
    memory: &MemoryContext,
    download_assets: &Map<String, Value>,
) {
    match value {
        Value::Object(map) => {
            if map.get("quantization").and_then(Value::as_object).is_some() {
                annotate_model_quantizations(map, system, catalog_backend, memory, download_assets);
                return;
            }
            for child in map.values_mut() {
                annotate_catalog(child, system, catalog_backend, memory, download_assets);
            }
        }
        Value::Array(items) => {
            for child in items {
                annotate_catalog(child, system, catalog_backend, memory, download_assets);
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
    download_assets: &Map<String, Value>,
) {
    let (vision_memory_gib, vision_size_bytes) = model
        .get_mut("vision")
        .and_then(Value::as_object_mut)
        .map(|vision| {
            annotate_download_metadata(vision, download_assets);
            (
                parse_optional_gib(vision.get("memory_estimate_gib")),
                vision.get("size_bytes").and_then(Value::as_u64),
            )
        })
        .unwrap_or((Some(0.0), Some(0)));
    let Some(quantizations) = model.get_mut("quantization").and_then(Value::as_object_mut) else {
        return;
    };
    for quant_info in quantizations.values_mut() {
        let Some(quant_map) = quant_info.as_object_mut() else {
            continue;
        };
        annotate_download_metadata(quant_map, download_assets);
        let main_size_bytes = quant_map.get("size_bytes").and_then(Value::as_u64);
        let bundle_size_bytes = main_size_bytes
            .zip(vision_size_bytes)
            .and_then(|(main, vision)| main.checked_add(vision));
        quant_map.insert(
            "bundle_size_bytes".to_string(),
            bundle_size_bytes.map_or(Value::Null, |value| json!(value)),
        );
        quant_map.insert(
            "bundle_size_gib".to_string(),
            bundle_size_bytes.map_or(Value::Null, |value| json!(bytes_to_gib(value))),
        );

        let required = parse_optional_gib(quant_map.get("memory_estimate_gib"))
            .zip(vision_memory_gib)
            .map(|(main, vision)| round_gib(main + vision));
        quant_map.insert(
            "required_memory_gib".to_string(),
            required.map_or(Value::Null, |value| json!(value)),
        );
        let available = available_memory_for_catalog_backend(system, catalog_backend, memory);
        let margin = safety_margin_gib(system, catalog_backend);
        let memory_status = match (available, required) {
            (Some(value), Some(required)) if value >= round_gib(required + margin) => "sufficient",
            (Some(_), Some(_)) => "insufficient",
            _ => "unknown",
        };
        quant_map.insert("suitable".to_string(), json!(memory_status == "sufficient"));
        quant_map.insert(
            "available_memory_gib".to_string(),
            available.map_or(Value::Null, |value| json!(value)),
        );
        quant_map.insert("memory_status".to_string(), json!(memory_status));
    }
}

fn annotate_download_metadata(
    artifact: &mut Map<String, Value>,
    download_assets: &Map<String, Value>,
) {
    let size_bytes = artifact
        .get("download")
        .and_then(|download| exact_download_size(download, download_assets));
    let size_gib = size_bytes.map(bytes_to_gib);
    artifact.insert(
        "size_bytes".to_string(),
        size_bytes.map_or(Value::Null, |value| json!(value)),
    );
    artifact.insert(
        "size_gib".to_string(),
        size_gib.map_or(Value::Null, |value| json!(value)),
    );
    // Compatibility for clients released before the explicit-unit fields.
    // This value is always derived from exact bytes and never drives admission.
    artifact.insert(
        "size".to_string(),
        size_gib.map_or(Value::Null, |value| json!(value)),
    );
}

fn exact_download_size(download: &Value, download_assets: &Map<String, Value>) -> Option<u64> {
    match download {
        Value::String(url) => download_assets
            .get(url)
            .and_then(|metadata| metadata.get("size_bytes"))
            .and_then(Value::as_u64),
        Value::Array(urls) if !urls.is_empty() => urls.iter().try_fold(0_u64, |total, url| {
            let size = url
                .as_str()
                .and_then(|url| download_assets.get(url))
                .and_then(|metadata| metadata.get("size_bytes"))
                .and_then(Value::as_u64)?;
            total.checked_add(size)
        }),
        _ => None,
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
        target_quant.insert("backend".to_string(), Value::String(best.backend.clone()));
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
    resolve_catalog_backend_id_for_machine(system, backend_id, std::env::consts::ARCH)
}

fn resolve_catalog_backend_id_for_machine(
    system: HostSystem,
    backend_id: &str,
    machine: &str,
) -> String {
    match (system, backend_id) {
        (HostSystem::Linux, "llama.cpp-cuda") => "llama.cpp-linux-cuda".to_string(),
        (HostSystem::Linux, "llama.cpp-vulkan") => "llama.cpp-linux-vulkan".to_string(),
        (HostSystem::Linux, "llama.cpp-openvino") => "llama.cpp-linux-openvino".to_string(),
        (HostSystem::Linux, "llama.cpp-linux") if machine == "s390x" => {
            "llama.cpp-linux-s390x".to_string()
        }
        (HostSystem::Mac, "llama.cpp-cpu") if matches!(machine, "x86_64" | "amd64") => {
            "llama.cpp-mac-intel".to_string()
        }
        (HostSystem::Mac, "llama.cpp-cpu") => "llama.cpp-mac".to_string(),
        (HostSystem::Windows, "llama.cpp-cpu") if matches!(machine, "aarch64" | "arm64") => {
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
    ram_available_gib: Option<f64>,
    cuda_available_gib: Option<f64>,
}

impl MemoryContext {
    fn detect() -> Self {
        Self {
            ram_available_gib: available_ram_gib(),
            cuda_available_gib: available_cuda_gib(),
        }
    }
}

fn available_memory_for_catalog_backend(
    system: HostSystem,
    catalog_backend: &str,
    memory: &MemoryContext,
) -> Option<f64> {
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
            | "freetoken-linux-cuda"
            | "vllm-wsl2-cuda"
            | "vllm-wsl2-rocm"
            | "llama.cpp-cuda"
            | "llama.cpp-vulkan"
            | "llama.cpp-sycl"
            | "llama.cpp-hip"
            | "ik_llama.cpp-cuda"
    )
}

fn available_ram_gib() -> Option<f64> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available = system.available_memory();
    (available > 0).then(|| round_gib(available as f64 / 1024.0 / 1024.0 / 1024.0))
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

fn parse_optional_gib(value: Option<&Value>) -> Option<f64> {
    let parsed = match value? {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }?;
    parsed.is_finite().then_some(parsed)
}

fn bytes_to_gib(value: u64) -> f64 {
    round_gib(value as f64 / 1024.0 / 1024.0 / 1024.0)
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
        assert_eq!(quant["memory_estimate_gib"], json!(0.49));
        assert_eq!(quant["size_bytes"], Value::Null);
        assert_eq!(quant["size_gib"], Value::Null);
        assert_eq!(quant["size"], Value::Null);
        assert!(quant.get("suitable").and_then(Value::as_bool).is_some());
        assert!(quant.get("memory_status").and_then(Value::as_str).is_some());
    }

    fn mac_qwen_quant(catalog: &Value) -> &Value {
        catalog
            .get("llama.cpp-mac")
            .and_then(|value| value.get("Qwen3.5"))
            .and_then(|value| value.get("Qwen3.5-0.8B"))
            .and_then(|value| value.get("quantization"))
            .and_then(|value| value.get("Q4_K_M"))
            .unwrap()
    }

    #[test]
    fn mac_catalog_uses_injected_available_memory() {
        let mut catalog = bundled_catalog(HostSystem::Mac).unwrap();
        let download_assets = bundled_download_assets().unwrap();
        let memory = MemoryContext {
            ram_available_gib: Some(8.0),
            cuda_available_gib: None,
        };
        annotate_catalog_root(&mut catalog, HostSystem::Mac, &memory, &download_assets);
        let quant = mac_qwen_quant(&catalog);
        assert_eq!(quant["available_memory_gib"], json!(8.0));
        assert_eq!(quant["memory_status"], json!("sufficient"));
        assert_eq!(quant["suitable"], json!(true));
    }

    #[test]
    fn unknown_mac_memory_preserves_installed_backend() {
        let mut catalog = bundled_catalog(HostSystem::Mac).unwrap();
        let download_assets = bundled_download_assets().unwrap();
        let memory = MemoryContext {
            ram_available_gib: None,
            cuda_available_gib: None,
        };
        annotate_catalog_root(&mut catalog, HostSystem::Mac, &memory, &download_assets);
        let quant = mac_qwen_quant(&catalog);
        assert_eq!(quant["available_memory_gib"], Value::Null);
        assert_eq!(quant["memory_status"], json!("unknown"));
        assert_eq!(quant["suitable"], json!(false));

        let merged =
            merge_best_supported_models(HostSystem::Mac, catalog, &["llama.cpp-mac".to_string()]);
        assert_eq!(
            merged["Qwen3.5"]["Qwen3.5-0.8B"]["quantization"]["Q4_K_M"]["backend"],
            json!("llama.cpp-mac")
        );
    }

    #[test]
    fn insufficient_mac_memory_preserves_installed_backend() {
        let mut catalog = bundled_catalog(HostSystem::Mac).unwrap();
        let download_assets = bundled_download_assets().unwrap();
        let memory = MemoryContext {
            ram_available_gib: Some(1.0),
            cuda_available_gib: None,
        };
        annotate_catalog_root(&mut catalog, HostSystem::Mac, &memory, &download_assets);
        let quant = mac_qwen_quant(&catalog);
        assert_eq!(quant["memory_status"], json!("insufficient"));
        assert_eq!(quant["suitable"], json!(false));

        let merged =
            merge_best_supported_models(HostSystem::Mac, catalog, &["llama.cpp-mac".to_string()]);
        assert_eq!(
            merged["Qwen3.5"]["Qwen3.5-0.8B"]["quantization"]["Q4_K_M"]["backend"],
            json!("llama.cpp-mac")
        );
    }

    #[test]
    fn resolves_mac_backend_for_each_architecture() {
        assert_eq!(
            resolve_catalog_backend_id_for_machine(HostSystem::Mac, "llama.cpp-cpu", "arm64"),
            "llama.cpp-mac"
        );
        assert_eq!(
            resolve_catalog_backend_id_for_machine(HostSystem::Mac, "llama.cpp-cpu", "aarch64"),
            "llama.cpp-mac"
        );
        assert_eq!(
            resolve_catalog_backend_id_for_machine(HostSystem::Mac, "llama.cpp-cpu", "x86_64"),
            "llama.cpp-mac-intel"
        );
    }

    #[test]
    fn gemma_4_12b_catalog_includes_vision_projector() {
        let catalog = list_supported_models("linux").unwrap();
        let model = catalog
            .get("llama.cpp-linux-cuda")
            .and_then(|value| value.get("Gemma4"))
            .and_then(|value| value.get("gemma-4-12B-it"))
            .unwrap();
        assert_eq!(
            model["quantization"]["Q4_K_M"]["download"],
            json!(
                "https://modelscope.cn/models/unsloth/gemma-4-12B-it-GGUF/resolve/master/gemma-4-12b-it-Q4_K_M.gguf"
            )
        );
        assert_eq!(
            model["vision"]["download"],
            json!(
                "https://modelscope.cn/models/unsloth/gemma-4-12B-it-GGUF/resolve/master/mmproj-F16.gguf"
            )
        );
    }

    #[test]
    fn qwen35_4b_exact_download_metadata_is_shared_across_platforms() {
        let platform_catalogs = [
            (
                "linux",
                vec![
                    "llama.cpp-linux",
                    "llama.cpp-linux-cuda",
                    "llama.cpp-linux-vulkan",
                ],
            ),
            ("mac", vec!["llama.cpp-mac"]),
            (
                "windows",
                vec!["llama.cpp-cpu", "llama.cpp-cuda", "llama.cpp-hip"],
            ),
        ];

        for (system, backends) in platform_catalogs {
            let catalog = list_supported_models(system).unwrap();
            for backend in backends {
                let model = &catalog[backend]["Qwen3.5"]["Qwen3.5-4B"];
                let quant = &model["quantization"]["Q4_K_M"];
                assert_eq!(quant["size_bytes"], json!(2_740_937_888_u64));
                assert_eq!(quant["size_gib"], json!(2.55));
                assert_eq!(quant["size"], json!(2.55));
                assert_eq!(model["vision"]["size_bytes"], json!(672_423_616_u64));
                assert_eq!(model["vision"]["size_gib"], json!(0.63));
                assert_eq!(quant["bundle_size_bytes"], json!(3_413_361_504_u64));
                assert_eq!(quant["bundle_size_gib"], json!(3.18));
                assert_eq!(quant["required_memory_gib"], json!(3.18));
            }
        }
    }

    #[test]
    fn best_catalog_preserves_exact_download_metadata() {
        let mut catalog = bundled_catalog(HostSystem::Mac).unwrap();
        let download_assets = bundled_download_assets().unwrap();
        let memory = MemoryContext {
            ram_available_gib: Some(8.0),
            cuda_available_gib: None,
        };
        annotate_catalog_root(&mut catalog, HostSystem::Mac, &memory, &download_assets);
        let best =
            merge_best_supported_models(HostSystem::Mac, catalog, &["llama.cpp-mac".to_string()]);
        let quant = &best["Qwen3.5"]["Qwen3.5-4B"]["quantization"]["Q4_K_M"];
        assert_eq!(quant["size_bytes"], json!(2_740_937_888_u64));
        assert_eq!(quant["bundle_size_bytes"], json!(3_413_361_504_u64));
        assert_eq!(quant["required_memory_gib"], json!(3.18));
        assert_eq!(quant["backend"], json!("llama.cpp-mac"));
    }

    #[test]
    fn memory_admission_does_not_read_download_sizes() {
        let mut catalog = bundled_catalog(HostSystem::Mac).unwrap();
        catalog["llama.cpp-mac"]["Qwen3.5"]["Qwen3.5-4B"]["quantization"]["Q4_K_M"]["memory_estimate_gib"] =
            json!(7.25);
        let download_assets = bundled_download_assets().unwrap();
        let memory = MemoryContext {
            ram_available_gib: Some(8.0),
            cuda_available_gib: None,
        };
        annotate_catalog_root(&mut catalog, HostSystem::Mac, &memory, &download_assets);
        let quant = &catalog["llama.cpp-mac"]["Qwen3.5"]["Qwen3.5-4B"]["quantization"]["Q4_K_M"];
        assert_eq!(quant["bundle_size_gib"], json!(3.18));
        assert_eq!(quant["required_memory_gib"], json!(7.88));
        assert_eq!(quant["memory_status"], json!("insufficient"));
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
