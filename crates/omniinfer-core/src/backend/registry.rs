use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Value, json};

use super::detection::{embedded_module_exists, is_hardware_compatible};
#[cfg(test)]
use super::detection::{gpu_backend_ids, output_mentions_amd_gpu, parse_nvidia_driver_branch};
use super::templates::backend_templates;
use crate::{config, local_state, paths};

const LLAMA_CPP_CACHE_RAM_MIB: &str = "8192";
const LLAMA_CPP_CACHE_SAFETY_ARGS: &[&str] =
    &["--slot-prompt-similarity", "0", "--cache-idle-slots"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendScope {
    Installed,
    Compatible,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostSystem {
    Linux,
    Windows,
    Mac,
    Android,
    Ios,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostInfo {
    pub system: HostSystem,
    pub machine: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendTemplate {
    pub id: &'static str,
    pub label: &'static str,
    pub family: &'static str,
    pub runtime_dir_name: &'static str,
    pub launcher_name: Option<&'static str>,
    pub description: &'static str,
    pub capabilities: &'static [&'static str],
    pub env_prefix: &'static str,
    pub default_ngl: Option<&'static str>,
    pub default_extra_args: &'static [&'static str],
    pub fallback_runtime_dir_names: &'static [&'static str],
    pub runtime_mode: &'static str,
    pub model_artifact: &'static str,
    pub supports_mmproj: bool,
    pub supports_ctx_size: bool,
    pub python_modules: &'static [&'static str],
    pub external_server_protocol: Option<&'static str>,
    pub log_file_name: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendSpec {
    pub id: String,
    pub label: String,
    pub family: String,
    pub runtime_dir: String,
    pub launcher_path: Option<String>,
    pub models_dir: Option<String>,
    pub catalog_url: Option<String>,
    pub description: String,
    pub capabilities: Vec<String>,
    pub default_args: Vec<String>,
    pub runtime_mode: String,
    pub model_artifact: String,
    pub supports_mmproj: bool,
    pub supports_ctx_size: bool,
    pub python_modules: Vec<String>,
    pub external_server_protocol: Option<String>,
    pub log_file_name: String,
}

impl BackendSpec {
    pub fn binary_exists(&self) -> bool {
        if self.runtime_mode == "embedded" {
            return self
                .python_modules
                .iter()
                .all(|module| embedded_module_exists(Path::new(&self.runtime_dir), module));
        }
        self.launcher_path
            .as_deref()
            .map(|path| Path::new(path).is_file())
            .unwrap_or(false)
    }

    pub fn to_api_payload(
        &self,
        selected: bool,
        loaded_model: Option<&str>,
        compatibility: Option<&str>,
        priority: Option<i32>,
    ) -> Value {
        let binary_exists = self.binary_exists();
        let mut payload = json!({
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "selected": selected,
            "binary_exists": binary_exists,
            "installed": binary_exists,
            "models_dir": self.models_dir,
            "capabilities": self.capabilities,
            "description": self.description,
            "loaded_model": if selected { loaded_model } else { None },
            "runtime_dir": self.runtime_dir,
            "launcher_path": self.launcher_path,
            "catalog_url": self.catalog_url,
            "default_args": self.default_args,
            "runtime_mode": self.runtime_mode,
            "model_artifact": self.model_artifact,
            "supports_mmproj": self.supports_mmproj,
            "supports_ctx_size": self.supports_ctx_size,
            "external_server_protocol": self.external_server_protocol,
            "log_file_name": self.log_file_name,
        });
        if let Some(compatibility) = compatibility {
            payload["compatibility"] = Value::String(compatibility.to_string());
            payload["hardware_compatible"] = Value::Bool(compatibility == "compatible");
        }
        if let Some(priority) = priority {
            payload["priority"] = Value::from(priority);
        }
        payload
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRegistry {
    specs: BTreeMap<String, BackendSpec>,
    host: HostInfo,
}

impl BackendRegistry {
    pub fn load_current() -> Self {
        let config = config::load_app_config().unwrap_or_default();
        let raw_config = config::load_raw_config().ok().flatten();
        let overrides = raw_config
            .as_ref()
            .and_then(|value| value.get("backends"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let host = HostInfo::current();
        Self::build(host, &config.runtime_root, &Value::Object(overrides))
    }

    pub fn build(host: HostInfo, requested_runtime_root: &str, overrides: &Value) -> Self {
        let runtime_root = discover_runtime_root(host, requested_runtime_root);
        let override_map = overrides.as_object();
        let specs = backend_templates(host)
            .iter()
            .map(|template| {
                let override_value = override_map
                    .and_then(|items| items.get(template.id))
                    .unwrap_or(&Value::Null);
                let spec = build_backend_spec(template, &runtime_root, override_value);
                (spec.id.clone(), spec)
            })
            .collect::<BTreeMap<_, _>>();
        Self { specs, host }
    }

    pub fn get(&self, backend_id: &str) -> Option<&BackendSpec> {
        self.specs.get(backend_id)
    }

    pub fn rows(&self, scope: BackendScope) -> Vec<Value> {
        let state = local_state::load_state().unwrap_or_default();
        let loaded_model = state
            .selected_model
            .as_ref()
            .map(|model| model.model.as_str());
        let selected_backend = state.selected_backend.as_deref();
        self.specs
            .values()
            .filter_map(|spec| {
                let compatible = is_hardware_compatible(self.host, spec);
                let installed = spec.binary_exists();
                let include = match scope {
                    BackendScope::Installed => installed,
                    BackendScope::Compatible => compatible,
                    BackendScope::All => true,
                };
                include.then(|| {
                    spec.to_api_payload(
                        selected_backend == Some(spec.id.as_str()),
                        loaded_model,
                        Some(if compatible {
                            "compatible"
                        } else {
                            "incompatible"
                        }),
                        Some(backend_priority(&spec.id)),
                    )
                })
            })
            .collect()
    }

    pub fn api_payload(&self, scope: BackendScope) -> Value {
        let rows = self.rows(scope);
        let recommended = recommended_backend(&rows);
        json!({
            "data": rows,
            "recommended": recommended,
        })
    }
}

impl HostInfo {
    pub fn current() -> Self {
        let system = match std::env::consts::OS {
            "windows" => HostSystem::Windows,
            "macos" => HostSystem::Mac,
            "android" => HostSystem::Android,
            "ios" => HostSystem::Ios,
            _ => HostSystem::Linux,
        };
        Self {
            system,
            machine: std::env::consts::ARCH,
        }
    }

    fn runtime_folder_name(self) -> &'static str {
        match self.system {
            HostSystem::Linux => "linux",
            HostSystem::Windows => "windows",
            HostSystem::Mac => "macos",
            HostSystem::Android => "android",
            HostSystem::Ios => "ios",
        }
    }
}

pub fn backend_priority(backend_id: &str) -> i32 {
    match backend_id {
        "llama.cpp-mac" => 0,
        "llama.cpp-mac-intel" => 1,
        "turboquant-mac" => 0,
        "mlx-mac" => 0,
        "llama.cpp-cuda" => 0,
        "llama.cpp-vulkan" => 0,
        "stable-diffusion.cpp-vulkan" => 5,
        "llama.cpp-sycl" => 0,
        "llama.cpp-hip" => 0,
        "llama.cpp-linux-cuda" => 0,
        "llama.cpp-linux-rocm" => 0,
        "llama.cpp-linux-vulkan" => 0,
        "stable-diffusion.cpp-linux-vulkan" => 5,
        "omniinfer-native-linux" => 0,
        "llama.cpp-linux-openvino" => 0,
        "llama.cpp-linux" => 1,
        "llama.cpp-linux-s390x" => 1,
        "vla.cpp-linux-cuda" => 1,
        "vla.cpp-linux" => 2,
        "vllm-linux-cuda" => 2,
        "freetoken-linux-cuda" => 3,
        "vllm-wsl2-cuda" => 2,
        "vllm-wsl2-rocm" => 2,
        "llama.cpp-cpu" => 1,
        "llama.cpp-windows-arm64" => 1,
        "llama.cpp-ios" => 0,
        "mlx-ios" => 0,
        "ik_llama.cpp-linux" => 11,
        "ik_llama.cpp-linux-cuda" => 10,
        "ik_llama.cpp-cpu" => 11,
        "ik_llama.cpp-cuda" => 10,
        _ => 99,
    }
}

fn build_backend_spec(
    template: &BackendTemplate,
    runtime_root: &Path,
    override_value: &Value,
) -> BackendSpec {
    let runtime_dir = resolve_runtime_dir(template, runtime_root, override_value);
    let models_dir = resolve_models_dir(template, override_value);
    let launcher_path = template.launcher_name.map(|launcher_name| {
        let default = runtime_dir.join("bin").join(launcher_name);
        let launcher = env_value(&format!("{}_LAUNCHER_PATH", template.env_prefix))
            .or_else(|| env_value(&format!("{}_SERVER_PATH", template.env_prefix)))
            .or_else(|| override_string(override_value, "launcher_path"))
            .or_else(|| override_string(override_value, "server_path"))
            .map(PathBuf::from)
            .unwrap_or(default);
        resolve_app_path(launcher)
    });
    BackendSpec {
        id: template.id.to_string(),
        label: template.label.to_string(),
        family: template.family.to_string(),
        runtime_dir: resolve_app_path(runtime_dir).display().to_string(),
        launcher_path: launcher_path.map(|path| path.display().to_string()),
        models_dir: models_dir.map(|path| path.display().to_string()),
        catalog_url: override_string(override_value, "catalog_url"),
        description: template.description.to_string(),
        capabilities: template
            .capabilities
            .iter()
            .map(|item| item.to_string())
            .collect(),
        default_args: backend_server_args(template, override_value),
        runtime_mode: template.runtime_mode.to_string(),
        model_artifact: template.model_artifact.to_string(),
        supports_mmproj: template.supports_mmproj,
        supports_ctx_size: template.supports_ctx_size,
        python_modules: template
            .python_modules
            .iter()
            .map(|item| item.to_string())
            .collect(),
        external_server_protocol: template.external_server_protocol.map(str::to_string),
        log_file_name: template.log_file_name.to_string(),
    }
}

fn discover_runtime_root(host: HostInfo, requested_runtime_root: &str) -> PathBuf {
    if let Some(root) = paths::runtime_root_override() {
        return root;
    }
    let requested = requested_runtime_root.trim();
    if !requested.is_empty() {
        let requested_path = resolve_app_path(PathBuf::from(requested));
        if requested_path.is_dir() {
            return requested_path;
        }
    }

    let portable_root = paths::repo_root().join("runtime");
    if portable_root.is_dir() {
        return portable_root;
    }

    paths::local_dir()
        .join("runtime")
        .join(host.runtime_folder_name())
}

fn resolve_runtime_dir(
    template: &BackendTemplate,
    runtime_root: &Path,
    override_value: &Value,
) -> PathBuf {
    if let Some(runtime_override) = override_string(override_value, "runtime_dir") {
        return resolve_app_path(PathBuf::from(runtime_override));
    }
    let primary = runtime_root.join(template.runtime_dir_name);
    if primary.exists() {
        return primary;
    }
    template
        .fallback_runtime_dir_names
        .iter()
        .map(|fallback| runtime_root.join(fallback))
        .find(|candidate| candidate.exists())
        .unwrap_or(primary)
}

fn resolve_models_dir(template: &BackendTemplate, override_value: &Value) -> Option<PathBuf> {
    if let Some(env_value) = env_value(&format!("{}_MODELS_DIR", template.env_prefix)) {
        return Some(resolve_app_path(PathBuf::from(env_value)));
    }
    if let Some(value) = override_value.get("models_dir") {
        if value.is_null() || value.as_str().is_some_and(|text| text.trim().is_empty()) {
            return None;
        }
        return Some(resolve_app_path(PathBuf::from(value_to_string(value))));
    }
    Some(paths::local_dir().join("models"))
}

fn backend_server_args(template: &BackendTemplate, override_value: &Value) -> Vec<String> {
    let official_llama_cpp =
        template.family == "llama.cpp" && template.id.starts_with("llama.cpp-");
    let mut args = if official_llama_cpp {
        LLAMA_CPP_CACHE_SAFETY_ARGS
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    args.extend(
        template
            .default_extra_args
            .iter()
            .map(|value| value.to_string()),
    );
    let ngl = env_value(&format!("{}_NGL", template.env_prefix))
        .or_else(|| override_string(override_value, "ngl"))
        .or_else(|| template.default_ngl.map(str::to_string));
    if let Some(ngl) = ngl
        && !ngl.trim().is_empty()
    {
        args.extend(["-ngl".to_string(), ngl]);
    }
    push_optional_int_arg(
        &mut args,
        "-c",
        env_value(&format!("{}_CTX_SIZE", template.env_prefix))
            .or_else(|| override_string(override_value, "ctx_size")),
    );
    push_optional_int_arg(
        &mut args,
        "-np",
        env_value(&format!("{}_PARALLEL", template.env_prefix))
            .or_else(|| override_string(override_value, "parallel")),
    );
    let cache_ram = env_value(&format!("{}_CACHE_RAM", template.env_prefix))
        .or_else(|| override_string(override_value, "cache_ram"))
        .or_else(|| official_llama_cpp.then(|| LLAMA_CPP_CACHE_RAM_MIB.to_string()));
    push_optional_int_arg(&mut args, "--cache-ram", cache_ram);
    args.extend(parse_extra_args(override_value.get("extra_args")));
    args
}

fn push_optional_int_arg(args: &mut Vec<String>, flag: &str, value: Option<String>) {
    let Some(value) = value else {
        return;
    };
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if value.parse::<i64>().is_ok() {
        args.extend([flag.to_string(), value.to_string()]);
    }
}

fn parse_extra_args(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => split_extra_args(text),
        Some(Value::Array(items)) => items
            .iter()
            .map(value_to_string)
            .filter(|text| !text.trim().is_empty())
            .collect(),
        Some(value) => {
            let text = value_to_string(value);
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![text]
            }
        }
    }
}

fn split_extra_args(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for ch in text.chars() {
        match quote {
            Some(active) if ch == active => quote = None,
            Some(_) => current.push(ch),
            None if ch == '"' || ch == '\'' => quote = Some(ch),
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                }
            }
            None => current.push(ch),
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

fn recommended_backend(rows: &[Value]) -> Option<String> {
    rows.iter()
        .filter(|row| backend_payload_has_capability(row, "chat"))
        .filter(|row| {
            row.get("binary_exists")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .filter(|row| {
            row.get("hardware_compatible")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .min_by_key(|row| {
            (
                row.get("priority")
                    .and_then(Value::as_i64)
                    .unwrap_or(
                        backend_priority(row.get("id").and_then(Value::as_str).unwrap_or(""))
                            as i64,
                    ),
                row.get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .and_then(|row| row.get("id").and_then(Value::as_str))
        .map(str::to_string)
}

fn backend_payload_has_capability(row: &Value, wanted: &str) -> bool {
    row.get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|capability| capability == wanted)
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn override_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).map(value_to_string).filter(|text| {
        let trimmed = text.trim();
        !trimmed.is_empty() && trimmed != "null"
    })
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn resolve_app_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    let text = path.to_string_lossy();
    if let Some(home) = text.strip_prefix("~/") {
        if let Some(home_dir) = home_dir() {
            return home_dir.join(home);
        }
    }
    paths::repo_root().join(path)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests;
