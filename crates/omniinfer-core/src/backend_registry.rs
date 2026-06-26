use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Value, json};

use crate::{config, local_state, paths};

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
        "llama.cpp-sycl" => 0,
        "llama.cpp-hip" => 0,
        "llama.cpp-linux-cuda" => 0,
        "llama.cpp-linux-rocm" => 0,
        "llama.cpp-linux-vulkan" => 0,
        "omniinfer-native-linux" => 0,
        "llama.cpp-linux-openvino" => 0,
        "llama.cpp-linux" => 1,
        "llama.cpp-linux-s390x" => 1,
        "vllm-linux-cuda" => 2,
        "llama.cpp-cpu" => 1,
        "llama.cpp-windows-arm64" => 1,
        "llama.cpp-ios" => 0,
        "mlx-ios" => 0,
        "ik_llama.cpp-linux" => 1,
        "ik_llama.cpp-linux-cuda" => 0,
        "ik_llama.cpp-cpu" => 1,
        "ik_llama.cpp-cuda" => 0,
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
    let mut args = template
        .default_extra_args
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    if let Some(default_ngl) = template.default_ngl {
        let ngl = env_value(&format!("{}_NGL", template.env_prefix))
            .or_else(|| override_string(override_value, "ngl"))
            .unwrap_or_else(|| default_ngl.to_string());
        if !ngl.trim().is_empty() {
            args.extend(["-ngl".to_string(), ngl]);
        }
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
    push_optional_int_arg(
        &mut args,
        "-cram",
        env_value(&format!("{}_CACHE_RAM", template.env_prefix))
            .or_else(|| override_string(override_value, "cache_ram")),
    );
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

fn is_hardware_compatible(host: HostInfo, spec: &BackendSpec) -> bool {
    let caps = spec
        .capabilities
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if caps.contains(&"arm64") && !matches!(host.machine, "aarch64" | "arm64") {
        return false;
    }
    if caps.contains(&"s390x") && host.machine != "s390x" {
        return false;
    }
    if caps.contains(&"openvino") || caps.contains(&"eagle3") {
        return spec.binary_exists();
    }
    if !gpu_backend_ids(host).contains(&spec.id.as_str()) {
        return true;
    }
    if caps.contains(&"cuda") {
        return cuda_detected();
    }
    if caps.contains(&"rocm") || caps.contains(&"hip") {
        return rocm_detected();
    }
    if caps.contains(&"metal") {
        return host.system == HostSystem::Mac || host.system == HostSystem::Ios;
    }
    if caps.contains(&"vulkan") {
        return vulkan_detected();
    }
    spec.binary_exists()
}

fn cuda_detected() -> bool {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false)
}

fn rocm_detected() -> bool {
    std::process::Command::new("rocm-smi")
        .arg("--showmeminfo")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn vulkan_detected() -> bool {
    std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn embedded_module_exists(runtime_dir: &Path, module_name: &str) -> bool {
    embedded_site_roots(runtime_dir)
        .iter()
        .any(|root| module_path_exists(root, module_name))
}

fn embedded_site_roots(runtime_dir: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let bases = [runtime_dir.to_path_buf(), runtime_dir.join("venv")];
    for base in bases {
        for candidate in [
            base.join("Lib").join("site-packages"),
            base.join("lib").join("site-packages"),
        ] {
            if candidate.is_dir() && !roots.contains(&candidate) {
                roots.push(candidate);
            }
        }
        let pattern_root = base.join("lib");
        if let Ok(entries) = std::fs::read_dir(pattern_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !name.starts_with("python") {
                    continue;
                }
                for site_name in ["site-packages", "dist-packages"] {
                    let candidate = path.join(site_name);
                    if candidate.is_dir() && !roots.contains(&candidate) {
                        roots.push(candidate);
                    }
                }
            }
        }
    }
    roots
}

fn module_path_exists(site_root: &Path, module_name: &str) -> bool {
    let module_path = module_name
        .split('.')
        .fold(site_root.to_path_buf(), |path, item| path.join(item));
    if module_path.is_dir() || module_path.with_extension("py").is_file() {
        return true;
    }
    let Some(parent) = module_path.parent() else {
        return false;
    };
    let Some(name) = module_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    std::fs::read_dir(parent)
        .map(|entries| {
            entries.flatten().any(|entry| {
                let path = entry.path();
                path.file_stem().and_then(|stem| stem.to_str()) == Some(name)
                    && matches!(
                        path.extension().and_then(|ext| ext.to_str()),
                        Some("so" | "pyd" | "dll" | "dylib")
                    )
            })
        })
        .unwrap_or(false)
}

fn recommended_backend(rows: &[Value]) -> Option<String> {
    rows.iter()
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

fn gpu_backend_ids(host: HostInfo) -> &'static [&'static str] {
    match host.system {
        HostSystem::Linux => &[
            "llama.cpp-linux-cuda",
            "llama.cpp-linux-rocm",
            "llama.cpp-linux-vulkan",
            "omniinfer-native-linux",
            "ik_llama.cpp-linux-cuda",
            "vllm-linux-cuda",
        ],
        HostSystem::Windows => &[
            "llama.cpp-cuda",
            "llama.cpp-vulkan",
            "llama.cpp-sycl",
            "llama.cpp-hip",
            "ik_llama.cpp-cuda",
        ],
        _ => &[],
    }
}

fn backend_templates(host: HostInfo) -> &'static [BackendTemplate] {
    match host.system {
        HostSystem::Linux => LINUX_TEMPLATES,
        HostSystem::Windows => WINDOWS_TEMPLATES,
        HostSystem::Mac => MAC_TEMPLATES,
        HostSystem::Android => ANDROID_TEMPLATES,
        HostSystem::Ios => IOS_TEMPLATES,
    }
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
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-linux-cuda",
            "llama.cpp Linux CUDA",
            "llama.cpp",
            "llama.cpp-linux-cuda",
            Some("llama-server"),
            "llama.cpp Linux CUDA backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "cuda", "linux"],
            "OMNIINFER_LLAMA_CPP_LINUX_CUDA",
        )
    },
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
    BackendTemplate {
        default_ngl: Some("999"),
        ..template(
            "llama.cpp-cuda",
            "llama.cpp CUDA",
            "llama.cpp",
            "llama.cpp-cuda",
            Some("llama-server.exe"),
            "llama.cpp CUDA backend managed by OmniInfer",
            &["chat", "vision", "stream", "gpu", "cuda"],
            "OMNIINFER_LLAMA_CPP_CUDA",
        )
    },
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(registry.get("mnn-linux").is_some());
    }

    #[test]
    fn cuda_backend_has_default_ngl() {
        let registry = BackendRegistry::build(
            HostInfo {
                system: HostSystem::Linux,
                machine: "x86_64",
            },
            "runtime",
            &Value::Null,
        );
        let backend = registry.get("llama.cpp-linux-cuda").unwrap();
        assert_eq!(backend.default_args, vec!["-ngl", "999"]);
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
                "-ngl",
                "12",
                "-c",
                "4096",
                "-np",
                "2",
                "-cram",
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
}
