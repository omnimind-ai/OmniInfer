use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use omniinfer_core::backend_args::parse_backend_load_extra_args;
use omniinfer_core::backend_registry::{self, BackendRegistry, BackendScope};
use omniinfer_core::local_state;
use omniinfer_core::model_artifacts::{discover_llama_cpp_model_artifacts, maybe_auto_mmproj};
use omniinfer_core::model_load::DEFAULT_LOAD_CONTEXT_SIZE;
use omniinfer_core::runtime_plan::{ExternalRuntimeRequest, build_external_runtime_plan};
use omniinfer_core::runtime_process::{RuntimeProcess, RuntimeProcessOptions};
use serde_json::{Value, json};

use super::gpu_status::runtime_env_for_backend;

#[derive(Default)]
pub(super) struct RustRuntimeManager {
    selected_backend: Option<String>,
    loaded: BTreeMap<String, LoadedRustRuntime>,
    default_model_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeProxyTarget {
    pub(super) base_url: String,
    pub(super) backend_id: String,
    pub(super) model: Option<String>,
}

struct LoadedRustRuntime {
    model_key: String,
    owner_admin_id: Option<String>,
    backend_id: String,
    model: String,
    public_model_id: Option<String>,
    mmproj: Option<String>,
    ctx_size: Option<u32>,
    launch_args: Vec<String>,
    cuda_visible_devices: Option<String>,
    cuda_warning: Option<String>,
    process: RuntimeProcess,
    proxy_model_ref: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct LoadedRuntimeSummary {
    pub(super) id: String,
    pub(super) owner_admin_id: Option<String>,
    pub(super) backend_pid: u32,
}

impl RustRuntimeManager {
    pub(super) fn select_backend(&mut self, backend_id: &str) -> Result<Value> {
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(backend_id)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {backend_id}"))?;
        if self.selected_backend.as_deref() != Some(backend_id) {
            self.stop_runtime()?;
        }
        self.selected_backend = Some(backend_id.to_string());
        local_state::save_selected_backend(backend_id)?;
        Ok(json!({
            "ok": true,
            "selected_backend": backend_id,
            "binary_exists": backend.binary_exists(),
            "models_dir": backend.models_dir,
        }))
    }

    pub(super) fn stop_runtime(&mut self) -> Result<Value> {
        for (_, mut loaded) in std::mem::take(&mut self.loaded) {
            loaded.process.stop(Duration::from_secs(8))?;
        }
        self.default_model_key = None;
        Ok(json!({
            "ok": true,
            "stopped": true,
            "selected_backend": self.selected_backend,
        }))
    }

    pub(super) fn has_loaded_runtime(&self) -> bool {
        !self.loaded.is_empty()
    }

    pub(super) fn load_model(
        &mut self,
        payload: Value,
        backend_host: String,
        startup_timeout: Duration,
        owner_admin_id: Option<String>,
    ) -> Result<Value> {
        let model = json_required_str(&payload, "model")?.to_string();
        let public_model_id = payload
            .get("public_model_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);
        let requested_model_key = public_model_id.clone().unwrap_or_else(|| model.clone());
        if self.loaded.contains_key(&requested_model_key) {
            anyhow::bail!("model is already loaded: {requested_model_key}");
        }
        let requested_backend = self.resolve_requested_backend(&payload)?;
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(&requested_backend)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
        if backend.runtime_mode != "external_server" {
            anyhow::bail!(
                "{} is an embedded backend. Python control-plane fallback has been removed; use an external-server backend or a backend adapter service.",
                backend.id
            );
        }
        if !backend.binary_exists() {
            anyhow::bail!(
                "backend launcher not found: {}",
                backend.launcher_path.as_deref().unwrap_or("(unset)")
            );
        }
        let resolved_model = resolve_model_for_backend(&model, backend)?;
        let explicit_mmproj = payload
            .get("mmproj")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(|value| resolve_path_for_backend(value, backend, "mmproj file"))
            .transpose()?;
        let mmproj_path = explicit_mmproj.or(resolved_model.mmproj_path).or_else(|| {
            maybe_auto_mmproj(backend.models_dir.as_deref(), &resolved_model.model_path)
        });
        if mmproj_path.is_some() && !backend.supports_mmproj {
            anyhow::bail!("{} does not support mmproj inputs", backend.id);
        }
        let requested_ctx_size = payload
            .get("ctx_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        let launch_args = payload
            .get("launch_args")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            });
        let effective_launch_args = launch_args
            .clone()
            .unwrap_or_else(|| backend.default_args.clone());
        let launch_args_have_ctx =
            launch_args_have_ctx_size(&backend.family, &effective_launch_args);
        let launch_args_ctx_size =
            parse_backend_load_extra_args(&backend.id, &backend.family, &effective_launch_args)
                .ok()
                .and_then(|parsed| parsed.ctx_size);
        let ctx_size = requested_ctx_size.or(launch_args_ctx_size).or_else(|| {
            (backend.supports_ctx_size && !launch_args_have_ctx)
                .then_some(DEFAULT_LOAD_CONTEXT_SIZE)
        });
        let port = payload
            .get("backend_port")
            .and_then(Value::as_u64)
            .filter(|value| (1..=u64::from(u16::MAX)).contains(value))
            .and_then(|value| u16::try_from(value).ok())
            .map(Ok)
            .unwrap_or_else(|| pick_runtime_port(&backend_host))?;
        let backend_payload = serde_json::to_value(backend)?;
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend: backend_payload,
            model_path: resolved_model.model_path.clone(),
            mmproj_path: mmproj_path.clone(),
            host: backend_host.clone(),
            port,
            ctx_size,
            launch_args,
        })?;
        let log_path = PathBuf::from(&backend.runtime_dir)
            .join("logs")
            .join(model_log_file_name(
                &plan.log_file_name,
                &requested_model_key,
            ));
        let (runtime_env, cuda_selection) =
            runtime_env_for_backend(backend, &effective_launch_args);
        let process = RuntimeProcess::start(
            &plan,
            RuntimeProcessOptions {
                log_path,
                env: runtime_env,
                startup_timeout,
                health_host: backend_host.clone(),
            },
        )?;
        let info = process.info().clone();
        self.selected_backend = Some(backend.id.clone());
        local_state::save_selected_backend(&backend.id)?;
        local_state::save_selected_model(
            &resolved_model.model_path,
            mmproj_path.as_deref(),
            plan.ctx_size,
        )?;
        self.loaded.insert(
            requested_model_key.clone(),
            LoadedRustRuntime {
                model_key: requested_model_key.clone(),
                owner_admin_id: owner_admin_id.clone(),
                backend_id: backend.id.clone(),
                model: resolved_model.model_path.clone(),
                public_model_id: public_model_id.clone(),
                mmproj: mmproj_path.clone(),
                ctx_size: plan.ctx_size,
                launch_args: effective_launch_args,
                cuda_visible_devices: cuda_selection
                    .as_ref()
                    .map(|selection| selection.visible_devices.clone()),
                cuda_warning: cuda_selection
                    .as_ref()
                    .and_then(|selection| selection.warning.clone()),
                proxy_model_ref: plan.proxy_model_ref.clone(),
                process,
            },
        );
        self.default_model_key = Some(requested_model_key.clone());
        let mut response = json!({
            "ok": true,
            "model": requested_model_key,
            "owner_admin_id": owner_admin_id,
            "selected_backend": backend.id,
            "selected_model": resolved_model.model_path,
            "selected_public_model_id": public_model_id,
            "selected_mmproj": mmproj_path,
            "selected_ctx_size": plan.ctx_size,
            "backend_pid": info.pid,
            "backend_port": info.port,
            "launch_command": info.command,
            "log_path": info.log_path.display().to_string(),
        });
        if let Some(selection) = cuda_selection {
            response["cuda_visible_devices"] = json!(selection.visible_devices);
            if let Some(warning) = selection.warning {
                response["warning"] = json!(warning);
            }
        }
        Ok(response)
    }

    pub(super) fn unload_model(&mut self, model: &str, admin_id: Option<&str>) -> Result<Value> {
        let model_key = self
            .resolve_loaded_model_key(model)
            .ok_or_else(|| anyhow::anyhow!("model is not loaded: {model}"))?;
        let owner = self
            .loaded
            .get(&model_key)
            .and_then(|runtime| runtime.owner_admin_id.as_deref())
            .map(str::to_string);
        if let Some(owner) = owner.as_deref()
            && let Some(admin_id) = admin_id
            && owner != admin_id
        {
            anyhow::bail!(
                "model '{model_key}' is owned by admin '{owner}' and cannot be unloaded by admin '{admin_id}'"
            );
        }
        let Some(mut loaded) = self.loaded.remove(&model_key) else {
            anyhow::bail!("model is not loaded: {model}");
        };
        loaded.process.stop(Duration::from_secs(8))?;
        if self.default_model_key.as_deref() == Some(&model_key) {
            self.default_model_key = self.loaded.keys().next_back().cloned();
        }
        Ok(json!({
            "ok": true,
            "unloaded": true,
            "model": model_key,
            "owner_admin_id": owner,
        }))
    }

    pub(super) fn resolve_requested_backend(&self, payload: &Value) -> Result<String> {
        payload
            .get("backend")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .or_else(|| self.selected_backend.clone())
            .or_else(|| {
                BackendRegistry::load_current()
                    .api_payload(BackendScope::Installed)
                    .get("recommended")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| anyhow::anyhow!("no installed backend available"))
    }

    pub(super) fn proxy_base_for_model(&self, requested_model: Option<&str>) -> Option<String> {
        self.proxy_target_for_model(requested_model)
            .map(|target| target.base_url)
    }

    pub(super) fn proxy_target_for_model(
        &self,
        requested_model: Option<&str>,
    ) -> Option<RuntimeProxyTarget> {
        let key = self.resolve_proxy_model_key(requested_model)?;
        let loaded = self.loaded.get(&key)?;
        Some(RuntimeProxyTarget {
            base_url: format!("http://127.0.0.1:{}", loaded.process.info().port),
            backend_id: loaded.backend_id.clone(),
            model: loaded.proxy_model_ref.clone(),
        })
    }

    fn resolve_proxy_model_key(&self, requested_model: Option<&str>) -> Option<String> {
        match requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some("omniinfer" | "local") => self.default_model_key.clone(),
            Some(model) => self.resolve_loaded_model_key(model),
            None => self.default_model_key.clone(),
        }
    }

    fn resolve_loaded_model_key(&self, requested: &str) -> Option<String> {
        let requested = requested.trim();
        if requested.is_empty() {
            return None;
        }
        if self.loaded.contains_key(requested) {
            return Some(requested.to_string());
        }
        self.loaded.iter().find_map(|(key, loaded)| {
            (loaded.public_model_id.as_deref() == Some(requested)
                || loaded.model == requested
                || loaded.proxy_model_ref.as_deref() == Some(requested))
            .then(|| key.clone())
        })
    }

    pub(super) fn loaded_models_payload(&self) -> Value {
        json!({
            "object": "list",
            "data": self.loaded.values().map(loaded_runtime_payload).collect::<Vec<_>>(),
        })
    }

    pub(super) fn loaded_runtime_summaries(&self) -> Vec<LoadedRuntimeSummary> {
        self.loaded
            .values()
            .map(|loaded| LoadedRuntimeSummary {
                id: loaded.model_key.clone(),
                owner_admin_id: loaded.owner_admin_id.clone(),
                backend_pid: loaded.process.info().pid,
            })
            .collect()
    }

    pub(super) fn snapshot(&self) -> Value {
        let selected_backend = self.selected_backend.clone().or_else(|| {
            local_state::load_state()
                .ok()
                .and_then(|state| state.selected_backend)
        });
        let loaded_models = self
            .loaded
            .values()
            .map(loaded_runtime_payload)
            .collect::<Vec<_>>();
        let Some(default_key) = self.default_model_key.as_ref() else {
            return json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "public_model_id": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
                "backend_pid": null,
                "backend_port": null,
                "launch_args": [],
                "cuda_visible_devices": null,
                "warning": null,
                "launch_command": [],
                "proxy_model": null,
                "backend_log": null,
                "effective_parameters": {},
                "runtime": null,
                "loaded_models": loaded_models,
                "default_model": null,
            });
        };
        let Some(loaded) = self.loaded.get(default_key) else {
            return json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "public_model_id": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
                "backend_pid": null,
                "backend_port": null,
                "launch_args": [],
                "cuda_visible_devices": null,
                "warning": null,
                "launch_command": [],
                "proxy_model": null,
                "backend_log": null,
                "effective_parameters": {},
                "runtime": null,
                "loaded_models": loaded_models,
                "default_model": null,
            });
        };
        let info = loaded.process.info();
        json!({
            "backend": loaded.backend_id,
            "backend_ready": true,
            "model": loaded.model_key,
            "model_path": loaded.model,
            "public_model_id": loaded.public_model_id,
            "owner_admin_id": loaded.owner_admin_id,
            "mmproj": loaded.mmproj,
            "ctx_size": loaded.ctx_size,
            "request_defaults": {},
            "runtime_mode": "external_server",
            "backend_pid": info.pid,
            "backend_port": info.port,
            "launch_args": loaded.launch_args,
            "cuda_visible_devices": loaded.cuda_visible_devices,
            "warning": loaded.cuda_warning,
            "launch_command": info.command,
            "proxy_model": loaded.proxy_model_ref,
            "backend_log": info.log_path.display().to_string(),
            "effective_parameters": {},
            "runtime": {
                "mode": "external_server",
                "host": "127.0.0.1",
                "port": info.port,
                "pid": info.pid,
                "cuda_visible_devices": loaded.cuda_visible_devices,
                "launch_command": info.command,
                "log_path": info.log_path.display().to_string(),
                "proxy_model_ref": loaded.proxy_model_ref,
            },
            "log_path": info.log_path.display().to_string(),
            "loaded_models": loaded_models,
            "default_model": loaded.model_key,
        })
    }
}

fn loaded_runtime_payload(loaded: &LoadedRustRuntime) -> Value {
    let info = loaded.process.info();
    json!({
        "id": loaded.model_key,
        "owner_admin_id": loaded.owner_admin_id,
        "backend": loaded.backend_id,
        "model": loaded.model_key,
        "model_path": loaded.model,
        "public_model_id": loaded.public_model_id,
        "mmproj": loaded.mmproj,
        "ctx_size": loaded.ctx_size,
        "runtime_mode": "external_server",
        "backend_pid": info.pid,
        "backend_port": info.port,
        "launch_args": loaded.launch_args,
        "cuda_visible_devices": loaded.cuda_visible_devices,
        "warning": loaded.cuda_warning,
        "launch_command": info.command,
        "proxy_model": loaded.proxy_model_ref,
        "backend_log": info.log_path.display().to_string(),
    })
}

fn model_log_file_name(base: &str, model_key: &str) -> String {
    let sanitized = model_key
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    match base.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => {
            format!("{stem}-{sanitized}.{ext}")
        }
        _ => format!("{base}-{sanitized}.log"),
    }
}

fn json_required_str<'a>(payload: &'a Value, key: &'static str) -> Result<&'a str> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("field '{key}' is required"))
}

fn resolve_model_for_backend(
    model: &str,
    backend: &backend_registry::BackendSpec,
) -> Result<omniinfer_core::model_artifacts::ResolvedModelArtifacts> {
    if backend.model_artifact == "reference" {
        return Ok(omniinfer_core::model_artifacts::ResolvedModelArtifacts {
            model_path: model.to_string(),
            mmproj_path: None,
        });
    }
    let path = resolve_path_for_backend(model, backend, "model")?;
    if backend.model_artifact == "file" && PathBuf::from(&path).is_dir() {
        return Ok(discover_llama_cpp_model_artifacts(&PathBuf::from(path))?);
    }
    Ok(omniinfer_core::model_artifacts::ResolvedModelArtifacts {
        model_path: path,
        mmproj_path: None,
    })
}

fn resolve_path_for_backend(
    text: &str,
    backend: &backend_registry::BackendSpec,
    label: &str,
) -> Result<String> {
    let mut path = expand_home(PathBuf::from(text.trim()));
    if !path.is_absolute() {
        let Some(models_dir) = backend.models_dir.as_deref() else {
            anyhow::bail!("relative {label} path requires a configured models_dir");
        };
        path = PathBuf::from(models_dir).join(path);
    }
    if label == "model" && backend.model_artifact == "directory" {
        if !path.is_dir() {
            anyhow::bail!("model directory not found: {}", path.display());
        }
    } else if !path.exists() {
        anyhow::bail!("{label} not found: {}", path.display());
    }
    Ok(path.display().to_string())
}

fn expand_home(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path
}

fn launch_args_have_ctx_size(family: &str, args: &[String]) -> bool {
    args.iter().any(|arg| {
        let flag = arg.split_once('=').map(|(flag, _)| flag).unwrap_or(arg);
        match family {
            "vllm" => flag == "--max-model-len",
            "llama.cpp" | "turboquant" => matches!(flag, "-c" | "--ctx-size"),
            _ => matches!(flag, "-c" | "--ctx-size" | "--max-model-len"),
        }
    })
}

pub(super) fn pick_runtime_port(host: &str) -> Result<u16> {
    let listener = std::net::TcpListener::bind((host, 0))?;
    Ok(listener.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
