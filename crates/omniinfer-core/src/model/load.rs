use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use thiserror::Error;

use crate::backend_args::{parse_backend_chat_extra_args, parse_backend_load_extra_args};
use crate::backend_profiles::BackendProfile;

pub const DEFAULT_LOAD_CONTEXT_SIZE: u32 = 8192;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelLoadRequest {
    pub model: String,
    pub mmproj: Option<String>,
    pub no_mmproj: bool,
    pub ctx_size: Option<u32>,
    pub backend_port: Option<u16>,
    pub resource_budget_bytes: Option<u64>,
    pub config: Option<String>,
    pub backend_extra_args: Vec<String>,
    pub request_defaults: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelLoadPlan {
    pub payload: Value,
    pub backend: String,
    pub auto_selected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelLoadEvent {
    Status(String),
    Log(String),
    Done(Value),
}

#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error(
        "No installed backend found.\nBuild or install a backend first, then run:\n  omniinfer backend list\n  omniinfer backend select <backend>"
    )]
    NoInstalledBackend,
    #[error("Selected backend is no longer available locally: {0}")]
    SelectedBackendMissing(String),
    #[error("Model reference must not be empty.")]
    EmptyModel,
    #[error("Model path does not exist: {0}")]
    ModelMissing(String),
    #[error("vla.cpp model must be a checkpoint file, not a directory: {0}")]
    VlaModelMustBeFile(String),
    #[error("mmproj file does not exist: {0}")]
    MmprojMissing(String),
    #[error("--no-mmproj cannot be combined with --mmproj")]
    MmprojConflict,
    #[error("--ctx-size must be a positive integer")]
    InvalidCtxSize,
    #[error(
        "Backend config {path} belongs to {backend}, but the current selected backend is {selected}."
    )]
    ProfileBackendMismatch {
        path: String,
        backend: String,
        selected: String,
    },
    #[error("{0}")]
    BackendName(#[from] crate::backend::names::ResolveError),
    #[error("{0}")]
    BackendArgs(#[from] crate::backend_args::BackendArgError),
    #[error("Failed to load the model.")]
    MissingResult,
    #[error("model loading failed: {0}")]
    LoadFailed(String),
    #[error("model load response JSON parse failed: {0}")]
    ResponseJson(#[from] serde_json::Error),
}

pub fn build_model_load_payload(
    request: &ModelLoadRequest,
    backends: &[Value],
    recommended_backend: Option<&str>,
    selected_backend: Option<&str>,
    profile: Option<&BackendProfile>,
    cwd: &Path,
) -> Result<ModelLoadPlan, ModelLoadError> {
    let (mut backend_id, auto_selected) =
        select_backend(backends, recommended_backend, selected_backend)?;
    backend_id = crate::backend::names::resolve_rows(backends, &backend_id)?.to_string();
    let mut backend = find_backend(backends, &backend_id)
        .ok_or_else(|| ModelLoadError::SelectedBackendMissing(backend_id.clone()))?;

    if let Some(profile) = profile {
        if let Some(profile_backend) = profile.backend_id.as_deref() {
            let profile_backend = crate::backend::names::resolve_rows(backends, profile_backend)?;
            if profile_backend != backend_id {
                if selected_backend.is_some() {
                    return Err(ModelLoadError::ProfileBackendMismatch {
                        path: profile.path.display().to_string(),
                        backend: profile_backend.to_string(),
                        selected: backend_id,
                    });
                }
                backend_id = profile_backend.to_string();
                backend = find_backend(backends, &backend_id)
                    .ok_or_else(|| ModelLoadError::SelectedBackendMissing(backend_id.clone()))?;
            }
        }
    }

    let family = json_str(backend, "family").unwrap_or("");
    let mut load_tokens = profile
        .map(|profile| profile.load_extra_args.clone())
        .unwrap_or_default();
    load_tokens.extend(request.backend_extra_args.clone());
    let load_args = parse_backend_load_extra_args(&backend_id, family, &load_tokens)?;

    if request.no_mmproj && request.mmproj.is_some() {
        return Err(ModelLoadError::MmprojConflict);
    }

    let model = resolve_model_reference(&request.model, family, cwd)?;
    if family == "vla.cpp" && Path::new(&model).is_dir() {
        return Err(ModelLoadError::VlaModelMustBeFile(model));
    }
    let mmproj = match request.mmproj.as_deref() {
        Some(mmproj) => Some(resolve_existing_path(mmproj, cwd, "mmproj file")?),
        None => None,
    };
    let ctx_size = request
        .ctx_size
        .or(load_args.ctx_size)
        .or_else(|| backend_supports_ctx_size(backend).then_some(DEFAULT_LOAD_CONTEXT_SIZE));
    if ctx_size == Some(0) {
        return Err(ModelLoadError::InvalidCtxSize);
    }

    let mut payload = Map::new();
    payload.insert("model".to_string(), Value::String(model));
    payload.insert("no_mmproj".to_string(), Value::Bool(request.no_mmproj));
    if let Some(mmproj) = mmproj {
        payload.insert("mmproj".to_string(), Value::String(mmproj));
    }
    if let Some(ctx_size) = ctx_size {
        payload.insert(
            "ctx_size".to_string(),
            Value::Number(u64::from(ctx_size).into()),
        );
    }
    if let Some(backend_port) = request.backend_port {
        payload.insert(
            "backend_port".to_string(),
            Value::Number(u64::from(backend_port).into()),
        );
    }
    if let Some(resource_budget_bytes) = request.resource_budget_bytes {
        payload.insert(
            "resource_budget_bytes".to_string(),
            Value::Number(resource_budget_bytes.into()),
        );
    }
    payload.insert("backend".to_string(), Value::String(backend_id.clone()));
    if !load_args.launch_args.is_empty() {
        payload.insert(
            "launch_args".to_string(),
            Value::Array(
                load_args
                    .launch_args
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    let request_defaults = if let Some(explicit_defaults) = request.request_defaults.as_ref() {
        explicit_defaults.clone()
    } else if let Some(profile) = profile {
        parse_backend_chat_extra_args(family, &profile.infer_extra_args)?.request_overrides
    } else {
        Map::new()
    };
    if !request_defaults.is_empty() || request.request_defaults.is_some() {
        payload.insert(
            "request_defaults".to_string(),
            Value::Object(request_defaults),
        );
    }

    Ok(ModelLoadPlan {
        payload: Value::Object(payload),
        backend: backend_id,
        auto_selected,
    })
}

pub fn parse_model_load_response(
    content_type: Option<&str>,
    body: &str,
) -> Result<(Value, Vec<ModelLoadEvent>), ModelLoadError> {
    if !content_type
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        return Ok((serde_json::from_str(body.trim())?, Vec::new()));
    }

    let mut result = None;
    let mut events = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        if data.is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(data)?;
        match json_str(&event, "type").unwrap_or("") {
            "done" => {
                result = Some(event.clone());
                events.push(ModelLoadEvent::Done(event));
            }
            "error" => {
                let message = json_str(&event, "message")
                    .unwrap_or("model loading failed")
                    .to_string();
                return Err(ModelLoadError::LoadFailed(message));
            }
            "log" => {
                if let Some(message) = json_str(&event, "message") {
                    events.push(ModelLoadEvent::Log(message.to_string()));
                }
            }
            _ => {
                if let Some(message) = json_str(&event, "message") {
                    events.push(ModelLoadEvent::Status(message.to_string()));
                }
            }
        }
    }
    result
        .map(|result| (result, events))
        .ok_or(ModelLoadError::MissingResult)
}

fn select_backend(
    backends: &[Value],
    recommended_backend: Option<&str>,
    selected_backend: Option<&str>,
) -> Result<(String, bool), ModelLoadError> {
    if let Some(selected_backend) = selected_backend.filter(|value| !value.trim().is_empty()) {
        return Ok((selected_backend.to_string(), false));
    }
    if let Some(recommended_backend) = recommended_backend.filter(|value| !value.trim().is_empty())
    {
        return Ok((recommended_backend.to_string(), true));
    }
    let recommended = backends
        .iter()
        .find(|backend| json_bool(backend, "binary_exists") == Some(true))
        .and_then(|backend| json_str(backend, "id"))
        .ok_or(ModelLoadError::NoInstalledBackend)?;
    Ok((recommended.to_string(), true))
}

fn find_backend<'a>(backends: &'a [Value], backend_id: &str) -> Option<&'a Value> {
    backends
        .iter()
        .find(|backend| json_str(backend, "id") == Some(backend_id))
}

fn resolve_model_reference(text: &str, family: &str, cwd: &Path) -> Result<String, ModelLoadError> {
    let text = normalize_path_text(text);
    if text.is_empty() {
        return Err(ModelLoadError::EmptyModel);
    }
    let path = absolute_path_from_text(&text, cwd);
    if matches!(family, "vllm" | "freetoken") && !path.exists() {
        return Ok(text);
    }
    if !path.exists() {
        return Err(ModelLoadError::ModelMissing(path.display().to_string()));
    }
    Ok(path.display().to_string())
}

fn resolve_existing_path(text: &str, cwd: &Path, label: &str) -> Result<String, ModelLoadError> {
    let path = absolute_path_from_text(text, cwd);
    if !path.exists() {
        return match label {
            "mmproj file" => Err(ModelLoadError::MmprojMissing(path.display().to_string())),
            _ => Err(ModelLoadError::ModelMissing(path.display().to_string())),
        };
    }
    Ok(path.display().to_string())
}

fn normalize_path_text(text: &str) -> String {
    let mut text = text.trim();
    if text.len() >= 2 {
        let first = text.as_bytes()[0] as char;
        let last = text.as_bytes()[text.len() - 1] as char;
        if first == last && matches!(first, '"' | '\'') {
            text = text[1..text.len() - 1].trim();
        }
    }
    text.to_string()
}

fn absolute_path_from_text(text: &str, cwd: &Path) -> PathBuf {
    let path = PathBuf::from(normalize_path_text(text));
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn json_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
}

fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn backend_supports_ctx_size(backend: &Value) -> bool {
    json_bool(backend, "supports_ctx_size").unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(id: &str, family: &str, installed: bool) -> Value {
        serde_json::json!({
            "id": id,
            "family": family,
            "binary_exists": installed,
            "supports_ctx_size": true
        })
    }

    fn backend_without_ctx(id: &str, family: &str, installed: bool) -> Value {
        serde_json::json!({
            "id": id,
            "family": family,
            "binary_exists": installed,
            "supports_ctx_size": false
        })
    }

    #[test]
    fn rejects_vla_model_directory() {
        let cwd = temp_dir("vla-model-directory");
        let model_dir = cwd.join("smolvla");
        std::fs::create_dir_all(&model_dir).unwrap();
        let backend = backend_without_ctx("vla.cpp-linux", "vla.cpp", true);
        let error = build_model_load_payload(
            &ModelLoadRequest {
                model: model_dir.display().to_string(),
                ..ModelLoadRequest::default()
            },
            &[backend],
            None,
            Some("vla.cpp-linux"),
            None,
            &cwd,
        )
        .unwrap_err();
        assert!(matches!(error, ModelLoadError::VlaModelMustBeFile(_)));
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn builds_vla_payload_without_default_ctx() {
        let cwd = temp_dir("vla-payload");
        let model = cwd.join("smolvla.gguf");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(&model, "").unwrap();
        let backend = backend_without_ctx("vla.cpp-linux", "vla.cpp", true);
        let plan = build_model_load_payload(
            &ModelLoadRequest {
                model: model.display().to_string(),
                backend_extra_args: vec!["--timing-detail".to_string(), "phase".to_string()],
                ..ModelLoadRequest::default()
            },
            &[backend],
            None,
            Some("vla.cpp-linux"),
            None,
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["backend"], serde_json::json!("vla.cpp-linux"));
        assert_eq!(
            plan.payload["model"],
            serde_json::json!(model.display().to_string())
        );
        assert_eq!(
            plan.payload["launch_args"],
            serde_json::json!(["--timing-detail", "phase"])
        );
        assert!(plan.payload.get("ctx_size").is_none());
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn public_selector_and_legacy_profile_produce_identical_load_plans() {
        let cwd = temp_dir("selector-profile");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(cwd.join("model.gguf"), "").unwrap();
        let mut row = backend("llama.cpp-linux-cuda", "llama.cpp", true);
        row["selector"] = serde_json::json!("llama.cpp-cuda");
        let rows = [row];
        let request = ModelLoadRequest {
            model: "model.gguf".into(),
            ..Default::default()
        };
        let mut profile = BackendProfile {
            path: cwd.join("profile.json"),
            backend_id: Some("llama.cpp-linux-cuda".into()),
            family: Some("llama.cpp".into()),
            load_extra_args: vec!["-ngl".into(), "99".into()],
            infer_extra_args: vec![],
        };
        let legacy = build_model_load_payload(
            &request,
            &rows,
            None,
            Some("llama.cpp-linux-cuda"),
            Some(&profile),
            &cwd,
        )
        .unwrap();
        let alias = build_model_load_payload(
            &request,
            &rows,
            None,
            Some("llama.cpp-cuda"),
            Some(&profile),
            &cwd,
        )
        .unwrap();
        assert_eq!(legacy, alias);
        profile.backend_id = Some("llama.cpp-cuda".into());
        assert_eq!(
            legacy,
            build_model_load_payload(
                &request,
                &rows,
                None,
                Some("llama.cpp-linux-cuda"),
                Some(&profile),
                &cwd
            )
            .unwrap()
        );
        profile.backend_id = Some("other-backend".into());
        assert!(
            build_model_load_payload(
                &request,
                &rows,
                None,
                Some("llama.cpp-cuda"),
                Some(&profile),
                &cwd
            )
            .is_err()
        );
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn builds_llama_payload_with_profile_and_cli_extras() {
        let cwd = temp_dir("payload");
        let model = cwd.join("model.gguf");
        let mmproj = cwd.join("mmproj.gguf");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(&model, "").unwrap();
        std::fs::write(&mmproj, "").unwrap();
        let profile = BackendProfile {
            path: cwd.join("profile.json"),
            backend_id: Some("llama.cpp-linux-cuda".to_string()),
            family: Some("llama.cpp".to_string()),
            load_extra_args: vec!["--ctx-size".to_string(), "4096".to_string()],
            infer_extra_args: vec!["--temp".to_string(), "0.2".to_string()],
        };
        let request = ModelLoadRequest {
            model: "model.gguf".to_string(),
            mmproj: Some("mmproj.gguf".to_string()),
            backend_extra_args: vec!["-ngl".to_string(), "999".to_string()],
            ..ModelLoadRequest::default()
        };

        let plan = build_model_load_payload(
            &request,
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            None,
            Some("llama.cpp-linux-cuda"),
            Some(&profile),
            &cwd,
        )
        .unwrap();

        assert!(!plan.auto_selected);
        assert_eq!(plan.backend, "llama.cpp-linux-cuda");
        assert_eq!(plan.payload["model"], model.display().to_string());
        assert_eq!(plan.payload["mmproj"], mmproj.display().to_string());
        assert_eq!(plan.payload["ctx_size"], serde_json::json!(4096));
        assert_eq!(plan.payload["backend"], "llama.cpp-linux-cuda");
        assert_eq!(
            plan.payload["launch_args"],
            serde_json::json!(["-ngl", "999"])
        );
        assert_eq!(
            plan.payload["request_defaults"]["temperature"],
            serde_json::json!(0.2)
        );
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn no_mmproj_payload_disables_projector_and_conflicts_with_explicit_path() {
        let cwd = temp_dir("no-mmproj");
        let model = cwd.join("model.gguf");
        let mmproj = cwd.join("mmproj.gguf");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(&model, "").unwrap();
        std::fs::write(&mmproj, "").unwrap();

        let plan = build_model_load_payload(
            &ModelLoadRequest {
                model: model.display().to_string(),
                no_mmproj: true,
                ..ModelLoadRequest::default()
            },
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            None,
            Some("llama.cpp-linux-cuda"),
            None,
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["no_mmproj"], true);
        assert!(plan.payload.get("mmproj").is_none());

        let error = build_model_load_payload(
            &ModelLoadRequest {
                model: model.display().to_string(),
                mmproj: Some(mmproj.display().to_string()),
                no_mmproj: true,
                ..ModelLoadRequest::default()
            },
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            None,
            Some("llama.cpp-linux-cuda"),
            None,
            &cwd,
        )
        .unwrap_err();
        assert!(matches!(error, ModelLoadError::MmprojConflict));
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn explicit_request_defaults_replace_profile_defaults_for_restore() {
        let cwd = temp_dir("request-defaults");
        let model = cwd.join("model.gguf");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(&model, "").unwrap();
        let profile = BackendProfile {
            path: cwd.join("profile.json"),
            backend_id: None,
            family: Some("llama.cpp".to_string()),
            load_extra_args: Vec::new(),
            infer_extra_args: vec![
                "--temp".to_string(),
                "0.2".to_string(),
                "--top-p".to_string(),
                "0.9".to_string(),
            ],
        };
        let request = ModelLoadRequest {
            model: model.display().to_string(),
            request_defaults: Some(
                serde_json::from_value(serde_json::json!({
                    "temperature": 0.7,
                    "max_tokens": 64
                }))
                .unwrap(),
            ),
            ..ModelLoadRequest::default()
        };

        let plan = build_model_load_payload(
            &request,
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            None,
            Some("llama.cpp-linux-cuda"),
            Some(&profile),
            &cwd,
        )
        .unwrap();

        assert_eq!(plan.payload["request_defaults"]["temperature"], 0.7);
        assert_eq!(plan.payload["request_defaults"]["max_tokens"], 64);
        assert!(plan.payload["request_defaults"].get("top_p").is_none());
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn cli_ctx_overrides_profile_ctx() {
        let cwd = temp_dir("ctx");
        std::fs::create_dir_all(&cwd).unwrap();
        let model = cwd.join("model.gguf");
        std::fs::write(&model, "").unwrap();
        let profile = BackendProfile {
            path: cwd.join("profile.json"),
            backend_id: None,
            family: None,
            load_extra_args: vec!["--ctx-size".to_string(), "4096".to_string()],
            infer_extra_args: Vec::new(),
        };
        let request = ModelLoadRequest {
            model: model.display().to_string(),
            ctx_size: Some(8192),
            backend_port: Some(12345),
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            None,
            Some("llama.cpp-linux-cuda"),
            Some(&profile),
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["ctx_size"], serde_json::json!(8192));
        assert_eq!(plan.payload["backend_port"], serde_json::json!(12345));
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn defaults_ctx_for_backends_that_support_context_size() {
        let cwd = temp_dir("default-ctx");
        std::fs::create_dir_all(&cwd).unwrap();
        let model = cwd.join("model.gguf");
        std::fs::write(&model, "").unwrap();
        let request = ModelLoadRequest {
            model: model.display().to_string(),
            ..ModelLoadRequest::default()
        };

        let plan = build_model_load_payload(
            &request,
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            None,
            Some("llama.cpp-linux-cuda"),
            None,
            &cwd,
        )
        .unwrap();

        assert_eq!(
            plan.payload["ctx_size"],
            serde_json::json!(DEFAULT_LOAD_CONTEXT_SIZE)
        );
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn skips_default_ctx_for_backends_without_context_size_support() {
        let cwd = temp_dir("no-default-ctx");
        std::fs::create_dir_all(&cwd).unwrap();
        let model = cwd.join("model.mnn");
        std::fs::write(&model, "").unwrap();
        let request = ModelLoadRequest {
            model: model.display().to_string(),
            ..ModelLoadRequest::default()
        };

        let plan = build_model_load_payload(
            &request,
            &[backend_without_ctx("mnn-linux", "mnn", true)],
            None,
            Some("mnn-linux"),
            None,
            &cwd,
        )
        .unwrap();

        assert!(plan.payload.get("ctx_size").is_none());
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn auto_selects_first_installed_backend() {
        let cwd = temp_dir("auto");
        std::fs::create_dir_all(&cwd).unwrap();
        let model = cwd.join("model.gguf");
        std::fs::write(&model, "").unwrap();
        let request = ModelLoadRequest {
            model: model.display().to_string(),
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[
                backend("llama.cpp-linux-vulkan", "llama.cpp", false),
                backend("llama.cpp-linux-cuda", "llama.cpp", true),
            ],
            None,
            None,
            None,
            &cwd,
        )
        .unwrap();
        assert!(plan.auto_selected);
        assert_eq!(plan.backend, "llama.cpp-linux-cuda");
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn auto_selects_recommended_backend_when_available() {
        let cwd = temp_dir("recommended");
        std::fs::create_dir_all(&cwd).unwrap();
        let model = cwd.join("model.gguf");
        std::fs::write(&model, "").unwrap();
        let request = ModelLoadRequest {
            model: model.display().to_string(),
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[
                backend("llama.cpp-linux", "llama.cpp", true),
                backend("llama.cpp-linux-cuda", "llama.cpp", true),
            ],
            Some("llama.cpp-linux-cuda"),
            None,
            None,
            &cwd,
        )
        .unwrap();
        assert!(plan.auto_selected);
        assert_eq!(plan.backend, "llama.cpp-linux-cuda");
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn keeps_vllm_model_reference_when_path_is_missing() {
        let cwd = temp_dir("vllm");
        std::fs::create_dir_all(&cwd).unwrap();
        let request = ModelLoadRequest {
            model: "Qwen/Qwen3".to_string(),
            resource_budget_bytes: Some(7_516_192_768),
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[backend("vllm-linux-cuda", "vllm", true)],
            None,
            Some("vllm-linux-cuda"),
            None,
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["model"], "Qwen/Qwen3");
        assert_eq!(plan.payload["resource_budget_bytes"], 7_516_192_768_u64);
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn keeps_freetoken_model_reference_when_path_is_missing() {
        let cwd = temp_dir("freetoken");
        std::fs::create_dir_all(&cwd).unwrap();
        let request = ModelLoadRequest {
            model: "Qwen/Qwen3.6-35B-A3B".to_string(),
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[backend("freetoken-linux-cuda", "freetoken", true)],
            None,
            Some("freetoken-linux-cuda"),
            None,
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["model"], "Qwen/Qwen3.6-35B-A3B");
        assert_eq!(plan.payload["ctx_size"], DEFAULT_LOAD_CONTEXT_SIZE);
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn parses_json_model_load_response() {
        let (result, events) = parse_model_load_response(
            Some("application/json; charset=utf-8"),
            r#"{"selected_backend":"llama.cpp-linux-cuda"}"#,
        )
        .unwrap();
        assert_eq!(result["selected_backend"], "llama.cpp-linux-cuda");
        assert!(events.is_empty());
    }

    #[test]
    fn parses_sse_model_load_response() {
        let body = concat!(
            r#"data: {"type":"status","message":"Resolving model files..."}"#,
            "\n\n",
            r#"data: {"type":"log","message":"backend log"}"#,
            "\n\n",
            r#"data: {"type":"done","selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf"}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let (result, events) =
            parse_model_load_response(Some("text/event-stream; charset=utf-8"), body).unwrap();
        assert_eq!(result["selected_backend"], "llama.cpp-linux-cuda");
        assert_eq!(result["selected_model"], "/tmp/model.gguf");
        assert_eq!(
            events,
            vec![
                ModelLoadEvent::Status("Resolving model files...".to_string()),
                ModelLoadEvent::Log("backend log".to_string()),
                ModelLoadEvent::Done(result),
            ]
        );
    }

    #[test]
    fn rejects_sse_error_events() {
        let error = parse_model_load_response(
            Some("text/event-stream"),
            r#"data: {"type":"error","message":"bad model"}"#,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "model loading failed: bad model");
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "omniinfer-model-load-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
