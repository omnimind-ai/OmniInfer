use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use thiserror::Error;

use crate::backend_args::{parse_backend_chat_extra_args, parse_backend_load_extra_args};
use crate::backend_profiles::BackendProfile;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelLoadRequest {
    pub model: String,
    pub mmproj: Option<String>,
    pub ctx_size: Option<u32>,
    pub config: Option<String>,
    pub backend_extra_args: Vec<String>,
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
    #[error("mmproj file does not exist: {0}")]
    MmprojMissing(String),
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
    selected_backend: Option<&str>,
    profile: Option<&BackendProfile>,
    cwd: &Path,
) -> Result<ModelLoadPlan, ModelLoadError> {
    let (mut backend_id, auto_selected) = select_backend(backends, selected_backend)?;
    let mut backend = find_backend(backends, &backend_id)
        .ok_or_else(|| ModelLoadError::SelectedBackendMissing(backend_id.clone()))?;

    if let Some(profile) = profile {
        if let Some(profile_backend) = profile.backend_id.as_deref() {
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

    let model = resolve_model_reference(&request.model, family, cwd)?;
    let mmproj = match request.mmproj.as_deref() {
        Some(mmproj) => Some(resolve_existing_path(mmproj, cwd, "mmproj file")?),
        None => None,
    };
    let ctx_size = request.ctx_size.or(load_args.ctx_size);
    if ctx_size == Some(0) {
        return Err(ModelLoadError::InvalidCtxSize);
    }

    let mut payload = Map::new();
    payload.insert("model".to_string(), Value::String(model));
    if let Some(mmproj) = mmproj {
        payload.insert("mmproj".to_string(), Value::String(mmproj));
    }
    if let Some(ctx_size) = ctx_size {
        payload.insert(
            "ctx_size".to_string(),
            Value::Number(u64::from(ctx_size).into()),
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
    if let Some(profile) = profile {
        let chat_args = parse_backend_chat_extra_args(family, &profile.infer_extra_args)?;
        if !chat_args.request_overrides.is_empty() {
            payload.insert(
                "request_defaults".to_string(),
                Value::Object(chat_args.request_overrides),
            );
        }
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
    selected_backend: Option<&str>,
) -> Result<(String, bool), ModelLoadError> {
    if let Some(selected_backend) = selected_backend.filter(|value| !value.trim().is_empty()) {
        return Ok((selected_backend.to_string(), false));
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
    if family == "vllm" && !path.exists() {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(id: &str, family: &str, installed: bool) -> Value {
        serde_json::json!({
            "id": id,
            "family": family,
            "binary_exists": installed
        })
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
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[backend("llama.cpp-linux-cuda", "llama.cpp", true)],
            Some("llama.cpp-linux-cuda"),
            Some(&profile),
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["ctx_size"], serde_json::json!(8192));
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
            ..ModelLoadRequest::default()
        };
        let plan = build_model_load_payload(
            &request,
            &[backend("vllm-linux-cuda", "vllm", true)],
            Some("vllm-linux-cuda"),
            None,
            &cwd,
        )
        .unwrap();
        assert_eq!(plan.payload["model"], "Qwen/Qwen3");
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
