use std::fs;
use std::path::PathBuf;

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::paths;

const PROFILE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Error)]
pub enum BackendProfileError {
    #[error("backend payload is missing field: {0}")]
    MissingField(&'static str),
    #[error("failed to create backend profile directory {path}: {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write backend profile {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode backend profile {path}: {source}")]
    Encode {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to read backend profile {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse backend profile {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("backend config must be a JSON object: {0}")]
    NotObject(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendProfileResult {
    pub path: PathBuf,
    pub created: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendProfile {
    pub path: PathBuf,
    pub backend_id: Option<String>,
    pub family: Option<String>,
    pub load_extra_args: Vec<String>,
    pub infer_extra_args: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BackendProfileTemplate<'a> {
    schema_version: u32,
    backend: &'a str,
    family: &'a str,
    description: &'static str,
    load: BackendProfileSection,
    infer: BackendProfileSection,
}

#[derive(Debug, Serialize)]
struct BackendProfileSection {
    extra_args: Vec<String>,
}

pub fn ensure_backend_profile_template(
    backend: &Value,
) -> Result<BackendProfileResult, BackendProfileError> {
    let backend_id = required_str(backend, "id")?;
    let family = required_str(backend, "family")?;
    let path = paths::backend_profile_file(backend_id);
    if path.is_file() {
        return Ok(BackendProfileResult {
            path,
            created: false,
        });
    }

    fs::create_dir_all(paths::backend_profile_dir()).map_err(|source| {
        BackendProfileError::CreateDir {
            path: paths::backend_profile_dir().display().to_string(),
            source,
        }
    })?;
    let template = BackendProfileTemplate {
        schema_version: PROFILE_SCHEMA_VERSION,
        backend: backend_id,
        family,
        description: "Advanced backend-native parameters for OmniInfer. Keep basic user inputs such as model path, message, image, and mmproj on the CLI, and only store backend-specific extra parameters here.",
        load: BackendProfileSection {
            extra_args: Vec::new(),
        },
        infer: BackendProfileSection {
            extra_args: Vec::new(),
        },
    };
    let raw =
        serde_json::to_string_pretty(&template).map_err(|source| BackendProfileError::Encode {
            path: path.display().to_string(),
            source,
        })?;
    fs::write(&path, format!("{raw}\n")).map_err(|source| BackendProfileError::Write {
        path: path.display().to_string(),
        source,
    })?;
    Ok(BackendProfileResult {
        path,
        created: true,
    })
}

pub fn load_backend_profile(path: PathBuf) -> Result<BackendProfile, BackendProfileError> {
    let raw = fs::read_to_string(&path).map_err(|source| BackendProfileError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let payload: Value =
        serde_json::from_str(&raw).map_err(|source| BackendProfileError::Parse {
            path: path.display().to_string(),
            source,
        })?;
    let Some(map) = payload.as_object() else {
        return Err(BackendProfileError::NotObject(path.display().to_string()));
    };
    let load_section = map
        .get("load")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let infer_section = map
        .get("infer")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut load_extra_args = parse_extra_args(load_section.get("extra_args"));
    if load_extra_args.is_empty() && load_section.contains_key("launcher_args") {
        load_extra_args = parse_extra_args(load_section.get("launcher_args"));
    }

    let mut infer_extra_args = parse_extra_args(infer_section.get("extra_args"));
    if infer_extra_args.is_empty() {
        infer_extra_args = legacy_infer_args(&infer_section);
    }

    Ok(BackendProfile {
        path,
        backend_id: optional_string(map.get("backend")),
        family: optional_string(map.get("family")),
        load_extra_args,
        infer_extra_args,
    })
}

fn required_str<'a>(value: &'a Value, key: &'static str) -> Result<&'a str, BackendProfileError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or(BackendProfileError::MissingField(key))
}

fn parse_extra_args(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) if text.trim().is_empty() => Vec::new(),
        Some(Value::String(text)) => split_extra_args(text),
        Some(Value::Array(items)) => items
            .iter()
            .map(value_to_arg)
            .filter(|text| !text.trim().is_empty())
            .collect(),
        _ => Vec::new(),
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

fn legacy_infer_args(infer_section: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut args = Vec::new();
    for key in ["temperature", "max_tokens", "stream", "think"] {
        if let Some(value) = infer_section.get(key) {
            let flag = format!("--{}", key.replace('_', "-"));
            if let Some(value) = value.as_bool() {
                args.push(flag);
                if matches!(key, "stream" | "think") {
                    args.push(if value { "true" } else { "false" }.to_string());
                }
            } else {
                args.push(flag);
                args.push(value_to_arg(value));
            }
        }
    }
    if let Some(Value::Object(overrides)) = infer_section.get("request_overrides") {
        for (key, value) in overrides {
            let flag = format!("--{}", key.replace('_', "-"));
            match value {
                Value::Bool(value) => {
                    args.push(flag);
                    args.push(if *value { "true" } else { "false" }.to_string());
                }
                Value::Array(items) => {
                    for item in items {
                        args.push(flag.clone());
                        args.push(value_to_arg(item));
                    }
                }
                other => {
                    args.push(flag);
                    args.push(value_to_arg(other));
                }
            }
        }
    }
    args
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    value
        .map(value_to_arg)
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn value_to_arg(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_template_matches_python_schema() {
        let backend = serde_json::json!({
            "id": "llama.cpp-linux-cuda",
            "family": "llama.cpp"
        });
        let backend_id = required_str(&backend, "id").unwrap();
        let family = required_str(&backend, "family").unwrap();
        let template = BackendProfileTemplate {
            schema_version: PROFILE_SCHEMA_VERSION,
            backend: backend_id,
            family,
            description: "Advanced backend-native parameters for OmniInfer. Keep basic user inputs such as model path, message, image, and mmproj on the CLI, and only store backend-specific extra parameters here.",
            load: BackendProfileSection {
                extra_args: Vec::new(),
            },
            infer: BackendProfileSection {
                extra_args: Vec::new(),
            },
        };
        let payload: Value =
            serde_json::from_str(&serde_json::to_string(&template).unwrap()).unwrap();
        assert_eq!(payload["schema_version"], 2);
        assert_eq!(payload["backend"], "llama.cpp-linux-cuda");
        assert_eq!(payload["family"], "llama.cpp");
        assert_eq!(payload["load"]["extra_args"], serde_json::json!([]));
        assert_eq!(payload["infer"]["extra_args"], serde_json::json!([]));
    }

    #[test]
    fn parses_profile_extra_args() {
        let payload = serde_json::json!({
            "backend": "llama.cpp-linux-cuda",
            "family": "llama.cpp",
            "load": { "extra_args": ["-ngl", 999] },
            "infer": {
                "request_overrides": {
                    "temperature": 0.2,
                    "stop": ["A", "B"]
                }
            }
        });
        let map = payload.as_object().unwrap();
        let load_section = map.get("load").and_then(Value::as_object).unwrap();
        let infer_section = map.get("infer").and_then(Value::as_object).unwrap();

        assert_eq!(
            parse_extra_args(load_section.get("extra_args")),
            vec!["-ngl".to_string(), "999".to_string()]
        );
        let infer_args = legacy_infer_args(infer_section);
        assert!(
            infer_args
                .windows(2)
                .any(|item| item == ["--temperature", "0.2"])
        );
        assert!(infer_args.windows(2).any(|item| item == ["--stop", "A"]));
        assert!(infer_args.windows(2).any(|item| item == ["--stop", "B"]));
        assert_eq!(
            optional_string(map.get("backend")).as_deref(),
            Some("llama.cpp-linux-cuda")
        );
    }

    #[test]
    fn splits_quoted_extra_args() {
        assert_eq!(
            split_extra_args(r#"--grammar "root ::= hello world" --flag"#),
            vec![
                "--grammar".to_string(),
                "root ::= hello world".to_string(),
                "--flag".to_string(),
            ]
        );
    }

    #[test]
    fn load_profile_uses_launcher_args_fallback() {
        let path = std::env::temp_dir().join(format!(
            "omniinfer-profile-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"{"backend":"b","family":"f","load":{"launcher_args":["--foo","bar"]}}"#,
        )
        .unwrap();
        let profile = load_backend_profile(path.clone()).unwrap();
        assert_eq!(profile.backend_id.as_deref(), Some("b"));
        assert_eq!(profile.family.as_deref(), Some("f"));
        assert_eq!(profile.load_extra_args, vec!["--foo", "bar"]);
        fs::remove_file(path).ok();
    }
}
