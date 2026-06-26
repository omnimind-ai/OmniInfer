use std::fs;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::paths;

#[derive(Debug, Error)]
pub enum StateError {
    #[error("failed to read state file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse state file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to write state file {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode state file {path}: {source}")]
    Encode {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedModel {
    pub model: String,
    pub mmproj: Option<String>,
    pub ctx_size: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalState {
    pub selected_backend: Option<String>,
    pub selected_model: Option<SelectedModel>,
    pub default_thinking: Option<bool>,
    pub tui_show_reasoning: bool,
}

pub fn load_state() -> Result<LocalState, StateError> {
    let value = load_state_value()?;
    Ok(parse_state_value(&value))
}

pub fn save_selected_backend(backend: &str) -> Result<(), StateError> {
    let backend = backend.trim();
    if backend.is_empty() {
        return Ok(());
    }
    let mut value = load_state_value().unwrap_or_else(|_| serde_json::json!({}));
    if !value.is_object() {
        value = serde_json::json!({});
    }
    let map = value
        .as_object_mut()
        .expect("state value was normalized to object");
    map.insert(
        "selected_backend".to_string(),
        Value::String(backend.to_string()),
    );
    save_state_value(&value)
}

pub fn save_selected_model(
    model: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
) -> Result<(), StateError> {
    let model = model.trim();
    if model.is_empty() {
        return Ok(());
    }
    let mut value = load_state_value().unwrap_or_else(|_| serde_json::json!({}));
    if !value.is_object() {
        value = serde_json::json!({});
    }
    let map = value
        .as_object_mut()
        .expect("state value was normalized to object");
    map.insert(
        "selected_model".to_string(),
        Value::String(model.to_string()),
    );
    match mmproj.map(str::trim).filter(|value| !value.is_empty()) {
        Some(mmproj) => {
            map.insert(
                "selected_mmproj".to_string(),
                Value::String(mmproj.to_string()),
            );
        }
        None => {
            map.remove("selected_mmproj");
        }
    }
    match ctx_size.filter(|value| *value > 0) {
        Some(ctx_size) => {
            map.insert(
                "selected_ctx_size".to_string(),
                Value::Number(u64::from(ctx_size).into()),
            );
        }
        None => {
            map.remove("selected_ctx_size");
        }
    }
    save_state_value(&value)
}

pub fn save_tui_show_reasoning(enabled: bool) -> Result<(), StateError> {
    save_boolish_state_field("tui_show_reasoning", enabled)
}

pub fn save_default_thinking(enabled: bool) -> Result<(), StateError> {
    save_boolish_state_field("default_thinking", enabled)
}

fn save_boolish_state_field(key: &str, enabled: bool) -> Result<(), StateError> {
    let mut value = load_state_value().unwrap_or_else(|_| serde_json::json!({}));
    if !value.is_object() {
        value = serde_json::json!({});
    }
    let map = value
        .as_object_mut()
        .expect("state value was normalized to object");
    map.insert(
        key.to_string(),
        Value::String(if enabled { "on" } else { "off" }.to_string()),
    );
    save_state_value(&value)
}

fn load_state_value() -> Result<Value, StateError> {
    let path = if paths::state_file().is_file() {
        paths::state_file()
    } else if paths::legacy_state_file().is_file() {
        paths::legacy_state_file()
    } else {
        return Ok(serde_json::json!({}));
    };

    let raw = fs::read_to_string(&path).map_err(|source| StateError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let value: Value = serde_json::from_str(&raw).map_err(|source| StateError::Parse {
        path: path.display().to_string(),
        source,
    })?;
    Ok(value)
}

fn save_state_value(value: &Value) -> Result<(), StateError> {
    let path = paths::state_file();
    let legacy = paths::legacy_state_file();
    fs::create_dir_all(paths::local_config_dir()).map_err(|source| StateError::Write {
        path: paths::local_config_dir().display().to_string(),
        source,
    })?;
    let tmp = path.with_extension("tmp");
    let raw = serde_json::to_string_pretty(value).map_err(|source| StateError::Encode {
        path: path.display().to_string(),
        source,
    })?;
    fs::write(&tmp, format!("{raw}\n")).map_err(|source| StateError::Write {
        path: tmp.display().to_string(),
        source,
    })?;
    fs::rename(&tmp, &path).map_err(|source| StateError::Write {
        path: path.display().to_string(),
        source,
    })?;
    if legacy.is_file() {
        fs::remove_file(&legacy).map_err(|source| StateError::Write {
            path: legacy.display().to_string(),
            source,
        })?;
    }
    Ok(())
}

fn parse_state_value(value: &Value) -> LocalState {
    let Some(map) = value.as_object() else {
        return LocalState::default();
    };

    let selected_backend = string_field(map.get("selected_backend"));
    let selected_model = string_field(map.get("selected_model")).map(|model| SelectedModel {
        model,
        mmproj: string_field(map.get("selected_mmproj")),
        ctx_size: map
            .get("selected_ctx_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0),
    });

    LocalState {
        selected_backend,
        selected_model,
        default_thinking: boolish_field(map.get("default_thinking")),
        tui_show_reasoning: boolish_field(map.get("tui_show_reasoning")).unwrap_or(false),
    }
}

fn string_field(value: Option<&Value>) -> Option<String> {
    let text = match value {
        Some(Value::String(text)) => text.trim(),
        Some(other) => return Some(other.to_string()).filter(|text| !text.trim().is_empty()),
        None => return None,
    };
    (!text.is_empty()).then(|| text.to_string())
}

fn boolish_field(value: Option<&Value>) -> Option<bool> {
    match value {
        Some(Value::Bool(value)) => Some(*value),
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" | "enabled" | "show" | "shown" | "visible" => Some(true),
            "off" | "false" | "0" | "no" | "disabled" | "hide" | "hidden" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_python_state_shape() {
        let value = serde_json::json!({
            "selected_backend": "llama.cpp-linux-cuda",
            "selected_model": "/models/model.gguf",
            "selected_mmproj": "/models/mmproj.gguf",
            "selected_ctx_size": 8192,
            "default_thinking": "off",
            "tui_show_reasoning": "on"
        });

        let state = parse_state_value(&value);
        assert_eq!(
            state.selected_backend.as_deref(),
            Some("llama.cpp-linux-cuda")
        );
        assert_eq!(
            state.selected_model,
            Some(SelectedModel {
                model: "/models/model.gguf".to_string(),
                mmproj: Some("/models/mmproj.gguf".to_string()),
                ctx_size: Some(8192),
            })
        );
        assert_eq!(state.default_thinking, Some(false));
        assert!(state.tui_show_reasoning);
    }

    #[test]
    fn save_selected_backend_preserves_unknown_fields() {
        let value = serde_json::json!({
            "selected_backend": "old",
            "future": { "keep": true }
        });
        let mut value = value;
        value.as_object_mut().unwrap().insert(
            "selected_backend".to_string(),
            Value::String("new".to_string()),
        );
        let state = parse_state_value(&value);
        assert_eq!(state.selected_backend.as_deref(), Some("new"));
        assert_eq!(value["future"]["keep"], true);
    }

    #[test]
    fn tui_reasoning_state_uses_python_shape() {
        let mut value = serde_json::json!({
            "future": { "keep": true }
        });
        value.as_object_mut().unwrap().insert(
            "tui_show_reasoning".to_string(),
            Value::String("on".to_string()),
        );
        let state = parse_state_value(&value);
        assert!(state.tui_show_reasoning);
        assert_eq!(value["future"]["keep"], true);
    }

    #[test]
    fn selected_model_state_shape_matches_python() {
        let value = serde_json::json!({
            "selected_model": "/models/model.gguf",
            "selected_mmproj": "/models/mmproj.gguf",
            "selected_ctx_size": 8192,
            "future": { "keep": true }
        });
        let state = parse_state_value(&value);
        assert_eq!(
            state.selected_model,
            Some(SelectedModel {
                model: "/models/model.gguf".to_string(),
                mmproj: Some("/models/mmproj.gguf".to_string()),
                ctx_size: Some(8192),
            })
        );
        assert_eq!(value["future"]["keep"], true);
    }

    #[test]
    fn saves_default_thinking_as_python_shape() {
        let root = temp_root("default-thinking");
        let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", root.display().to_string());
        save_default_thinking(true).expect("save");
        let state = load_state().expect("load");
        assert_eq!(state.default_thinking, Some(true));
        let raw = std::fs::read_to_string(crate::paths::state_file()).expect("read state");
        assert!(raw.contains(r#""default_thinking": "on""#));
        std::fs::remove_dir_all(root).ok();
    }

    fn temp_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("omniinfer-rs-local-state-{name}-{nanos}"))
    }

    struct EnvGuard {
        key: &'static str,
        old: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: String) -> Self {
            let old = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, old }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(old) = &self.old {
                    std::env::set_var(self.key, old);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }
}
