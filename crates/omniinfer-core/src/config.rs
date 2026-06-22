use std::fs;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{local_state, paths};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid config value: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    pub host: String,
    pub port: u16,
    pub default_backend: String,
    pub default_thinking: String,
    pub window_mode: String,
    pub startup_timeout: f64,
    pub runtime_root: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9000,
            default_backend: String::new(),
            default_thinking: "off".to_string(),
            window_mode: "hidden".to_string(),
            startup_timeout: 60.0,
            runtime_root: "runtime".to_string(),
        }
    }
}

impl AppConfig {
    pub fn service_host(&self) -> &str {
        match self.host.trim() {
            "" | "0.0.0.0" | "::" => "127.0.0.1",
            other => other,
        }
    }

    pub fn service_base_url(&self) -> String {
        format!("http://{}:{}", self.service_host(), self.port)
    }
}

pub fn load_app_config() -> Result<AppConfig, ConfigError> {
    let mut config = AppConfig::default();
    let config_path = paths::repo_root().join("config").join("omniinfer.json");
    if config_path.is_file() {
        let raw = fs::read_to_string(&config_path).map_err(|source| ConfigError::Read {
            path: config_path.display().to_string(),
            source,
        })?;
        let value: Value = serde_json::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: config_path.display().to_string(),
            source,
        })?;
        apply_config_value(&mut config, &value)?;
    }

    if let Some(default_thinking) = local_state::load_state()
        .ok()
        .and_then(|state| state.default_thinking)
    {
        config.default_thinking = if default_thinking { "on" } else { "off" }.to_string();
    }
    validate_config(&config)?;
    Ok(config)
}

fn apply_config_value(config: &mut AppConfig, value: &Value) -> Result<(), ConfigError> {
    let Some(map) = value.as_object() else {
        return Ok(());
    };
    if let Some(Value::String(host)) = map.get("host") {
        config.host = host.clone();
    }
    if let Some(port) = map.get("port").and_then(Value::as_u64) {
        config.port = u16::try_from(port)
            .map_err(|_| ConfigError::Invalid(format!("port {port} is outside 1-65535")))?;
    }
    if let Some(Value::String(default_backend)) = map.get("default_backend") {
        config.default_backend = default_backend.clone();
    }
    if let Some(Value::String(default_thinking)) = map.get("default_thinking") {
        config.default_thinking = default_thinking.clone();
    }
    if let Some(Value::String(window_mode)) = map.get("window_mode") {
        config.window_mode = window_mode.clone();
    }
    if let Some(timeout) = map.get("startup_timeout").and_then(Value::as_f64) {
        config.startup_timeout = timeout;
    }
    if let Some(Value::String(runtime_root)) = map.get("runtime_root") {
        config.runtime_root = runtime_root.clone();
    }
    Ok(())
}

fn validate_config(config: &AppConfig) -> Result<(), ConfigError> {
    if config.port == 0 {
        return Err(ConfigError::Invalid("port must be 1-65535".to_string()));
    }
    if config.host.trim().is_empty() {
        return Err(ConfigError::Invalid("host must be non-empty".to_string()));
    }
    if config.startup_timeout <= 0.0 {
        return Err(ConfigError::Invalid("startup_timeout must be positive".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_wildcard_host_for_local_clients() {
        let config = AppConfig {
            host: "0.0.0.0".to_string(),
            ..AppConfig::default()
        };
        assert_eq!(config.service_base_url(), "http://127.0.0.1:9000");
    }
}
