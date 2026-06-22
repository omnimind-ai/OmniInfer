use std::fs;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::paths;

#[derive(Debug, Error)]
pub enum ServeStateError {
    #[error("failed to read serve state file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse serve state file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to create serve state directory {path}: {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write serve state file {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode serve state file {path}: {source}")]
    Encode {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ServePidInfo {
    pub pid: Option<u32>,
    pub port: Option<u16>,
    pub log: Option<String>,
    pub public_url: Option<String>,
    pub openai_base_url: Option<String>,
    pub backend: Option<String>,
    pub model: Option<String>,
    pub mmproj: Option<String>,
    pub ctx_size: Option<u32>,
    pub backend_ready: Option<bool>,
    pub backend_pid: Option<u32>,
    pub backend_port: Option<u16>,
}

pub fn load_serve_pid_info(port: u16) -> Result<Option<ServePidInfo>, ServeStateError> {
    let path = paths::serve_pid_file(port);
    if !path.is_file() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).map_err(|source| ServeStateError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let info = serde_json::from_str(&raw).map_err(|source| ServeStateError::Parse {
        path: path.display().to_string(),
        source,
    })?;
    Ok(Some(info))
}

pub fn save_serve_pid_info(info: &ServePidInfo) -> Result<(), ServeStateError> {
    let port = info.port.unwrap_or(9000);
    let path = paths::serve_pid_file(port);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| ServeStateError::CreateDir {
            path: parent.display().to_string(),
            source,
        })?;
    }
    let raw = serde_json::to_string_pretty(info).map_err(|source| ServeStateError::Encode {
        path: path.display().to_string(),
        source,
    })?;
    fs::write(&path, format!("{raw}\n")).map_err(|source| ServeStateError::Write {
        path: path.display().to_string(),
        source,
    })
}

pub fn remove_serve_pid_info(port: u16) -> Result<(), std::io::Error> {
    let path = paths::serve_pid_file(port);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_python_compatible_pid_info() {
        let info = ServePidInfo {
            pid: Some(123),
            port: Some(9000),
            log: Some("/tmp/serve.log".to_string()),
            public_url: Some("https://example.trycloudflare.com".to_string()),
            openai_base_url: Some("https://example.trycloudflare.com/v1".to_string()),
            backend: Some("llama.cpp-linux-cuda".to_string()),
            model: Some("/models/model.gguf".to_string()),
            mmproj: None,
            ctx_size: Some(8192),
            backend_ready: Some(true),
            backend_pid: Some(456),
            backend_port: Some(12345),
        };
        let value = serde_json::to_value(&info).unwrap();
        assert_eq!(value["pid"], 123);
        assert_eq!(value["port"], 9000);
        assert_eq!(value["public_url"], "https://example.trycloudflare.com");
        assert_eq!(
            value["openai_base_url"],
            "https://example.trycloudflare.com/v1"
        );
        assert_eq!(value["backend_ready"], true);
    }
}
