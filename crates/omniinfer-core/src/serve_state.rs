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
