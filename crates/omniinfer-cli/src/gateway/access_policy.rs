use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::Result;
use omniinfer_core::gateway_auth::{GatewayAccessPolicy, GatewayAdminApiKey};
use serde_json::Value;

pub(super) struct DynamicAccessPolicy {
    base: GatewayAccessPolicy,
    path: PathBuf,
    loaded_mtime: Option<SystemTime>,
    file_admin_keys: Vec<GatewayAdminApiKey>,
}

impl DynamicAccessPolicy {
    pub(super) fn new(base: GatewayAccessPolicy, path: PathBuf) -> Self {
        Self {
            base,
            path,
            loaded_mtime: None,
            file_admin_keys: Vec::new(),
        }
    }

    pub(super) fn effective_policy(&mut self) -> GatewayAccessPolicy {
        self.reload_if_changed();
        let mut policy = self.base.clone();
        policy
            .admin_api_keys
            .extend(self.file_admin_keys.iter().cloned());
        policy
    }

    fn reload_if_changed(&mut self) {
        let Ok(metadata) = std::fs::metadata(&self.path) else {
            if self.loaded_mtime.is_some() || !self.file_admin_keys.is_empty() {
                self.loaded_mtime = None;
                self.file_admin_keys.clear();
            }
            return;
        };
        let modified = metadata.modified().ok();
        if modified.is_some() && modified == self.loaded_mtime {
            return;
        }
        let Ok(raw) = std::fs::read_to_string(&self.path) else {
            return;
        };
        let Ok(keys) = parse_admin_keys_file(&raw) else {
            return;
        };
        self.loaded_mtime = modified;
        self.file_admin_keys = keys;
    }
}

fn parse_admin_keys_file(raw: &str) -> Result<Vec<GatewayAdminApiKey>> {
    let value: Value = serde_json::from_str(raw)?;
    let source = value.get("keys").unwrap_or(&value);
    let mut keys = Vec::new();
    match source {
        Value::Object(map) => {
            for (id, key) in map {
                let Some(key) = key.as_str().map(str::trim).filter(|key| !key.is_empty()) else {
                    continue;
                };
                keys.push(GatewayAdminApiKey {
                    id: id.trim().to_string(),
                    key: key.to_string(),
                });
            }
        }
        Value::Array(items) => {
            for item in items {
                let Some(id) = item
                    .get("id")
                    .or_else(|| item.get("name"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                else {
                    continue;
                };
                let Some(key) = item
                    .get("key")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|key| !key.is_empty())
                else {
                    continue;
                };
                keys.push(GatewayAdminApiKey {
                    id: id.to_string(),
                    key: key.to_string(),
                });
            }
        }
        _ => {}
    }
    Ok(keys)
}
