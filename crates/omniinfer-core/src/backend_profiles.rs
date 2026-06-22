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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendProfileResult {
    pub path: PathBuf,
    pub created: bool,
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

fn required_str<'a>(value: &'a Value, key: &'static str) -> Result<&'a str, BackendProfileError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or(BackendProfileError::MissingField(key))
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
}
