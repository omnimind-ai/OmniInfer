use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

const MANIFEST_NAME: &str = "omni-model.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicModelManifest {
    pub id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub display_name: String,
    #[serde(default)]
    pub backend: Option<String>,
    pub model: String,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub launch_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicModelEntry {
    pub manifest: PublicModelManifest,
    pub directory: PathBuf,
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub native_ctx_size: Option<u32>,
}

#[derive(Debug, Error)]
pub enum PublicModelError {
    #[error("public model root is not configured")]
    RootNotConfigured,
    #[error("public model root does not exist or is not a directory: {0}")]
    RootMissing(String),
    #[error("public model id is not available: {0}")]
    ModelNotFound(String),
    #[error("invalid public model id '{0}': use lowercase letters, digits, '.', '_' or '-'")]
    InvalidId(String),
    #[error(
        "invalid public model path '{0}': paths must be relative files under the model directory"
    )]
    InvalidRelativePath(String),
    #[error("public model manifest parse failed at {path}: {message}")]
    ManifestParse { path: String, message: String },
    #[error("duplicate public model id or alias: {0}")]
    DuplicateId(String),
    #[error("public model file not found: {0}")]
    ModelFileMissing(String),
    #[error("public mmproj file not found: {0}")]
    MmprojFileMissing(String),
    #[error("{0}")]
    Io(String),
}

pub fn list_public_models(root: Option<&Path>) -> Result<Vec<PublicModelEntry>, PublicModelError> {
    let root = root.ok_or(PublicModelError::RootNotConfigured)?;
    if !root.is_dir() {
        return Err(PublicModelError::RootMissing(root.display().to_string()));
    }
    let mut entries = Vec::new();
    let mut ids = Vec::<String>::new();
    for child in fs::read_dir(root).map_err(io_error)? {
        let child = child.map_err(io_error)?;
        let directory = child.path();
        if !directory.is_dir() {
            continue;
        }
        let manifest_path = directory.join(MANIFEST_NAME);
        if !manifest_path.is_file() {
            continue;
        }
        let entry = read_manifest(&directory, &manifest_path)?;
        for id in std::iter::once(&entry.manifest.id).chain(entry.manifest.aliases.iter()) {
            validate_model_id(id)?;
            if ids.iter().any(|seen| seen == id) {
                return Err(PublicModelError::DuplicateId(id.clone()));
            }
            ids.push(id.clone());
        }
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.manifest.id.cmp(&right.manifest.id));
    Ok(entries)
}

pub fn resolve_public_model(
    root: Option<&Path>,
    requested: &str,
) -> Result<PublicModelEntry, PublicModelError> {
    let requested = requested.trim();
    validate_model_id(requested)?;
    list_public_models(root)?
        .into_iter()
        .find(|entry| {
            entry.manifest.id == requested
                || entry.manifest.aliases.iter().any(|id| id == requested)
        })
        .ok_or_else(|| PublicModelError::ModelNotFound(requested.to_string()))
}

pub fn public_models_payload(entries: &[PublicModelEntry]) -> Value {
    json!({
        "object": "list",
        "data": entries.iter().map(public_model_payload).collect::<Vec<_>>(),
    })
}

pub fn public_model_payload(entry: &PublicModelEntry) -> Value {
    json!({
        "id": entry.manifest.id,
        "aliases": entry.manifest.aliases,
        "display_name": entry.manifest.display_name,
        "backend": entry.manifest.backend,
        "modalities": entry.manifest.modalities,
        "quant": entry.manifest.quant,
        "ctx_size": entry.native_ctx_size,
        "native_ctx_size": entry.native_ctx_size,
        "default_ctx_size": entry.manifest.ctx_size,
        "model": entry.model_path.display().to_string(),
        "mmproj": entry.mmproj_path.as_ref().map(|path| path.display().to_string()),
    })
}

pub fn looks_like_public_model_id(value: &str) -> bool {
    validate_model_id(value.trim()).is_ok()
}

fn read_manifest(
    directory: &Path,
    manifest_path: &Path,
) -> Result<PublicModelEntry, PublicModelError> {
    let text = fs::read_to_string(manifest_path).map_err(io_error)?;
    let manifest: PublicModelManifest =
        serde_json::from_str(&text).map_err(|error| PublicModelError::ManifestParse {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        })?;
    validate_model_id(&manifest.id)?;
    for alias in &manifest.aliases {
        validate_model_id(alias)?;
    }
    let model_path = resolve_manifest_file(directory, &manifest.model)?;
    if !model_path.is_file() {
        return Err(PublicModelError::ModelFileMissing(
            model_path.display().to_string(),
        ));
    }
    let mmproj_path = manifest
        .mmproj
        .as_deref()
        .map(|path| resolve_manifest_file(directory, path))
        .transpose()?;
    if let Some(mmproj) = mmproj_path.as_ref()
        && !mmproj.is_file()
    {
        return Err(PublicModelError::MmprojFileMissing(
            mmproj.display().to_string(),
        ));
    }
    Ok(PublicModelEntry {
        manifest,
        directory: directory.to_path_buf(),
        native_ctx_size: read_gguf_context_length(&model_path).ok().flatten(),
        model_path,
        mmproj_path,
    })
}

fn read_gguf_context_length(path: &Path) -> Result<Option<u32>, PublicModelError> {
    let mut file = fs::File::open(path).map_err(io_error)?;
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).map_err(io_error)?;
    if &magic != b"GGUF" {
        return Ok(None);
    }
    let version = read_u32(&mut file)?;
    if version < 2 {
        return Ok(None);
    }
    let _tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if key.ends_with(".context_length") {
            return read_metadata_u32_value(&mut file, value_type);
        }
        skip_metadata_value(&mut file, value_type)?;
    }
    Ok(None)
}

fn read_metadata_u32_value<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
) -> Result<Option<u32>, PublicModelError> {
    match value_type {
        0 => read_u8(reader).map(u32::from).map(Some),
        2 => read_u16(reader).map(u32::from).map(Some),
        4 => read_u32(reader).map(Some),
        5 => read_i32(reader).map(|value| u32::try_from(value).ok()),
        10 => read_u64(reader).map(|value| Some(value.min(u64::from(u32::MAX)) as u32)),
        11 => read_i64(reader).map(|value| u32::try_from(value).ok()),
        _ => {
            skip_metadata_value(reader, value_type)?;
            Ok(None)
        }
    }
}

fn skip_metadata_value<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
) -> Result<(), PublicModelError> {
    match value_type {
        0 | 1 | 7 => seek_forward(reader, 1),
        2 | 3 => seek_forward(reader, 2),
        4 | 5 | 6 => seek_forward(reader, 4),
        8 => {
            let len = read_u64(reader)?;
            seek_forward(reader, len)
        }
        9 => {
            let item_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            for _ in 0..len {
                skip_metadata_value(reader, item_type)?;
            }
            Ok(())
        }
        10 | 11 | 12 => seek_forward(reader, 8),
        _ => Ok(()),
    }
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, PublicModelError> {
    let len = read_u64(reader)?;
    let len = usize::try_from(len).map_err(|error| PublicModelError::Io(error.to_string()))?;
    let mut bytes = vec![0_u8; len];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, PublicModelError> {
    let mut bytes = [0_u8; 1];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(bytes[0])
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, PublicModelError> {
    let mut bytes = [0_u8; 2];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, PublicModelError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, PublicModelError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, PublicModelError> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64, PublicModelError> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(i64::from_le_bytes(bytes))
}

fn seek_forward<R: Seek>(reader: &mut R, bytes: u64) -> Result<(), PublicModelError> {
    let offset = i64::try_from(bytes).map_err(|error| PublicModelError::Io(error.to_string()))?;
    reader.seek(SeekFrom::Current(offset)).map_err(io_error)?;
    Ok(())
}

fn resolve_manifest_file(directory: &Path, value: &str) -> Result<PathBuf, PublicModelError> {
    let path = Path::new(value.trim());
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PublicModelError::InvalidRelativePath(value.to_string()));
    }
    Ok(directory.join(path))
}

fn validate_model_id(value: &str) -> Result<(), PublicModelError> {
    let value = value.trim();
    if value.is_empty()
        || value.bytes().any(|ch| {
            !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && !matches!(ch, b'.' | b'_' | b'-')
        })
    {
        return Err(PublicModelError::InvalidId(value.to_string()));
    }
    Ok(())
}

fn io_error(error: std::io::Error) -> PublicModelError {
    PublicModelError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn lists_manifest_models() {
        let root = temp_root("public-models-list");
        let dir = root.join("qwen3.5-4b-q4_k_m");
        fs::create_dir_all(&dir).unwrap();
        write_test_gguf(&dir.join("model.gguf"), "qwen35.context_length", 131_072);
        fs::write(
            dir.join(MANIFEST_NAME),
            r#"{
                "id": "qwen3.5-4b-q4_k_m",
                "aliases": ["qwen35-4b"],
                "display_name": "Qwen3.5 4B Q4_K_M",
                "backend": "llama.cpp-linux-cuda",
                "model": "model.gguf",
                "ctx_size": 8192,
                "modalities": ["text"],
                "quant": "Q4_K_M",
                "launch_args": ["-ngl", "999"]
            }"#,
        )
        .unwrap();

        let entries = list_public_models(Some(&root)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].manifest.id, "qwen3.5-4b-q4_k_m");
        assert_eq!(entries[0].manifest.ctx_size, Some(8192));
        assert_eq!(entries[0].native_ctx_size, Some(131_072));
        assert_eq!(entries[0].model_path, dir.join("model.gguf"));
        assert_eq!(
            public_model_payload(&entries[0])["default_ctx_size"],
            json!(8192)
        );
        assert_eq!(public_model_payload(&entries[0])["ctx_size"], json!(131072));
        assert_eq!(
            public_model_payload(&entries[0])["native_ctx_size"],
            json!(131072)
        );
        assert_eq!(
            resolve_public_model(Some(&root), "qwen35-4b")
                .unwrap()
                .manifest
                .id,
            "qwen3.5-4b-q4_k_m"
        );
    }

    #[test]
    fn tolerates_models_without_gguf_context_metadata() {
        let root = temp_root("public-models-no-context");
        let dir = root.join("plain");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.gguf"), b"GGUF").unwrap();
        fs::write(
            dir.join(MANIFEST_NAME),
            r#"{
                "id": "plain",
                "display_name": "Plain",
                "model": "model.gguf",
                "ctx_size": 4096
            }"#,
        )
        .unwrap();

        let entries = list_public_models(Some(&root)).unwrap();
        assert_eq!(entries[0].native_ctx_size, None);
        assert_eq!(public_model_payload(&entries[0])["ctx_size"], Value::Null);
        assert_eq!(
            public_model_payload(&entries[0])["default_ctx_size"],
            json!(4096)
        );
    }

    #[test]
    fn rejects_path_traversal() {
        let root = temp_root("public-models-traversal");
        let dir = root.join("bad-model");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join(MANIFEST_NAME),
            r#"{
                "id": "bad-model",
                "display_name": "Bad Model",
                "model": "../secret.gguf"
            }"#,
        )
        .unwrap();

        assert!(matches!(
            list_public_models(Some(&root)).unwrap_err(),
            PublicModelError::InvalidRelativePath(_)
        ));
    }

    #[test]
    fn rejects_duplicate_aliases() {
        let root = temp_root("public-models-duplicate");
        for name in ["one", "two"] {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("model.gguf"), b"gguf").unwrap();
            fs::write(
                dir.join(MANIFEST_NAME),
                format!(
                    r#"{{
                        "id": "{name}",
                        "aliases": ["same"],
                        "display_name": "{name}",
                        "model": "model.gguf"
                    }}"#
                ),
            )
            .unwrap();
        }

        assert!(matches!(
            list_public_models(Some(&root)).unwrap_err(),
            PublicModelError::DuplicateId(id) if id == "same"
        ));
    }

    fn temp_root(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("omniinfer-rs-{name}-{nanos}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_test_gguf(path: &Path, context_key: &str, context_length: u32) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        write_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        write_gguf_string(&mut bytes, "qwen35");
        write_gguf_string(&mut bytes, context_key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&context_length.to_le_bytes());
        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
