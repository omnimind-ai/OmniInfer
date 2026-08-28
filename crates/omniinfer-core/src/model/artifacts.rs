use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelArtifacts {
    pub model_path: String,
    pub mmproj_path: Option<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelArtifactError {
    #[error("model directory not found: {0}")]
    ModelDirMissing(String),
    #[error("no GGUF files were found under model directory: {0}")]
    NoGguf(String),
    #[error("no text model GGUF file was found under model directory: {0}")]
    NoTextModel(String),
    #[error(
        "multiple text model GGUF files were found under {0}; please keep a single model GGUF in that directory or set load.model explicitly"
    )]
    MultipleTextModels(String),
    #[error(
        "multiple mmproj GGUF files were found under {0}; please keep a single mmproj GGUF in that directory or set load.mmproj explicitly"
    )]
    MultipleMmproj(String),
}

pub fn discover_llama_cpp_model_artifacts(
    model_dir: &Path,
) -> Result<ResolvedModelArtifacts, ModelArtifactError> {
    if !model_dir.is_dir() {
        return Err(ModelArtifactError::ModelDirMissing(
            model_dir.display().to_string(),
        ));
    }
    let mut gguf_files = Vec::new();
    collect_gguf_files(model_dir, &mut gguf_files);
    gguf_files.sort();
    if gguf_files.is_empty() {
        return Err(ModelArtifactError::NoGguf(model_dir.display().to_string()));
    }
    let (mmproj_candidates, model_files): (Vec<_>, Vec<_>) =
        gguf_files.into_iter().partition(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.to_ascii_lowercase().contains("mmproj"))
                .unwrap_or(false)
        });
    let model_candidates = model_files
        .into_iter()
        .filter(|path| !is_secondary_gguf_shard(path))
        .collect::<Vec<_>>();
    if model_candidates.is_empty() {
        return Err(ModelArtifactError::NoTextModel(
            model_dir.display().to_string(),
        ));
    }
    if model_candidates.len() > 1 {
        return Err(ModelArtifactError::MultipleTextModels(
            model_dir.display().to_string(),
        ));
    }
    if mmproj_candidates.len() > 1 {
        return Err(ModelArtifactError::MultipleMmproj(
            model_dir.display().to_string(),
        ));
    }
    Ok(ResolvedModelArtifacts {
        model_path: model_candidates[0].display().to_string(),
        mmproj_path: mmproj_candidates
            .first()
            .map(|path| path.display().to_string()),
    })
}

pub fn maybe_auto_mmproj(models_dir: Option<&str>, model_path: &str) -> Option<String> {
    let model_file = PathBuf::from(model_path);
    for name in [
        "mmproj-F32.gguf",
        "mmproj-f32.gguf",
        "mmproj-F16.gguf",
        "mmproj-f16.gguf",
    ] {
        let candidate = model_file.with_file_name(name);
        if candidate.is_file() {
            return Some(candidate.display().to_string());
        }
    }
    let root = models_dir.map(PathBuf::from)?;
    for name in [
        "mmproj-F32.gguf",
        "mmproj-f32.gguf",
        "mmproj-F16.gguf",
        "mmproj-f16.gguf",
    ] {
        let candidate = root.join(name);
        if candidate.is_file() {
            return Some(candidate.display().to_string());
        }
    }
    None
}

fn collect_gguf_files(root: &Path, output: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, output);
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            output.push(path);
        }
    }
}

fn is_secondary_gguf_shard(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let Some((prefix, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, index)) = prefix.rsplit_once('-') else {
        return false;
    };
    if index.len() != 5
        || total.len() != 5
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }
    let Ok(index) = index.parse::<u32>() else {
        return false;
    };
    let Ok(total) = total.parse::<u32>() else {
        return false;
    };
    index > 1 && index <= total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_single_model_and_mmproj() {
        let root = temp_root("artifacts-single");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.gguf"), b"").unwrap();
        std::fs::write(root.join("mmproj-F16.gguf"), b"").unwrap();
        let resolved = discover_llama_cpp_model_artifacts(&root).unwrap();
        assert!(resolved.model_path.ends_with("model.gguf"));
        assert!(resolved.mmproj_path.unwrap().ends_with("mmproj-F16.gguf"));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rejects_multiple_text_models() {
        let root = temp_root("artifacts-multiple");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.gguf"), b"").unwrap();
        std::fs::write(root.join("b.gguf"), b"").unwrap();
        let error = discover_llama_cpp_model_artifacts(&root).unwrap_err();
        assert!(matches!(error, ModelArtifactError::MultipleTextModels(_)));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn discovers_first_shard_of_split_model() {
        let root = temp_root("artifacts-split");
        std::fs::create_dir_all(&root).unwrap();
        for index in 1..=3 {
            std::fs::write(root.join(format!("model-0000{index}-of-00003.gguf")), b"").unwrap();
        }
        std::fs::write(root.join("mmproj-F16.gguf"), b"").unwrap();
        let resolved = discover_llama_cpp_model_artifacts(&root).unwrap();
        assert!(resolved.model_path.ends_with("model-00001-of-00003.gguf"));
        assert!(resolved.mmproj_path.unwrap().ends_with("mmproj-F16.gguf"));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rejects_multiple_split_model_sets() {
        let root = temp_root("artifacts-multiple-split");
        std::fs::create_dir_all(&root).unwrap();
        for quantization in ["Q4", "Q8"] {
            for index in 1..=2 {
                std::fs::write(
                    root.join(format!("model-{quantization}-0000{index}-of-00002.gguf")),
                    b"",
                )
                .unwrap();
            }
        }
        let error = discover_llama_cpp_model_artifacts(&root).unwrap_err();
        assert!(matches!(error, ModelArtifactError::MultipleTextModels(_)));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn finds_sibling_auto_mmproj() {
        let root = temp_root("artifacts-auto-mmproj");
        std::fs::create_dir_all(&root).unwrap();
        let model = root.join("model.gguf");
        std::fs::write(&model, b"").unwrap();
        std::fs::write(root.join("mmproj-F32.gguf"), b"").unwrap();
        let mmproj = maybe_auto_mmproj(None, &model.display().to_string()).unwrap();
        assert!(mmproj.ends_with("mmproj-F32.gguf"));
        std::fs::remove_dir_all(root).ok();
    }

    fn temp_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}"))
    }
}
