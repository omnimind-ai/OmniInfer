use super::*;

#[derive(Debug, Clone)]
pub(super) struct LocalModel {
    pub(super) path: PathBuf,
    pub(super) label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct AdvisorModelSummary {
    pub(super) fit: Option<String>,
    pub(super) backend: Option<String>,
    pub(super) evidence: Option<String>,
    pub(super) confidence: Option<String>,
}

pub(super) fn discover_local_models(config: &config::AppConfig) -> Result<Vec<LocalModel>> {
    let _ = config;
    let payload = rust_backend_payload(BackendScope::All);
    let mut roots = Vec::new();
    for backend in payload
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(path) = json_str(backend, "models_dir").map(PathBuf::from)
            && path.is_dir()
            && is_under_local_dir(&path)
        {
            roots.push(path);
        }
    }
    roots.sort();
    roots.dedup();
    Ok(discover_models_in_roots(&roots))
}

pub(super) fn discover_models_in_roots(roots: &[PathBuf]) -> Vec<LocalModel> {
    let mut seen = std::collections::BTreeSet::new();
    let mut models = Vec::new();
    for root in roots {
        visit_model_root(root, root, &mut seen, &mut models);
    }
    models.sort_by(|left, right| left.label.to_lowercase().cmp(&right.label.to_lowercase()));
    models
}

fn visit_model_root(
    root: &Path,
    current: &Path,
    seen: &mut std::collections::BTreeSet<PathBuf>,
    models: &mut Vec<LocalModel>,
) {
    let Ok(entries) = std::fs::read_dir(current) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_model_root(root, &path, seen, models);
        } else if is_model_file(&path) && !is_mmproj_file(&path) {
            let resolved = path.canonicalize().unwrap_or_else(|_| path.clone());
            if seen.insert(resolved) {
                models.push(LocalModel {
                    label: model_label(root, &path),
                    path,
                });
            }
        }
    }
}

pub(super) fn prompt_model_path() -> Result<Option<PathBuf>> {
    loop {
        let text = prompt_default("Model path", "")?;
        if text.trim().is_empty() {
            return Ok(None);
        }
        let path = expand_path(text.trim());
        if !path.exists() {
            notice(
                &format!("Model path does not exist: {}", path.display()),
                NoticeKind::Warning,
            );
            continue;
        }
        if path.is_dir() {
            let candidates = detect_model_files_in_directory(&path);
            if candidates.is_empty() {
                notice(
                    &format!("No GGUF model files found under {}", path.display()),
                    NoticeKind::Warning,
                );
                continue;
            }
            if candidates.len() == 1 {
                return link_model_into_managed_models(&candidates[0].path, Some(&path));
            }
            let items = candidates
                .iter()
                .map(|model| MenuItem {
                    label: model.label.clone(),
                    details: Vec::new(),
                    selected: false,
                })
                .collect::<Vec<_>>();
            if let Some(index) = select_menu("Models", "Detected model files", &items, 0)? {
                return link_model_into_managed_models(&candidates[index].path, Some(&path));
            }
            return Ok(None);
        }
        return link_model_into_managed_models(
            &path,
            Some(path.parent().unwrap_or(Path::new("."))),
        );
    }
}

fn detect_model_files_in_directory(directory: &Path) -> Vec<LocalModel> {
    let mut models = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    visit_model_root(directory, directory, &mut seen, &mut models);
    models.sort_by(|left, right| model_file_rank(&left.path).cmp(&model_file_rank(&right.path)));
    models
}

fn link_model_into_managed_models(
    source: &Path,
    model_root: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if !source.is_file() {
        return Ok(Some(source.to_path_buf()));
    }
    let target_root = paths::local_dir().join("models");
    std::fs::create_dir_all(&target_root)?;
    let target = managed_model_target(source, &target_root, model_root);
    if link_points_to(&target, source) {
        return Ok(Some(target));
    }
    let mut target = target;
    if target.exists() || target.is_symlink() {
        for index in 2..1000 {
            let candidate = managed_model_target_with_suffix(
                source,
                &target_root,
                model_root,
                &format!("-{index}"),
            );
            if link_points_to(&candidate, source) {
                return Ok(Some(candidate));
            }
            if !candidate.exists() && !candidate.is_symlink() {
                target = candidate;
                break;
            }
        }
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    create_model_link(source, &target)?;
    notice(
        &format!("Linked model: {}", target.display()),
        NoticeKind::Success,
    );
    Ok(Some(target))
}

fn create_model_link(source: &Path, target: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(source, target)?;
        Ok(())
    }
    #[cfg(windows)]
    {
        match std::os::windows::fs::symlink_file(source, target) {
            Ok(_) => Ok(()),
            Err(_) => {
                std::fs::hard_link(source, target)?;
                Ok(())
            }
        }
    }
}

pub(super) fn advisor_recommendation_map(
    config: &config::AppConfig,
    models: &[LocalModel],
) -> BTreeMap<String, Value> {
    let _ = config;
    if models.is_empty() {
        return BTreeMap::new();
    }
    let backends = rust_backend_payload(BackendScope::All);
    let payload = advisor::recommend_payload(None, models.len().max(20) as u32, None, backends);
    let mut result = BTreeMap::new();
    for row in payload
        .get("recommendations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let model = row.get("model").unwrap_or(&Value::Null);
        for key in ["input", "model", "model_path"] {
            if let Some(value) = json_str(model, key) {
                result.insert(advisor_path_key(value), row.clone());
            }
        }
    }
    result
}

pub(super) fn advisor_model_summary(
    model_path: &Path,
    recommendations: &BTreeMap<String, Value>,
) -> Option<AdvisorModelSummary> {
    let row = recommendations.get(&advisor_path_key(&model_path.display().to_string()))?;
    let recommended = row.get("recommended").unwrap_or(&Value::Null);
    let evidence = recommended.get("evidence").unwrap_or(&Value::Null);
    Some(AdvisorModelSummary {
        fit: json_str(recommended, "fit").map(str::to_string),
        backend: json_str(recommended, "backend").map(str::to_string),
        evidence: json_str(evidence, "level").map(str::to_string),
        confidence: json_str(recommended, "recommendation_confidence")
            .or_else(|| json_str(evidence, "confidence"))
            .map(str::to_string),
    })
}

pub(super) fn model_quant_label(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    [
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q3_k_m", "q3_k_s", "q2_k",
        "bf16", "f16", "f32",
    ]
    .iter()
    .find(|quant| name.contains(**quant))
    .map(|quant| quant.to_ascii_uppercase())
    .unwrap_or_else(|| "-".to_string())
}

pub(super) fn model_size_label(path: &Path) -> String {
    let Ok(metadata) = std::fs::metadata(path) else {
        return "-".to_string();
    };
    format_bytes(metadata.len())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.2} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.1} MiB", value / MIB)
    } else if value >= KIB {
        format!("{:.1} KiB", value / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "gguf" | "ggml" | "bin"))
}

fn is_mmproj_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("mmproj")
}

fn is_under_local_dir(path: &Path) -> bool {
    let root = paths::local_dir()
        .canonicalize()
        .unwrap_or_else(|_| paths::local_dir());
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    path.starts_with(root)
}

fn model_label(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn model_file_rank(path: &Path) -> (usize, usize, String) {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let quant_order = [
        "q4_k_m", "q4_0", "q5_k_m", "q6_k", "q8_0", "f16", "q3_k_m", "q2_k",
    ];
    let quant_rank = quant_order
        .iter()
        .position(|quant| name.contains(quant))
        .unwrap_or(quant_order.len());
    (path.components().count(), quant_rank, name)
}

fn expand_path(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

pub(super) fn same_path(left: &Path, right: &Path) -> bool {
    left.canonicalize().unwrap_or_else(|_| left.to_path_buf())
        == right.canonicalize().unwrap_or_else(|_| right.to_path_buf())
}

pub(super) fn managed_model_target(
    source: &Path,
    target_root: &Path,
    model_root: Option<&Path>,
) -> PathBuf {
    managed_model_target_with_suffix(source, target_root, model_root, "")
}

fn managed_model_target_with_suffix(
    source: &Path,
    target_root: &Path,
    model_root: Option<&Path>,
    suffix: &str,
) -> PathBuf {
    let root_name = model_root
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .map(safe_path_component)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            source
                .parent()
                .and_then(Path::file_name)
                .and_then(|value| value.to_str())
                .map(safe_path_component)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "model".to_string())
        });
    target_root.join(format!("{root_name}{suffix}")).join(
        source
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf")),
    )
}

fn safe_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches([' ', '.', '-', '_'])
        .to_string()
}

fn link_points_to(path: &Path, source: &Path) -> bool {
    path.exists() && source.exists() && path.canonicalize().ok() == source.canonicalize().ok()
}

pub(super) fn advisor_path_key(value: &str) -> String {
    expand_path(value)
        .canonicalize()
        .unwrap_or_else(|_| expand_path(value))
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_models_and_skips_mmproj_files() {
        let root = std::env::temp_dir().join(unique_name("tui-discover"));
        let family = root.join("Qwen3.5-4B");
        std::fs::create_dir_all(&family).expect("create model dir");
        let model = family.join("Qwen3.5-4B-Q4_K_M.gguf");
        let mmproj = family.join("mmproj-Qwen3.5-4B.gguf");
        std::fs::write(&model, "").expect("write model");
        std::fs::write(mmproj, "").expect("write mmproj");

        let rows = discover_models_in_roots(std::slice::from_ref(&root));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].path, model);
        assert_eq!(rows[0].label, "Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn advisor_badges_include_fit_backend_evidence_and_confidence() {
        let model = PathBuf::from("/tmp/Qwen3.5-4B-Q4_K_M.gguf");
        let mut rows = BTreeMap::new();
        rows.insert(
            advisor_path_key(&model.display().to_string()),
            serde_json::json!({
                "recommended": {
                    "backend": "llama.cpp-linux-cuda",
                    "fit": "good",
                    "recommendation_confidence": "high",
                    "evidence": { "level": "direct" }
                }
            }),
        );
        assert_eq!(
            advisor_model_summary(&model, &rows),
            Some(AdvisorModelSummary {
                fit: Some("good".to_string()),
                backend: Some("llama.cpp-linux-cuda".to_string()),
                evidence: Some("direct".to_string()),
                confidence: Some("high".to_string()),
            })
        );
    }

    #[test]
    fn managed_target_uses_model_root_name() {
        let source = PathBuf::from("/models/qwen/Qwen3.5-4B-Q4_K_M.gguf");
        let target_root = PathBuf::from("/repo/.local/models");
        let model_root = PathBuf::from("/models/qwen");
        assert_eq!(
            managed_model_target(&source, &target_root, Some(&model_root)),
            PathBuf::from("/repo/.local/models/qwen/Qwen3.5-4B-Q4_K_M.gguf")
        );
    }

    #[test]
    fn model_labels_include_quant_and_size() {
        let root = std::env::temp_dir().join(unique_name("tui-model-labels"));
        std::fs::create_dir_all(&root).expect("create model dir");
        let model = root.join("Qwen3.5-4B-Q4_K_M.gguf");
        std::fs::write(&model, vec![0_u8; 2048]).expect("write model");

        assert_eq!(model_quant_label(&model), "Q4_K_M");
        assert_eq!(model_size_label(&model), "2.0 KiB");
        std::fs::remove_dir_all(root).ok();
    }

    fn unique_name(prefix: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        format!("omniinfer-rs-{prefix}-{nanos}")
    }
}
