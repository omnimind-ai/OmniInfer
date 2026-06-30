use super::*;

const QUANT_PATTERNS: &[&str] = &[
    "UD-Q8_K_XL",
    "UD-Q8_K_L",
    "UD-Q8_K_M",
    "UD-Q8_K_S",
    "UD-Q6_K_XL",
    "UD-Q6_K_L",
    "UD-Q6_K_M",
    "UD-Q6_K_S",
    "UD-Q5_K_XL",
    "UD-Q5_K_L",
    "UD-Q5_K_M",
    "UD-Q5_K_S",
    "UD-Q4_K_XL",
    "UD-Q4_K_L",
    "UD-Q4_K_M",
    "UD-Q4_K_S",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "Q4_0",
    "Q3_K_M",
    "Q2_K",
    "BF16",
    "F16",
    "F32",
];

pub fn inspect_payload(model: &str, mmproj: Option<&str>, ctx_size: Option<u32>) -> Result<Value> {
    let (resolved_model, model_path, resolved_mmproj, artifact_kind, warnings) =
        resolve_model_artifact(model, mmproj)?;
    let size_gib = model_path.as_deref().and_then(path_size_gib);
    let mmproj_size_gib = resolved_mmproj.as_deref().and_then(path_size_gib);
    let quantization = infer_quantization(&resolved_model);
    let params_b = infer_params_b(&resolved_model);
    let capabilities = infer_model_capabilities(&resolved_model, resolved_mmproj.as_deref());
    let format = model_format(&artifact_kind, model_path.as_deref(), &resolved_model);
    Ok(json!({
        "object": "advisor.model",
        "input": model,
        "model": resolved_model,
        "model_path": model_path.as_ref().map(|path| path.display().to_string()),
        "mmproj": resolved_mmproj.as_ref().map(|path| path.display().to_string()),
        "format": format,
        "artifact_kind": artifact_kind,
        "exists": model_path.as_deref().map(Path::exists).unwrap_or(false),
        "size_gib": size_gib,
        "mmproj_size_gib": mmproj_size_gib,
        "quantization": quantization,
        "params_b": params_b,
        "capabilities": capabilities,
        "estimate": memory_estimate(size_gib, mmproj_size_gib, params_b, ctx_size.unwrap_or(DEFAULT_CONTEXT_SIZE)),
        "warnings": warnings,
    }))
}

fn resolve_model_artifact(
    model: &str,
    mmproj: Option<&str>,
) -> Result<(String, Option<PathBuf>, Option<PathBuf>, String, Vec<Value>)> {
    let text = model.trim();
    if text.is_empty() {
        anyhow::bail!("model reference must not be empty");
    }
    let path = expand_path(text);
    let resolved_mmproj = mmproj
        .filter(|value| !value.trim().is_empty())
        .map(expand_path)
        .map(|path| path.canonicalize().unwrap_or(path));
    let mut warnings = Vec::new();
    if path.exists() {
        let path = path.canonicalize().unwrap_or(path);
        if path.is_dir() {
            return Ok((
                path.display().to_string(),
                Some(path),
                resolved_mmproj,
                "directory".to_string(),
                warnings,
            ));
        }
        return Ok((
            path.display().to_string(),
            Some(path),
            resolved_mmproj,
            "file".to_string(),
            warnings,
        ));
    }
    if (text.contains('/') || text.contains(':')) && !text.to_ascii_lowercase().ends_with(".gguf") {
        return Ok((
            text.to_string(),
            None,
            resolved_mmproj,
            "reference".to_string(),
            warnings,
        ));
    }
    warnings.push(Value::String(format!(
        "model path does not exist locally: {}",
        path.display()
    )));
    Ok((
        path.display().to_string(),
        Some(path),
        resolved_mmproj,
        "missing".to_string(),
        warnings,
    ))
}

fn expand_path(value: &str) -> PathBuf {
    let text = value.trim();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    let path = PathBuf::from(text);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn model_format(artifact_kind: &str, model_path: Option<&Path>, model_ref: &str) -> String {
    if artifact_kind == "reference" {
        return "hf-reference".to_string();
    }
    if model_path.is_some_and(Path::is_dir) {
        return "directory".to_string();
    }
    if model_path
        .and_then(Path::extension)
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
        || model_ref.to_ascii_lowercase().ends_with(".gguf")
    {
        return "gguf".to_string();
    }
    if artifact_kind == "directory" {
        return "directory".to_string();
    }
    "unknown".to_string()
}

fn path_size_gib(path: &Path) -> Option<f64> {
    path.is_file()
        .then(|| path.metadata().ok())
        .flatten()
        .map(|metadata| round_gib(metadata.len() as f64 / 1024.0 / 1024.0 / 1024.0))
}

fn infer_quantization(text: &str) -> Option<String> {
    let upper = Path::new(text)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(text)
        .to_ascii_uppercase();
    QUANT_PATTERNS
        .iter()
        .find(|quant| upper.contains(**quant))
        .map(|value| (*value).to_string())
}

fn infer_params_b(text: &str) -> Option<f64> {
    let name = Path::new(text)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(text)
        .as_bytes();
    let mut best = None;
    let mut index = 0;
    while index < name.len() {
        if !name[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < name.len() && (name[index].is_ascii_digit() || name[index] == b'.') {
            index += 1;
        }
        if index >= name.len() || !matches!(name[index], b'B' | b'b' | b'M' | b'm') {
            continue;
        }
        let Ok(mut value) = std::str::from_utf8(&name[start..index])
            .unwrap_or("")
            .parse::<f64>()
        else {
            continue;
        };
        if matches!(name[index], b'M' | b'm') {
            value /= 1000.0;
        }
        best = Some(best.map_or(value, |current: f64| current.max(value)));
        index += 1;
    }
    best
}

fn infer_model_capabilities(text: &str, mmproj_path: Option<&Path>) -> Vec<String> {
    let lower = Path::new(text)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(text)
        .to_ascii_lowercase();
    let mut capabilities = vec!["chat".to_string()];
    if mmproj_path.is_some()
        || ["vl", "vision", "mmproj", "multimodal"]
            .iter()
            .any(|token| lower.contains(token))
    {
        capabilities.push("vision".to_string());
    }
    if ["embed", "embedding", "bge", "nomic"]
        .iter()
        .any(|token| lower.contains(token))
    {
        capabilities.push("embedding".to_string());
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

pub(super) fn memory_estimate(
    size_gib: Option<f64>,
    mmproj_size_gib: Option<f64>,
    params_b: Option<f64>,
    ctx_size: u32,
) -> Value {
    let Some(size_gib) = size_gib else {
        return json!({
            "estimated_gpu_memory_gib": null,
            "estimated_ram_gib": null,
            "estimated_kv_cache_gib": null,
            "breakdown": unknown_memory_breakdown(ctx_size),
            "estimate_source": "unknown",
            "confidence": "low",
            "notes": ["local model size is unknown; fit cannot be estimated safely"],
        });
    };
    let base = size_gib + mmproj_size_gib.unwrap_or(0.0);
    let ctx_factor = f64::from(ctx_size.max(1)) / f64::from(DEFAULT_CONTEXT_SIZE);
    let param_factor = params_b.unwrap_or_else(|| f64::max(base * 2.0, 1.0));
    let weights = round_gib(size_gib);
    let mmproj = round_gib(mmproj_size_gib.unwrap_or(0.0));
    let kv_cache = round_gib(f64::max(0.25, param_factor * 0.03 * ctx_factor));
    let activation = round_gib(f64::max(0.12, param_factor * 0.01 * ctx_factor.min(4.0)));
    let framework_overhead = round_gib(f64::max(0.35, base * 0.08));
    let allocator_slack = round_gib(f64::max(0.15, base * 0.04));
    let runtime_overhead = round_gib(framework_overhead + allocator_slack);
    let required = round_gib(weights + mmproj + kv_cache + activation + runtime_overhead);
    let confidence = if params_b.is_some() { "medium" } else { "low" };
    let breakdown = json!({
        "weights_gib": weights,
        "mmproj_gib": mmproj,
        "kv_cache_gib": kv_cache,
        "activation_gib": activation,
        "framework_overhead_gib": framework_overhead,
        "allocator_slack_gib": allocator_slack,
        "runtime_overhead_gib": runtime_overhead,
        "total_gib": required,
        "context_size": ctx_size,
        "assumptions": [
            "weights are approximated from local artifact file size",
            "KV cache is estimated from inferred parameter count and requested context",
            "activation and framework overhead include conservative runtime buffers and allocator slack"
        ],
    });
    json!({
        "estimated_gpu_memory_gib": required,
        "estimated_ram_gib": required,
        "estimated_kv_cache_gib": kv_cache,
        "weight_and_projector_gib": round_gib(weights + mmproj),
        "activation_gib": activation,
        "framework_overhead_gib": framework_overhead,
        "allocator_slack_gib": allocator_slack,
        "overhead_gib": runtime_overhead,
        "breakdown": breakdown,
        "context_size": ctx_size,
        "estimate_source": "file_size_heuristic",
        "confidence": confidence,
        "notes": ["Estimate uses local file size plus KV cache, activation, framework overhead, and allocator slack; backend logs or benchmark results are authoritative."],
    })
}

fn unknown_memory_breakdown(ctx_size: u32) -> Value {
    json!({
        "weights_gib": null,
        "mmproj_gib": null,
        "kv_cache_gib": null,
        "activation_gib": null,
        "framework_overhead_gib": null,
        "allocator_slack_gib": null,
        "runtime_overhead_gib": null,
        "total_gib": null,
        "context_size": ctx_size,
        "assumptions": ["local model size is unknown"],
    })
}
