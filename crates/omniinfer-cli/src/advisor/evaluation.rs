use super::*;

pub(super) fn backend_fit_payload(
    backend: &Value,
    model_info: &Value,
    estimate: &Value,
    ctx_size: u32,
    system: &Value,
    explicitly_selected: bool,
) -> Value {
    let (compatible, reasons) = backend_model_compatible(backend, model_info, explicitly_selected);
    let hardware_ok = json_bool(backend, "hardware_compatible").unwrap_or(false);
    let installed = json_bool(backend, "installed").unwrap_or(false);
    let memory_kind = if is_gpu_backend(backend) {
        "gpu"
    } else {
        "ram"
    };
    let available_gib = available_memory_for_backend(backend, system);
    let required_gib = if memory_kind == "gpu" {
        estimate
            .get("estimated_gpu_memory_gib")
            .and_then(Value::as_f64)
    } else {
        estimate.get("estimated_ram_gib").and_then(Value::as_f64)
    };
    let margin = if memory_kind == "gpu" {
        GPU_MEMORY_MARGIN_GIB
    } else {
        CPU_MEMORY_MARGIN_GIB
    };
    let fit = fit_level(required_gib, available_gib, margin);
    let launch_args = advisor_launch_args(backend, model_info, ctx_size);
    let mut notes = reasons;
    if !installed {
        notes.push("backend runtime is not installed".to_string());
    }
    if !hardware_ok {
        notes.push("backend hardware probe did not pass".to_string());
    }
    let evidence = candidate_evidence(backend, model_info, estimate, compatible, hardware_ok);
    json!({
        "backend": json_str(backend, "id"),
        "label": json_str(backend, "label"),
        "family": json_str(backend, "family"),
        "capabilities": backend.get("capabilities").cloned().unwrap_or_else(|| json!([])),
        "supports_ctx_size": json_bool(backend, "supports_ctx_size").unwrap_or(false),
        "installed": installed,
        "hardware_compatible": hardware_ok,
        "compatible": compatible && hardware_ok,
        "fit": fit,
        "memory_required_gib": required_gib,
        "memory_available_gib": available_gib,
        "memory_margin_gib": margin,
        "memory_kind": memory_kind,
        "memory_breakdown": memory_breakdown_for_backend(estimate, memory_kind),
        "launch_args": launch_args,
        "priority": json_u64(backend, "priority").unwrap_or(999),
        "evidence": evidence,
        "recommendation_confidence": evidence.get("confidence").cloned().unwrap_or(Value::String("low".to_string())),
        "notes": notes,
        "why_not": why_not_candidate(backend, compatible, hardware_ok, &fit),
    })
}

pub(super) fn advisor_launch_args(
    backend: &Value,
    model_info: &Value,
    ctx_size: u32,
) -> Vec<String> {
    if is_action_backend(backend) {
        return Vec::new();
    }
    let mut launch_args = Vec::new();
    if json_bool(backend, "supports_ctx_size").unwrap_or(true) {
        launch_args.extend(["--ctx-size".to_string(), ctx_size.to_string()]);
    }
    if let Some(mmproj) = json_str(model_info, "mmproj") {
        launch_args.extend(["--mmproj".to_string(), mmproj.to_string()]);
    }
    launch_args
}

pub(super) fn backend_model_compatible(
    backend: &Value,
    model_info: &Value,
    explicitly_selected: bool,
) -> (bool, Vec<String>) {
    let format = json_str(model_info, "format").unwrap_or("");
    let artifact_kind = json_str(model_info, "artifact_kind").unwrap_or("");
    let family = json_str(backend, "family").unwrap_or("");
    let caps = capabilities(backend);
    let action_model = model_has_capability(model_info, "action");
    if is_action_backend(backend) {
        if !explicitly_selected && !action_model {
            return (
                false,
                vec![
                    "action backends require an identified VLA artifact or explicit selection"
                        .to_string(),
                ],
            );
        }
        return match format {
            "gguf" | "safetensors" if artifact_kind == "file" => (true, Vec::new()),
            _ => (
                false,
                vec!["vla.cpp requires a GGUF or safetensors checkpoint file".to_string()],
            ),
        };
    }
    if action_model {
        return (
            false,
            vec!["VLA action models require an action-capable backend".to_string()],
        );
    }
    match format {
        "gguf" => {
            if family == "llama.cpp" || family == "turboquant" {
                (true, Vec::new())
            } else {
                (
                    false,
                    vec!["GGUF models require a llama.cpp-compatible backend".to_string()],
                )
            }
        }
        "hf-reference" => {
            if family == "vllm" {
                (true, Vec::new())
            } else {
                (
                    false,
                    vec!["HF references require vLLM or an explicit local model path".to_string()],
                )
            }
        }
        "directory" => {
            if caps.contains(&"mnn".to_string()) || family == "mlx" || family == "vllm" {
                (true, Vec::new())
            } else {
                (
                    false,
                    vec!["directory models require a directory/path backend".to_string()],
                )
            }
        }
        _ if artifact_kind == "file" => (family == "llama.cpp", Vec::new()),
        _ => (
            false,
            vec!["unknown model format; only reference backends are considered safe".to_string()],
        ),
    }
}

pub(super) fn is_action_backend(backend: &Value) -> bool {
    capabilities(backend)
        .iter()
        .any(|capability| capability == "action")
}

pub(super) fn model_has_capability(model_info: &Value, wanted: &str) -> bool {
    model_info
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|capability| capability == wanted)
}

pub(super) fn is_gpu_backend(backend: &Value) -> bool {
    let caps = capabilities(backend);
    caps.iter().any(|cap| {
        matches!(
            cap.as_str(),
            "gpu" | "cuda" | "rocm" | "vulkan" | "metal" | "hip" | "sycl"
        )
    })
}

pub(super) fn capabilities(backend: &Value) -> Vec<String> {
    backend
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

pub(super) fn available_memory_for_backend(backend: &Value, system: &Value) -> Option<f64> {
    let caps = capabilities(backend);
    if caps.iter().any(|cap| cap == "cuda") {
        return system
            .get("cuda")
            .and_then(|cuda| cuda.get("visible_devices"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|device| device.get("free_gib").and_then(Value::as_f64))
            .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    }
    if is_gpu_backend(backend) && !caps.iter().any(|cap| cap == "shared-memory") {
        return None;
    }
    system
        .get("host")
        .and_then(|host| host.get("available_ram_gib"))
        .and_then(Value::as_f64)
}

pub(super) fn fit_level(
    required_gib: Option<f64>,
    available_gib: Option<f64>,
    margin_gib: f64,
) -> String {
    match (required_gib, available_gib) {
        (Some(required), Some(available)) if required + margin_gib <= available => {
            "good".to_string()
        }
        (Some(required), Some(available)) if required <= available => "marginal".to_string(),
        (Some(_), Some(_)) => "too_tight".to_string(),
        _ => "unknown".to_string(),
    }
}

pub(super) fn memory_breakdown_for_backend(estimate: &Value, memory_kind: &str) -> Value {
    let mut result = estimate
        .get("breakdown")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(map) = result.as_object_mut() {
        map.insert(
            "memory_kind".to_string(),
            Value::String(memory_kind.to_string()),
        );
        let total = if memory_kind == "gpu" {
            estimate.get("estimated_gpu_memory_gib")
        } else {
            estimate.get("estimated_ram_gib")
        }
        .cloned()
        .unwrap_or(Value::Null);
        map.insert("total_gib".to_string(), total);
    }
    result
}

pub(super) fn candidate_evidence(
    backend: &Value,
    model_info: &Value,
    estimate: &Value,
    compatible: bool,
    hardware_ok: bool,
) -> Value {
    let exists = json_bool(model_info, "exists").unwrap_or(false);
    let installed = json_bool(backend, "installed").unwrap_or(false);
    let format = json_str(model_info, "format").unwrap_or("");
    let family = json_str(backend, "family").unwrap_or("");
    let confidence = json_str(estimate, "confidence").unwrap_or("low");
    let level = if !compatible || !hardware_ok {
        "none"
    } else if exists && installed && hardware_ok && matches!(confidence, "medium" | "high") {
        "direct"
    } else if format == "hf-reference" && family == "vllm" {
        "self_reported"
    } else if matches!(format, "gguf" | "directory") && compatible {
        "variant"
    } else {
        "none"
    };
    let confidence_label = match level {
        "direct" => "high",
        "variant" | "self_reported" => "medium",
        _ => "low",
    };
    let mut sources = Vec::new();
    if exists {
        sources.push("local_model_file");
    }
    if installed {
        sources.push("installed_backend");
    }
    if hardware_ok {
        sources.push("hardware_probe");
    }
    if let Some(source) = json_str(estimate, "estimate_source") {
        sources.push(source);
    }
    sources.sort();
    sources.dedup();
    json!({
        "level": level,
        "confidence": confidence_label,
        "sources": sources,
        "notes": evidence_notes(level, backend, estimate),
    })
}

pub(super) fn evidence_notes(level: &str, backend: &Value, estimate: &Value) -> Vec<String> {
    match level {
        "direct" => vec![
            "local model artifact exists".to_string(),
            format!(
                "{} runtime is installed and hardware-compatible",
                json_str(backend, "id").unwrap_or("backend")
            ),
            format!(
                "memory estimate confidence is {}",
                json_str(estimate, "confidence").unwrap_or("unknown")
            ),
        ],
        "variant" => vec![
            "model format is compatible with backend family".to_string(),
            "estimate is inferred from local artifact metadata rather than a measured run"
                .to_string(),
        ],
        "self_reported" => vec![
            "model is a remote or external reference accepted by the backend".to_string(),
            "local artifact size is unavailable unless the backend downloads or resolves it"
                .to_string(),
        ],
        _ => vec!["insufficient compatible local evidence".to_string()],
    }
}

pub(super) fn why_not_candidate(
    backend: &Value,
    compatible: bool,
    hardware_ok: bool,
    fit: &str,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !compatible {
        reasons.push("model format is not compatible with this backend".to_string());
    }
    if !json_bool(backend, "installed").unwrap_or(false) {
        reasons.push("runtime is not installed".to_string());
    }
    if !hardware_ok {
        reasons.push("hardware probe did not pass".to_string());
    }
    if fit == "too_tight" {
        reasons.push("estimated memory requirement exceeds available memory".to_string());
    }
    if fit == "marginal" {
        reasons.push("estimated memory fits but has little headroom".to_string());
    }
    reasons
}

pub(super) fn recommended_candidate(candidates: &[Value]) -> Option<Value> {
    candidates.iter().cloned().min_by_key(|candidate| {
        let fit_rank = match json_str(candidate, "fit").unwrap_or("unknown") {
            "good" => 0,
            "marginal" => 1,
            "too_tight" => 2,
            _ => 3,
        };
        let installed_rank = if json_bool(candidate, "installed").unwrap_or(false) {
            0
        } else {
            1
        };
        let priority = json_u64(candidate, "priority").unwrap_or(999);
        let ik_rank = if json_str(candidate, "backend")
            .unwrap_or("")
            .starts_with("ik_llama.cpp")
        {
            1
        } else {
            0
        };
        (
            fit_rank,
            installed_rank,
            priority,
            ik_rank,
            json_str(candidate, "backend").unwrap_or("").to_string(),
        )
    })
}

pub(super) fn why_recommended(candidate: &Value, model_info: &Value) -> Vec<String> {
    let backend = json_str(candidate, "backend").unwrap_or("backend");
    let fit = json_str(candidate, "fit").unwrap_or("unknown");
    let mut reasons = vec![format!(
        "{backend} has the best ranked fit among compatible backends ({fit})"
    )];
    if json_bool(candidate, "installed").unwrap_or(false) {
        reasons.push("runtime is already installed".to_string());
    }
    if json_bool(candidate, "hardware_compatible").unwrap_or(false) {
        reasons.push("hardware probe passed".to_string());
    }
    if let Some(level) = candidate
        .get("evidence")
        .and_then(|value| json_str(value, "level"))
    {
        reasons.push(format!("recommendation evidence level is {level}"));
    }
    if let (Some(required), Some(available)) = (
        candidate.get("memory_required_gib").and_then(Value::as_f64),
        candidate
            .get("memory_available_gib")
            .and_then(Value::as_f64),
    ) {
        reasons.push(format!(
            "estimated memory {required} GiB fits available {available} GiB"
        ));
    }
    let capabilities = model_info
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if !capabilities.is_empty() {
        reasons.push(format!("model capabilities: {capabilities}"));
    }
    reasons
}

pub(super) fn with_why_not(candidate: &Value, recommended: Option<&Value>) -> Value {
    let mut result = candidate.clone();
    let needs_reason = result
        .get("why_not")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty);
    let reason = recommended.map(|recommended| rank_difference_reason(&result, recommended));
    if needs_reason && let (Some(map), Some(reason)) = (result.as_object_mut(), reason) {
        map.insert(
            "why_not".to_string(),
            Value::Array(vec![Value::String(reason)]),
        );
    }
    result
}

pub(super) fn rank_difference_reason(candidate: &Value, recommended: &Value) -> String {
    if json_str(candidate, "fit") != json_str(recommended, "fit") {
        return format!(
            "recommended backend has better fit ({} vs {})",
            json_str(recommended, "fit").unwrap_or("-"),
            json_str(candidate, "fit").unwrap_or("-")
        );
    }
    if json_bool(candidate, "installed") != json_bool(recommended, "installed") {
        return "recommended backend is already installed".to_string();
    }
    if json_u64(candidate, "priority").unwrap_or(999)
        > json_u64(recommended, "priority").unwrap_or(999)
    {
        return "recommended backend has higher product priority".to_string();
    }
    "ranked below the recommended backend by deterministic tie-breakers".to_string()
}

pub(super) fn next_load_command(
    model_info: &Value,
    recommended: &Value,
    ctx_size: u32,
) -> Option<String> {
    let backend = json_str(recommended, "backend")?;
    let model = shell_quote(json_str(model_info, "model")?);
    let mut parts = vec![
        "omniinfer".to_string(),
        "model".to_string(),
        "load".to_string(),
        "-m".to_string(),
        model,
    ];
    if !is_action_backend(recommended)
        && json_bool(recommended, "supports_ctx_size").unwrap_or(true)
    {
        parts.extend(["--ctx-size".to_string(), ctx_size.to_string()]);
    }
    if let Some(mmproj) = json_str(model_info, "mmproj") {
        parts.extend(["--mmproj".to_string(), shell_quote(mmproj)]);
    }
    Some(format!(
        "omniinfer backend select {} && {}",
        shell_quote(backend),
        parts.join(" ")
    ))
}
