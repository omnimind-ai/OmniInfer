use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use omniinfer_core::{backend_registry::backend_priority, model_load::DEFAULT_LOAD_CONTEXT_SIZE};
use serde_json::{Value, json};

use crate::{current_system_name, json_bool, json_str, json_u64, prebuilt_catalog};

mod model;
mod output;
mod system;

pub use model::inspect_payload;
pub use output::{print_fit, print_inspect, print_plan, print_recommend, print_system};
pub use system::system_payload;

use model::memory_estimate;

const DEFAULT_CONTEXT_SIZE: u32 = DEFAULT_LOAD_CONTEXT_SIZE;
const GPU_MEMORY_MARGIN_GIB: f64 = 0.5;
const CPU_MEMORY_MARGIN_GIB: f64 = 1.0;
pub fn fit_payload(
    model: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
    backend_filter: Option<&str>,
    backends_payload: Value,
) -> Result<Value> {
    let context = ctx_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
    let mut model_info = inspect_payload(model, mmproj, Some(context))?;
    let estimate = memory_estimate(
        model_info.get("size_gib").and_then(Value::as_f64),
        model_info.get("mmproj_size_gib").and_then(Value::as_f64),
        model_info.get("params_b").and_then(Value::as_f64),
        context,
    );
    model_info["estimate"] = estimate.clone();
    let system = system_payload(backends_payload);
    let backends = system
        .get("backends")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(backend) = backend_filter
        && !backends
            .iter()
            .any(|item| json_str(item, "id") == Some(backend))
    {
        anyhow::bail!("Unsupported backend: {backend}");
    }
    let candidates = backends
        .iter()
        .filter(|backend| {
            backend_filter.is_none_or(|wanted| json_str(backend, "id") == Some(wanted))
        })
        .map(|backend| backend_fit_payload(backend, &model_info, &estimate, context, &system))
        .collect::<Vec<_>>();
    let compatible = candidates
        .iter()
        .filter(|candidate| json_bool(candidate, "compatible").unwrap_or(false))
        .cloned()
        .collect::<Vec<_>>();
    let mut recommended = recommended_candidate(&compatible);
    if let Some(recommended) = recommended.as_mut() {
        let why = why_recommended(recommended, &model_info)
            .into_iter()
            .map(Value::String)
            .collect();
        if let Some(map) = recommended.as_object_mut() {
            map.insert("why_recommended".to_string(), Value::Array(why));
        }
    }
    let warnings = model_info
        .get("warnings")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(json!({
        "object": "advisor.fit",
        "model": model_info,
        "context_size": context,
        "recommended": recommended,
        "alternatives": compatible.iter()
            .filter(|candidate| recommended.as_ref().and_then(|r| json_str(r, "backend")) != json_str(candidate, "backend"))
            .map(|candidate| with_why_not(candidate, recommended.as_ref()))
            .collect::<Vec<_>>(),
        "all_backends": candidates,
        "next_command": recommended.as_ref().and_then(|candidate| next_load_command(&model_info, candidate, context)),
        "warnings": warnings,
    }))
}

pub fn plan_payload(
    model: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
    gpu_vram_gib: Option<f64>,
    ram_gib: Option<f64>,
    cpu_cores: Option<u32>,
    backends_payload: Value,
) -> Result<Value> {
    let context = ctx_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
    let mut model_info = inspect_payload(model, mmproj, Some(context))?;
    let estimate = memory_estimate(
        model_info.get("size_gib").and_then(Value::as_f64),
        model_info.get("mmproj_size_gib").and_then(Value::as_f64),
        model_info.get("params_b").and_then(Value::as_f64),
        context,
    );
    model_info["estimate"] = estimate.clone();
    let system = system_payload(backends_payload);
    let current = current_hardware(&system);
    let planning = apply_hardware_overrides(current.clone(), gpu_vram_gib, ram_gib, cpu_cores);
    let paths = vec![
        plan_run_path("gpu", &estimate, &planning),
        plan_run_path("cpu_offload", &estimate, &planning),
        plan_run_path("cpu_only", &estimate, &planning),
    ];
    let recommended_path = recommended_plan_path(&paths);
    Ok(json!({
        "object": "advisor.plan",
        "model": model_info,
        "context_size": context,
        "current_hardware": current,
        "planning_hardware": planning,
        "run_paths": paths,
        "recommended_path": recommended_path,
        "upgrade_deltas": upgrade_deltas(&paths, &planning),
        "estimate_notice": "Hardware planning uses local advisor heuristics; backend logs and benchmark runs remain authoritative.",
        "next_commands": plan_next_commands(&model_info, context, recommended_path.as_ref()),
        "warnings": model_info.get("warnings").and_then(Value::as_array).cloned().unwrap_or_default(),
    }))
}

pub fn recommend_payload(
    task: Option<&str>,
    limit: u32,
    ctx_size: Option<u32>,
    backends_payload: Value,
) -> Value {
    let context = ctx_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
    let system = system_payload(backends_payload.clone());
    let model_dirs = system
        .get("backends")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|backend| json_str(backend, "models_dir"))
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut recommendations = Vec::new();
    for dir in &model_dirs {
        for model_path in iter_local_models(dir) {
            if !seen.insert(model_path.clone()) {
                continue;
            }
            let model_text = model_path.display().to_string();
            let Ok(fit) = fit_payload(
                &model_text,
                None,
                Some(context),
                None,
                backends_payload.clone(),
            ) else {
                continue;
            };
            let Some(recommended) = fit.get("recommended").filter(|value| value.is_object()) else {
                continue;
            };
            let model_info = fit.get("model").cloned().unwrap_or_else(|| json!({}));
            if !task_matches_model(task.unwrap_or("any"), &model_info) {
                continue;
            }
            let score = recommendation_score(recommended, &model_info);
            recommendations.push(json!({
                "score": score,
                "model": model_info,
                "recommended": recommended,
                "evidence": recommended.get("evidence").cloned().unwrap_or_else(|| json!({})),
                "recommendation_confidence": recommended.get("recommendation_confidence").cloned().unwrap_or(Value::Null),
                "why_recommended": recommended.get("why_recommended").cloned().unwrap_or_else(|| json!([])),
                "next_command": fit.get("next_command").cloned().unwrap_or(Value::Null),
                "warnings": fit.get("warnings").cloned().unwrap_or_else(|| json!([])),
            }));
        }
    }
    recommendations.sort_by(|left, right| {
        let left_score = left
            .get("score")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        let right_score = right
            .get("score")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                json_str(left.get("model").unwrap_or(&Value::Null), "model")
                    .unwrap_or("")
                    .cmp(
                        json_str(right.get("model").unwrap_or(&Value::Null), "model").unwrap_or(""),
                    )
            })
    });
    let returned = usize::min(limit as usize, recommendations.len());
    json!({
        "object": "advisor.recommend",
        "task": task.unwrap_or("any"),
        "context_size": context,
        "models_scanned": seen.len(),
        "returned": returned,
        "recommendations": recommendations.into_iter().take(returned).collect::<Vec<_>>(),
    })
}

fn backend_fit_payload(
    backend: &Value,
    model_info: &Value,
    estimate: &Value,
    ctx_size: u32,
    system: &Value,
) -> Value {
    let (compatible, reasons) = backend_model_compatible(backend, model_info);
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
    let mut launch_args = vec!["--ctx-size".to_string(), ctx_size.to_string()];
    if is_gpu_backend(backend) && json_str(backend, "id").is_some_and(|id| id.contains("cuda")) {
        launch_args.extend(["-ngl".to_string(), "999".to_string()]);
    }
    if let Some(mmproj) = json_str(model_info, "mmproj") {
        launch_args.extend(["--mmproj".to_string(), mmproj.to_string()]);
    }
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

fn backend_model_compatible(backend: &Value, model_info: &Value) -> (bool, Vec<String>) {
    let format = json_str(model_info, "format").unwrap_or("");
    let artifact_kind = json_str(model_info, "artifact_kind").unwrap_or("");
    let family = json_str(backend, "family").unwrap_or("");
    let caps = capabilities(backend);
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

fn is_gpu_backend(backend: &Value) -> bool {
    let caps = capabilities(backend);
    caps.iter().any(|cap| {
        matches!(
            cap.as_str(),
            "gpu" | "cuda" | "rocm" | "vulkan" | "metal" | "hip" | "sycl"
        )
    })
}

fn capabilities(backend: &Value) -> Vec<String> {
    backend
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn available_memory_for_backend(backend: &Value, system: &Value) -> Option<f64> {
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

fn fit_level(required_gib: Option<f64>, available_gib: Option<f64>, margin_gib: f64) -> String {
    match (required_gib, available_gib) {
        (Some(required), Some(available)) if required + margin_gib <= available => {
            "good".to_string()
        }
        (Some(required), Some(available)) if required <= available => "marginal".to_string(),
        (Some(_), Some(_)) => "too_tight".to_string(),
        _ => "unknown".to_string(),
    }
}

fn memory_breakdown_for_backend(estimate: &Value, memory_kind: &str) -> Value {
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

fn candidate_evidence(
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

fn evidence_notes(level: &str, backend: &Value, estimate: &Value) -> Vec<String> {
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

fn why_not_candidate(
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

fn recommended_candidate(candidates: &[Value]) -> Option<Value> {
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

fn why_recommended(candidate: &Value, model_info: &Value) -> Vec<String> {
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

fn with_why_not(candidate: &Value, recommended: Option<&Value>) -> Value {
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

fn rank_difference_reason(candidate: &Value, recommended: &Value) -> String {
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

fn next_load_command(model_info: &Value, recommended: &Value, ctx_size: u32) -> Option<String> {
    let backend = json_str(recommended, "backend")?;
    let model = shell_quote(json_str(model_info, "model")?);
    let mut parts = vec![
        "omniinfer".to_string(),
        "model".to_string(),
        "load".to_string(),
        "-m".to_string(),
        model,
        "--ctx-size".to_string(),
        ctx_size.to_string(),
    ];
    if let Some(mmproj) = json_str(model_info, "mmproj") {
        parts.extend(["--mmproj".to_string(), shell_quote(mmproj)]);
    }
    Some(format!(
        "omniinfer backend select {} && {}",
        shell_quote(backend),
        parts.join(" ")
    ))
}

fn current_hardware(system: &Value) -> Value {
    let host = system.get("host").unwrap_or(&Value::Null);
    let cuda = system.get("cuda").unwrap_or(&Value::Null);
    let devices = cuda
        .get("visible_devices")
        .and_then(Value::as_array)
        .or_else(|| cuda.get("devices").and_then(Value::as_array));
    let best = cuda.get("best_free_device").unwrap_or(&Value::Null);
    json!({
        "available_ram_gib": host.get("available_ram_gib").cloned().unwrap_or(Value::Null),
        "total_ram_gib": host.get("total_ram_gib").cloned().unwrap_or(Value::Null),
        "cpu_cores": host.get("cpu_cores").cloned().unwrap_or(Value::Null),
        "gpu_vram_free_gib": best.get("free_gib").cloned().unwrap_or(Value::Null),
        "gpu_vram_total_gib": best.get("total_gib").cloned().unwrap_or(Value::Null),
        "gpu_name": best.get("name").cloned().unwrap_or(Value::Null),
        "gpu_count": devices.map(Vec::len).unwrap_or(0),
    })
}

fn apply_hardware_overrides(
    mut current: Value,
    gpu_vram_gib: Option<f64>,
    ram_gib: Option<f64>,
    cpu_cores: Option<u32>,
) -> Value {
    let map = current
        .as_object_mut()
        .expect("current hardware is an object");
    if let Some(vram) = gpu_vram_gib {
        map.insert("gpu_vram_free_gib".to_string(), json!(vram));
        map.insert("gpu_vram_total_gib".to_string(), json!(vram));
        map.insert("simulated_gpu_vram_gib".to_string(), json!(vram));
    }
    if let Some(ram) = ram_gib {
        map.insert("available_ram_gib".to_string(), json!(ram));
        map.insert("total_ram_gib".to_string(), json!(ram));
        map.insert("simulated_ram_gib".to_string(), json!(ram));
    }
    if let Some(cpu_cores) = cpu_cores {
        map.insert("cpu_cores".to_string(), json!(cpu_cores));
        map.insert("simulated_cpu_cores".to_string(), json!(cpu_cores));
    }
    current
}

fn plan_run_path(path: &str, estimate: &Value, hardware: &Value) -> Value {
    let required = estimate
        .get("estimated_gpu_memory_gib")
        .or_else(|| estimate.get("estimated_ram_gib"))
        .and_then(Value::as_f64);
    let cpu_cores = hardware
        .get("cpu_cores")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1) as u64
        });
    let required_value = required.unwrap_or(0.0);
    let (available, minimum, recommended, speed, notes, margin) = match path {
        "gpu" => (
            hardware.get("gpu_vram_free_gib").and_then(Value::as_f64),
            json!({"vram_gib": round_gib(required_value), "ram_gib": round_gib(f64::max(4.0, required_value * 0.25)), "cpu_cores": u64::max(2, u64::min(cpu_cores, 4))}),
            json!({"vram_gib": round_gib(required_value * 1.2 + GPU_MEMORY_MARGIN_GIB), "ram_gib": round_gib(f64::max(8.0, required_value * 0.35)), "cpu_cores": u64::max(4, u64::min(cpu_cores, 8))}),
            "fast",
            vec!["fastest path when the selected backend can fully or mostly use GPU memory"],
            GPU_MEMORY_MARGIN_GIB,
        ),
        "cpu_offload" => (
            hardware.get("available_ram_gib").and_then(Value::as_f64),
            json!({"vram_gib": 2.0, "ram_gib": round_gib(required_value), "cpu_cores": u64::max(4, u64::min(cpu_cores, 8))}),
            json!({"vram_gib": 4.0, "ram_gib": round_gib(required_value * 1.25 + CPU_MEMORY_MARGIN_GIB), "cpu_cores": u64::max(8, u64::min(cpu_cores, 16))}),
            "medium",
            vec![
                "uses system RAM as the primary pool and GPU for partial acceleration when backend supports it",
            ],
            CPU_MEMORY_MARGIN_GIB,
        ),
        _ => (
            hardware.get("available_ram_gib").and_then(Value::as_f64),
            json!({"vram_gib": null, "ram_gib": round_gib(required_value), "cpu_cores": u64::max(4, u64::min(cpu_cores, 8))}),
            json!({"vram_gib": null, "ram_gib": round_gib(required_value * 1.35 + CPU_MEMORY_MARGIN_GIB), "cpu_cores": u64::max(8, u64::min(cpu_cores, 32))}),
            if cpu_cores < 16 {
                "slow"
            } else {
                "medium-slow"
            },
            vec!["lowest GPU requirement, usually slowest for chat generation"],
            CPU_MEMORY_MARGIN_GIB,
        ),
    };
    let fit = fit_level(required, available, margin);
    let feasible = matches!(fit.as_str(), "good" | "marginal") && available.is_some();
    json!({
        "path": path,
        "feasible_now": feasible,
        "fit": fit,
        "memory_required_gib": required.map(round_gib),
        "memory_available_gib": available,
        "minimum": minimum,
        "recommended": recommended,
        "estimated_relative_speed": speed,
        "notes": notes,
    })
}

fn recommended_plan_path(paths: &[Value]) -> Option<Value> {
    paths.iter().cloned().min_by_key(|path| {
        let feasible_rank = if json_bool(path, "feasible_now").unwrap_or(false) {
            0
        } else {
            1
        };
        let fit_rank = match json_str(path, "fit").unwrap_or("unknown") {
            "good" => 0,
            "marginal" => 1,
            "too_tight" => 2,
            _ => 3,
        };
        let path_rank = match json_str(path, "path").unwrap_or("") {
            "gpu" => 0,
            "cpu_offload" => 1,
            "cpu_only" => 2,
            _ => 9,
        };
        (feasible_rank, fit_rank, path_rank)
    })
}

fn upgrade_deltas(paths: &[Value], hardware: &Value) -> Vec<Value> {
    let mut result = Vec::new();
    for path in paths {
        let recommended = path.get("recommended").unwrap_or(&Value::Null);
        if json_str(path, "path") == Some("gpu") {
            let current = hardware
                .get("gpu_vram_free_gib")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if let Some(target) = recommended.get("vram_gib").and_then(Value::as_f64)
                && current < target
            {
                result.push(json!({
                    "path": "gpu",
                    "resource": "vram",
                    "add_gib": round_gib(target - current),
                    "target_gib": target,
                    "description": format!("add about {} GiB free VRAM for the recommended GPU path", round_gib(target - current)),
                }));
            }
        }
        let current_ram = hardware
            .get("available_ram_gib")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if let Some(target_ram) = recommended.get("ram_gib").and_then(Value::as_f64)
            && current_ram < target_ram
        {
            result.push(json!({
                "path": json_str(path, "path"),
                "resource": "ram",
                "add_gib": round_gib(target_ram - current_ram),
                "target_gib": target_ram,
                "description": format!("add about {} GiB available RAM for {}", round_gib(target_ram - current_ram), json_str(path, "path").unwrap_or("-")),
            }));
        }
    }
    result
}

fn plan_next_commands(
    model_info: &Value,
    ctx_size: u32,
    recommended: Option<&Value>,
) -> Vec<String> {
    let Some(recommended) = recommended else {
        return Vec::new();
    };
    let Some(model) = json_str(model_info, "model").map(shell_quote) else {
        return Vec::new();
    };
    match json_str(recommended, "path").unwrap_or("") {
        "cpu_only" => vec![format!(
            "omniinfer backend select llama.cpp-linux && omniinfer load -m {model} --ctx-size {ctx_size}"
        )],
        _ => vec![
            format!("omniinfer advisor fit {model} --ctx-size {ctx_size}"),
            format!("omniinfer load -m {model} --ctx-size {ctx_size}"),
        ],
    }
}

fn iter_local_models(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    visit_models(root, &mut result);
    result.sort();
    result
}

fn visit_models(root: &Path, result: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_models(&path, result);
        } else if path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
            && !path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .contains("mmproj")
        {
            result.push(path);
        }
    }
}

fn task_matches_model(task: &str, model_info: &Value) -> bool {
    let normalized = task.trim().to_ascii_lowercase();
    let model_name = json_str(model_info, "model")
        .unwrap_or("")
        .to_ascii_lowercase();
    let caps = model_info
        .get("capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    match normalized.as_str() {
        "any" | "chat" | "general" => true,
        "vision" | "multimodal" => caps.contains(&"vision"),
        "embedding" | "embeddings" => caps.contains(&"embedding"),
        "coding" => ["coder", "code", "deepseek", "qwen"]
            .iter()
            .any(|token| model_name.contains(token)),
        _ => true,
    }
}

fn recommendation_score(candidate: &Value, model_info: &Value) -> f64 {
    let fit_score = match json_str(candidate, "fit").unwrap_or("unknown") {
        "good" => 100.0,
        "marginal" => 65.0,
        "too_tight" => 10.0,
        _ => 20.0,
    };
    let installed_bonus = if json_bool(candidate, "installed").unwrap_or(false) {
        10.0
    } else {
        0.0
    };
    let priority_penalty = json_u64(candidate, "priority").unwrap_or(0) as f64 * 2.0;
    let size_bonus = model_info
        .get("size_gib")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .min(30.0)
        / 3.0;
    round_gib(fit_score + installed_bonus + size_bonus - priority_penalty)
}

fn shell_quote(value: &str) -> String {
    if value.chars().all(|ch| {
        ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '/' | ':' | '=' | '+' | '-')
    }) {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }
}

fn round_gib(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn format_gib(value: f64) -> String {
    let rounded = round_gib(value);
    if (rounded.fract()).abs() < f64::EPSILON {
        format!("{rounded:.1}")
    } else {
        format!("{rounded:.2}")
    }
}
