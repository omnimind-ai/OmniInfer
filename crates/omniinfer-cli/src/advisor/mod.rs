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
    let backend_filter = backend_filter
        .map(|name| omniinfer_core::backend::names::resolve_rows(&backends, name))
        .transpose()?;
    let candidates = backends
        .iter()
        .filter(|backend| {
            backend_filter.is_none_or(|wanted| json_str(backend, "id") == Some(wanted))
        })
        .map(|backend| {
            backend_fit_payload(
                backend,
                &model_info,
                &estimate,
                context,
                &system,
                backend_filter.is_some(),
            )
        })
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

mod evaluation;

use evaluation::*;
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

mod discovery;

use discovery::*;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_vla_fit_uses_action_contract_without_chat_arguments() {
        let model = temp_model("explicit-vla-fit", "gguf");
        let payload = fit_payload(
            model.to_str().unwrap(),
            None,
            Some(8192),
            Some("vla.cpp-linux-cuda"),
            json!({
                "data": [vla_backend(true)],
            }),
        )
        .unwrap();

        assert_eq!(payload["recommended"]["backend"], "vla.cpp-linux-cuda");
        assert_eq!(payload["recommended"]["compatible"], true);
        assert_eq!(payload["recommended"]["launch_args"], json!([]));
        let command = payload["next_command"].as_str().unwrap();
        assert!(!command.contains("--ctx-size"));
        assert!(!command.contains("-ngl"));

        std::fs::remove_file(model).ok();
    }

    #[test]
    fn action_models_are_isolated_from_chat_backends() {
        let model_info = json!({
            "format": "gguf",
            "artifact_kind": "file",
            "capabilities": ["action", "robotics", "vision"],
        });
        assert!(backend_model_compatible(&vla_backend(true), &model_info, false).0);
        let llama = json!({
            "id": "llama.cpp-linux",
            "family": "llama.cpp",
            "model_artifact": "file",
            "capabilities": ["chat", "vision"],
        });
        let (compatible, reasons) = backend_model_compatible(&llama, &model_info, false);
        assert!(!compatible);
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("action-capable"))
        );
    }

    #[test]
    fn unidentified_artifacts_require_explicit_vla_selection() {
        let model_info = json!({
            "format": "gguf",
            "artifact_kind": "file",
            "capabilities": ["chat"],
        });
        assert!(!backend_model_compatible(&vla_backend(true), &model_info, false).0);
        assert!(backend_model_compatible(&vla_backend(true), &model_info, true).0);
    }

    fn vla_backend(installed: bool) -> Value {
        json!({
            "id": "vla.cpp-linux-cuda",
            "label": "vla.cpp Linux CUDA",
            "family": "vla.cpp",
            "binary_exists": installed,
            "compatibility": "compatible",
            "model_artifact": "vla-artifact",
            "supports_ctx_size": false,
            "capabilities": ["vision", "action", "robotics", "gpu", "cuda"],
        })
    }

    fn temp_model(name: &str, extension: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}.{extension}"));
        std::fs::write(&path, b"test").unwrap();
        path
    }
}
