use std::time::Duration;

use anyhow::Result;
use omniinfer_core::{backend_profiles, config, local_state, model_catalog, model_load, paths};

use crate::{
    BackendScope, ModelLoadArgs, current_system_name, json_bool, json_str, json_u64,
    post_local_model_load_for_config_with_autostart, rust_backend_payload,
};

pub(crate) fn print_model_list(all: bool, best: bool) -> Result<()> {
    let best = best || !all;
    let system = current_system_name();
    let payload = if best {
        model_catalog::list_supported_models_best(system)?
    } else {
        model_catalog::list_supported_models(system)?
    };
    println!("Supported models ({system})");
    if best {
        print_best_model_catalog(&payload);
    } else {
        print_full_model_catalog(&payload);
    }
    Ok(())
}

pub(crate) fn load_model(args: &ModelLoadArgs) -> Result<()> {
    let request = model_load::ModelLoadRequest {
        model: args.model.clone(),
        mmproj: args.mmproj.clone(),
        ctx_size: args.ctx_size,
        backend_port: None,
        config: args.config.clone(),
        backend_extra_args: args.backend_extra_args.clone(),
    };
    let (response, plan) = load_model_with_request(&request, args.verbose)?;
    if plan.auto_selected {
        println!("Auto-selected backend: {}", plan.backend);
    }
    print_model_loaded(&response, &plan)?;
    Ok(())
}

pub(crate) fn load_model_with_request(
    request: &model_load::ModelLoadRequest,
    verbose: bool,
) -> Result<(serde_json::Value, model_load::ModelLoadPlan)> {
    let config = config::load_app_config().unwrap_or_default();
    load_model_with_request_for_config(request, verbose, &config)
}

pub(crate) fn load_model_with_request_for_config(
    request: &model_load::ModelLoadRequest,
    verbose: bool,
    config: &config::AppConfig,
) -> Result<(serde_json::Value, model_load::ModelLoadPlan)> {
    load_model_with_request_for_config_and_autostart(request, verbose, config, true)
}

pub(crate) fn load_model_with_request_for_config_and_autostart(
    request: &model_load::ModelLoadRequest,
    verbose: bool,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<(serde_json::Value, model_load::ModelLoadPlan)> {
    let backends = rust_backend_payload(BackendScope::All);
    let rows = backends
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Unable to read backend list."))?;
    let state = local_state::load_state().unwrap_or_default();
    let profile = match &request.config {
        Some(path) => Some(backend_profiles::load_backend_profile(
            std::path::PathBuf::from(path),
        )?),
        None => state
            .selected_backend
            .as_deref()
            .map(paths::backend_profile_file)
            .filter(|path| path.is_file())
            .map(backend_profiles::load_backend_profile)
            .transpose()?,
    };
    let plan = model_load::build_model_load_payload(
        request,
        rows,
        json_str(&backends, "recommended"),
        state.selected_backend.as_deref(),
        profile.as_ref(),
        &std::env::current_dir()?,
    )?;
    println!("Loading model...");
    let response = post_local_model_load_for_config_with_autostart(
        &plan.payload,
        verbose,
        Duration::from_secs(600),
        config,
        autostart,
    )?;
    let selected_backend = json_str(&response, "selected_backend").unwrap_or(&plan.backend);
    local_state::save_selected_backend(selected_backend)?;
    let selected_model = json_str(&response, "selected_model")
        .or_else(|| json_str(&plan.payload, "model"))
        .ok_or_else(|| anyhow::anyhow!("Model load response did not include a selected model."))?;
    let selected_mmproj =
        json_str(&response, "selected_mmproj").or_else(|| json_str(&plan.payload, "mmproj"));
    let selected_ctx_size = json_u64(&response, "selected_ctx_size")
        .or_else(|| json_u64(&plan.payload, "ctx_size"))
        .and_then(|value| u32::try_from(value).ok());
    local_state::save_selected_model(selected_model, selected_mmproj, selected_ctx_size)?;
    Ok((response, plan))
}

pub(crate) fn print_model_loaded(
    response: &serde_json::Value,
    plan: &model_load::ModelLoadPlan,
) -> Result<()> {
    let selected_backend = json_str(response, "selected_backend").unwrap_or(&plan.backend);
    let selected_model = json_str(response, "selected_model")
        .or_else(|| json_str(&plan.payload, "model"))
        .ok_or_else(|| anyhow::anyhow!("Model load response did not include a selected model."))?;
    let selected_mmproj =
        json_str(response, "selected_mmproj").or_else(|| json_str(&plan.payload, "mmproj"));
    let selected_ctx_size = json_u64(response, "selected_ctx_size")
        .or_else(|| json_u64(&plan.payload, "ctx_size"))
        .and_then(|value| u32::try_from(value).ok());
    println!("Model loaded");
    println!("Backend: {selected_backend}");
    println!("Model: {selected_model}");
    println!("mmproj: {}", selected_mmproj.unwrap_or("-"));
    println!(
        "ctx-size: {}",
        selected_ctx_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    if let Some(devices) = json_str(response, "cuda_visible_devices") {
        println!("CUDA_VISIBLE_DEVICES: {devices}");
    }
    if let Some(warning) = json_str(response, "warning") {
        println!("{warning}");
    }
    Ok(())
}

fn print_best_model_catalog(payload: &serde_json::Value) {
    let Some(families) = payload.as_object() else {
        println!("No models are available to display.");
        return;
    };
    if families.is_empty() {
        println!("No models are available to display.");
        return;
    }
    for (family, models) in families {
        println!("\n[{family}]");
        let Some(models) = models.as_object() else {
            continue;
        };
        for (model_name, model_info) in models {
            println!("  {model_name}");
            print_quantization_rows(model_info, true);
        }
    }
}

fn print_full_model_catalog(payload: &serde_json::Value) {
    let Some(backends) = payload.as_object() else {
        return;
    };
    for (backend, backend_payload) in backends {
        println!("\n[{backend}]");
        let Some(families) = backend_payload.as_object() else {
            continue;
        };
        for (family, models) in families {
            println!("  {family}");
            let Some(models) = models.as_object() else {
                continue;
            };
            for (model_name, model_info) in models {
                println!("    {model_name}");
                print_quantization_rows(model_info, false);
            }
        }
    }
}

fn print_quantization_rows(model_info: &serde_json::Value, include_backend: bool) {
    let quantizations = model_info
        .get("quantization")
        .and_then(serde_json::Value::as_object);
    let Some(quantizations) = quantizations else {
        return;
    };
    for (quant_name, quant_info) in quantizations {
        let suitable = match json_str(quant_info, "memory_status") {
            Some("unknown") => "unknown",
            _ if json_bool(quant_info, "suitable").unwrap_or(false) => "yes",
            _ => "no",
        };
        let memory = quant_info
            .get("required_memory_gib")
            .map(|value| format!("{value} GiB"))
            .unwrap_or_else(|| "-".to_string());
        if include_backend {
            let backend = json_str(quant_info, "backend").unwrap_or("-");
            println!("    - {quant_name}: backend={backend}, suitable={suitable}, memory={memory}");
        } else {
            println!("      - {quant_name}: suitable={suitable}, memory={memory}");
        }
    }
}
