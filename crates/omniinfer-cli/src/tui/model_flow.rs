use super::*;

pub(super) fn setup_model_flow(config: &config::AppConfig) -> Result<String> {
    let backend = choose_backend()?.ok_or_else(|| anyhow::anyhow!("No backend selected."))?;
    activate_backend(config, &backend)?;
    loop {
        let model = choose_model(config, false, Some(&backend))?
            .ok_or_else(|| anyhow::anyhow!("No model selected."))?;
        match load_model_interactive(config, model.to_string_lossy().as_ref()) {
            Ok(loaded) => return Ok(loaded),
            Err(error) => {
                notice(&format!("Model load failed: {error}"), NoticeKind::Warning);
                notice(
                    "Choose another model or cancel with q/Esc.",
                    NoticeKind::Warning,
                );
                println!();
            }
        }
    }
}

pub(super) fn load_remembered_model(
    config: &config::AppConfig,
    model: &local_state::SelectedModel,
) -> Result<String> {
    print_section("Resume", "Loading your last selected backend and model");
    print_kv("Model", &model.model);
    let request = model_load::ModelLoadRequest {
        model: model.model.clone(),
        mmproj: model.mmproj.clone(),
        no_mmproj: model.no_mmproj,
        ctx_size: model.ctx_size,
        backend_port: None,
        resource_budget_bytes: None,
        config: None,
        backend_extra_args: Vec::new(),
        request_defaults: Some(model.request_defaults.clone()),
    };
    if let Some(backend) = reuse_loaded_remembered_model(config, model) {
        notice("Reusing already loaded model", NoticeKind::Success);
        print_kv("Model loaded", &model.model);
        println!();
        return Ok(backend);
    }
    let (response, plan) = match load_model_with_request_for_config(&request, false, config) {
        Ok(result) => result,
        Err(error) if error.to_string().contains("model is already loaded") => {
            if let Some(backend) = reuse_loaded_remembered_model(config, model) {
                notice("Reusing already loaded model", NoticeKind::Success);
                print_kv("Model loaded", &model.model);
                println!();
                return Ok(backend);
            }
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let backend = json_str(&response, "selected_backend").unwrap_or(&plan.backend);
    notice("Backend ready", NoticeKind::Success);
    print_kv(
        "Model loaded",
        json_str(&response, "selected_model").unwrap_or(&model.model),
    );
    println!();
    Ok(backend.to_string())
}

pub(super) fn reuse_loaded_remembered_model(
    config: &config::AppConfig,
    model: &local_state::SelectedModel,
) -> Option<String> {
    let state = get_running_state(config)?;
    if !json_bool(&state, "backend_ready").unwrap_or(false) {
        return None;
    }
    let backend = json_str(&state, "backend")?.to_string();
    if state_matches_remembered_model(&state, model) {
        return Some(backend);
    }
    for row in state
        .get("loaded_models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if state_matches_remembered_model(row, model) {
            return Some(backend);
        }
    }
    None
}

pub(super) fn state_matches_remembered_model(
    state: &Value,
    model: &local_state::SelectedModel,
) -> bool {
    state_matches_model(state, &model.model)
        && state
            .get("request_defaults")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
            == model.request_defaults
}

pub(super) fn get_running_state(config: &config::AppConfig) -> Option<Value> {
    let url = format!("{}/omni/state", config.service_base_url());
    let response = http_client::get_json(&url, Duration::from_secs(2)).ok()?;
    (response.status == 200).then_some(response.body)
}

pub(super) fn state_matches_model(state: &Value, requested: &str) -> bool {
    ["model_path", "model", "selected_model"]
        .iter()
        .filter_map(|key| json_str(state, key))
        .any(|candidate| model_reference_matches(candidate, requested))
}

pub(super) fn model_reference_matches(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let left_path = Path::new(left);
    let right_path = Path::new(right);
    left_path.exists() && right_path.exists() && same_path(left_path, right_path)
}

pub(super) fn choose_backend() -> Result<Option<String>> {
    loop {
        let payload = rust_backend_payload(BackendScope::Compatible);
        let rows = payload
            .get("data")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if rows.is_empty() {
            anyhow::bail!("No compatible backends are available.");
        }
        let items = rows
            .iter()
            .map(|row| MenuItem {
                label: json_str(row, "id").unwrap_or("-").to_string(),
                details: vec![if json_bool(row, "binary_exists").unwrap_or(false) {
                    "installed".to_string()
                } else {
                    "not installed".to_string()
                }],
                selected: json_bool(row, "selected").unwrap_or(false),
            })
            .collect::<Vec<_>>();
        let default = rows
            .iter()
            .position(|row| json_bool(row, "selected").unwrap_or(false))
            .unwrap_or(0);
        let Some(index) = select_menu(
            "Backends",
            "Choose the runtime used for model loading",
            &items,
            default,
        )?
        else {
            return Ok(None);
        };
        let backend = json_str(&rows[index], "id").unwrap_or("").to_string();
        if backend.is_empty() {
            notice("Invalid backend.", NoticeKind::Warning);
            continue;
        }
        if !json_bool(&rows[index], "binary_exists").unwrap_or(false) {
            notice(
                &format!("Backend is not installed: {backend}"),
                NoticeKind::Warning,
            );
            if prompt_yes_no("Install prebuilt backend now?", true)? {
                match backend_installer::install_backend(backend_installer::InstallOptions {
                    backend: backend.clone(),
                    dry_run: false,
                    from_source: false,
                    json: false,
                    wsl_distro: None,
                }) {
                    Ok(()) => {
                        notice(
                            &format!("Installed backend: {backend}"),
                            NoticeKind::Success,
                        );
                    }
                    Err(error) => {
                        notice(
                            &format!("Backend install failed: {error}"),
                            NoticeKind::Warning,
                        );
                    }
                }
                println!();
            }
            continue;
        }
        return Ok(Some(backend));
    }
}

pub(super) fn activate_backend(config: &config::AppConfig, backend: &str) -> Result<()> {
    select_backend_for_config(backend, config)?;
    notice(&format!("Selected backend: {backend}"), NoticeKind::Success);
    println!();
    Ok(())
}

pub(super) fn prompt_yes_no(label: &str, default: bool) -> Result<bool> {
    let default_text = if default { "Y" } else { "n" };
    loop {
        let answer = prompt_default(label, default_text)?;
        match answer.trim().to_ascii_lowercase().as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => notice("Please answer y or n.", NoticeKind::Warning),
        }
    }
}

pub(super) fn choose_model(
    config: &config::AppConfig,
    mark_last_selected: bool,
    backend: Option<&str>,
) -> Result<Option<PathBuf>> {
    let backends_payload = rust_backend_payload(BackendScope::All);
    let menu_context = model_menu_context(&backends_payload, backend);
    let models = discover_local_models(config)?;
    let selected_backend = selected_backend_info(&backends_payload, backend);
    let recommendations = advisor_recommendation_map(config, &models);
    let remembered = if mark_last_selected {
        local_state::load_state()
            .ok()
            .and_then(|state| state.selected_model)
    } else {
        None
    };
    let remembered_path = remembered.as_ref().map(|model| PathBuf::from(&model.model));
    let mut items = Vec::<ModelMenuItem>::new();
    let mut choices = Vec::<Option<PathBuf>>::new();
    let mut default = 0;
    for model in models
        .iter()
        .filter(|model| model_supported_by_backend(&model.path, selected_backend.as_ref()))
    {
        let selected = remembered_path
            .as_ref()
            .is_some_and(|path| same_path(path, &model.path));
        if selected {
            default = items.len();
        }
        let summary = advisor_model_summary(&model.path, &recommendations).unwrap_or_default();
        items.push(ModelMenuItem {
            label: model.label.clone(),
            provider: model_provider_label(&model.path),
            quant: model_quant_label(&model.path),
            disk: model_size_label(&model.path),
            ctx: model_context_label(&model.path),
            fit: summary.fit.unwrap_or_else(|| "-".to_string()),
            backend: summary.backend.unwrap_or_else(|| "-".to_string()),
            evidence: evidence_label(summary.evidence, summary.confidence),
            selected,
        });
        choices.push(Some(model.path.clone()));
    }
    if let Some(path) = remembered_path
        .filter(|path| path.exists())
        .filter(|path| model_supported_by_backend(path, selected_backend.as_ref()))
        && !models.iter().any(|model| same_path(&model.path, &path))
    {
        default = items.len();
        items.push(ModelMenuItem {
            label: path.display().to_string(),
            provider: model_provider_label(&path),
            quant: model_quant_label(&path),
            disk: model_size_label(&path),
            ctx: model_context_label(&path),
            fit: "-".to_string(),
            backend: "-".to_string(),
            evidence: "last selected".to_string(),
            selected: true,
        });
        choices.push(Some(path));
    }
    items.push(ModelMenuItem {
        label: "Enter path manually".to_string(),
        provider: "manual".to_string(),
        quant: "-".to_string(),
        disk: "-".to_string(),
        ctx: "-".to_string(),
        fit: "manual".to_string(),
        backend: "-".to_string(),
        evidence: "link local file".to_string(),
        selected: false,
    });
    choices.push(None);
    let Some(index) = select_model_menu("Models", "", &items, default, &menu_context)? else {
        return Ok(None);
    };
    if let Some(path) = &choices[index] {
        return Ok(Some(path.clone()));
    }
    prompt_model_path()
}

#[derive(Debug, Clone)]
pub(super) struct SelectedBackendInfo {
    pub(super) family: String,
    pub(super) model_artifact: String,
}

pub(super) fn selected_backend_info(
    backends_payload: &Value,
    backend: Option<&str>,
) -> Option<SelectedBackendInfo> {
    let row = backends_payload
        .get("data")
        .and_then(Value::as_array)?
        .iter()
        .find(|row| {
            backend
                .map(|backend| json_str(row, "id") == Some(backend))
                .unwrap_or_else(|| json_bool(row, "selected").unwrap_or(false))
        })?;
    Some(SelectedBackendInfo {
        family: json_str(row, "family").unwrap_or("").to_string(),
        model_artifact: json_str(row, "model_artifact").unwrap_or("").to_string(),
    })
}

pub(super) fn model_supported_by_backend(
    path: &Path,
    backend: Option<&SelectedBackendInfo>,
) -> bool {
    let Some(backend) = backend else {
        return true;
    };
    if path.is_dir() {
        return backend.model_artifact == "directory"
            || matches!(backend.family.as_str(), "mnn" | "mlx" | "vllm");
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "gguf" => {
            backend.model_artifact == "vla-artifact"
                || matches!(backend.family.as_str(), "llama.cpp" | "turboquant")
        }
        "mnn" => backend.family == "mnn",
        "safetensors" => backend.model_artifact == "vla-artifact",
        "bin" => backend.model_artifact == "file",
        _ => backend.model_artifact == "file",
    }
}

pub(super) fn model_menu_context(
    backends_payload: &Value,
    backend: Option<&str>,
) -> ModelMenuContext {
    let system = advisor::system_payload(backends_payload.clone());
    let host = system.get("host").unwrap_or(&Value::Null);
    let cuda = system.get("cuda").unwrap_or(&Value::Null);
    let mut hardware_lines = Vec::new();
    hardware_lines.push(format!(
        "CPU: {} threads | RAM: {} / {} GiB",
        json_u64(host, "cpu_cores")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string()),
        json_f64(host, "available_ram_gib")
            .map(format_one_decimal)
            .unwrap_or_else(|| "-".to_string()),
        json_f64(host, "total_ram_gib")
            .map(format_one_decimal)
            .unwrap_or_else(|| "-".to_string())
    ));
    hardware_lines.push(gpu_summary_line(cuda));

    ModelMenuContext {
        hardware_lines,
        backend_line: selected_backend_line(backends_payload, backend),
    }
}

pub(super) fn selected_backend_line(backends_payload: &Value, backend: Option<&str>) -> String {
    let selected = backends_payload
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|row| {
            backend
                .map(|backend| json_str(row, "id") == Some(backend))
                .unwrap_or_else(|| json_bool(row, "selected").unwrap_or(false))
        });
    let Some(row) = selected else {
        return "Backend: none selected".to_string();
    };
    let id = json_str(row, "id").unwrap_or("-");
    let state = if json_bool(row, "installed").unwrap_or(false)
        && json_bool(row, "hardware_compatible").unwrap_or(false)
    {
        "installed, compatible"
    } else if json_bool(row, "installed").unwrap_or(false) {
        "installed, hardware unavailable"
    } else {
        "not installed"
    };
    format!("Backend: {id} ({state})")
}

pub(super) fn gpu_summary_line(cuda: &Value) -> String {
    let devices = cuda
        .get("visible_devices")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .or_else(|| cuda.get("devices").and_then(Value::as_array));
    let Some(devices) = devices.filter(|items| !items.is_empty()) else {
        return "GPU: none detected".to_string();
    };
    let primary = &devices[0];
    let name = json_str(primary, "name").unwrap_or("GPU");
    let matching = devices
        .iter()
        .filter(|device| {
            json_str(device, "name") == Some(name)
                && json_f64(device, "total_gib") == json_f64(primary, "total_gib")
        })
        .count();
    let count = matching.max(1);
    let single = json_f64(primary, "total_gib");
    let other = devices.len().saturating_sub(count);
    let mut line = if let Some(single) = single {
        format!(
            "GPU: {name} x{count} (total VRAM = {} GiB x {count} = {} GiB)",
            format_one_decimal(single),
            format_one_decimal(single * count as f64)
        )
    } else {
        format!("GPU: {name} x{count}")
    };
    if other > 0 {
        line.push_str(&format!(" + {other} other"));
    }
    line
}

pub(super) fn json_f64<'a>(value: &'a Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

pub(super) fn format_one_decimal(value: f64) -> String {
    format!("{value:.1}")
}

pub(super) fn evidence_label(evidence: Option<String>, confidence: Option<String>) -> String {
    match (evidence, confidence) {
        (Some(evidence), Some(confidence)) => format!("{evidence}/{confidence}"),
        (Some(evidence), None) => evidence,
        (None, Some(confidence)) => confidence,
        (None, None) => "-".to_string(),
    }
}

pub(super) fn load_model_interactive(config: &config::AppConfig, model: &str) -> Result<String> {
    println!();
    print_section("Load model", "Starting the selected runtime");
    print_kv("Model", model);
    let request = model_load::ModelLoadRequest {
        model: model.to_string(),
        mmproj: None,
        no_mmproj: false,
        ctx_size: None,
        backend_port: None,
        resource_budget_bytes: None,
        config: None,
        backend_extra_args: Vec::new(),
        request_defaults: None,
    };
    let (response, plan) = load_model_with_request_for_config(&request, false, config)?;
    if plan.auto_selected {
        notice(
            &format!("Auto-selected backend: {}", plan.backend),
            NoticeKind::Success,
        );
    }
    notice("Backend ready", NoticeKind::Success);
    print_kv(
        "Model loaded",
        json_str(&response, "selected_model").unwrap_or(model),
    );
    println!();
    Ok(json_str(&response, "selected_backend")
        .or(Some(&plan.backend))
        .unwrap_or(&plan.backend)
        .to_string())
}
