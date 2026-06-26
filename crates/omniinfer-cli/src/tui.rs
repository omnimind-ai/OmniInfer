use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use omniinfer_core::{
    chat_stream, config, http_client, local_state, model_load, paths, serve_state,
};
use serde_json::Value;

use crate::{
    BackendScope, ServeArgs, advisor, get_local_json_for_config, json_bool, json_str, json_u64,
    load_model_with_request_for_config, post_local_json_for_config, print_chat_performance,
    rust_backend_payload, select_backend_for_config, serve_orchestrated, shutdown_service,
};

#[derive(Debug, Clone)]
struct MenuItem {
    label: String,
    details: Vec<String>,
    selected: bool,
}

#[derive(Debug, Clone)]
struct LocalModel {
    path: PathBuf,
    label: String,
}

#[derive(Debug)]
struct ChatSession {
    backend: String,
    reasoning_visible: bool,
    messages: Vec<Value>,
    last_usage: Option<Value>,
}

pub fn run() -> Result<()> {
    if !is_interactive() {
        anyhow::bail!("OmniInfer TUI requires an interactive terminal.");
    }
    clear_screen();
    print_header("OmniInfer", "Local inference console");
    let config = config::load_app_config().unwrap_or_default();
    let state = local_state::load_state().unwrap_or_default();
    let backend = match state.selected_model.clone() {
        Some(model) if Path::new(&model.model).exists() => {
            match load_remembered_model(&config, &model) {
                Ok(backend) => backend,
                Err(error) => {
                    notice(
                        &format!("Could not load previous model: {error}"),
                        NoticeKind::Warning,
                    );
                    setup_model_flow(&config)?
                }
            }
        }
        _ => setup_model_flow(&config)?,
    };
    chat_loop(&config, backend)?;
    let _ = shutdown_service();
    Ok(())
}

pub fn run_server(args: &ServeArgs) -> Result<()> {
    if !is_interactive() {
        return serve_orchestrated(args);
    }
    clear_screen();
    print_header("OmniInfer Server", "Interactive gateway launcher");
    let config = config::load_app_config().unwrap_or_default();
    let backend =
        choose_backend(&config)?.ok_or_else(|| anyhow::anyhow!("No backend selected."))?;
    let model =
        choose_model(&config, true)?.ok_or_else(|| anyhow::anyhow!("No model selected."))?;
    let mut args = args.clone();
    args.backend = Some(backend);
    args.model = Some(model.display().to_string());
    serve_orchestrated(&args)
}

fn setup_model_flow(config: &config::AppConfig) -> Result<String> {
    let backend = choose_backend(config)?.ok_or_else(|| anyhow::anyhow!("No backend selected."))?;
    let model =
        choose_model(config, false)?.ok_or_else(|| anyhow::anyhow!("No model selected."))?;
    let loaded =
        load_model_interactive(config, model.to_string_lossy().as_ref(), true)?.unwrap_or(backend);
    Ok(loaded)
}

fn load_remembered_model(
    config: &config::AppConfig,
    model: &local_state::SelectedModel,
) -> Result<String> {
    print_section("Resume", "Loading your last selected backend and model");
    print_kv("Model", &model.model);
    let request = model_load::ModelLoadRequest {
        model: model.model.clone(),
        mmproj: model.mmproj.clone(),
        ctx_size: model.ctx_size,
        backend_port: None,
        config: None,
        backend_extra_args: Vec::new(),
    };
    let (response, plan) = load_model_with_request_for_config(&request, false, config)?;
    let backend = json_str(&response, "selected_backend").unwrap_or(&plan.backend);
    notice("Backend ready", NoticeKind::Success);
    print_kv(
        "Model loaded",
        json_str(&response, "selected_model").unwrap_or(&model.model),
    );
    println!();
    Ok(backend.to_string())
}

fn choose_backend(config: &config::AppConfig) -> Result<Option<String>> {
    loop {
        let payload = get_local_json_for_config(
            "/omni/backends?scope=compatible",
            Duration::from_secs(10),
            config,
        )?;
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
            continue;
        }
        select_backend_for_config(&backend, config)?;
        notice(&format!("Selected backend: {backend}"), NoticeKind::Success);
        println!();
        return Ok(Some(backend));
    }
}

fn choose_model(config: &config::AppConfig, mark_last_selected: bool) -> Result<Option<PathBuf>> {
    let models = discover_local_models(config)?;
    let recommendations = advisor_recommendation_map(config, &models);
    let remembered = if mark_last_selected {
        local_state::load_state()
            .ok()
            .and_then(|state| state.selected_model)
    } else {
        None
    };
    let remembered_path = remembered.as_ref().map(|model| PathBuf::from(&model.model));
    let mut items = Vec::new();
    let mut choices = Vec::<Option<PathBuf>>::new();
    let mut default = 0;
    for model in &models {
        let mut details = Vec::new();
        let selected = remembered_path
            .as_ref()
            .is_some_and(|path| same_path(path, &model.path));
        if selected {
            details.push("last selected".to_string());
            default = items.len();
        }
        details.extend(advisor_model_details(&model.path, &recommendations));
        items.push(MenuItem {
            label: model.label.clone(),
            details,
            selected,
        });
        choices.push(Some(model.path.clone()));
    }
    if let Some(path) = remembered_path.filter(|path| path.exists())
        && !models.iter().any(|model| same_path(&model.path, &path))
    {
        default = items.len();
        items.push(MenuItem {
            label: path.display().to_string(),
            details: vec!["last selected".to_string()],
            selected: true,
        });
        choices.push(Some(path));
    }
    items.push(MenuItem {
        label: "Enter path manually".to_string(),
        details: vec!["link into .local/models".to_string()],
        selected: false,
    });
    choices.push(None);
    let Some(index) = select_menu(
        "Models",
        "Pick a managed model or link a new local file",
        &items,
        default,
    )?
    else {
        return Ok(None);
    };
    if let Some(path) = &choices[index] {
        return Ok(Some(path.clone()));
    }
    prompt_model_path()
}

fn load_model_interactive(
    config: &config::AppConfig,
    model: &str,
    advisor_preflight: bool,
) -> Result<Option<String>> {
    if advisor_preflight && !advisor_preflight_flow(config, model)? {
        return Ok(None);
    }
    println!();
    print_kv("Model", model);
    let request = model_load::ModelLoadRequest {
        model: model.to_string(),
        mmproj: None,
        ctx_size: None,
        backend_port: None,
        config: None,
        backend_extra_args: Vec::new(),
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
        .map(str::to_string))
}

fn advisor_preflight_flow(config: &config::AppConfig, model: &str) -> Result<bool> {
    let _ = config;
    let backends = rust_backend_payload(BackendScope::All);
    let payload = advisor::fit_payload(model, None, None, None, backends)?;
    let recommended = payload.get("recommended").filter(|value| value.is_object());
    let Some(recommended) = recommended else {
        print_warnings(&payload);
        return Ok(true);
    };
    print_advisor_preflight_summary(&payload);
    loop {
        let choice = prompt_default("Advisor", "")?;
        match choice.trim().to_ascii_lowercase().as_str() {
            "" | "y" | "yes" | "load" => {
                apply_advisor_backend(config, recommended)?;
                return Ok(true);
            }
            "a" | "advisor" | "details" => {
                advisor::print_fit(&payload, false)?;
            }
            "b" | "backend" => {
                let _ = choose_backend(config)?;
                return Ok(true);
            }
            "s" | "skip" | "current" => return Ok(true),
            "q" | "quit" | "cancel" => return Ok(false),
            _ => notice(
                "Use Enter to load, A for details, B to change backend, or S to keep current.",
                NoticeKind::Warning,
            ),
        }
    }
}

fn apply_advisor_backend(config: &config::AppConfig, recommended: &Value) -> Result<()> {
    let Some(backend) = json_str(recommended, "backend") else {
        return Ok(());
    };
    if !json_bool(recommended, "installed").unwrap_or(false) {
        notice(
            &format!("Advisor recommended {backend}, but it is not installed."),
            NoticeKind::Warning,
        );
        return Ok(());
    }
    select_backend_for_config(backend, config)?;
    notice(
        &format!("Advisor selected backend: {backend}"),
        NoticeKind::Success,
    );
    Ok(())
}

fn print_advisor_preflight_summary(payload: &Value) {
    let recommended = payload.get("recommended").unwrap_or(&Value::Null);
    print_section("Advisor", "Load preflight");
    print_kv(
        "Recommended",
        &format!(
            "{} ({})",
            json_str(recommended, "backend").unwrap_or("-"),
            json_str(recommended, "fit").unwrap_or("unknown")
        ),
    );
    let evidence = recommended.get("evidence").unwrap_or(&Value::Null);
    print_kv(
        "Evidence",
        &format!(
            "{} / {}",
            json_str(evidence, "level").unwrap_or("-"),
            json_str(recommended, "recommendation_confidence")
                .or_else(|| json_str(evidence, "confidence"))
                .unwrap_or("-")
        ),
    );
    print_kv(
        "Memory",
        &format!(
            "{} required / {} available",
            format_gib(recommended.get("memory_required_gib")),
            format_gib(recommended.get("memory_available_gib"))
        ),
    );
    if let Some(text) =
        memory_breakdown_text(recommended.get("memory_breakdown").unwrap_or(&Value::Null))
    {
        print_kv("Breakdown", &text);
    }
    print_kv(
        "Action",
        "Enter load recommendation | A details | B backend | S keep current",
    );
    print_warnings(payload);
}

fn print_warnings(payload: &Value) {
    if let Some(warnings) = payload.get("warnings").and_then(Value::as_array) {
        for warning in warnings.iter().filter_map(Value::as_str).take(2) {
            notice(&format!("Advisor: {warning}"), NoticeKind::Warning);
        }
    }
}

fn chat_loop(config: &config::AppConfig, backend: String) -> Result<()> {
    let mut session = ChatSession {
        backend,
        reasoning_visible: local_state::load_state()
            .map(|state| state.tui_show_reasoning)
            .unwrap_or(false),
        messages: Vec::new(),
        last_usage: None,
    };
    print_chat_header(&session);
    loop {
        let message = prompt_default("You", "")?;
        let message = message.trim();
        if message.is_empty() {
            continue;
        }
        match message {
            "/exit" => return Ok(()),
            "/backend" => {
                if let Some(backend) = choose_backend(config)? {
                    session.backend = backend;
                    session.messages.clear();
                    if let Some(model) = choose_model(config, true)? {
                        if let Some(loaded) =
                            load_model_interactive(config, model.to_string_lossy().as_ref(), true)?
                        {
                            session.backend = loaded;
                        }
                    }
                    print_chat_header(&session);
                }
            }
            "/model" => {
                if let Some(model) = choose_model(config, true)?
                    && let Some(loaded) =
                        load_model_interactive(config, model.to_string_lossy().as_ref(), true)?
                {
                    session.backend = loaded;
                    session.messages.clear();
                    print_chat_header(&session);
                }
            }
            "/clear" => {
                clear_screen();
                print_header("OmniInfer", "Local inference console");
                print_chat_header(&session);
            }
            "/status" => print_status(config, &session)?,
            "/help" => print_help(),
            "/think" | "/thinking" => toggle_thinking(config)?,
            "/reasoning" => toggle_reasoning(&mut session)?,
            _ if message.starts_with("/think ") || message.starts_with("/thinking ") => {
                set_thinking(config, message.split_whitespace().nth(1))?;
            }
            _ if message.starts_with("/reasoning ") => {
                set_reasoning(&mut session, message.split_whitespace().nth(1))?;
            }
            _ => send_chat_message(config, &mut session, message)?,
        }
    }
}

fn send_chat_message(
    config: &config::AppConfig,
    session: &mut ChatSession,
    message: &str,
) -> Result<()> {
    println!("You:");
    println!("  {message}");
    let state = get_local_json_for_config("/omni/state", Duration::from_secs(10), config)?;
    if json_str(&state, "model").is_none() {
        anyhow::bail!("No model is currently loaded. Use /model first.");
    }
    let mut payload = state
        .get("request_defaults")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut messages = session.messages.clone();
    messages.push(serde_json::json!({ "role": "user", "content": message }));
    payload.insert("messages".to_string(), Value::Array(messages));
    payload.insert("stream".to_string(), serde_json::json!(true));
    payload.insert(
        "stream_options".to_string(),
        serde_json::json!({ "include_usage": true }),
    );
    payload
        .entry("temperature")
        .or_insert(serde_json::json!(0.2));
    payload
        .entry("max_tokens")
        .or_insert(serde_json::json!(2048));
    println!("Assistant:");
    let assistant_text = stream_chat_response(config, &Value::Object(payload), session)?;
    if !assistant_text.trim().is_empty() {
        session
            .messages
            .push(serde_json::json!({ "role": "user", "content": message }));
        session
            .messages
            .push(serde_json::json!({ "role": "assistant", "content": assistant_text }));
    }
    Ok(())
}

fn stream_chat_response(
    config: &config::AppConfig,
    payload: &Value,
    session: &mut ChatSession,
) -> Result<String> {
    let url = format!("{}/v1/chat/completions", config.service_base_url());
    let mut filter = chat_stream::StreamPrefixFilter::new();
    let mut final_payload = None;
    let mut assistant_text = String::new();
    let response = http_client::post_streaming_lines(
        &url,
        payload,
        "text/event-stream, application/json",
        Duration::from_secs(3600),
        |line| {
            for chunk in chat_stream::parse_chat_stream_line(line).unwrap_or_default() {
                match chunk {
                    chat_stream::ChatStreamChunk::Text(text) => {
                        if let Some(text) = filter.push(&text)
                            && !text.is_empty()
                        {
                            assistant_text.push_str(&text);
                            print!("{text}");
                            let _ = io::stdout().flush();
                        }
                    }
                    chat_stream::ChatStreamChunk::Reasoning(text) => {
                        if session.reasoning_visible && !text.trim().is_empty() {
                            print!("\nReasoning:\n  {text}\nAssistant:\n");
                            let _ = io::stdout().flush();
                        }
                    }
                    chat_stream::ChatStreamChunk::Final(payload) => {
                        final_payload = Some(payload);
                    }
                }
            }
        },
    )?;
    if response.status >= 400 {
        anyhow::bail!("Streaming inference failed with status {}", response.status);
    }
    if let Some(text) = filter.finish()
        && !text.is_empty()
    {
        assistant_text.push_str(&text);
        print!("{text}");
    }
    println!();
    if let Some(payload) = final_payload {
        if let Some(usage) = payload.get("usage") {
            session.last_usage = Some(usage.clone());
        }
        print_chat_performance(&payload);
    }
    Ok(assistant_text)
}

fn print_status(config: &config::AppConfig, session: &ChatSession) -> Result<()> {
    let state = get_local_json_for_config("/omni/state", Duration::from_secs(10), config)?;
    print_section("Status", "Current OmniInfer session");
    print_kv(
        "Backend",
        json_str(&state, "backend").unwrap_or(&session.backend),
    );
    print_kv(
        "State",
        if json_bool(&state, "backend_ready").unwrap_or(false) {
            "ready"
        } else {
            "not ready"
        },
    );
    print_kv("Model", json_str(&state, "model").unwrap_or("-"));
    print_kv(
        "Context size",
        &json_u64(&state, "ctx_size")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    print_kv(
        "Reasoning display",
        if session.reasoning_visible {
            "show"
        } else {
            "hide"
        },
    );
    if let Some(usage) = &session.last_usage {
        print_kv(
            "Last usage",
            &format!(
                "prompt={}, completion={}, total={}",
                json_u64(usage, "prompt_tokens")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                json_u64(usage, "completion_tokens")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                json_u64(usage, "total_tokens")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string())
            ),
        );
    }
    println!();
    Ok(())
}

fn toggle_thinking(config: &config::AppConfig) -> Result<()> {
    let current = get_thinking(config).unwrap_or(false);
    set_thinking_value(config, !current)
}

fn set_thinking(config: &config::AppConfig, value: Option<&str>) -> Result<()> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        Some("on") => set_thinking_value(config, true),
        Some("off") => set_thinking_value(config, false),
        _ => {
            notice(
                "Usage: /think, /think on, or /think off",
                NoticeKind::Warning,
            );
            Ok(())
        }
    }
}

fn get_thinking(config: &config::AppConfig) -> Result<bool> {
    let payload = get_local_json_for_config("/omni/thinking", Duration::from_secs(10), config)?;
    Ok(json_bool(&payload, "default_enabled").unwrap_or(false))
}

fn set_thinking_value(config: &config::AppConfig, enabled: bool) -> Result<()> {
    let payload = post_local_json_for_config(
        "/omni/thinking/select",
        &serde_json::json!({ "enabled": enabled }),
        Duration::from_secs(10),
        config,
    )?;
    notice(
        &format!(
            "Thinking mode: {}",
            if json_bool(&payload, "default_enabled").unwrap_or(false) {
                "on"
            } else {
                "off"
            }
        ),
        NoticeKind::Success,
    );
    Ok(())
}

fn toggle_reasoning(session: &mut ChatSession) -> Result<()> {
    set_reasoning_value(session, !session.reasoning_visible)
}

fn set_reasoning(session: &mut ChatSession, value: Option<&str>) -> Result<()> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        Some("on" | "show") => set_reasoning_value(session, true),
        Some("off" | "hide") => set_reasoning_value(session, false),
        _ => {
            notice(
                "Usage: /reasoning, /reasoning on, or /reasoning off",
                NoticeKind::Warning,
            );
            Ok(())
        }
    }
}

fn set_reasoning_value(session: &mut ChatSession, enabled: bool) -> Result<()> {
    session.reasoning_visible = enabled;
    local_state::save_tui_show_reasoning(enabled)?;
    notice(
        &format!(
            "Reasoning display: {}",
            if enabled { "show" } else { "hide" }
        ),
        NoticeKind::Success,
    );
    Ok(())
}

fn print_help() {
    print_section("Help", "Conversation commands");
    let rows = [
        ("/backend", "switch the selected runtime"),
        ("/model", "load a different managed model"),
        (
            "/think",
            "toggle thinking mode; use /think on or /think off",
        ),
        (
            "/reasoning",
            "toggle visible reasoning; use /reasoning on or /reasoning off",
        ),
        ("/status", "show backend, model, and context usage"),
        ("/clear", "clear the terminal and redraw the chat header"),
        ("/help", "show this command reference"),
        ("/exit", "stop the OmniInfer service and leave the TUI"),
    ];
    for (name, description) in rows {
        print_kv(name, description);
    }
    println!();
}

fn discover_local_models(config: &config::AppConfig) -> Result<Vec<LocalModel>> {
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

fn discover_models_in_roots(roots: &[PathBuf]) -> Vec<LocalModel> {
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

fn prompt_model_path() -> Result<Option<PathBuf>> {
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

fn advisor_recommendation_map(
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

fn advisor_model_details(
    model_path: &Path,
    recommendations: &BTreeMap<String, Value>,
) -> Vec<String> {
    let Some(row) = recommendations.get(&advisor_path_key(&model_path.display().to_string()))
    else {
        return Vec::new();
    };
    let recommended = row.get("recommended").unwrap_or(&Value::Null);
    let mut details = Vec::new();
    if let Some(fit) = json_str(recommended, "fit") {
        details.push(format!("advisor {fit}"));
    }
    if let Some(backend) = json_str(recommended, "backend") {
        details.push(backend.to_string());
    }
    let evidence = recommended.get("evidence").unwrap_or(&Value::Null);
    if let Some(level) = json_str(evidence, "level") {
        details.push(level.to_string());
    }
    if let Some(confidence) = json_str(recommended, "recommendation_confidence")
        .or_else(|| json_str(evidence, "confidence"))
    {
        details.push(confidence.to_string());
    }
    details
}

fn select_menu(
    title: &str,
    subtitle: &str,
    items: &[MenuItem],
    default_index: usize,
) -> Result<Option<usize>> {
    if items.is_empty() {
        return Ok(None);
    }
    print_section(title, subtitle);
    for (index, item) in items.iter().enumerate() {
        let marker = if item.selected { "*" } else { " " };
        let details = if item.details.is_empty() {
            String::new()
        } else {
            format!("  {}", item.details.join(" | "))
        };
        println!("{:>2}. [{}] {}{}", index + 1, marker, item.label, details);
    }
    println!("Press Enter to keep the default, or type q to cancel.");
    loop {
        let choice = prompt_default("Select", &(default_index + 1).to_string())?;
        if matches!(
            choice.trim().to_ascii_lowercase().as_str(),
            "q" | "quit" | "cancel" | "esc"
        ) {
            return Ok(None);
        }
        if let Ok(index) = choice.trim().parse::<usize>()
            && (1..=items.len()).contains(&index)
        {
            return Ok(Some(index - 1));
        }
        notice("Invalid selection.", NoticeKind::Warning);
    }
}

fn prompt_default(label: &str, default: &str) -> Result<String> {
    if default.is_empty() {
        print!("{label}: ");
    } else {
        print!("{label} [{default}]: ");
    }
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let text = input.trim();
    if text.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(text.to_string())
    }
}

fn print_header(title: &str, subtitle: &str) {
    println!("{title}");
    println!("{subtitle}");
    println!("{}", "-".repeat(64));
    println!();
}

fn print_section(title: &str, subtitle: &str) {
    println!("{title}");
    if !subtitle.is_empty() {
        println!("{subtitle}");
    }
    println!("{}", "-".repeat(title.len().max(24)));
}

fn print_chat_header(session: &ChatSession) {
    print_section("Chat", &format!("Backend: {}", session.backend));
    println!("Commands: /backend /model /think /reasoning /status /clear /help /exit");
    println!();
}

fn print_kv(label: &str, value: &str) {
    println!("  {label}: {value}");
}

#[derive(Debug, Clone, Copy)]
enum NoticeKind {
    Success,
    Warning,
}

fn notice(message: &str, kind: NoticeKind) {
    let prefix = match kind {
        NoticeKind::Success => "ok",
        NoticeKind::Warning => "warn",
    };
    println!("  {prefix}: {message}");
}

fn clear_screen() {
    print!("\x1b[2J\x1b[H");
    let _ = io::stdout().flush();
}

fn is_interactive() -> bool {
    use std::io::IsTerminal;
    io::stdin().is_terminal() && io::stdout().is_terminal()
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

fn same_path(left: &Path, right: &Path) -> bool {
    left.canonicalize().unwrap_or_else(|_| left.to_path_buf())
        == right.canonicalize().unwrap_or_else(|_| right.to_path_buf())
}

fn managed_model_target(source: &Path, target_root: &Path, model_root: Option<&Path>) -> PathBuf {
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

fn advisor_path_key(value: &str) -> String {
    expand_path(value)
        .canonicalize()
        .unwrap_or_else(|_| expand_path(value))
        .display()
        .to_string()
}

fn format_gib(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_f64)
        .map(|value| format!("{value:.2} GiB"))
        .unwrap_or_else(|| "-".to_string())
}

fn memory_breakdown_text(breakdown: &Value) -> Option<String> {
    let fields = [
        ("weights", "weights_gib"),
        ("mmproj", "mmproj_gib"),
        ("kv", "kv_cache_gib"),
        ("act", "activation_gib"),
        ("runtime", "runtime_overhead_gib"),
    ];
    let parts = fields
        .iter()
        .filter_map(|(label, key)| {
            breakdown
                .get(*key)
                .and_then(Value::as_f64)
                .map(|value| format!("{label} {value:.2} GiB"))
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join(" | "))
}

#[allow(dead_code)]
fn _loaded_services() -> Vec<serve_state::ServePidInfo> {
    serve_state::list_serve_pid_infos().unwrap_or_default()
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
            advisor_model_details(&model, &rows),
            vec!["advisor good", "llama.cpp-linux-cuda", "direct", "high"]
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
    fn memory_breakdown_text_uses_expected_labels() {
        let breakdown = serde_json::json!({
            "weights_gib": 2.55,
            "kv_cache_gib": 0.25,
            "runtime_overhead_gib": 0.5
        });
        assert_eq!(
            memory_breakdown_text(&breakdown).as_deref(),
            Some("weights 2.55 GiB | kv 0.25 GiB | runtime 0.50 GiB")
        );
    }

    fn unique_name(prefix: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        format!("omniinfer-rs-{prefix}-{nanos}")
    }
}
