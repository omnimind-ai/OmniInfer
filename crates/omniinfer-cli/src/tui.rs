use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::Result;
use omniinfer_core::{
    chat_stream, config, http_client, local_state, model_load, paths, serve_state,
};
use serde_json::Value;

mod models;
mod render;

use crate::{
    BackendScope, ServeArgs, advisor, get_local_json_for_config, json_bool, json_str, json_u64,
    load_model_with_request_for_config, post_local_json_for_config, print_chat_performance,
    rust_backend_payload, select_backend_for_config, serve_orchestrated, stop_serve,
    wait_for_gateway_ready,
};

use models::{
    advisor_model_summary, advisor_recommendation_map, discover_local_models, model_context_label,
    model_provider_label, model_quant_label, model_size_label, prompt_model_path, same_path,
};
use render::{
    ModelMenuItem, NoticeKind, clear_screen, is_interactive, notice, print_chat_header,
    print_header, print_help, print_kv, print_section, prompt_default, select_menu,
    select_model_menu,
};

#[derive(Debug, Clone)]
struct MenuItem {
    label: String,
    details: Vec<String>,
    selected: bool,
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
    let _gateway = TuiGatewayGuard::ensure(&config)?;
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
    Ok(())
}

struct TuiGatewayGuard {
    port: u16,
    owned: bool,
    child: Option<Child>,
    stopped: Arc<AtomicBool>,
}

impl TuiGatewayGuard {
    fn ensure(config: &config::AppConfig) -> Result<Self> {
        if get_running_state(config).is_some() {
            return Ok(Self {
                port: config.port,
                owned: false,
                child: None,
                stopped: Arc::new(AtomicBool::new(false)),
            });
        }
        print_section("Service", "Starting local OmniInfer gateway");
        print_kv("Port", &config.port.to_string());
        let mut command = ProcessCommand::new(std::env::current_exe()?);
        command
            .arg("gateway")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(config.port.to_string())
            .current_dir(paths::repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command.spawn()?;
        let guard = Self {
            port: config.port,
            owned: true,
            child: Some(child),
            stopped: Arc::new(AtomicBool::new(false)),
        };
        wait_for_gateway_ready(config)?;
        guard.install_ctrl_c_handler();
        notice("Local gateway ready", NoticeKind::Success);
        println!();
        Ok(guard)
    }

    fn install_ctrl_c_handler(&self) {
        if !self.owned {
            return;
        }
        let port = self.port;
        let stopped = Arc::clone(&self.stopped);
        std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            runtime.block_on(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    stop_tui_owned_gateway(port, &stopped);
                    std::process::exit(130);
                }
            });
        });
    }
}

impl Drop for TuiGatewayGuard {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        stop_tui_owned_gateway(self.port, &self.stopped);
        if let Some(child) = self.child.as_mut() {
            let _ = child.try_wait();
        }
    }
}

fn stop_tui_owned_gateway(port: u16, stopped: &AtomicBool) {
    if stopped.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = stop_serve(port);
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
    choose_backend(config)?.ok_or_else(|| anyhow::anyhow!("No backend selected."))?;
    loop {
        let model =
            choose_model(config, false)?.ok_or_else(|| anyhow::anyhow!("No model selected."))?;
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

fn reuse_loaded_remembered_model(
    config: &config::AppConfig,
    model: &local_state::SelectedModel,
) -> Option<String> {
    let state = get_running_state(config)?;
    if !json_bool(&state, "backend_ready").unwrap_or(false) {
        return None;
    }
    let backend = json_str(&state, "backend")?.to_string();
    if state_matches_model(&state, &model.model) {
        return Some(backend);
    }
    for row in state
        .get("loaded_models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if state_matches_model(row, &model.model) {
            return Some(backend);
        }
    }
    None
}

fn get_running_state(config: &config::AppConfig) -> Option<Value> {
    let url = format!("{}/omni/state", config.service_base_url());
    let response = http_client::get_json(&url, Duration::from_secs(2)).ok()?;
    (response.status == 200).then_some(response.body)
}

fn state_matches_model(state: &Value, requested: &str) -> bool {
    ["model_path", "model", "selected_model"]
        .iter()
        .filter_map(|key| json_str(state, key))
        .any(|candidate| model_reference_matches(candidate, requested))
}

fn model_reference_matches(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let left_path = Path::new(left);
    let right_path = Path::new(right);
    left_path.exists() && right_path.exists() && same_path(left_path, right_path)
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
    let mut items = Vec::<ModelMenuItem>::new();
    let mut choices = Vec::<Option<PathBuf>>::new();
    let mut default = 0;
    for model in &models {
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
    if let Some(path) = remembered_path.filter(|path| path.exists())
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
    let Some(index) = select_model_menu(
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

fn evidence_label(evidence: Option<String>, confidence: Option<String>) -> String {
    match (evidence, confidence) {
        (Some(evidence), Some(confidence)) => format!("{evidence}/{confidence}"),
        (Some(evidence), None) => evidence,
        (None, Some(confidence)) => confidence,
        (None, None) => "-".to_string(),
    }
}

fn load_model_interactive(config: &config::AppConfig, model: &str) -> Result<String> {
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
        .unwrap_or(&plan.backend)
        .to_string())
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
                    session.messages.clear();
                    if let Some(model) = choose_model(config, true)? {
                        load_model_for_chat(
                            config,
                            &mut session,
                            model.to_string_lossy().as_ref(),
                        )?;
                    } else {
                        session.backend = backend;
                        print_chat_header(&session);
                    }
                }
            }
            "/model" => {
                if let Some(model) = choose_model(config, true)? {
                    load_model_for_chat(config, &mut session, model.to_string_lossy().as_ref())?;
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

fn load_model_for_chat(
    config: &config::AppConfig,
    session: &mut ChatSession,
    model: &str,
) -> Result<()> {
    match load_model_interactive(config, model) {
        Ok(loaded) => {
            session.backend = loaded;
            session.messages.clear();
        }
        Err(error) => {
            notice(&format!("Model load failed: {error}"), NoticeKind::Warning);
            notice(
                "Still in chat. Use /model to pick another model.",
                NoticeKind::Warning,
            );
        }
    }
    print_chat_header(session);
    Ok(())
}

fn send_chat_message(
    config: &config::AppConfig,
    session: &mut ChatSession,
    message: &str,
) -> Result<()> {
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
    println!();
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

#[allow(dead_code)]
fn _loaded_services() -> Vec<serve_state::ServePidInfo> {
    serve_state::list_serve_pid_infos().unwrap_or_default()
}
