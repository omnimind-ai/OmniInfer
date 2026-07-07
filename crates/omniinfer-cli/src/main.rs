use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::{Shell, generate};
use std::time::Duration;

use omniinfer_core::{config, gateway_auth, http_client, local_state, paths, serve_state};

mod advisor;
mod backend_commands;
mod chat;
mod cli;
mod cloudflare;
mod gateway;
mod local_gateway;
mod model_commands;
mod serve;
mod tui;

pub(crate) use backend_commands::{
    print_backend_list, rust_backend_payload, select_backend, select_backend_for_config,
    select_backend_for_config_with_autostart,
};
use chat::{print_chat, print_chat_performance};
use cli::*;
pub(crate) use local_gateway::{
    ensure_local_gateway_running, get_local_json, get_local_json_for_config, parse_http_error_body,
    post_local_json, post_local_json_for_config, post_local_json_for_config_with_autostart,
    post_local_model_load_for_config_with_autostart, wait_for_gateway_ready,
};
pub(crate) use model_commands::{
    load_model, load_model_with_request_for_config,
    load_model_with_request_for_config_and_autostart, print_model_list, print_model_loaded,
};
pub(crate) use serve::{
    can_serve_locally, detach_child_process, expand_home_path, hide_child_window,
    parse_admin_api_keys, print_serve_status, serve_orchestrated, should_run_server_tui,
    stop_serve,
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.as_ref() {
        None => tui::run()?,
        Some(Command::Status) => print_status(),
        Some(Command::Completion { shell }) => print_completion(shell.clone()),
        Some(Command::Gateway(args)) => run_gateway_command(args)?,
        Some(command) => run_ported_command(command)?,
    }
    Ok(())
}

fn run_ported_command(command: &Command) -> Result<()> {
    match command {
        Command::Backend {
            command: BackendCommand::List { scope },
        } => print_backend_list(scope.clone()),
        Command::Backend {
            command: BackendCommand::Select { backend },
        } => select_backend(backend),
        Command::Backend {
            command: BackendCommand::Stop,
        } => stop_backend(),
        Command::Build { .. } if !source_build_scripts_available() => {
            anyhow::bail!(
                "Backend builds are only available from a source checkout, not packaged releases."
            )
        }
        Command::Ps { json } => print_ps(*json),
        Command::Model {
            command: ModelCommand::List { all, best },
        } => print_model_list(*all, *best),
        Command::Model {
            command: ModelCommand::Load(args),
        }
        | Command::Load(args) => load_model(args),
        Command::Advisor {
            command: AdvisorCommand::System { json },
        } => print_advisor_system(*json),
        Command::Advisor {
            command:
                AdvisorCommand::Inspect {
                    model,
                    mmproj,
                    json,
                },
        } => print_advisor_inspect(model, mmproj.as_deref(), *json),
        Command::Advisor {
            command:
                AdvisorCommand::Fit {
                    model,
                    mmproj,
                    ctx_size,
                    backend,
                    json,
                },
        } => print_advisor_fit(
            model,
            mmproj.as_deref(),
            *ctx_size,
            backend.as_deref(),
            *json,
        ),
        Command::Advisor {
            command:
                AdvisorCommand::Plan {
                    model,
                    mmproj,
                    ctx_size,
                    gpu_vram,
                    ram,
                    cpu_cores,
                    json,
                },
        } => print_advisor_plan(
            model,
            mmproj.as_deref(),
            *ctx_size,
            *gpu_vram,
            *ram,
            *cpu_cores,
            *json,
        ),
        Command::Advisor {
            command:
                AdvisorCommand::Recommend {
                    task,
                    limit,
                    ctx_size,
                    json,
                },
        } => print_advisor_recommend(task.as_deref(), *limit, *ctx_size, *json),
        Command::Thinking {
            command: ThinkingCommand::Show,
        } => {
            print_thinking_show();
            Ok(())
        }
        Command::Thinking {
            command: ThinkingCommand::Set { mode },
        } => print_thinking_set(mode.clone()),
        Command::Chat(args) => print_chat(args),
        Command::Shutdown => shutdown_service(),
        Command::Serve(ServeArgs {
            command: Some(ServeCommand::Status { port }),
            ..
        }) => {
            print_serve_status(*port);
            Ok(())
        }
        Command::Serve(ServeArgs {
            command: Some(ServeCommand::Stop { port }),
            ..
        }) => stop_serve(*port),
        Command::Serve(args) if can_serve_locally(args) && should_run_server_tui(args) => {
            tui::run_server(args)
        }
        Command::Serve(args) if can_serve_locally(args) => serve_orchestrated(args),
        Command::Gateway(args) => run_gateway_command(args),
        other => unsupported_rust_command(other),
    }
}

fn run_gateway_command(args: &GatewayArgs) -> Result<()> {
    gateway::run_gateway_blocking(gateway::GatewayConfig {
        listen_host: args.host.clone(),
        listen_port: args.port,
        access_policy: gateway_auth::GatewayAccessPolicy {
            api_key: args.api_key.clone().unwrap_or_default(),
            admin_api_key: args.admin_api_key.clone().unwrap_or_default(),
            admin_api_keys: parse_admin_api_keys(args.admin_api_keys.as_deref())?,
            allow_insecure_lan: args.allow_insecure_lan,
            allow_remote_management: args.allow_remote_management,
            trust_proxy_headers: args.trust_proxy_headers,
        },
        public_model_root: args.public_model_root.as_deref().map(expand_home_path),
    })
}

fn print_status() {
    let config = config::load_app_config().unwrap_or_default();
    let state = local_state::load_state().unwrap_or_default();
    println!("OmniInfer Rust Status");
    println!("Repo root: {}", paths::repo_root().display());
    println!("Local service: {}", service_status_line(&config));
    println!(
        "Selected backend: {}",
        state.selected_backend.as_deref().unwrap_or("not selected")
    );
    if let Some(model) = state.selected_model {
        println!("Selected model: {}", model.model);
        println!(
            "Selected mmproj: {}",
            model.mmproj.as_deref().unwrap_or("not selected")
        );
        println!(
            "Selected ctx-size: {}",
            model
                .ctx_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "not selected".to_string())
        );
    } else {
        println!("Selected model: not selected");
    }
}

fn service_status_line(config: &config::AppConfig) -> String {
    let url = format!("{}/omni/state", config.service_base_url());
    match http_client::get_json(&url, Duration::from_millis(600)) {
        Ok(response) if response.status == 200 => {
            let backend_ready = response
                .body
                .get("backend_ready")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let model = response
                .body
                .get("model")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or("not loaded");
            format!(
                "running (backend_ready={}, model={})",
                yes_no(backend_ready),
                model
            )
        }
        Ok(response) => format!("unhealthy (HTTP {})", response.status),
        Err(_) => "not running".to_string(),
    }
}

pub(crate) fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn print_completion(shell: CompletionShell) {
    let mut command = Cli::command();
    match shell {
        CompletionShell::Bash => generate(
            Shell::Bash,
            &mut command,
            "omniinfer-rs",
            &mut std::io::stdout(),
        ),
    }
}

fn source_build_scripts_available() -> bool {
    paths::repo_root()
        .join("scripts")
        .join("platforms")
        .is_dir()
}

fn print_advisor_system(json_output: bool) -> Result<()> {
    let backends = rust_backend_payload(BackendScope::All);
    let payload = advisor::system_payload(backends);
    advisor::print_system(&payload, json_output)
}

fn print_advisor_inspect(model: &str, mmproj: Option<&str>, json_output: bool) -> Result<()> {
    let payload = advisor::inspect_payload(model, mmproj, None)?;
    advisor::print_inspect(&payload, json_output)
}

fn print_advisor_fit(
    model: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
    backend: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let backends = rust_backend_payload(BackendScope::All);
    let payload = advisor::fit_payload(model, mmproj, ctx_size, backend, backends)?;
    advisor::print_fit(&payload, json_output)
}

#[allow(clippy::too_many_arguments)]
fn print_advisor_plan(
    model: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
    gpu_vram_gib: Option<f64>,
    ram_gib: Option<f64>,
    cpu_cores: Option<u32>,
    json_output: bool,
) -> Result<()> {
    let backends = rust_backend_payload(BackendScope::All);
    let payload = advisor::plan_payload(
        model,
        mmproj,
        ctx_size,
        gpu_vram_gib,
        ram_gib,
        cpu_cores,
        backends,
    )?;
    advisor::print_plan(&payload, json_output)
}

fn print_advisor_recommend(
    task: Option<&str>,
    limit: u32,
    ctx_size: Option<u32>,
    json_output: bool,
) -> Result<()> {
    let backends = rust_backend_payload(BackendScope::All);
    let payload = advisor::recommend_payload(task, limit, ctx_size, backends);
    advisor::print_recommend(&payload, json_output)
}

fn print_thinking_show() {
    let config = config::load_app_config().unwrap_or_default();
    println!(
        "Default thinking: {}",
        match config.default_thinking.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" | "enabled" => "on",
            _ => "off",
        }
    );
}

fn print_thinking_set(mode: ThinkingMode) -> Result<()> {
    let enabled = matches!(mode, ThinkingMode::On);
    let config = config::load_app_config().unwrap_or_default();
    let payload = match post_local_json_for_config_with_autostart(
        "/omni/thinking/select",
        &serde_json::json!({ "enabled": enabled }),
        Duration::from_secs(10),
        &config,
        false,
    ) {
        Ok(payload) => payload,
        Err(_) => {
            local_state::save_default_thinking(enabled)?;
            serde_json::json!({ "default_enabled": enabled })
        }
    };
    println!(
        "Default thinking set to: {}",
        if json_bool(&payload, "default_enabled").unwrap_or(false) {
            "on"
        } else {
            "off"
        }
    );
    Ok(())
}

pub(crate) fn shutdown_service() -> Result<()> {
    let config = config::load_app_config().unwrap_or_default();
    let url = format!("{}/omni/shutdown", config.service_base_url());
    match http_client::post_json(&url, &serde_json::json!({}), Duration::from_secs(10)) {
        Ok(response) if response.status < 400 => {
            println!("OmniInfer service stopped");
            Ok(())
        }
        _ => {
            println!("OmniInfer service is not running");
            Ok(())
        }
    }
}

fn stop_backend() -> Result<()> {
    post_local_json(
        "/omni/backend/stop",
        &serde_json::json!({}),
        Duration::from_secs(30),
    )?;
    println!("Current backend process stopped");
    Ok(())
}

fn print_ps(json_output: bool) -> Result<()> {
    let mut services = Vec::new();
    for info in serve_state::list_serve_pid_infos()? {
        let port = info.port.unwrap_or(9000);
        let mut config = config::load_app_config().unwrap_or_default();
        config.port = port;
        let state = http_client::get_json(
            &format!("{}/omni/state", config.service_base_url()),
            Duration::from_secs(2),
        )
        .ok()
        .filter(|response| response.status == 200)
        .map(|response| response.body);
        let service = serde_json::json!({
            "port": port,
            "pid": info.pid,
            "cloudflared_pid": info.cloudflared_pid,
            "status": "running",
            "backend": state.as_ref().and_then(|state| json_str(state, "backend")).or(info.backend.as_deref()).unwrap_or("unknown"),
            "backend_ready": state.as_ref().and_then(|state| json_bool(state, "backend_ready")).or(info.backend_ready).unwrap_or(false),
            "model": state.as_ref().and_then(|state| json_str(state, "model")).or(info.model.as_deref()).unwrap_or("not loaded"),
            "mmproj": state.as_ref().and_then(|state| json_str(state, "mmproj")).or(info.mmproj.as_deref()),
            "ctx_size": state.as_ref().and_then(|state| json_u64(state, "ctx_size")).or(info.ctx_size.map(u64::from)),
            "public_url": info.public_url,
            "openai_base_url": info.openai_base_url,
            "log": info.log,
        });
        services.push(service);
    }

    if json_output {
        println!("{}", serde_json::to_string_pretty(&services)?);
        return Ok(());
    }
    if services.is_empty() {
        println!("No running OmniInfer services found.");
        return Ok(());
    }
    println!("Running OmniInfer Services:");
    println!();
    for service in services {
        let port = json_u64(&service, "port")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!("  Port {port}:");
        println!(
            "    Status: {}",
            json_str(&service, "status").unwrap_or("running")
        );
        if let Some(pid) = json_u64(&service, "pid") {
            println!("    PID: {pid}");
        }
        if let Some(url) = json_str(&service, "openai_base_url") {
            println!("    OpenAI Base URL: {url}");
        }
        println!(
            "    Backend: {}",
            json_str(&service, "backend").unwrap_or("unknown")
        );
        println!(
            "    Backend Ready: {}",
            yes_no(json_bool(&service, "backend_ready").unwrap_or(false))
        );
        println!(
            "    Model: {}",
            json_str(&service, "model").unwrap_or("not loaded")
        );
        if let Some(mmproj) = json_str(&service, "mmproj") {
            println!("    MMProj: {mmproj}");
        }
        if let Some(ctx_size) = json_u64(&service, "ctx_size") {
            println!("    Context Size: {ctx_size}");
        }
        if let Some(log) = json_str(&service, "log") {
            println!("    Log: {log}");
        }
        println!();
    }
    Ok(())
}

pub(crate) fn json_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

pub(crate) fn json_bool(value: &serde_json::Value, key: &str) -> Option<bool> {
    value.get(key).and_then(serde_json::Value::as_bool)
}

pub(crate) fn json_u64(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(serde_json::Value::as_u64)
}

pub(crate) fn current_system_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "mac",
        "windows" => "windows",
        _ => "linux",
    }
}

fn unsupported_rust_command(command: &Command) -> Result<()> {
    anyhow::bail!(
        "command is not available in the Rust control plane: {command:?}. Python control-plane fallback has been removed."
    )
}
