use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{Shell, generate};
use rand::Rng;
use rand::distr::Alphanumeric;
use std::env;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use omniinfer_core::{
    backend_profiles, backend_registry, chat_stream, config, gateway_auth, http_client,
    local_state, model_catalog, model_load, paths, serve_state, version,
};

mod advisor;
mod gateway;
mod tui;

#[derive(Debug, Parser)]
#[command(name = "omniinfer-rs")]
#[command(version = version::VERSION)]
#[command(about = "Rust control-plane prototype for OmniInfer")]
#[command(long_about = "\
Rust control-plane prototype for OmniInfer.

This binary is intentionally experimental. It mirrors the Python OmniInfer CLI
surface while gateway, runtime, and TUI features are migrated incrementally.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Manage inference runtimes.
    Backend {
        #[command(subcommand)]
        command: BackendCommand,
    },
    /// Build a backend from this source checkout.
    Build {
        backend: String,
        #[arg(long)]
        prebuilt: bool,
        #[arg(long)]
        from_source: bool,
    },
    /// Show current status.
    Status,
    /// List running OmniInfer services.
    Ps {
        #[arg(long)]
        json: bool,
    },
    /// Discover and load models.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Load a model.
    Load(ModelLoadArgs),
    /// Inspect hardware, estimate fit, and plan deployments.
    Advisor {
        #[command(subcommand)]
        command: AdvisorCommand,
    },
    /// Manage the default thinking mode.
    Thinking {
        #[command(subcommand)]
        command: ThinkingCommand,
    },
    /// Run inference on the loaded model.
    Chat(ChatArgs),
    /// Stop the OmniInfer service.
    Shutdown,
    /// Start and manage the OmniInfer gateway.
    Serve(ServeArgs),
    /// Print shell completion.
    Completion { shell: CompletionShell },
    #[command(hide = true)]
    Gateway(GatewayArgs),
}

#[derive(Debug, Subcommand)]
enum BackendCommand {
    /// List backends available on this system.
    List {
        #[arg(long, default_value = "compatible")]
        scope: BackendScope,
    },
    /// Select a backend.
    Select { backend: String },
    /// Stop the current backend process.
    Stop,
}

#[derive(Debug, Clone, ValueEnum)]
enum BackendScope {
    Installed,
    Compatible,
    All,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// List supported models.
    List {
        #[arg(long)]
        all: bool,
        #[arg(long)]
        best: bool,
    },
    /// Load a model.
    Load(ModelLoadArgs),
}

#[derive(Debug, Args)]
struct ModelLoadArgs {
    #[arg(short = 'm', long)]
    model: String,
    #[arg(long = "mmproj")]
    mmproj: Option<String>,
    #[arg(long)]
    ctx_size: Option<u32>,
    #[arg(long)]
    config: Option<String>,
    #[arg(long)]
    verbose: bool,
    #[arg(last = true, allow_hyphen_values = true)]
    backend_extra_args: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum AdvisorCommand {
    /// Inspect local hardware and OmniInfer runtimes.
    System {
        #[arg(long)]
        json: bool,
    },
    /// Inspect a model reference or local artifact.
    Inspect {
        model: String,
        #[arg(long = "mmproj")]
        mmproj: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Recommend a backend and launch shape for a model.
    Fit {
        model: String,
        #[arg(long = "mmproj")]
        mmproj: Option<String>,
        #[arg(long)]
        ctx_size: Option<u32>,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Estimate hardware requirements for a model.
    Plan {
        model: String,
        #[arg(long = "mmproj")]
        mmproj: Option<String>,
        #[arg(long)]
        ctx_size: Option<u32>,
        #[arg(long)]
        gpu_vram: Option<f64>,
        #[arg(long)]
        ram: Option<f64>,
        #[arg(long)]
        cpu_cores: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Recommend from locally managed model files.
    Recommend {
        #[arg(long)]
        task: Option<String>,
        #[arg(short = 'n', long, default_value_t = 5)]
        limit: u32,
        #[arg(long)]
        ctx_size: Option<u32>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ThinkingCommand {
    /// Show the default thinking state.
    Show,
    /// Set the default thinking state.
    Set { mode: ThinkingMode },
}

#[derive(Debug, Clone, ValueEnum)]
enum ThinkingMode {
    On,
    Off,
}

#[derive(Debug, Args)]
struct ChatArgs {
    prompt: Option<String>,
    #[arg(long)]
    message: Option<String>,
    #[arg(long, conflicts_with = "no_stream")]
    stream: bool,
    #[arg(long)]
    no_stream: bool,
    #[arg(long)]
    image: Option<String>,
    #[arg(long)]
    think: Option<ThinkingMode>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ServeArgs {
    #[command(subcommand)]
    command: Option<ServeCommand>,
    #[arg(short = 'm', long)]
    model: Option<String>,
    #[arg(long = "mmproj")]
    mmproj: Option<String>,
    #[arg(long)]
    ctx_size: Option<u32>,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long)]
    cloudflare: bool,
    #[arg(long)]
    cloudflared_path: Option<String>,
    #[arg(long)]
    cloudflare_no_print_key: bool,
    #[arg(long)]
    lan: bool,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    admin_api_key: Option<String>,
    #[arg(long, value_name = "ID:KEY[,ID:KEY...]")]
    admin_api_keys: Option<String>,
    #[arg(long)]
    allow_insecure_lan: bool,
    #[arg(long)]
    allow_remote_management: bool,
    #[arg(long)]
    behind_proxy: bool,
    #[arg(long)]
    public_model_root: Option<String>,
    #[arg(long)]
    detach: bool,
    #[arg(long)]
    smoke_test: bool,
    #[arg(long)]
    no_smoke_test: bool,
    #[arg(long, default_value_t = 9000)]
    port: u16,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    backend_host: Option<String>,
    #[arg(long)]
    backend_port: Option<u16>,
    #[arg(long)]
    default_backend: Option<String>,
    #[arg(long)]
    default_thinking: Option<ThinkingMode>,
    #[arg(long)]
    force_backend: Option<String>,
    #[arg(long)]
    window_mode: Option<WindowMode>,
    #[arg(long)]
    startup_timeout: Option<u32>,
    #[arg(long)]
    log_level: Option<LogLevel>,
    #[arg(long)]
    verbose: bool,
    #[arg(long)]
    debug_body: bool,
}

#[derive(Debug, Args)]
struct GatewayArgs {
    #[arg(long)]
    host: String,
    #[arg(long)]
    port: u16,
    #[arg(long)]
    upstream_host: String,
    #[arg(long)]
    upstream_port: u16,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    admin_api_key: Option<String>,
    #[arg(long, value_name = "ID:KEY[,ID:KEY...]")]
    admin_api_keys: Option<String>,
    #[arg(long)]
    allow_insecure_lan: bool,
    #[arg(long)]
    allow_remote_management: bool,
    #[arg(long)]
    trust_proxy_headers: bool,
    #[arg(long)]
    public_model_root: Option<String>,
}

#[derive(Debug, Clone, Subcommand)]
enum ServeCommand {
    /// Show service status for a port.
    Status {
        #[arg(long, default_value_t = 9000)]
        port: u16,
    },
    /// Stop service on a port.
    Stop {
        #[arg(long, default_value_t = 9000)]
        port: u16,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum CompletionShell {
    Bash,
}

#[derive(Debug, Clone, ValueEnum)]
enum WindowMode {
    Visible,
    Hidden,
}

#[derive(Debug, Clone, ValueEnum)]
enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

fn main() -> Result<()> {
    if should_force_python() {
        return exec_python(env::args().skip(1));
    }
    let cli = Cli::parse();
    match cli.command.as_ref() {
        None => tui::run()?,
        Some(Command::Status) => print_status(),
        Some(Command::Completion { shell }) => print_completion(shell.clone()),
        Some(Command::Gateway(args)) => run_gateway_command(args)?,
        Some(command) => {
            if let Err(error) = run_ported_command(command) {
                if should_strict_rust() {
                    return Err(error);
                }
                return exec_python(env::args().skip(1));
            }
        }
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
        other => fallback_to_python(other),
    }
}

fn run_gateway_command(args: &GatewayArgs) -> Result<()> {
    gateway::run_gateway_blocking(gateway::GatewayConfig {
        listen_host: args.host.clone(),
        listen_port: args.port,
        upstream_host: args.upstream_host.clone(),
        upstream_port: args.upstream_port,
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

fn yes_no(value: bool) -> &'static str {
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

fn print_backend_list(scope: BackendScope) -> Result<()> {
    let payload = rust_backend_payload(scope.clone());
    let rows = payload
        .get("data")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if rows.is_empty() {
        anyhow::bail!("No backends are available on this system.");
    }

    let title = match scope {
        BackendScope::Compatible => "Compatible backends",
        BackendScope::Installed => "Installed backends",
        BackendScope::All => "Available backends",
    };
    println!("{title}");
    let width = rows
        .iter()
        .filter_map(|item| json_str(item, "id"))
        .map(str::len)
        .chain(std::iter::once("Backend".len()))
        .max()
        .unwrap_or("Backend".len());
    println!("{:<width$}  Selected  Installed", "Backend");
    println!("{:<width$}  --------  ---------", "-".repeat(width));
    for item in rows {
        let backend = json_str(&item, "id").unwrap_or("");
        let selected = if json_bool(&item, "selected").unwrap_or(false) {
            "yes"
        } else {
            ""
        };
        let installed = if json_bool(&item, "binary_exists").unwrap_or(false) {
            "yes"
        } else {
            ""
        };
        println!("{backend:<width$}  {selected:<8}  {installed:<9}");
    }
    Ok(())
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

pub(crate) fn rust_backend_payload(scope: BackendScope) -> serde_json::Value {
    backend_registry::BackendRegistry::load_current().api_payload(match scope {
        BackendScope::Installed => backend_registry::BackendScope::Installed,
        BackendScope::Compatible => backend_registry::BackendScope::Compatible,
        BackendScope::All => backend_registry::BackendScope::All,
    })
}

fn select_backend(backend: &str) -> Result<()> {
    let config = config::load_app_config().unwrap_or_default();
    select_backend_for_config(backend, &config)
}

pub(crate) fn select_backend_for_config(backend: &str, config: &config::AppConfig) -> Result<()> {
    select_backend_for_config_with_autostart(backend, config, true)
}

fn select_backend_for_config_with_autostart(
    backend: &str,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<()> {
    let backends = rust_backend_payload(BackendScope::All);
    let rows = backends
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Unable to read backend list."))?;
    let backend_payload = rows
        .iter()
        .find(|item| json_str(item, "id") == Some(backend))
        .ok_or_else(|| {
            let available = rows
                .iter()
                .filter_map(|item| json_str(item, "id"))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::anyhow!("Unsupported backend: {backend}\nAvailable backends: {available}")
        })?;

    let _payload = post_local_json_for_config_with_autostart(
        "/omni/backend/select",
        &serde_json::json!({ "backend": backend }),
        Duration::from_secs(30),
        config,
        autostart,
    )?;
    local_state::save_selected_backend(backend)?;
    let profile = backend_profiles::ensure_backend_profile_template(backend_payload)?;
    println!("Selected backend: {backend}");
    if let Some(models_dir) = json_str(backend_payload, "models_dir") {
        println!("Models directory: {models_dir}");
    }
    println!(
        "Backend config: {} ({})",
        profile.path.display(),
        if profile.created {
            "created"
        } else {
            "already exists"
        }
    );
    Ok(())
}

fn print_model_list(all: bool, best: bool) -> Result<()> {
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

fn load_model(args: &ModelLoadArgs) -> Result<()> {
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

fn load_model_with_request(
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

fn load_model_with_request_for_config_and_autostart(
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

fn print_model_loaded(
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
        let suitable = if json_bool(quant_info, "suitable").unwrap_or(false) {
            "yes"
        } else {
            "no"
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
    let payload = post_local_json(
        "/omni/thinking/select",
        &serde_json::json!({ "enabled": enabled }),
        Duration::from_secs(10),
    )?;
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

fn stop_serve(port: u16) -> Result<()> {
    let mut config = config::load_app_config().unwrap_or_default();
    config.port = port;
    let info = serve_state::load_serve_pid_info(port).ok().flatten();
    let url = format!("{}/omni/shutdown", config.service_base_url());
    let stopped =
        match http_client::post_json(&url, &serde_json::json!({}), Duration::from_secs(10)) {
            Ok(response) => response.status < 400,
            Err(_) => false,
        };
    if let Some(pid) = info.and_then(|info| info.cloudflared_pid) {
        stop_process(pid);
    }
    let _ = serve_state::remove_serve_pid_info(port);
    if stopped {
        println!("OmniInfer service stopped on port {port}");
    } else {
        println!("OmniInfer service is not running on port {port}");
    }
    Ok(())
}

fn stop_process(pid: u32) {
    #[cfg(unix)]
    {
        let _ = ProcessCommand::new("kill").arg(pid.to_string()).status();
    }
    #[cfg(windows)]
    {
        let _ = ProcessCommand::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
    }
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

fn can_serve_locally(args: &ServeArgs) -> bool {
    args.command.is_none()
}

fn should_run_server_tui(args: &ServeArgs) -> bool {
    use std::io::IsTerminal;
    args.command.is_none()
        && !args.detach
        && args.model.is_none()
        && !env_flag("OMNIINFER_SERVE_DIRECT")
        && std::io::stdin().is_terminal()
        && std::io::stdout().is_terminal()
}

pub(crate) fn serve_orchestrated(args: &ServeArgs) -> Result<()> {
    if args.no_smoke_test && args.smoke_test {
        anyhow::bail!("Use either --smoke-test or --no-smoke-test, not both.");
    }
    validate_serve_remote_access_args(args)?;
    let restore_model = resolve_serve_restore_model(args);
    let mut config = config::load_app_config().unwrap_or_default();
    config.port = args.port;
    if let Some(host) = args
        .host
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        config.host = host.to_string();
    }
    if let Some(default_backend) = args
        .default_backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        config.default_backend = default_backend.to_string();
    }
    if let Some(default_thinking) = &args.default_thinking {
        config.default_thinking = match default_thinking {
            ThinkingMode::On => "on",
            ThinkingMode::Off => "off",
        }
        .to_string();
    }
    if let Some(window_mode) = &args.window_mode {
        config.window_mode = match window_mode {
            WindowMode::Visible => "visible",
            WindowMode::Hidden => "hidden",
        }
        .to_string();
    }
    if let Some(timeout) = args.startup_timeout {
        config.startup_timeout = f64::from(timeout);
    }

    if args.lan && args.host.is_none() {
        config.host = "0.0.0.0".to_string();
    } else if args.cloudflare && args.host.is_none() {
        config.host = "127.0.0.1".to_string();
    }

    let remote_bind = !args.cloudflare && !is_loopback_host(&config.host);
    let generate_session_key = args.cloudflare || (remote_bind && !args.allow_insecure_lan);
    let api_key = resolve_serve_api_key(args, generate_session_key)?;
    let admin_api_key = resolve_serve_admin_api_key(args)?;
    let admin_api_keys = resolve_serve_admin_api_keys(args)?;
    if remote_bind && api_key.is_none() && !args.allow_insecure_lan {
        anyhow::bail!(
            "Refusing to expose OmniInfer on a non-loopback host without an API key. Use --lan to generate a session key, --api-key/OMNIINFER_API_KEY to set one, or --allow-insecure-lan for trusted test networks."
        );
    }
    if args.behind_proxy && api_key.is_none() && !args.allow_insecure_lan {
        anyhow::bail!(
            "--behind-proxy exposes OmniInfer through trusted proxy headers and requires --api-key or OMNIINFER_API_KEY"
        );
    }
    if args.allow_remote_management && admin_api_key.is_none() && admin_api_keys.is_empty() {
        anyhow::bail!(
            "--allow-remote-management requires --admin-api-key, --admin-api-keys, OMNIINFER_ADMIN_API_KEY, or OMNIINFER_ADMIN_API_KEYS"
        );
    }
    let public_model_root = args.public_model_root.as_deref().map(expand_home_path);
    if args.allow_remote_management && public_model_root.is_none() {
        anyhow::bail!("--allow-remote-management requires --public-model-root");
    }
    if let Some(root) = public_model_root.as_ref()
        && !root.is_dir()
    {
        anyhow::bail!("public model root does not exist: {}", root.display());
    }
    let use_rust_gateway = env::var("OMNIINFER_RUST_DISABLE_GATEWAY_PROXY")
        .map(|value| value.trim() != "1")
        .unwrap_or(true);
    let use_python_compat_upstream = use_rust_gateway && should_use_python_compat_upstream(args)?;
    let public_config = config.clone();
    let mut upstream_config = config.clone();
    if use_rust_gateway {
        // Keep fallback proxy traffic off the public listener. In pure Rust mode no
        // upstream is started, but the address must still be distinct to avoid
        // accidental self-proxy recursion for unhandled endpoints.
        upstream_config.host = "127.0.0.1".to_string();
        upstream_config.port = choose_upstream_port(&public_config)?;
    }
    let log_path = paths::local_logs_dir().join(format!("serve-{}.log", public_config.port));
    println!("Starting OmniInfer service on port {}...", config.port);
    println!("Log: {}", log_path.display());
    let upstream_child = if use_python_compat_upstream || !use_rust_gateway {
        let child = start_serve_child(&upstream_config, args, &log_path, api_key.as_deref())?;
        wait_for_gateway_ready(&upstream_config)?;
        Some(child)
    } else {
        None
    };
    let rust_gateway = if use_rust_gateway {
        let child = start_rust_gateway_child(
            &public_config,
            &upstream_config,
            args,
            &log_path,
            api_key.as_deref(),
            admin_api_key.as_deref(),
            &admin_api_keys,
            public_model_root.as_deref(),
        )?;
        wait_for_gateway_ready(&public_config)?;
        Some(child)
    } else {
        None
    };
    let mut cloudflared_child = None;
    let mut public_url = None;
    if args.cloudflare {
        let cloudflared = resolve_cloudflared(args.cloudflared_path.as_deref())?;
        let local_url = format!("http://127.0.0.1:{}", config.port);
        let (child, url) =
            start_cloudflare_quick_tunnel(&cloudflared, &local_url, &log_path, args.detach)?;
        cloudflared_child = Some(child);
        public_url = Some(url);
    }
    let configure_result = (|| -> Result<()> {
        if let Some(backend) = args
            .backend
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            select_backend_for_config_with_autostart(backend, &public_config, false)?;
        }
        if let Some(model) = args
            .model
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| ServeModelRequest {
                model: value.to_string(),
                mmproj: args.mmproj.clone(),
                ctx_size: args.ctx_size,
                backend_port: args.backend_port,
                restored: false,
            })
            .or_else(|| restore_model.clone())
        {
            if model.restored {
                println!("Restoring last model: {}", model.model);
            }
            let request = model_load::ModelLoadRequest {
                model: model.model,
                mmproj: model.mmproj,
                ctx_size: model.ctx_size,
                backend_port: model.backend_port,
                config: None,
                backend_extra_args: Vec::new(),
            };
            let (response, plan) = load_model_with_request_for_config_and_autostart(
                &request,
                false,
                &public_config,
                false,
            )?;
            if plan.auto_selected {
                println!("Auto-selected backend: {}", plan.backend);
            }
            print_model_loaded(&response, &plan)?;
        }
        Ok(())
    })();
    match configure_result {
        Ok(_) => {}
        Err(error) => {
            if let Some(mut child) = cloudflared_child {
                let _ = child.kill();
                let _ = child.wait();
            }
            if let Some(rust_gateway) = &rust_gateway {
                stop_process(rust_gateway.id());
            }
            if upstream_child.is_some() {
                let _ = http_client::post_json(
                    &format!("{}/omni/shutdown", upstream_config.service_base_url()),
                    &serde_json::json!({}),
                    Duration::from_secs(10),
                );
            }
            return Err(error);
        }
    }
    let state = get_serve_health_state(&public_config)?;
    let mut smoke_text = None;
    let mut smoke_failed = false;
    if args.smoke_test {
        let local_base_url = format!("http://127.0.0.1:{}", public_config.port);
        match serve_smoke(&local_base_url, api_key.as_deref()) {
            Ok(local_text) => {
                if let Some(public_url) = public_url.as_deref() {
                    match serve_smoke_with_retry(public_url, api_key.as_deref()) {
                        Ok(public_text) => {
                            smoke_text =
                                Some(format!("local ok: {local_text}; public ok: {public_text}"));
                        }
                        Err(error) => {
                            let transient_public_error = is_transient_public_smoke_error(&error);
                            smoke_failed = !transient_public_error;
                            let public_status = if transient_public_error {
                                "public warning"
                            } else {
                                "public failed"
                            };
                            smoke_text =
                                Some(format!("local ok: {local_text}; {public_status}: {error}"));
                        }
                    }
                } else {
                    smoke_text = Some(local_text);
                }
            }
            Err(error) => {
                smoke_failed = true;
                smoke_text = Some(format!("local failed: {error}"));
            }
        }
    }
    serve_state::save_serve_pid_info(&serve_state::ServePidInfo {
        pid: Some(
            rust_gateway
                .as_ref()
                .map(std::process::Child::id)
                .or_else(|| upstream_child.as_ref().map(std::process::Child::id))
                .ok_or_else(|| anyhow::anyhow!("serve has no process pid to record"))?,
        ),
        cloudflared_pid: cloudflared_child.as_ref().map(std::process::Child::id),
        port: Some(public_config.port),
        log: Some(log_path.display().to_string()),
        public_url: public_url.clone(),
        openai_base_url: public_url
            .as_ref()
            .map(|url| format!("{}/v1", url.trim_end_matches('/'))),
        backend: json_str(&state, "backend").map(str::to_string),
        model: json_str(&state, "model").map(str::to_string),
        mmproj: json_str(&state, "mmproj").map(str::to_string),
        ctx_size: json_u64(&state, "ctx_size").and_then(|value| u32::try_from(value).ok()),
        backend_ready: json_bool(&state, "backend_ready"),
        backend_pid: json_u64(&state, "backend_pid").and_then(|value| u32::try_from(value).ok()),
        backend_port: json_u64(&state, "backend_port").and_then(|value| u16::try_from(value).ok()),
    })?;
    print_serve_ready(
        public_config.port,
        &state,
        public_url.as_deref(),
        &lan_base_urls(&public_config, args.lan),
        api_key.as_deref(),
        admin_api_key.as_deref(),
        public_model_root.as_deref(),
        args.allow_remote_management,
        !args.cloudflare_no_print_key,
        &log_path,
        smoke_text.as_deref(),
    );
    if smoke_failed {
        anyhow::bail!("smoke test failed");
    }
    if !args.detach {
        println!("Press Ctrl+C to stop.");
        let status = wait_for_foreground_service(
            rust_gateway,
            upstream_child,
            cloudflared_child,
            public_config.port,
        )?;
        if !status.success() {
            anyhow::bail!("OmniInfer service exited with status {status}");
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ServeModelRequest {
    model: String,
    mmproj: Option<String>,
    ctx_size: Option<u32>,
    backend_port: Option<u16>,
    restored: bool,
}

fn resolve_serve_restore_model(args: &ServeArgs) -> Option<ServeModelRequest> {
    if args
        .model
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return None;
    }
    if args.mmproj.is_some() || args.ctx_size.is_some() {
        return None;
    }
    if should_run_server_tui(args) {
        return None;
    }
    let selected = local_state::load_state().ok()?.selected_model?;
    if selected.model.trim().is_empty() {
        return None;
    }
    Some(ServeModelRequest {
        model: selected.model,
        mmproj: selected.mmproj,
        ctx_size: selected.ctx_size,
        backend_port: args.backend_port,
        restored: true,
    })
}

fn wait_for_foreground_service(
    rust_gateway: Option<std::process::Child>,
    mut upstream_child: Option<std::process::Child>,
    cloudflared_child: Option<std::process::Child>,
    port: u16,
) -> Result<std::process::ExitStatus> {
    let status = if let Some(mut rust_gateway) = rust_gateway {
        let status = rust_gateway.wait()?;
        if let Some(upstream_child) = upstream_child.as_mut() {
            let _ = upstream_child.kill();
            let _ = upstream_child.wait();
        }
        status
    } else {
        upstream_child
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("serve has no foreground process to wait for"))?
            .wait()?
    };
    if let Some(mut tunnel) = cloudflared_child {
        let _ = tunnel.kill();
        let _ = tunnel.wait();
    }
    let _ = serve_state::remove_serve_pid_info(port);
    Ok(status)
}

fn should_use_python_compat_upstream(args: &ServeArgs) -> Result<bool> {
    if env::var("OMNIINFER_RUST_FORCE_PYTHON_UPSTREAM")
        .map(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
    {
        return Ok(true);
    }
    let Some(backend_id) = resolve_serve_start_backend(args)? else {
        return Ok(false);
    };
    let registry = backend_registry::BackendRegistry::load_current();
    let Some(backend) = registry.get(&backend_id) else {
        return Ok(false);
    };
    Ok(backend.runtime_mode == "embedded")
}

fn resolve_serve_start_backend(args: &ServeArgs) -> Result<Option<String>> {
    if let Some(backend) = args
        .backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(Some(backend.to_string()));
    }
    if let Some(default_backend) = args
        .default_backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(Some(default_backend.to_string()));
    }
    if let Some(selected_backend) = local_state::load_state()
        .ok()
        .and_then(|state| state.selected_backend)
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(Some(selected_backend));
    }
    Ok(backend_registry::BackendRegistry::load_current()
        .api_payload(backend_registry::BackendScope::Installed)
        .get("recommended")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string))
}

fn choose_upstream_port(public_config: &config::AppConfig) -> Result<u16> {
    if public_config.port != 0 {
        for _ in 0..20 {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            if port != public_config.port {
                return Ok(port);
            }
        }
    }
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn validate_serve_remote_access_args(args: &ServeArgs) -> Result<()> {
    if args.cloudflare && !is_loopback_host(args.host.as_deref().unwrap_or("127.0.0.1")) {
        anyhow::bail!(
            "--cloudflare keeps OmniInfer on 127.0.0.1; do not combine it with a non-loopback --host."
        );
    }
    if args.cloudflare && args.allow_insecure_lan {
        anyhow::bail!(
            "--cloudflare requires an API key and cannot be combined with --allow-insecure-lan."
        );
    }
    if args.cloudflare && args.allow_remote_management {
        anyhow::bail!(
            "--cloudflare keeps /omni/* management endpoints local-only; do not use --allow-remote-management."
        );
    }
    if args.cloudflare && args.behind_proxy {
        anyhow::bail!(
            "--cloudflare already configures proxy headers; do not combine it with --behind-proxy."
        );
    }
    Ok(())
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host.trim(), "" | "127.0.0.1" | "localhost" | "::1")
}

fn resolve_serve_api_key(args: &ServeArgs, generate_session_key: bool) -> Result<Option<String>> {
    if generate_session_key {
        let config = config::load_app_config().unwrap_or_default();
        let configured = if let Some(value) = args
            .api_key
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            Some(value.to_string())
        } else if !config.api_key.trim().is_empty() {
            Some(config.api_key)
        } else if let Ok(value) = env::var("OMNIINFER_API_KEY") {
            Some(value.trim().to_string()).filter(|value| !value.is_empty())
        } else {
            None
        };

        if let Some(value) = configured {
            return if value.trim().eq_ignore_ascii_case("auto") {
                Ok(Some(generate_session_api_key()?))
            } else {
                Ok(Some(value))
            };
        }
        return Ok(Some(generate_session_api_key()?));
    }
    Ok(args
        .api_key
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string))
}

fn resolve_serve_admin_api_key(args: &ServeArgs) -> Result<Option<String>> {
    let value = args
        .admin_api_key
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            env::var("OMNIINFER_ADMIN_API_KEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        });
    match value.as_deref() {
        Some(value) if value.eq_ignore_ascii_case("auto") => Ok(Some(generate_session_api_key()?)),
        _ => Ok(value),
    }
}

fn resolve_serve_admin_api_keys(args: &ServeArgs) -> Result<Vec<gateway_auth::GatewayAdminApiKey>> {
    let raw = args
        .admin_api_keys
        .clone()
        .or_else(|| env::var("OMNIINFER_ADMIN_API_KEYS").ok());
    parse_admin_api_keys(raw.as_deref())
}

fn parse_admin_api_keys(raw: Option<&str>) -> Result<Vec<gateway_auth::GatewayAdminApiKey>> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    for item in raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let Some((id, key)) = item.split_once(':').or_else(|| item.split_once('=')) else {
            anyhow::bail!("invalid admin API key entry '{item}'; expected ID:KEY or ID=KEY");
        };
        let id = id.trim();
        let key = key.trim();
        if id.is_empty() || key.is_empty() {
            anyhow::bail!(
                "invalid admin API key entry '{item}'; admin id and key must be non-empty"
            );
        }
        entries.push(gateway_auth::GatewayAdminApiKey {
            id: id.to_string(),
            key: key.to_string(),
        });
    }
    Ok(entries)
}

fn lan_base_urls(config: &config::AppConfig, lan_enabled: bool) -> Vec<String> {
    if !lan_enabled {
        return Vec::new();
    }
    let host = config.host.trim();
    if !host.is_empty() && !matches!(host, "0.0.0.0" | "::") {
        return vec![format!(
            "http://{}:{}/v1",
            host.trim_matches(['[', ']']),
            config.port
        )];
    }
    detect_primary_lan_ipv4()
        .into_iter()
        .map(|ip| format!("http://{ip}:{}/v1", config.port))
        .collect()
}

fn detect_primary_lan_ipv4() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    let ip = addr.ip();
    if ip.is_loopback() || !ip.is_ipv4() {
        return None;
    }
    Some(ip.to_string())
}

fn generate_session_api_key() -> Result<String> {
    let token: String = rand::rng()
        .sample_iter(Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    Ok(format!("oi_{token}"))
}

fn resolve_cloudflared(explicit_path: Option<&str>) -> Result<PathBuf> {
    if let Some(path) = explicit_path.filter(|value| !value.trim().is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            anyhow::bail!("cloudflared was not found at {}", path.display());
        }
        return Ok(path);
    }

    let managed = managed_cloudflared_path();
    if managed.is_file() {
        return Ok(managed);
    }

    let mut command = ProcessCommand::new(python_executable());
    pass_rust_path_overrides(&mut command);
    let output = command
        .arg("-c")
        .arg("from service_core.remote_access import find_cloudflared; print(find_cloudflared())")
        .current_dir(paths::repo_root())
        .env("PYTHONPATH", paths::repo_root())
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        anyhow::bail!("failed to resolve cloudflared: {stderr}");
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        anyhow::bail!("cloudflared resolver returned an empty path");
    }
    Ok(PathBuf::from(path))
}

fn managed_cloudflared_path() -> PathBuf {
    let exe = if cfg!(windows) {
        "cloudflared.exe"
    } else {
        "cloudflared"
    };
    paths::local_dir()
        .join("tools")
        .join("cloudflared")
        .join(exe)
}

fn start_cloudflare_quick_tunnel(
    cloudflared: &Path,
    local_url: &str,
    log_path: &Path,
    detach: bool,
) -> Result<(std::process::Child, String)> {
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let mut command = ProcessCommand::new(cloudflared);
    command
        .args(["tunnel", "--url", local_url])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if detach {
        detach_child_process(&mut command);
    }
    let mut child = command.spawn()?;

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture cloudflared stdout"))?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture cloudflared stderr"))?;
    let (line_tx, line_rx) = mpsc::channel();
    spawn_cloudflared_reader(stdout_pipe, stdout, line_tx.clone());
    spawn_cloudflared_reader(stderr_pipe, stderr, line_tx);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut tail = Vec::new();
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!(
                "cloudflared exited before creating a Quick Tunnel with status {status}.{}",
                format_log_tail(&tail)
            );
        }
        match line_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                if tail.len() == 10 {
                    tail.remove(0);
                }
                tail.push(line.clone());
                if let Some(url) = parse_trycloudflare_url(&line) {
                    return Ok((child, url));
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    anyhow::bail!(
        "Timed out waiting for Cloudflare Quick Tunnel URL.{}",
        format_log_tail(&tail)
    )
}

fn spawn_cloudflared_reader<R: std::io::Read + Send + 'static>(
    stream: R,
    mut log: std::fs::File,
    line_tx: mpsc::Sender<String>,
) {
    thread::spawn(move || {
        let reader = BufReader::new(stream);
        for line in reader.lines().map_while(Result::ok) {
            use std::io::Write;
            let _ = writeln!(log, "{line}");
            let _ = log.flush();
            let _ = line_tx.send(line);
        }
    });
}

fn parse_trycloudflare_url(line: &str) -> Option<String> {
    line.split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | '(' | ')' | '[' | ']'))
        .find(|part| part.starts_with("https://") && part.contains(".trycloudflare.com"))
        .map(|part| {
            part.trim_end_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '/')
                .trim_end_matches('/')
                .to_string()
        })
}

fn format_log_tail(lines: &[String]) -> String {
    if lines.is_empty() {
        String::new()
    } else {
        format!("\ncloudflared log tail:\n{}", lines.join("\n"))
    }
}

fn start_serve_child(
    config: &config::AppConfig,
    args: &ServeArgs,
    log_path: &std::path::Path,
    resolved_api_key: Option<&str>,
) -> Result<std::process::Child> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let stderr = stdout.try_clone()?;
    let mut command = ProcessCommand::new(python_executable());
    pass_rust_path_overrides(&mut command);
    command
        .arg(paths::repo_root().join("omniinfer.py"))
        .arg("serve")
        .arg("--host")
        .arg(&config.host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--startup-timeout")
        .arg(format!("{:.0}", config.startup_timeout))
        .arg("--window-mode")
        .arg(&config.window_mode)
        .arg("--default-thinking")
        .arg(&config.default_thinking)
        .current_dir(paths::repo_root())
        .env("OMNIINFER_SERVE_DIRECT", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if !config.default_backend.trim().is_empty() {
        command
            .arg("--default-backend")
            .arg(&config.default_backend);
    }
    if let Some(api_key) = resolved_api_key.filter(|value| !value.trim().is_empty()) {
        command.arg("--api-key").arg(api_key);
        command.env("OMNIINFER_API_KEY", api_key);
    }
    if args.allow_insecure_lan {
        command.arg("--allow-insecure-lan");
    }
    if args.allow_remote_management {
        command.arg("--allow-remote-management");
    }
    if let Some(backend_host) = args
        .backend_host
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.arg("--backend-host").arg(backend_host);
    }
    if let Some(backend_port) = args.backend_port {
        command.arg("--backend-port").arg(backend_port.to_string());
    }
    if let Some(force_backend) = args
        .force_backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.arg("--force-backend").arg(force_backend);
    }
    if let Some(log_level) = &args.log_level {
        command.arg("--log-level").arg(match log_level {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
        });
    }
    if args.verbose {
        command.arg("--verbose");
    }
    if args.debug_body {
        command.arg("--debug-body");
    }
    if args.detach {
        detach_child_process(&mut command);
    }
    Ok(command.spawn()?)
}

fn start_rust_gateway_child(
    public_config: &config::AppConfig,
    upstream_config: &config::AppConfig,
    args: &ServeArgs,
    log_path: &std::path::Path,
    api_key: Option<&str>,
    admin_api_key: Option<&str>,
    admin_api_keys: &[gateway_auth::GatewayAdminApiKey],
    public_model_root: Option<&std::path::Path>,
) -> Result<std::process::Child> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let stderr = stdout.try_clone()?;
    let mut command = ProcessCommand::new(std::env::current_exe()?);
    command
        .arg("gateway")
        .arg("--host")
        .arg(&public_config.host)
        .arg("--port")
        .arg(public_config.port.to_string())
        .arg("--upstream-host")
        .arg(upstream_config.service_host())
        .arg("--upstream-port")
        .arg(upstream_config.port.to_string())
        .current_dir(paths::repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if let Some(api_key) = api_key.filter(|value| !value.trim().is_empty()) {
        command.arg("--api-key").arg(api_key);
    }
    if let Some(admin_api_key) = admin_api_key.filter(|value| !value.trim().is_empty()) {
        command.arg("--admin-api-key").arg(admin_api_key);
    }
    if !admin_api_keys.is_empty() {
        let raw = admin_api_keys
            .iter()
            .map(|entry| format!("{}:{}", entry.id, entry.key))
            .collect::<Vec<_>>()
            .join(",");
        command.arg("--admin-api-keys").arg(raw);
    }
    if let Some(public_model_root) = public_model_root {
        command.arg("--public-model-root").arg(public_model_root);
    }
    if args.allow_insecure_lan {
        command.arg("--allow-insecure-lan");
    }
    if args.allow_remote_management {
        command.arg("--allow-remote-management");
    }
    if args.cloudflare || args.behind_proxy {
        command.arg("--trust-proxy-headers");
    }
    if args.detach {
        detach_child_process(&mut command);
    }
    Ok(command.spawn()?)
}

fn detach_child_process(command: &mut ProcessCommand) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
}

fn expand_home_path(value: &str) -> PathBuf {
    let path = PathBuf::from(value.trim());
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path
}

fn get_serve_health_state(config: &config::AppConfig) -> Result<serde_json::Value> {
    let url = format!("{}/health?deep=true", config.service_base_url());
    let response = http_client::get_json(&url, Duration::from_secs(10))?;
    if response.status >= 400 {
        anyhow::bail!(
            "GET /health?deep=true failed with status {}",
            response.status
        );
    }
    Ok(response.body.get("omni").cloned().unwrap_or(response.body))
}

fn print_serve_ready(
    port: u16,
    state: &serde_json::Value,
    public_url: Option<&str>,
    lan_base_urls: &[String],
    api_key: Option<&str>,
    admin_api_key: Option<&str>,
    public_model_root: Option<&std::path::Path>,
    remote_management: bool,
    print_api_key: bool,
    log_path: &std::path::Path,
    smoke_text: Option<&str>,
) {
    println!();
    println!("OmniInfer service is ready");
    if let Some(public_url) = public_url {
        println!("OpenAI Base URL: {}/v1", public_url.trim_end_matches('/'));
        println!("Health URL: {}/health", public_url.trim_end_matches('/'));
    }
    for lan_base_url in lan_base_urls {
        println!("LAN Base URL: {lan_base_url}");
    }
    println!("Local Base URL: http://127.0.0.1:{port}/v1");
    if let Some(api_key) = api_key.filter(|_| print_api_key) {
        println!("API Key: {api_key}");
    }
    if remote_management {
        println!("Remote management: enabled");
        if let Some(admin_api_key) = admin_api_key.filter(|_| print_api_key) {
            println!("Admin API Key: {admin_api_key}");
        }
        if let Some(public_model_root) = public_model_root {
            println!("Public model root: {}", public_model_root.display());
            match omniinfer_core::public_models::list_public_models(Some(public_model_root)) {
                Ok(models) => println!("Public models: {}", models.len()),
                Err(error) => println!("Public models: unavailable ({error})"),
            }
        }
    }
    println!("Backend: {}", json_str(state, "backend").unwrap_or("-"));
    println!(
        "Backend ready: {}",
        yes_no(json_bool(state, "backend_ready").unwrap_or(false))
    );
    println!("Model: {}", json_str(state, "model").unwrap_or("-"));
    println!("mmproj: {}", json_str(state, "mmproj").unwrap_or("-"));
    println!(
        "ctx-size: {}",
        json_u64(state, "ctx_size")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    if let Some(smoke_text) = smoke_text {
        println!("Smoke: {smoke_text}");
    }
    println!("Log: {}", log_path.display());
    println!("Stop: ./omniinfer serve stop --port {port}");
    let remote_base_url = public_url
        .map(|url| format!("{}/v1", url.trim_end_matches('/')))
        .or_else(|| lan_base_urls.first().cloned());
    if let Some(remote_base_url) = remote_base_url {
        println!("Curl:");
        let auth = if let Some(api_key) = api_key.filter(|_| print_api_key) {
            format!(" -H 'Authorization: Bearer {api_key}'")
        } else {
            String::new()
        };
        println!(
            "  curl -sS{} -H 'Content-Type: application/json' {}/chat/completions -d '{{\"model\":\"omniinfer\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":false}}'",
            auth, remote_base_url
        );
        if remote_management {
            let management_base = remote_base_url.trim_end_matches("/v1");
            let admin_auth = if let Some(admin_api_key) = admin_api_key.filter(|_| print_api_key) {
                format!(" -H 'Authorization: Bearer {admin_api_key}'")
            } else {
                String::new()
            };
            println!(
                "  curl -sS{} {}/omni/public-models",
                admin_auth, management_base
            );
            println!(
                "  curl -sS{} -H 'Content-Type: application/json' {}/omni/model/select -d '{{\"model\":\"qwen3.5-4b-q4_k_m\"}}'",
                admin_auth, management_base
            );
        }
    }
}

fn serve_smoke(base_url: &str, api_key: Option<&str>) -> Result<String> {
    let mut payload = serde_json::Map::new();
    payload.insert("model".to_string(), serde_json::json!("omniinfer"));
    payload.insert(
        "messages".to_string(),
        serde_json::json!([{ "role": "user", "content": "Hello" }]),
    );
    payload.insert("temperature".to_string(), serde_json::json!(0));
    payload.insert("max_tokens".to_string(), serde_json::json!(16));
    payload.insert("stream".to_string(), serde_json::json!(false));

    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let response = http_client::post_json_with_bearer(
        &url,
        &serde_json::Value::Object(payload),
        api_key,
        Duration::from_secs(120),
    )?;
    if response.status >= 400 {
        anyhow::bail!(
            "POST /v1/chat/completions failed with status {}",
            response.status
        );
    }
    response
        .body
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("Smoke test returned an empty response."))
}

fn serve_smoke_with_retry(base_url: &str, api_key: Option<&str>) -> Result<String> {
    let deadline = Instant::now() + public_smoke_retry_duration();
    loop {
        match serve_smoke(base_url, api_key) {
            Ok(text) => return Ok(text),
            Err(error) => {
                if !is_transient_public_smoke_error(&error) || Instant::now() >= deadline {
                    return Err(error);
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}

fn public_smoke_retry_duration() -> Duration {
    env::var("OMNIINFER_RUST_PUBLIC_SMOKE_RETRY_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(45))
}

fn is_transient_public_smoke_error(error: &anyhow::Error) -> bool {
    let text = error.to_string().to_ascii_lowercase();
    text.contains("failed to lookup address")
        || text.contains("name or service not known")
        || text.contains("temporary failure in name resolution")
        || text.contains("connection refused")
        || text.contains("connection reset")
        || text.contains("timed out")
        || text.contains("operation timed out")
        || text.contains("http status: 520")
        || text.contains("http status: 521")
        || text.contains("http status: 522")
        || text.contains("http status: 523")
        || text.contains("http status: 524")
        || text.contains("http status: 530")
}

fn print_chat(args: &ChatArgs) -> Result<()> {
    let mut payload = build_chat_payload(args)?;
    if args.no_stream {
        payload.insert("stream".to_string(), serde_json::json!(false));
        let response = post_local_json(
            "/v1/chat/completions",
            &serde_json::Value::Object(payload),
            Duration::from_secs(600),
        )?;
        print_chat_response(&response);
        return Ok(());
    }
    payload.insert("stream".to_string(), serde_json::json!(true));
    payload.insert(
        "stream_options".to_string(),
        serde_json::json!({ "include_usage": true }),
    );
    print_chat_stream(&serde_json::Value::Object(payload))
}

fn build_chat_payload(args: &ChatArgs) -> Result<serde_json::Map<String, serde_json::Value>> {
    if args.message.is_some() && args.prompt.is_some() {
        anyhow::bail!("Use either positional prompt or --message, not both.");
    }
    let message = args
        .message
        .as_deref()
        .or(args.prompt.as_deref())
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("Please provide a message, for example: omniinfer chat \"Hello\".")
        })?;
    let state = get_local_json("/omni/state", Duration::from_secs(10))?;
    if json_str(&state, "model").is_none() {
        anyhow::bail!("No model is currently loaded.\nRun `omniinfer load -m <model>` first.");
    }

    let mut payload = state
        .get("request_defaults")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    payload.insert("messages".to_string(), build_chat_messages(message, args)?);
    payload.insert(
        "temperature".to_string(),
        serde_json::json!(args.temperature.unwrap_or_else(|| {
            payload
                .get("temperature")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.2) as f32
        })),
    );
    payload.insert(
        "max_tokens".to_string(),
        serde_json::json!(args.max_tokens.unwrap_or_else(|| {
            payload
                .get("max_tokens")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(2048)
        })),
    );
    payload.insert("stream".to_string(), serde_json::json!(false));
    if let Some(think) = &args.think {
        payload.insert(
            "think".to_string(),
            serde_json::json!(matches!(think, ThinkingMode::On)),
        );
    }
    Ok(payload)
}

pub(crate) fn build_chat_messages(message: &str, args: &ChatArgs) -> Result<serde_json::Value> {
    let Some(image) = args
        .image
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(serde_json::json!([{ "role": "user", "content": message }]));
    };
    let image_path = std::path::PathBuf::from(image);
    if !image_path.is_file() {
        anyhow::bail!("image file does not exist: {}", image_path.display());
    }
    let bytes = std::fs::read(&image_path)?;
    let image_b64 = BASE64_STANDARD.encode(bytes);
    let mime = image_mime_type(&image_path);
    Ok(serde_json::json!([
        {
            "role": "user",
            "content": [
                { "type": "text", "text": message },
                { "type": "image_url", "image_url": { "url": format!("data:{mime};base64,{image_b64}") } }
            ]
        }
    ]))
}

fn image_mime_type(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

fn print_chat_stream(payload: &serde_json::Value) -> Result<()> {
    let config = config::load_app_config().unwrap_or_default();
    ensure_local_gateway_running(&config)?;
    let url = format!("{}/v1/chat/completions", config.service_base_url());
    println!("Response");
    let mut filter = chat_stream::StreamPrefixFilter::new();
    let mut final_payload = None;
    let response = http_client::post_streaming_lines(
        &url,
        payload,
        "text/event-stream, application/json",
        Duration::from_secs(3600),
        |line| {
            let chunks = chat_stream::parse_chat_stream_line(line).unwrap_or_default();
            for chunk in chunks {
                match chunk {
                    chat_stream::ChatStreamChunk::Text(text) => {
                        if let Some(text) = filter.push(&text)
                            && !text.is_empty()
                        {
                            print!("{text}");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                        }
                    }
                    chat_stream::ChatStreamChunk::Reasoning(_) => {}
                    chat_stream::ChatStreamChunk::Final(payload) => {
                        final_payload = Some(payload);
                    }
                }
            }
        },
    )?;
    if response.status >= 400 {
        let message = parse_http_error_body(&response.body);
        anyhow::bail!(
            "Streaming inference failed with status {}: {}",
            response.status,
            message
        );
    }
    if let Some(text) = filter.finish()
        && !text.is_empty()
    {
        print!("{text}");
    }
    println!();
    if let Some(final_payload) = final_payload {
        print_chat_performance(&final_payload);
    }
    Ok(())
}

fn print_chat_response(response: &serde_json::Value) {
    let text = response
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"));
    println!("Response");
    match text {
        Some(serde_json::Value::String(content)) => println!("{content}"),
        Some(other) => println!("{other}"),
        None => println!("{response}"),
    }
    if response.get("usage").is_some() || response.get("timings").is_some() {
        println!();
        print_chat_performance(response);
    }
}

pub(crate) fn print_chat_performance(response: &serde_json::Value) {
    println!("Performance");
    println!("  Model: {}", json_str(response, "model").unwrap_or("-"));
    if let Some(usage) = response.get("usage") {
        let prompt = json_u64(usage, "prompt_tokens")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let completion = json_u64(usage, "completion_tokens")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let total = json_u64(usage, "total_tokens")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!("  Tokens: prompt={prompt}, completion={completion}, total={total}");
    }
}

fn print_serve_status(port: u16) {
    let mut config = config::load_app_config().unwrap_or_default();
    config.port = port;
    println!("OmniInfer Serve Status");
    println!("Port: {port}");
    match serve_state::load_serve_pid_info(port) {
        Ok(Some(info)) => {
            if let Some(pid) = info.pid {
                println!("PID: {pid}");
            }
            if let Some(public_url) = info
                .public_url
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                println!("OpenAI Base URL: {}/v1", public_url.trim_end_matches('/'));
            }
            if let Some(log) = info.log.as_deref().filter(|value| !value.trim().is_empty()) {
                println!("Log: {log}");
            }
        }
        Ok(None) => {}
        Err(error) => println!("Serve metadata: unavailable ({error})"),
    }

    let url = format!("{}/health?deep=true", config.service_base_url());
    match http_client::get_json(&url, Duration::from_secs(2)) {
        Ok(response) if response.status == 200 => {
            let state = response.body.get("omni").unwrap_or(&response.body);
            println!("Backend: {}", json_str(state, "backend").unwrap_or("-"));
            println!(
                "Backend ready: {}",
                yes_no(json_bool(state, "backend_ready").unwrap_or(false))
            );
            println!("Model: {}", json_str(state, "model").unwrap_or("-"));
            println!(
                "ctx-size: {}",
                json_u64(state, "ctx_size")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
        Ok(response) => println!("Service: unhealthy (HTTP {})", response.status),
        Err(error) => println!("OmniInfer service is not running on port {port}: {error}"),
    }
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

fn get_local_json(endpoint: &str, timeout: Duration) -> Result<serde_json::Value> {
    let config = config::load_app_config().unwrap_or_default();
    get_local_json_for_config(endpoint, timeout, &config)
}

pub(crate) fn get_local_json_for_config(
    endpoint: &str,
    timeout: Duration,
    config: &config::AppConfig,
) -> Result<serde_json::Value> {
    ensure_local_gateway_running(config)?;
    let url = format!("{}{}", config.service_base_url(), endpoint);
    let response = http_client::get_json(&url, timeout)?;
    if response.status >= 400 {
        anyhow::bail!("GET {endpoint} failed with status {}", response.status);
    }
    Ok(response.body)
}

fn post_local_json(
    endpoint: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let config = config::load_app_config().unwrap_or_default();
    post_local_json_for_config(endpoint, body, timeout, &config)
}

pub(crate) fn post_local_json_for_config(
    endpoint: &str,
    body: &serde_json::Value,
    timeout: Duration,
    config: &config::AppConfig,
) -> Result<serde_json::Value> {
    post_local_json_for_config_with_autostart(endpoint, body, timeout, config, true)
}

fn post_local_json_for_config_with_autostart(
    endpoint: &str,
    body: &serde_json::Value,
    timeout: Duration,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<serde_json::Value> {
    if autostart {
        ensure_local_gateway_running(config)?;
    }
    let url = format!("{}{}", config.service_base_url(), endpoint);
    let response = http_client::post_json(&url, body, timeout)?;
    if response.status >= 400 {
        anyhow::bail!("POST {endpoint} failed with status {}", response.status);
    }
    Ok(response.body)
}

fn post_local_model_load_for_config_with_autostart(
    body: &serde_json::Value,
    verbose: bool,
    timeout: Duration,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<serde_json::Value> {
    if autostart {
        ensure_local_gateway_running(config)?;
    }
    let url = format!("{}/omni/model/select", config.service_base_url());
    let response = http_client::post_streaming_lines(
        &url,
        body,
        "text/event-stream, application/json",
        timeout,
        |line| print_model_load_progress_line(line, verbose),
    )?;
    if response.status >= 400 {
        let message = parse_http_error_body(&response.body);
        anyhow::bail!(
            "POST /omni/model/select failed with status {}: {}",
            response.status,
            message
        );
    }
    let (result, _events) =
        model_load::parse_model_load_response(response.content_type.as_deref(), &response.body)?;
    Ok(result)
}

fn print_model_load_progress_line(line: &str, verbose: bool) {
    let Some(data) = line.trim().strip_prefix("data:") else {
        return;
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return;
    }
    let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    let event_type = json_str(&event, "type").unwrap_or("");
    let message = json_str(&event, "message").unwrap_or("");
    match event_type {
        "log" if verbose && !message.is_empty() => println!("  {message}"),
        "done" | "error" | "log" => {}
        _ if !message.is_empty() => println!("{message}"),
        _ => {}
    }
}

fn parse_http_error_body(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body.trim())
        .ok()
        .and_then(|payload| {
            payload
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .or_else(|| json_str(&payload, "message").map(str::to_string))
        })
        .unwrap_or_else(|| body.trim().to_string())
}

fn ensure_local_gateway_running(config: &config::AppConfig) -> Result<()> {
    if is_gateway_running(config) {
        return Ok(());
    }
    start_gateway_background(config)?;
    wait_for_gateway_ready(config)
}

fn is_gateway_running(config: &config::AppConfig) -> bool {
    let url = format!("{}/health", config.service_base_url());
    match http_client::get_json(&url, Duration::from_secs(2)) {
        Ok(response) if response.status == 200 => {
            response
                .body
                .get("status")
                .and_then(serde_json::Value::as_str)
                == Some("ok")
        }
        _ => false,
    }
}

fn start_gateway_background(config: &config::AppConfig) -> Result<()> {
    let log_dir = paths::local_logs_dir();
    std::fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join("gateway.log");
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let stderr = stdout.try_clone()?;
    let mut command = ProcessCommand::new(python_executable());
    pass_rust_path_overrides(&mut command);
    command
        .arg(paths::repo_root().join("omniinfer.py"))
        .arg("serve")
        .arg("--host")
        .arg(&config.host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--startup-timeout")
        .arg(format!("{:.0}", config.startup_timeout))
        .arg("--window-mode")
        .arg(&config.window_mode)
        .arg("--default-thinking")
        .arg(&config.default_thinking)
        .current_dir(paths::repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if !config.default_backend.trim().is_empty() {
        command
            .arg("--default-backend")
            .arg(&config.default_backend);
    }
    command.spawn()?;
    Ok(())
}

fn pass_rust_path_overrides(command: &mut ProcessCommand) {
    for key in ["OMNIINFER_RUST_REPO_ROOT", "OMNIINFER_RUST_STATE_ROOT"] {
        if let Some(value) = std::env::var_os(key).filter(|value| !value.is_empty()) {
            command.env(key, value);
        }
    }
}

fn wait_for_gateway_ready(config: &config::AppConfig) -> Result<()> {
    let timeout = Duration::from_secs(config.startup_timeout.max(10.0).round() as u64);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if is_gateway_running(config) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(500));
    }
    let log_path = paths::local_logs_dir().join("gateway.log");
    let tail = std::fs::read_to_string(&log_path)
        .map(|text| tail_chars(&text, 4000))
        .unwrap_or_default();
    anyhow::bail!("failed to start local OmniInfer service in time\n{tail}");
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars().rev().take(max_chars).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}

fn current_system_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "mac",
        "windows" => "windows",
        _ => "linux",
    }
}

fn fallback_to_python(command: &Command) -> Result<()> {
    if should_strict_rust() {
        println!("omniinfer-rs command parsed: {command:?}");
        println!("implementation pending; unset OMNIINFER_RUST_STRICT to fallback to Python");
        return Ok(());
    }
    exec_python(env::args().skip(1))
}

fn should_force_python() -> bool {
    env_flag("OMNIINFER_FORCE_PYTHON")
}

fn should_strict_rust() -> bool {
    env_flag("OMNIINFER_RUST_STRICT")
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn exec_python<I>(args: I) -> Result<()>
where
    I: IntoIterator,
    I::Item: AsRef<std::ffi::OsStr>,
{
    let script = paths::repo_root().join("omniinfer.py");
    let status = ProcessCommand::new(python_executable())
        .arg(script)
        .args(args)
        .status()?;
    std::process::exit(status.code().unwrap_or(1));
}

fn python_executable() -> std::ffi::OsString {
    env::var_os("OMNIINFER_PYTHON")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "python3".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloudflare_edge_530_is_transient_public_smoke_error() {
        let error = anyhow::anyhow!("HTTPS request failed: http status: 530");
        assert!(is_transient_public_smoke_error(&error));
    }

    #[test]
    fn auth_failures_are_not_transient_public_smoke_errors() {
        let error = anyhow::anyhow!("HTTPS request failed: http status: 401");
        assert!(!is_transient_public_smoke_error(&error));
    }
}
