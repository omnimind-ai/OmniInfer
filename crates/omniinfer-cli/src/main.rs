use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{Shell, generate};
use std::env;
use std::process::Command as ProcessCommand;
use std::time::Duration;

use omniinfer_core::{config, http_client, local_state, paths, serve_state, version};

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
    Ps,
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

#[derive(Debug, Args)]
struct ServeArgs {
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
    lan: bool,
    #[arg(long)]
    api_key: Option<String>,
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
}

#[derive(Debug, Subcommand)]
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

fn main() -> Result<()> {
    if should_force_python() {
        return exec_python(env::args().skip(1));
    }
    let cli = Cli::parse();
    match cli.command.as_ref() {
        None => print_tui_placeholder(),
        Some(Command::Status) => print_status(),
        Some(Command::Completion { shell }) => print_completion(shell.clone()),
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
            command: BackendCommand::Stop,
        } => stop_backend(),
        Command::Model {
            command: ModelCommand::List { all, best },
        } => print_model_list(*all, *best),
        Command::Thinking {
            command: ThinkingCommand::Show,
        } => {
            print_thinking_show();
            Ok(())
        }
        Command::Thinking {
            command: ThinkingCommand::Set { mode },
        } => print_thinking_set(mode.clone()),
        Command::Chat(args) if args.no_stream && args.image.is_none() => print_chat_no_stream(args),
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
        other => fallback_to_python(other),
    }
}

fn print_tui_placeholder() {
    if should_force_python() {
        let _ = exec_python(env::args().skip(1));
        return;
    }
    if should_strict_rust() {
        println!("OmniInfer Rust TUI is not implemented yet.");
        println!("Unset OMNIINFER_RUST_STRICT or use ./omniinfer for the current Python TUI.");
        return;
    }
    let _ = exec_python(env::args().skip(1));
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
    let scope_text = match scope {
        BackendScope::Installed => "installed",
        BackendScope::Compatible => "compatible",
        BackendScope::All => "all",
    };
    let payload = get_local_json(
        &format!("/omni/backends?scope={scope_text}"),
        Duration::from_secs(10),
    )?;
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

fn print_model_list(all: bool, best: bool) -> Result<()> {
    let best = best || !all;
    let system = current_system_name();
    let endpoint = if best {
        "/omni/supported-models/best"
    } else {
        "/omni/supported-models"
    };
    let payload = get_local_json(
        &format!("{endpoint}?system={system}"),
        Duration::from_secs(60),
    )?;
    println!("Supported models ({system})");
    if best {
        print_best_model_catalog(&payload);
    } else {
        print_full_model_catalog(&payload);
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

fn shutdown_service() -> Result<()> {
    match post_local_json(
        "/omni/shutdown",
        &serde_json::json!({}),
        Duration::from_secs(10),
    ) {
        Ok(_) => {
            println!("OmniInfer service stopped");
            Ok(())
        }
        Err(_) => {
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
    let url = format!("{}/omni/shutdown", config.service_base_url());
    let stopped =
        match http_client::post_json(&url, &serde_json::json!({}), Duration::from_secs(10)) {
            Ok(response) => response.status < 400,
            Err(_) => false,
        };
    let _ = serve_state::remove_serve_pid_info(port);
    if stopped {
        println!("OmniInfer service stopped on port {port}");
    } else {
        println!("OmniInfer service is not running on port {port}");
    }
    Ok(())
}

fn print_chat_no_stream(args: &ChatArgs) -> Result<()> {
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
    payload.insert(
        "messages".to_string(),
        serde_json::json!([{ "role": "user", "content": message }]),
    );
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

    let response = post_local_json(
        "/v1/chat/completions",
        &serde_json::Value::Object(payload),
        Duration::from_secs(600),
    )?;
    print_chat_response(&response);
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

fn print_chat_performance(response: &serde_json::Value) {
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

fn json_str<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

fn json_bool(value: &serde_json::Value, key: &str) -> Option<bool> {
    value.get(key).and_then(serde_json::Value::as_bool)
}

fn json_u64(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(serde_json::Value::as_u64)
}

fn get_local_json(endpoint: &str, timeout: Duration) -> Result<serde_json::Value> {
    let config = config::load_app_config().unwrap_or_default();
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
    let url = format!("{}{}", config.service_base_url(), endpoint);
    let response = http_client::post_json(&url, body, timeout)?;
    if response.status >= 400 {
        anyhow::bail!("POST {endpoint} failed with status {}", response.status);
    }
    Ok(response.body)
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
    let python = env::var_os("OMNIINFER_PYTHON")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "python3".into());
    let script = paths::repo_root().join("omniinfer.py");
    let status = ProcessCommand::new(python)
        .arg(script)
        .args(args)
        .status()?;
    std::process::exit(status.code().unwrap_or(1));
}
