use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::time::Duration;

use omniinfer_core::{config, http_client, local_state, paths, version};

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
    let cli = Cli::parse();
    match cli.command {
        None => print_tui_placeholder(),
        Some(Command::Status) => print_status(),
        Some(command) => {
            println!("omniinfer-rs command parsed: {command:?}");
            println!("implementation pending; use ./omniinfer for production behavior");
        }
    }
    Ok(())
}

fn print_tui_placeholder() {
    println!("OmniInfer Rust TUI is not implemented yet.");
    println!("Use ./omniinfer for the current Python TUI.");
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
        println!("Selected mmproj: {}", model.mmproj.as_deref().unwrap_or("not selected"));
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
            format!("running (backend_ready={}, model={})", yes_no(backend_ready), model)
        }
        Ok(response) => format!("unhealthy (HTTP {})", response.status),
        Err(_) => "not running".to_string(),
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
