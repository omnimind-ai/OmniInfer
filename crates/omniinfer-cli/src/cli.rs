use clap::{Args, Parser, Subcommand, ValueEnum};
use omniinfer_core::version;

#[derive(Debug, Parser)]
#[command(name = "omniinfer")]
#[command(version = version::VERSION)]
#[command(about = "Rust control-plane prototype for OmniInfer")]
#[command(long_about = "\
Rust control-plane prototype for OmniInfer.

This binary is intentionally experimental. It mirrors the Python OmniInfer CLI
surface while gateway, runtime, and TUI features are migrated incrementally.")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Manage inference runtimes.
    Backend {
        #[command(subcommand)]
        command: BackendCommand,
    },
    /// Compatibility alias for backend runtime install/build.
    Build {
        backend: String,
        /// Install a prebuilt runtime. This is the default for the compatibility alias.
        #[arg(long)]
        prebuilt: bool,
        /// Build from source. Requires a source checkout and platform build scripts.
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
pub(crate) enum BackendCommand {
    /// List backends available on this system.
    List {
        #[arg(long, default_value = "compatible")]
        scope: BackendScope,
    },
    /// Install a backend runtime.
    Install {
        backend: String,
        /// Explicitly install a prebuilt runtime. This is the default.
        #[arg(long)]
        prebuilt: bool,
        /// Reject with source-checkout guidance; source builds use `build --from-source`.
        #[arg(long)]
        from_source: bool,
        #[arg(long)]
        dry_run: bool,
    },
    /// Select a backend.
    Select { backend: String },
    /// Stop the current backend process.
    Stop,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum BackendScope {
    Installed,
    Compatible,
    All,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ModelCommand {
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
pub(crate) struct ModelLoadArgs {
    #[arg(short = 'm', long)]
    pub(crate) model: String,
    #[arg(long = "mmproj")]
    pub(crate) mmproj: Option<String>,
    #[arg(long)]
    pub(crate) ctx_size: Option<u32>,
    #[arg(long)]
    pub(crate) config: Option<String>,
    #[arg(long)]
    pub(crate) verbose: bool,
    #[arg(last = true, allow_hyphen_values = true)]
    pub(crate) backend_extra_args: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum AdvisorCommand {
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

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum ThinkingMode {
    On,
    Off,
}

#[derive(Debug, Args)]
pub(crate) struct ChatArgs {
    pub(crate) prompt: Option<String>,
    #[arg(long)]
    pub(crate) message: Option<String>,
    #[arg(long, conflicts_with = "no_stream")]
    pub(crate) stream: bool,
    #[arg(long)]
    pub(crate) no_stream: bool,
    #[arg(long)]
    pub(crate) image: Option<String>,
    #[arg(long)]
    pub(crate) think: Option<ThinkingMode>,
    #[arg(long)]
    pub(crate) temperature: Option<f32>,
    #[arg(long)]
    pub(crate) max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ServeArgs {
    #[command(subcommand)]
    pub(crate) command: Option<ServeCommand>,
    #[arg(short = 'm', long)]
    pub(crate) model: Option<String>,
    #[arg(long = "mmproj")]
    pub(crate) mmproj: Option<String>,
    #[arg(long)]
    pub(crate) ctx_size: Option<u32>,
    #[arg(long)]
    pub(crate) backend: Option<String>,
    #[arg(long)]
    pub(crate) cloudflare: bool,
    #[arg(long)]
    pub(crate) cloudflared_path: Option<String>,
    #[arg(long)]
    pub(crate) cloudflare_no_print_key: bool,
    #[arg(long)]
    pub(crate) lan: bool,
    #[arg(long)]
    pub(crate) api_key: Option<String>,
    #[arg(long)]
    pub(crate) admin_api_key: Option<String>,
    #[arg(long, value_name = "ID:KEY[,ID:KEY...]")]
    pub(crate) admin_api_keys: Option<String>,
    #[arg(long)]
    pub(crate) allow_insecure_lan: bool,
    #[arg(long)]
    pub(crate) allow_remote_management: bool,
    #[arg(long)]
    pub(crate) behind_proxy: bool,
    #[arg(long)]
    pub(crate) public_model_root: Option<String>,
    #[arg(long)]
    pub(crate) detach: bool,
    #[arg(long)]
    pub(crate) smoke_test: bool,
    #[arg(long)]
    pub(crate) no_restore_model: bool,
    #[arg(long, default_value_t = 9000)]
    pub(crate) port: u16,
    #[arg(long)]
    pub(crate) host: Option<String>,
    #[arg(long)]
    pub(crate) backend_host: Option<String>,
    #[arg(long)]
    pub(crate) backend_port: Option<u16>,
    #[arg(long)]
    pub(crate) default_backend: Option<String>,
    #[arg(long)]
    pub(crate) default_thinking: Option<ThinkingMode>,
    #[arg(long)]
    pub(crate) force_backend: Option<String>,
    #[arg(long)]
    pub(crate) window_mode: Option<WindowMode>,
    #[arg(long)]
    pub(crate) startup_timeout: Option<u32>,
    #[arg(long)]
    pub(crate) log_level: Option<LogLevel>,
    #[arg(long)]
    pub(crate) verbose: bool,
    #[arg(long)]
    pub(crate) debug_body: bool,
}

#[derive(Debug, Args)]
pub(crate) struct GatewayArgs {
    #[arg(long)]
    pub(crate) host: String,
    #[arg(long)]
    pub(crate) port: u16,
    #[arg(long)]
    pub(crate) api_key: Option<String>,
    #[arg(long)]
    pub(crate) admin_api_key: Option<String>,
    #[arg(long, value_name = "ID:KEY[,ID:KEY...]")]
    pub(crate) admin_api_keys: Option<String>,
    #[arg(long)]
    pub(crate) allow_insecure_lan: bool,
    #[arg(long)]
    pub(crate) allow_remote_management: bool,
    #[arg(long)]
    pub(crate) trust_proxy_headers: bool,
    #[arg(long)]
    pub(crate) public_model_root: Option<String>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum ServeCommand {
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
pub(crate) enum CompletionShell {
    Bash,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum WindowMode {
    Visible,
    Hidden,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}
