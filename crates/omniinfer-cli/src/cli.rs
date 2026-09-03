use clap::{Args, Parser, Subcommand, ValueEnum};
use omniinfer_core::{config::DEFAULT_STARTUP_TIMEOUT_SECONDS, version};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "omniinfer")]
#[command(version = version::VERSION)]
#[command(about = "Rust control-plane prototype for OmniInfer")]
#[command(long_about = "\
Rust control-plane prototype for OmniInfer.

This binary is intentionally experimental. It mirrors the Python OmniInfer CLI
surface while gateway, runtime, and TUI features are migrated incrementally.")]
pub(crate) struct Cli {
    /// Root for OmniInfer state, configuration, logs, models, and default runtimes.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) state_root: Option<PathBuf>,
    /// Root containing platform backend runtime directories.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) runtime_root: Option<PathBuf>,
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
        #[arg(long, conflicts_with = "from_source")]
        prebuilt: bool,
        /// Build from source. Requires a source checkout and platform build scripts.
        #[arg(long, conflicts_with = "prebuilt")]
        from_source: bool,
        /// Arguments passed unchanged to the platform source build script after `--`.
        #[arg(last = true, requires = "from_source")]
        build_args: Vec<String>,
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
    /// Measure the loaded model and archive submission-compatible benchmark JSON.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
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
        /// Emit newline-delimited JSON progress events on stdout.
        #[arg(long)]
        json: bool,
        /// WSL2 distribution used by managed Windows Linux runtimes.
        #[arg(long)]
        wsl_distro: Option<String>,
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
    #[arg(long, conflicts_with = "mmproj")]
    pub(crate) no_mmproj: bool,
    #[arg(long)]
    pub(crate) ctx_size: Option<u32>,
    /// Explicit runtime memory budget for remote model references whose size is unknown.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) resource_budget_bytes: Option<u64>,
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

#[derive(Debug, Subcommand)]
pub(crate) enum BenchCommand {
    /// Run a benchmark against the currently loaded model.
    Run(Box<BenchRunArgs>),
    /// List locally archived benchmark results.
    List {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Args)]
pub(crate) struct BenchRunArgs {
    /// Optional stable benchmark ID; a unique ID is generated when omitted.
    #[arg(long)]
    pub(crate) benchmark_id: Option<String>,
    /// Model ID used by the destination benchmark catalog.
    #[arg(long)]
    pub(crate) catalog_model_id: String,
    /// Optional human-readable model name.
    #[arg(long)]
    pub(crate) model_name: Option<String>,
    /// Model artifact format, for example GGUF, MLX, or Safetensors.
    #[arg(long = "format")]
    pub(crate) model_format: String,
    /// Quantization name used by the catalog.
    #[arg(long)]
    pub(crate) quantization: String,
    /// Stable public HTTPS URL for the measured model artifact.
    #[arg(long)]
    pub(crate) model_url: String,
    /// Human-readable tested device name; auto-detected for known local hardware when omitted.
    #[arg(long)]
    pub(crate) device_name: Option<String>,
    /// Catalog SoC/device ID; inferred for known local hardware when omitted.
    #[arg(long)]
    pub(crate) soc: Option<String>,
    /// Optional expected catalog backend ID; defaults to the loaded backend.
    #[arg(long)]
    pub(crate) backend_id: Option<String>,
    /// Optional human-readable backend name.
    #[arg(long)]
    pub(crate) backend_name: Option<String>,
    /// Accelerator used for Prefill. Set both phase accelerators for mixed execution.
    #[arg(long, value_enum)]
    pub(crate) prefill_accelerator: Option<BenchmarkAccelerator>,
    /// Accelerator used for Decode. Set both phase accelerators for mixed execution.
    #[arg(long, value_enum)]
    pub(crate) decode_accelerator: Option<BenchmarkAccelerator>,
    /// Privilege level of the measured runtime process.
    #[arg(long, value_enum, default_value_t = BenchmarkPrivilegeLevel::Standard)]
    pub(crate) privilege_level: BenchmarkPrivilegeLevel,
    /// Exact runtime/backend version; read from a managed prebuilt manifest when omitted.
    #[arg(long)]
    pub(crate) backend_version: Option<String>,
    /// Exact build/install command; inferred for a managed prebuilt runtime when omitted.
    #[arg(long)]
    pub(crate) build_command: Option<String>,
    /// Override the runtime launch command captured from OmniInfer state.
    #[arg(long)]
    pub(crate) run_command: Option<String>,
    /// Explicitly declare a baseline run with no optional optimization.
    #[arg(long)]
    pub(crate) baseline: bool,
    /// Confirm an optimization method that was active. Repeat for multiple methods.
    #[arg(long = "optimization")]
    pub(crate) optimizations: Vec<String>,
    /// Inline benchmark prompt. Conflicts with --prompt-file.
    #[arg(long, conflicts_with = "prompt_file")]
    pub(crate) prompt: Option<String>,
    /// UTF-8 file containing the benchmark prompt.
    #[arg(long, conflicts_with = "prompt")]
    pub(crate) prompt_file: Option<PathBuf>,
    /// Maximum output tokens requested from the runtime.
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..))]
    pub(crate) max_tokens: u32,
    /// Measured repetitions written to the result.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u16).range(3..=100))]
    pub(crate) runs: u16,
    /// Unrecorded warmup requests before measurement.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(0..=100))]
    pub(crate) warmup_runs: u16,
    /// Request that the runtime ignore end-of-sequence tokens.
    #[arg(long)]
    pub(crate) ignore_eos: bool,
    /// Context size; inferred from loaded runtime state when omitted.
    #[arg(long)]
    pub(crate) context_size: Option<u32>,
    /// Runtime batch size; inferred from known launch flags when omitted.
    #[arg(long)]
    pub(crate) batch_size: Option<u32>,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u32).range(1..=86400))]
    pub(crate) timeout_seconds: u32,
    /// Submitter name or stable community identity.
    #[arg(long)]
    pub(crate) submitter_name: String,
    /// Optional submitter organization.
    #[arg(long)]
    pub(crate) organization: Option<String>,
    /// Optional public source URL for supporting evidence.
    #[arg(long)]
    pub(crate) source_url: Option<String>,
    /// Optional methodology notes.
    #[arg(long)]
    pub(crate) notes: Option<String>,
    /// Optional result path. Defaults to .local/benchmarks/results/<id>.json.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    /// Also print the complete result JSON to stdout.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum BenchmarkAccelerator {
    Cpu,
    Gpu,
    Npu,
    Htp,
    Ane,
    Other,
}

impl BenchmarkAccelerator {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Npu => "npu",
            Self::Htp => "htp",
            Self::Ane => "ane",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub(crate) enum BenchmarkPrivilegeLevel {
    #[default]
    Standard,
    Elevated,
}

impl BenchmarkPrivilegeLevel {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Elevated => "elevated",
        }
    }
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
    #[arg(long, conflicts_with = "mmproj")]
    pub(crate) no_mmproj: bool,
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
    /// Run one inference check, then stop the gateway/backend and release their ports.
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
    /// Explicit runtime memory budget for remote model references whose size is unknown.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) resource_budget_bytes: Option<u64>,
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
    #[arg(long, default_value_t = DEFAULT_STARTUP_TIMEOUT_SECONDS)]
    pub(crate) startup_timeout: u64,
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
