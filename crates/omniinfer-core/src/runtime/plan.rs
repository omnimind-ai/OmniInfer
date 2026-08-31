use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalRuntimeRequest {
    pub backend: Value,
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub host: String,
    pub port: u16,
    pub ctx_size: Option<u32>,
    pub launch_args: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeReadinessProbe {
    HttpHealth,
    TcpConnectAndLog { marker: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalServerProtocol {
    LlamaCppServer,
    VlaCppZmqServer,
    StableDiffusionCppServer,
    FreeTokenOpenAiServer,
    VllmOpenAiServer,
    VllmWsl2OpenAiServer,
}

impl ExternalServerProtocol {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "llama.cpp-server" => Some(Self::LlamaCppServer),
            "vla.cpp-zmq-server" => Some(Self::VlaCppZmqServer),
            "stable-diffusion.cpp-server" => Some(Self::StableDiffusionCppServer),
            "freetoken-openai-server" => Some(Self::FreeTokenOpenAiServer),
            "vllm-openai-server" => Some(Self::VllmOpenAiServer),
            "vllm-wsl2-openai-server" => Some(Self::VllmWsl2OpenAiServer),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCppServer => "llama.cpp-server",
            Self::VlaCppZmqServer => "vla.cpp-zmq-server",
            Self::StableDiffusionCppServer => "stable-diffusion.cpp-server",
            Self::FreeTokenOpenAiServer => "freetoken-openai-server",
            Self::VllmOpenAiServer => "vllm-openai-server",
            Self::VllmWsl2OpenAiServer => "vllm-wsl2-openai-server",
        }
    }

    pub fn is_openai_compatible(self) -> bool {
        !matches!(self, Self::VlaCppZmqServer | Self::StableDiffusionCppServer)
    }

    pub fn is_http_transport(self) -> bool {
        !matches!(self, Self::VlaCppZmqServer)
    }

    pub fn supports_chat(self) -> bool {
        !matches!(self, Self::VlaCppZmqServer | Self::StableDiffusionCppServer)
    }

    pub fn client_endpoint(self, host: &str, port: u16) -> String {
        let endpoint_host = host
            .parse::<std::net::IpAddr>()
            .ok()
            .filter(std::net::IpAddr::is_ipv6)
            .map(|_| format!("[{host}]"))
            .unwrap_or_else(|| host.to_string());
        if matches!(self, Self::VlaCppZmqServer) {
            format!("tcp://{endpoint_host}:{port}")
        } else {
            format!("http://{endpoint_host}:{port}")
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalRuntimePlan {
    pub command: Vec<String>,
    pub stop_command: Option<Vec<String>>,
    pub cwd: PathBuf,
    pub port: u16,
    pub ctx_size: Option<u32>,
    pub log_file_name: String,
    pub proxy_model_ref: Option<String>,
    pub protocol: ExternalServerProtocol,
    pub client_endpoint: String,
    pub readiness_probe: RuntimeReadinessProbe,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RuntimePlanError {
    #[error("backend payload is missing field: {0}")]
    MissingBackendField(&'static str),
    #[error("backend launcher not found: {0}")]
    MissingLauncher(String),
    #[error("launch arg {0:?} is managed by OmniInfer and must not be set in backend config")]
    ReservedLaunchArg(String),
    #[error("unsupported external runtime protocol for {backend}: {protocol}")]
    UnsupportedProtocol { backend: String, protocol: String },
    #[error("port must be in 1-65535")]
    InvalidPort,
    #[error("vla.cpp ZeroMQ runtime must bind to a loopback host, got: {0}")]
    NonLoopbackVlaBind(String),
    #[error("stable-diffusion.cpp runtime must bind to a loopback host, got: {0}")]
    NonLoopbackDiffusionBind(String),
    #[error("MiniMax H3 requires the stable-diffusion.cpp launch arg {0}")]
    MissingH3Component(&'static str),
    #[error("stable-diffusion.cpp component not found for {flag}: {path}")]
    DiffusionComponentNotFound { flag: String, path: String },
    #[error("invalid WSL2 launcher manifest {path}: {message}")]
    InvalidWslLauncherManifest { path: String, message: String },
    #[error("WSL2 vLLM does not support this Windows model path: {0}")]
    UnsupportedWslModelPath(String),
}

#[derive(Debug, Deserialize)]
struct WslVllmLauncherManifest {
    schema_version: u32,
    backend: String,
    distribution: String,
    linux_launcher: String,
    linux_runner: String,
    linux_stopper: String,
    linux_pid_dir: String,
    automount_root: String,
}

pub fn build_external_runtime_plan(
    request: &ExternalRuntimeRequest,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if request.port == 0 {
        return Err(RuntimePlanError::InvalidPort);
    }
    let backend_id = required_str(&request.backend, "id")?;
    let protocol_text =
        optional_str(&request.backend, "external_server_protocol").unwrap_or("llama.cpp-server");
    let protocol = ExternalServerProtocol::parse(protocol_text).ok_or_else(|| {
        RuntimePlanError::UnsupportedProtocol {
            backend: backend_id.to_string(),
            protocol: protocol_text.to_string(),
        }
    })?;
    let launcher = optional_str(&request.backend, "launcher_path")
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| RuntimePlanError::MissingLauncher(backend_id.to_string()))?;
    let launcher_path = PathBuf::from(launcher);
    let mut server_args = request
        .launch_args
        .clone()
        .unwrap_or_else(|| string_array(&request.backend, "default_args"));
    validate_launch_args(&server_args)?;
    let ctx_flags = ctx_size_flags(protocol.as_str());
    if let Some(ctx_size) = request.ctx_size
        && !ctx_flags[0].is_empty()
    {
        server_args = with_server_arg(server_args, &ctx_flags, ctx_size.to_string());
    }
    let effective_ctx_size =
        extract_server_arg_value(&server_args, &ctx_flags).and_then(|value| value.parse().ok());
    let log_file_name = optional_str(&request.backend, "log_file_name")
        .unwrap_or("runtime.log")
        .to_string();

    match protocol {
        ExternalServerProtocol::LlamaCppServer => build_llama_cpp_plan(
            backend_id,
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        ExternalServerProtocol::VlaCppZmqServer => build_vla_cpp_plan(
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        ExternalServerProtocol::StableDiffusionCppServer => {
            build_stable_diffusion_cpp_plan(&launcher_path, request, server_args, log_file_name)
        }
        ExternalServerProtocol::FreeTokenOpenAiServer => build_freetoken_plan(
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        ExternalServerProtocol::VllmOpenAiServer => build_vllm_plan(
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
        ExternalServerProtocol::VllmWsl2OpenAiServer => build_wsl_vllm_plan(
            backend_id,
            &launcher_path,
            request,
            server_args,
            effective_ctx_size,
            log_file_name,
        ),
    }
}

fn build_stable_diffusion_cpp_plan(
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if !is_loopback_host(&request.host) {
        return Err(RuntimePlanError::NonLoopbackDiffusionBind(
            request.host.clone(),
        ));
    }
    validate_stable_diffusion_cpp_launch_args(&server_args, &request.model_path)?;
    let mut command = vec![
        launcher_path.display().to_string(),
        "--diffusion-model".to_string(),
        request.model_path.clone(),
        "--listen-ip".to_string(),
        request.host.clone(),
        "--listen-port".to_string(),
        request.port.to_string(),
    ];
    command.append(&mut server_args);
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: None,
        log_file_name,
        proxy_model_ref: None,
        protocol: ExternalServerProtocol::StableDiffusionCppServer,
        client_endpoint: ExternalServerProtocol::StableDiffusionCppServer
            .client_endpoint(&request.host, request.port),
        readiness_probe: RuntimeReadinessProbe::TcpConnectAndLog {
            marker: format!("listening on: http://{}:{}", request.host, request.port),
        },
    })
}

fn validate_stable_diffusion_cpp_launch_args(
    args: &[String],
    model_path: &str,
) -> Result<(), RuntimePlanError> {
    for token in args {
        let flag = token.split_once('=').map(|(flag, _)| flag).unwrap_or(token);
        if matches!(
            flag,
            "--diffusion-model" | "--listen-ip" | "--listen-port" | "--host" | "--port"
        ) {
            return Err(RuntimePlanError::ReservedLaunchArg(flag.to_string()));
        }
    }
    for flag in ["--llm", "--vae", "--audio-vae"] {
        if let Some(path) = extract_server_arg_value(args, &[flag])
            && !Path::new(&path).is_file()
        {
            return Err(RuntimePlanError::DiffusionComponentNotFound {
                flag: flag.to_string(),
                path,
            });
        }
    }
    let model_name = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_ascii_lowercase();
    if model_name.contains("minimax_h3") {
        for flag in ["--llm", "--vae"] {
            if extract_server_arg_value(args, &[flag]).is_none() {
                return Err(RuntimePlanError::MissingH3Component(flag));
            }
        }
    }
    Ok(())
}

fn build_freetoken_plan(
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if extract_server_arg_value(&server_args, &["--served-model-name"]).is_none() {
        server_args.splice(
            0..0,
            ["--served-model-name".to_string(), "local".to_string()],
        );
    }
    let proxy_model_ref = extract_server_arg_value(&server_args, &["--served-model-name"]);
    let mut command = vec![
        launcher_path.display().to_string(),
        "serve".to_string(),
        "--model".to_string(),
        request.model_path.clone(),
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    command.append(&mut server_args);
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref,
        protocol: ExternalServerProtocol::FreeTokenOpenAiServer,
        client_endpoint: ExternalServerProtocol::FreeTokenOpenAiServer
            .client_endpoint(&request.host, request.port),
        readiness_probe: RuntimeReadinessProbe::TcpConnectAndLog {
            marker: format!(
                "API server is ready to serve on {}:{}",
                request.host, request.port
            ),
        },
    })
}

fn build_llama_cpp_plan(
    backend_id: &str,
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    let mut command = vec![
        launcher_path.display().to_string(),
        "-m".to_string(),
        request.model_path.clone(),
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    if backend_id.starts_with("ik_llama.cpp") {
        command.extend(["--webui".to_string(), "none".to_string()]);
    } else {
        command.push("--no-webui".to_string());
    }
    let log_dir = runtime_dir(&request.backend).join("logs");
    command.extend([
        "--slot-save-path".to_string(),
        log_dir.display().to_string(),
    ]);
    command.extend(server_args);
    if let Some(mmproj) = request
        .mmproj_path
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        command.extend(["--mmproj".to_string(), mmproj.to_string()]);
    }
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref: None,
        protocol: ExternalServerProtocol::LlamaCppServer,
        client_endpoint: ExternalServerProtocol::LlamaCppServer
            .client_endpoint(&request.host, request.port),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    })
}

fn build_vla_cpp_plan(
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    _effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if !is_loopback_host(&request.host) {
        return Err(RuntimePlanError::NonLoopbackVlaBind(request.host.clone()));
    }
    validate_vla_cpp_launch_args(&server_args)?;
    let client_endpoint =
        ExternalServerProtocol::VlaCppZmqServer.client_endpoint(&request.host, request.port);
    let mut command = vec![
        launcher_path.display().to_string(),
        "--bind".to_string(),
        client_endpoint.clone(),
    ];
    command.append(&mut server_args);
    if let Some(mmproj) = request
        .mmproj_path
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        command.push(mmproj.to_string());
    }
    command.push(request.model_path.clone());
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: None,
        log_file_name,
        proxy_model_ref: None,
        protocol: ExternalServerProtocol::VlaCppZmqServer,
        client_endpoint: client_endpoint.clone(),
        readiness_probe: RuntimeReadinessProbe::TcpConnectAndLog {
            marker: format!("vla-server: bound to {client_endpoint}. ready."),
        },
    })
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn validate_vla_cpp_launch_args(args: &[String]) -> Result<(), RuntimePlanError> {
    for token in args {
        let flag = token.split_once('=').map(|(flag, _)| flag).unwrap_or(token);
        if matches!(flag, "-c" | "--ctx-size" | "--max-model-len") {
            return Err(RuntimePlanError::ReservedLaunchArg(flag.to_string()));
        }
    }
    Ok(())
}

fn build_vllm_plan(
    launcher_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    if extract_server_arg_value(&server_args, &["--served-model-name"]).is_none() {
        server_args.splice(
            0..0,
            ["--served-model-name".to_string(), "local".to_string()],
        );
    }
    let proxy_model_ref = extract_server_arg_value(&server_args, &["--served-model-name"]);
    let mut command = vec![
        launcher_path.display().to_string(),
        "serve".to_string(),
        request.model_path.clone(),
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    command.extend(server_args);
    Ok(ExternalRuntimePlan {
        command,
        stop_command: None,
        cwd: launcher_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref,
        protocol: ExternalServerProtocol::VllmOpenAiServer,
        client_endpoint: ExternalServerProtocol::VllmOpenAiServer
            .client_endpoint(&request.host, request.port),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    })
}

fn build_wsl_vllm_plan(
    backend_id: &str,
    manifest_path: &Path,
    request: &ExternalRuntimeRequest,
    mut server_args: Vec<String>,
    effective_ctx_size: Option<u32>,
    log_file_name: String,
) -> Result<ExternalRuntimePlan, RuntimePlanError> {
    let raw = std::fs::read_to_string(manifest_path).map_err(|error| {
        RuntimePlanError::InvalidWslLauncherManifest {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        }
    })?;
    let manifest: WslVllmLauncherManifest = serde_json::from_str(&raw).map_err(|error| {
        RuntimePlanError::InvalidWslLauncherManifest {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        }
    })?;
    validate_wsl_manifest(backend_id, manifest_path, &manifest)?;
    if extract_server_arg_value(&server_args, &["--served-model-name"]).is_none() {
        server_args.splice(
            0..0,
            ["--served-model-name".to_string(), "local".to_string()],
        );
    }
    let proxy_model_ref = extract_server_arg_value(&server_args, &["--served-model-name"]);
    let model = translate_wsl_model_ref(&request.model_path, &manifest.automount_root)?;
    let pid_file = format!(
        "{}/{}.pid",
        manifest.linux_pid_dir.trim_end_matches('/'),
        request.port
    );
    let wsl = std::env::var("OMNIINFER_WSL_EXE").unwrap_or_else(|_| "wsl.exe".to_string());
    let mut command = vec![
        wsl.clone(),
        "--distribution".to_string(),
        manifest.distribution.clone(),
        "--exec".to_string(),
        manifest.linux_runner.clone(),
        pid_file.clone(),
        manifest.linux_launcher.clone(),
        "serve".to_string(),
        model,
        "--host".to_string(),
        request.host.clone(),
        "--port".to_string(),
        request.port.to_string(),
    ];
    command.extend(server_args);
    let stop_command = vec![
        wsl,
        "--distribution".to_string(),
        manifest.distribution,
        "--exec".to_string(),
        manifest.linux_stopper,
        pid_file,
    ];
    Ok(ExternalRuntimePlan {
        command,
        stop_command: Some(stop_command),
        cwd: manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        port: request.port,
        ctx_size: effective_ctx_size,
        log_file_name,
        proxy_model_ref,
        protocol: ExternalServerProtocol::VllmWsl2OpenAiServer,
        client_endpoint: ExternalServerProtocol::VllmWsl2OpenAiServer
            .client_endpoint(&request.host, request.port),
        readiness_probe: RuntimeReadinessProbe::HttpHealth,
    })
}

fn validate_wsl_manifest(
    backend_id: &str,
    path: &Path,
    manifest: &WslVllmLauncherManifest,
) -> Result<(), RuntimePlanError> {
    let valid = manifest.schema_version == 1
        && manifest.backend == backend_id
        && !manifest.distribution.trim().is_empty()
        && !manifest.distribution.chars().any(char::is_control)
        && [
            manifest.linux_launcher.as_str(),
            manifest.linux_runner.as_str(),
            manifest.linux_stopper.as_str(),
            manifest.linux_pid_dir.as_str(),
            manifest.automount_root.as_str(),
        ]
        .into_iter()
        .all(valid_absolute_linux_path);
    if valid {
        Ok(())
    } else {
        Err(RuntimePlanError::InvalidWslLauncherManifest {
            path: path.display().to_string(),
            message: "unsupported schema or incomplete absolute Linux paths".to_string(),
        })
    }
}

fn valid_absolute_linux_path(value: &str) -> bool {
    value.starts_with('/')
        && !value.chars().any(char::is_control)
        && !value.split('/').any(|component| component == "..")
}

fn translate_wsl_model_ref(model: &str, automount_root: &str) -> Result<String, RuntimePlanError> {
    let bytes = model.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
    {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let suffix = model[3..].replace('\\', "/");
        return Ok(format!(
            "{}/{drive}/{}",
            automount_root.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        ));
    }
    if model.starts_with(r"\\") {
        return Err(RuntimePlanError::UnsupportedWslModelPath(model.to_string()));
    }
    Ok(model.to_string())
}

fn validate_launch_args(args: &[String]) -> Result<(), RuntimePlanError> {
    for token in args {
        let flag = token.split_once('=').map(|(flag, _)| flag).unwrap_or(token);
        if matches!(
            flag,
            "-m" | "--model" | "-mm" | "--mmproj" | "--host" | "--port" | "--bind" | "--no-webui"
        ) {
            return Err(RuntimePlanError::ReservedLaunchArg(flag.to_string()));
        }
    }
    Ok(())
}

fn with_server_arg(mut args: Vec<String>, flags: &[&str], value: String) -> Vec<String> {
    let mut updated = Vec::with_capacity(args.len() + 2);
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if token
            .split_once('=')
            .is_some_and(|(flag, _)| flags.contains(&flag))
        {
            index += 1;
            continue;
        }
        if flags.contains(&token.as_str()) {
            index += if index + 1 < args.len() { 2 } else { 1 };
            continue;
        }
        updated.push(std::mem::take(&mut args[index]));
        index += 1;
    }
    updated.extend([flags[0].to_string(), value]);
    updated
}

fn extract_server_arg_value(args: &[String], flags: &[&str]) -> Option<String> {
    let mut value = None;
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if let Some((flag, inline_value)) = token.split_once('=')
            && flags.contains(&flag)
        {
            value = Some(inline_value.to_string());
            index += 1;
            continue;
        }
        if flags.contains(&token.as_str()) {
            if let Some(next) = args.get(index + 1) {
                value = Some(next.clone());
            }
            index += 2;
            continue;
        }
        index += 1;
    }
    value
}

fn ctx_size_flags(protocol: &str) -> [&'static str; 2] {
    match protocol {
        "freetoken-openai-server" => ["--max-seq-len-override", ""],
        "vllm-openai-server" | "vllm-wsl2-openai-server" => ["--max-model-len", ""],
        "vla.cpp-zmq-server" | "stable-diffusion.cpp-server" => ["", ""],
        _ => ["-c", "--ctx-size"],
    }
}

fn required_str<'a>(value: &'a Value, key: &'static str) -> Result<&'a str, RuntimePlanError> {
    optional_str(value, key).ok_or(RuntimePlanError::MissingBackendField(key))
}

fn optional_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
}

fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn runtime_dir(backend: &Value) -> PathBuf {
    optional_str(backend, "runtime_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests;
