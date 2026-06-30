use std::env;
use std::fs::OpenOptions;
use std::net::{TcpStream, UdpSocket};
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use omniinfer_core::{
    backend_registry, config, gateway_auth, http_client, local_state, model_load, paths,
    serve_state,
};
use rand::Rng;
use rand::distr::Alphanumeric;

use crate::cloudflare::{resolve_cloudflared, start_cloudflare_quick_tunnel};
use crate::{
    ServeArgs, ThinkingMode, WindowMode, json_bool, json_str, json_u64,
    load_model_with_request_for_config_and_autostart, print_model_loaded,
    select_backend_for_config_with_autostart, wait_for_gateway_ready, yes_no,
};

pub(crate) fn stop_serve(port: u16) -> Result<()> {
    let mut config = config::load_app_config().unwrap_or_default();
    config.port = port;
    let info = serve_state::load_serve_pid_info(port).ok().flatten();
    let url = format!("{}/omni/shutdown", config.service_base_url());
    let stopped =
        match http_client::post_json(&url, &serde_json::json!({}), Duration::from_secs(10)) {
            Ok(response) => response.status < 400,
            Err(_) => false,
        };
    if stopped && !wait_for_local_port_closed(port, Duration::from_secs(3)) {
        if let Some(pid) = info.as_ref().and_then(|info| info.pid) {
            stop_process(pid);
            let _ = wait_for_local_port_closed(port, Duration::from_secs(3));
        }
    }
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

fn wait_for_local_port_closed(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_err() {
            return true;
        }
        thread::sleep(Duration::from_millis(100));
    }
    TcpStream::connect(("127.0.0.1", port)).is_err()
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

pub(crate) fn can_serve_locally(args: &ServeArgs) -> bool {
    args.command.is_none()
}

pub(crate) fn should_run_server_tui(args: &ServeArgs) -> bool {
    use std::io::IsTerminal;
    args.command.is_none()
        && !args.detach
        && args.model.is_none()
        && !env_flag("OMNIINFER_SERVE_DIRECT")
        && std::io::stdin().is_terminal()
        && std::io::stdout().is_terminal()
}

pub(crate) fn serve_orchestrated(args: &ServeArgs) -> Result<()> {
    validate_serve_remote_access_args(args)?;
    let restore_model = resolve_serve_restore_model(args);
    let mut config = config::load_app_config().unwrap_or_default();
    config.port = args.port;
    config.host = resolve_serve_listen_host(args);
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
    if args.allow_remote_management
        && admin_api_key.is_none()
        && admin_api_keys.is_empty()
        && !admin_keys_file_has_entries()
    {
        anyhow::bail!(
            "--allow-remote-management requires --admin-api-key, --admin-api-keys, OMNIINFER_ADMIN_API_KEY, OMNIINFER_ADMIN_API_KEYS, or .local/config/admin_keys.json"
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
    reject_embedded_serve_backend(args)?;
    let public_config = config.clone();
    let log_path = paths::local_logs_dir().join(format!("serve-{}.log", public_config.port));
    println!("Starting OmniInfer service on port {}...", config.port);
    println!("Log: {}", log_path.display());
    let rust_gateway = start_rust_gateway_child(
        &public_config,
        args,
        &log_path,
        api_key.as_deref(),
        admin_api_key.as_deref(),
        &admin_api_keys,
        public_model_root.as_deref(),
    )?;
    wait_for_gateway_ready(&public_config)?;
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
            stop_process(rust_gateway.id());
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
        pid: Some(rust_gateway.id()),
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
        let status =
            wait_for_foreground_service(rust_gateway, cloudflared_child, public_config.port)?;
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

fn resolve_serve_listen_host(args: &ServeArgs) -> String {
    args.host
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            if args.lan {
                "0.0.0.0".to_string()
            } else {
                "127.0.0.1".to_string()
            }
        })
}

fn resolve_serve_restore_model(args: &ServeArgs) -> Option<ServeModelRequest> {
    if args.no_restore_model {
        return None;
    }
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
    mut rust_gateway: std::process::Child,
    cloudflared_child: Option<std::process::Child>,
    port: u16,
) -> Result<std::process::ExitStatus> {
    let status = rust_gateway.wait()?;
    if let Some(mut tunnel) = cloudflared_child {
        let _ = tunnel.kill();
        let _ = tunnel.wait();
    }
    let _ = serve_state::remove_serve_pid_info(port);
    Ok(status)
}

fn reject_embedded_serve_backend(args: &ServeArgs) -> Result<()> {
    let Some(backend_id) = resolve_serve_start_backend(args)? else {
        return Ok(());
    };
    let registry = backend_registry::BackendRegistry::load_current();
    let Some(backend) = registry.get(&backend_id) else {
        return Ok(());
    };
    if backend.runtime_mode == "embedded" {
        anyhow::bail!(
            "{} is an embedded backend. Python control-plane fallback has been removed; use an external-server backend or a backend adapter service.",
            backend.id
        );
    }
    Ok(())
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

pub(crate) fn parse_admin_api_keys(
    raw: Option<&str>,
) -> Result<Vec<gateway_auth::GatewayAdminApiKey>> {
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

fn admin_keys_file_has_entries() -> bool {
    let path = paths::admin_keys_file();
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let source = value.get("keys").unwrap_or(&value);
    match source {
        serde_json::Value::Object(map) => map.values().any(|key| {
            key.as_str()
                .map(str::trim)
                .is_some_and(|key| !key.is_empty())
        }),
        serde_json::Value::Array(items) => items.iter().any(|item| {
            item.get("key")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .is_some_and(|key| !key.is_empty())
        }),
        _ => false,
    }
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

fn start_rust_gateway_child(
    public_config: &config::AppConfig,
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
    hide_child_window(&mut command);
    if args.detach {
        detach_child_process(&mut command);
    }
    Ok(command.spawn()?)
}

pub(crate) fn detach_child_process(command: &mut ProcessCommand) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
}

pub(crate) fn hide_child_window(command: &mut ProcessCommand) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

pub(crate) fn expand_home_path(value: &str) -> PathBuf {
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
        || text.contains("unknown host")
        || text.contains("no such host")
        || text.contains("os error 11001")
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

pub(crate) fn print_serve_status(port: u16) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloudflare_edge_530_is_transient_public_smoke_error() {
        let error = anyhow::anyhow!("HTTPS request failed: http status: 530");
        assert!(is_transient_public_smoke_error(&error));
    }

    #[test]
    fn windows_dns_lookup_failure_is_transient_public_smoke_error() {
        let error = anyhow::anyhow!("HTTPS request failed: io: unknown host (os error 11001)");
        assert!(is_transient_public_smoke_error(&error));
    }

    #[test]
    fn auth_failures_are_not_transient_public_smoke_errors() {
        let error = anyhow::anyhow!("HTTPS request failed: http status: 401");
        assert!(!is_transient_public_smoke_error(&error));
    }
}
