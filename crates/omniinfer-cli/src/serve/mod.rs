use std::env;
use std::fs::OpenOptions;
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
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

const SHUTDOWN_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
const FORCED_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

mod lifecycle;

pub(crate) use lifecycle::stop_process;
pub(crate) use lifecycle::stop_serve;
use lifecycle::{
    cleanup_failed_serve, cleanup_smoke_serve, ensure_serve_port_available, stop_serve_locked,
};
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
    let mut port_lock = Some(serve_state::try_lock_serve_port(public_config.port)?);
    if let Some(existing) = serve_state::load_serve_pid_info(public_config.port)? {
        stop_serve_locked(public_config.port, Some(existing), false).map_err(|error| {
            anyhow::anyhow!(
                "cannot replace the existing managed serve state on port {}: {error}",
                public_config.port
            )
        })?;
    }
    ensure_serve_port_available(&public_config)?;
    let cloudflared = if args.cloudflare {
        Some(resolve_cloudflared(args.cloudflared_path.as_deref())?)
    } else {
        None
    };
    let interrupt = (!args.detach)
        .then(|| install_foreground_ctrl_c_handler(public_config.port, true))
        .transpose()?;
    let log_path = paths::local_logs_dir().join(format!("serve-{}.log", public_config.port));
    println!("Starting OmniInfer service on port {}...", config.port);
    println!("Log: {}", log_path.display());
    let run_id = format!("serve-{:032x}", rand::random::<u128>());
    let mut rust_gateway = start_rust_gateway_child(
        &public_config,
        args,
        &log_path,
        api_key.as_deref(),
        admin_api_key.as_deref(),
        &admin_api_keys,
        public_model_root.as_deref(),
    )?;
    let gateway_process = match serve_state::capture_process_identity(rust_gateway.id()) {
        Some(identity) => identity,
        #[cfg(debug_assertions)]
        None if env_flag("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT") => {
            serve_state::ProcessIdentity {
                pid: rust_gateway.id(),
                start_time: 0,
                executable: None,
                name: "external-test-gateway".to_string(),
            }
        }
        None => {
            cleanup_failed_serve(&mut rust_gateway, None, public_config.port, &run_id);
            anyhow::bail!("gateway exited before its process identity could be recorded");
        }
    };
    let mut serve_info = serve_state::ServePidInfo {
        run_id: Some(run_id.clone()),
        phase: Some("starting".to_string()),
        pid: Some(rust_gateway.id()),
        gateway_process: Some(gateway_process),
        cloudflared_pid: None,
        cloudflared_process: None,
        port: Some(public_config.port),
        log: Some(log_path.display().to_string()),
        public_url: None,
        openai_base_url: None,
        backend: None,
        model: None,
        mmproj: None,
        ctx_size: None,
        backend_ready: Some(false),
        backend_pid: None,
        backend_process: None,
        backend_port: None,
        backend_process_owned: None,
    };
    if let Err(error) = serve_state::save_serve_pid_info(&serve_info) {
        cleanup_failed_serve(&mut rust_gateway, None, public_config.port, &run_id);
        return Err(error.into());
    }
    if interrupt
        .as_ref()
        .is_some_and(ForegroundCtrlCHandler::interrupted)
    {
        cleanup_failed_serve(&mut rust_gateway, None, public_config.port, &run_id);
        return Err(anyhow::anyhow!("startup interrupted"));
    }
    if let Some(interrupt) = &interrupt {
        interrupt.arm();
        if interrupt.interrupted() {
            cleanup_failed_serve(&mut rust_gateway, None, public_config.port, &run_id);
            return Err(anyhow::anyhow!("startup interrupted"));
        }
    }
    if let Err(error) = wait_for_gateway_ready(&public_config) {
        cleanup_failed_serve(&mut rust_gateway, None, public_config.port, &run_id);
        return Err(error);
    }
    let mut cloudflared_child = None;
    let mut public_url = None;
    if let Some(cloudflared) = cloudflared {
        let local_url = format!("http://127.0.0.1:{}", config.port);
        let (child, url) = match start_cloudflare_quick_tunnel(
            &cloudflared,
            &local_url,
            &log_path,
            args.detach,
            |child| {
                let Some(identity) = serve_state::capture_process_identity(child.id()) else {
                    anyhow::bail!(
                        "cloudflared exited before its process identity could be recorded"
                    );
                };
                serve_info.cloudflared_pid = Some(child.id());
                serve_info.cloudflared_process = Some(identity);
                serve_state::save_serve_pid_info(&serve_info)?;
                Ok(())
            },
        ) {
            Ok(result) => result,
            Err(error) => {
                cleanup_failed_serve(&mut rust_gateway, None, public_config.port, &run_id);
                return Err(error);
            }
        };
        cloudflared_child = Some(child);
        public_url = Some(url);
        serve_info.public_url = public_url.clone();
        serve_info.openai_base_url = public_url
            .as_ref()
            .map(|url| format!("{}/v1", url.trim_end_matches('/')));
        if let Err(error) = serve_state::save_serve_pid_info(&serve_info) {
            cleanup_failed_serve(
                &mut rust_gateway,
                cloudflared_child.as_mut(),
                public_config.port,
                &run_id,
            );
            return Err(error.into());
        }
    }
    if interrupt
        .as_ref()
        .is_some_and(ForegroundCtrlCHandler::interrupted)
    {
        cleanup_failed_serve(
            &mut rust_gateway,
            cloudflared_child.as_mut(),
            public_config.port,
            &run_id,
        );
        return Err(anyhow::anyhow!("startup interrupted"));
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
                no_mmproj: args.no_mmproj,
                ctx_size: args.ctx_size,
                backend_port: args.backend_port,
                resource_budget_bytes: args.resource_budget_bytes,
                request_defaults: None,
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
                no_mmproj: model.no_mmproj,
                ctx_size: model.ctx_size,
                backend_port: model.backend_port,
                resource_budget_bytes: model.resource_budget_bytes,
                config: None,
                backend_extra_args: Vec::new(),
                request_defaults: model.request_defaults,
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
            cleanup_failed_serve(
                &mut rust_gateway,
                cloudflared_child.as_mut(),
                public_config.port,
                &run_id,
            );
            return Err(error);
        }
    }
    let state = match get_serve_health_state(&public_config) {
        Ok(state) => state,
        Err(error) => {
            cleanup_failed_serve(
                &mut rust_gateway,
                cloudflared_child.as_mut(),
                public_config.port,
                &run_id,
            );
            return Err(error);
        }
    };
    let backend_pid = json_u64(&state, "backend_pid").and_then(|value| u32::try_from(value).ok());
    let backend_port = json_u64(&state, "backend_port").and_then(|value| u16::try_from(value).ok());
    serve_info.phase = Some("ready".to_string());
    serve_info.backend = json_str(&state, "backend").map(str::to_string);
    serve_info.model = json_str(&state, "model").map(str::to_string);
    serve_info.mmproj = json_str(&state, "mmproj").map(str::to_string);
    serve_info.ctx_size = json_u64(&state, "ctx_size").and_then(|value| u32::try_from(value).ok());
    serve_info.backend_ready = json_bool(&state, "backend_ready");
    serve_info.backend_pid = backend_pid;
    serve_info.backend_process = backend_pid.and_then(serve_state::capture_process_identity);
    serve_info.backend_port = backend_port;
    serve_info.backend_process_owned = json_bool(&state, "process_owned");
    if backend_pid.is_some() && serve_info.backend_process.is_none() {
        cleanup_failed_serve(
            &mut rust_gateway,
            cloudflared_child.as_mut(),
            public_config.port,
            &run_id,
        );
        anyhow::bail!("backend exited before its process identity could be recorded");
    }
    if let Err(error) = serve_state::save_serve_pid_info(&serve_info) {
        cleanup_failed_serve(
            &mut rust_gateway,
            cloudflared_child.as_mut(),
            public_config.port,
            &run_id,
        );
        return Err(error.into());
    }
    if !args.smoke_test {
        drop(port_lock.take());
    }
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
        !args.smoke_test,
        &log_path,
        smoke_text.as_deref(),
    );
    if args.smoke_test {
        let cleanup_result = cleanup_smoke_serve(
            &mut rust_gateway,
            cloudflared_child.as_mut(),
            public_config.port,
            backend_pid,
            backend_port,
            &run_id,
        );
        if let Err(cleanup_error) = cleanup_result {
            if smoke_failed {
                anyhow::bail!("smoke test failed; {cleanup_error}");
            }
            return Err(cleanup_error);
        }
        if smoke_failed {
            anyhow::bail!("smoke test failed");
        }
        drop(port_lock.take());
        return Ok(());
    }
    if !args.detach {
        println!("Press Ctrl+C to stop.");
        let status = wait_for_foreground_service(
            rust_gateway,
            cloudflared_child,
            public_config.port,
            &run_id,
        )?;
        drop(interrupt);
        if !status.success() {
            anyhow::bail!("OmniInfer service exited with status {status}");
        }
    }
    Ok(())
}

mod options;

pub(crate) use options::parse_admin_api_keys;
use options::*;
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
    paths::propagate_cli_roots(&mut command);
    command
        .arg("gateway")
        .arg("--host")
        .arg(&public_config.host)
        .arg("--port")
        .arg(public_config.port.to_string())
        .arg("--startup-timeout")
        .arg(public_config.startup_timeout.max(1.0).round().to_string())
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
    } else {
        isolate_child_process_group(&mut command);
    }
    Ok(command.spawn()?)
}

#[cfg(unix)]
fn isolate_child_process_group(command: &mut ProcessCommand) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
fn isolate_child_process_group(command: &mut ProcessCommand) {
    let _ = command;
}

pub(super) struct ForegroundCtrlCHandler {
    stopped: Arc<AtomicBool>,
    armed: Arc<AtomicBool>,
    interrupted: Arc<AtomicBool>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for ForegroundCtrlCHandler {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.armed.store(false, Ordering::SeqCst);
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl ForegroundCtrlCHandler {
    pub(super) fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }

    pub(super) fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }
}

pub(super) fn install_foreground_ctrl_c_handler(
    port: u16,
    exit_after_shutdown: bool,
) -> Result<ForegroundCtrlCHandler> {
    let stopped = Arc::new(AtomicBool::new(false));
    let armed = Arc::new(AtomicBool::new(false));
    let interrupted = Arc::new(AtomicBool::new(false));
    let stopped_for_task = Arc::clone(&stopped);
    let armed_for_task = Arc::clone(&armed);
    let interrupted_for_task = Arc::clone(&interrupted);
    let (cancel, cancel_rx) = tokio::sync::oneshot::channel();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = thread::spawn(move || {
        let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            let _ = ready_tx.send(Err("failed to create Ctrl+C listener runtime"));
            return;
        };
        #[cfg(unix)]
        runtime.block_on(async move {
            let mut signal =
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
                    Ok(signal) => signal,
                    Err(_) => {
                        let _ = ready_tx.send(Err("failed to register Ctrl+C listener"));
                        return;
                    }
                };
            let _ = ready_tx.send(Ok(()));
            tokio::select! {
                _ = signal.recv() => shutdown_after_interrupt(
                    &stopped_for_task,
                    &armed_for_task,
                    &interrupted_for_task,
                    port,
                    exit_after_shutdown,
                ),
                _ = cancel_rx => {}
            }
        });
        #[cfg(not(unix))]
        runtime.block_on(async move {
            let _ = ready_tx.send(Ok(()));
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    if result.is_ok() {
                        shutdown_after_interrupt(
                            &stopped_for_task,
                            &armed_for_task,
                            &interrupted_for_task,
                            port,
                            exit_after_shutdown,
                        );
                    }
                }
                _ = cancel_rx => {}
            }
        });
    });
    match ready_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => Ok(ForegroundCtrlCHandler {
            stopped,
            armed,
            interrupted,
            cancel: Some(cancel),
            thread: Some(thread),
        }),
        Ok(Err(error)) => {
            let _ = thread.join();
            anyhow::bail!("{error}")
        }
        Err(_) => {
            let _ = thread.join();
            anyhow::bail!("timed out registering Ctrl+C listener")
        }
    }
}

fn shutdown_after_interrupt(
    stopped: &AtomicBool,
    armed: &AtomicBool,
    interrupted: &AtomicBool,
    port: u16,
    exit_after_shutdown: bool,
) {
    if stopped.swap(true, Ordering::SeqCst) {
        return;
    }
    interrupted.store(true, Ordering::SeqCst);
    if armed.load(Ordering::SeqCst) {
        if let Err(error) = stop_serve_during_startup(port) {
            eprintln!("OmniInfer: Ctrl+C cleanup failed: {error}");
        }
        if exit_after_shutdown {
            std::process::exit(130);
        }
    }
}

fn stop_serve_during_startup(port: u16) -> Result<()> {
    let info = serve_state::load_serve_pid_info(port)?;
    stop_serve_locked(port, info, true)
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

mod health;

pub(crate) use health::print_serve_status;
use health::{
    get_serve_health_state, is_transient_public_smoke_error, print_serve_ready, serve_smoke,
    serve_smoke_with_retry,
};
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
    fn proxy_eof_is_transient_public_smoke_error() {
        let error = anyhow::anyhow!("HTTPS request failed: io: unexpected end of file");
        assert!(is_transient_public_smoke_error(&error));
    }

    #[test]
    fn auth_failures_are_not_transient_public_smoke_errors() {
        let error = anyhow::anyhow!("HTTPS request failed: http status: 401");
        assert!(!is_transient_public_smoke_error(&error));
    }
}
