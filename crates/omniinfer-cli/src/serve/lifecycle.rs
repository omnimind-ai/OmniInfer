use super::*;

pub(crate) fn stop_serve(port: u16) -> Result<()> {
    let _port_lock = serve_state::try_lock_serve_port(port)?;
    let info = serve_state::load_serve_pid_info(port)?;
    stop_serve_locked(port, info, true)
}

pub(super) fn stop_serve_locked(
    port: u16,
    info: Option<serve_state::ServePidInfo>,
    print_status: bool,
) -> Result<()> {
    let mut config = config::load_app_config().unwrap_or_default();
    config.port = port;
    let live_backend = info
        .as_ref()
        .filter(|value| {
            value.backend_port.is_some()
                && value.backend_process_owned.is_none()
                && value.backend_pid.is_none()
        })
        .and_then(|_| live_backend_shutdown_target(&config));
    let backend_process_owned =
        recorded_backend_process_owned(info.as_ref(), live_backend.as_ref());
    if let Some(info) = info.as_ref() {
        validate_recorded_processes(info, port, backend_process_owned)?;
    }
    let url = format!("{}/omni/shutdown", config.service_base_url());
    let shutdown_accepted =
        match http_client::post_json(&url, &serde_json::json!({}), SHUTDOWN_REQUEST_TIMEOUT) {
            Ok(response) => response.status < 400,
            Err(_) => false,
        };

    let mut gateway_closed = wait_for_local_port_closed(
        port,
        if shutdown_accepted {
            GRACEFUL_SHUTDOWN_TIMEOUT
        } else {
            Duration::ZERO
        },
    );
    gateway_closed = gateway_closed
        && wait_for_recorded_process_exit(
            info.as_ref().and_then(|value| value.pid),
            info.as_ref()
                .and_then(|value| value.gateway_process.as_ref()),
            LegacyProcessKind::Gateway,
            info.as_ref(),
            port,
            if shutdown_accepted {
                GRACEFUL_SHUTDOWN_TIMEOUT
            } else {
                Duration::ZERO
            },
        );
    let backend_port =
        managed_backend_port(info.as_ref(), live_backend.as_ref(), backend_process_owned);
    let mut backend_closed = backend_port
        .is_none_or(|backend_port| wait_for_local_port_closed(backend_port, Duration::ZERO));
    backend_closed = backend_closed
        && (backend_process_owned == Some(false)
            || recorded_process_exited(
                info.as_ref().and_then(|value| value.backend_pid),
                info.as_ref()
                    .and_then(|value| value.backend_process.as_ref()),
                LegacyProcessKind::Backend,
                info.as_ref(),
                port,
            ));
    if info.is_some() && (!gateway_closed || !backend_closed) {
        if backend_process_owned != Some(false)
            && let Some(pid) = info.as_ref().and_then(|value| value.backend_pid)
        {
            stop_recorded_process(
                pid,
                info.as_ref()
                    .and_then(|value| value.backend_process.as_ref()),
                LegacyProcessKind::Backend,
                info.as_ref(),
                port,
            )?;
        }
        if let Some(pid) = info.as_ref().and_then(|value| value.pid) {
            stop_recorded_process(
                pid,
                info.as_ref()
                    .and_then(|value| value.gateway_process.as_ref()),
                LegacyProcessKind::Gateway,
                info.as_ref(),
                port,
            )?;
        }
        gateway_closed = wait_for_local_port_closed(port, FORCED_SHUTDOWN_TIMEOUT);
        gateway_closed = gateway_closed
            && wait_for_recorded_process_exit(
                info.as_ref().and_then(|value| value.pid),
                info.as_ref()
                    .and_then(|value| value.gateway_process.as_ref()),
                LegacyProcessKind::Gateway,
                info.as_ref(),
                port,
                FORCED_SHUTDOWN_TIMEOUT,
            );
        backend_closed = backend_port.is_none_or(|backend_port| {
            wait_for_local_port_closed(backend_port, FORCED_SHUTDOWN_TIMEOUT)
        });
        backend_closed = backend_closed
            && (backend_process_owned == Some(false)
                || wait_for_recorded_process_exit(
                    info.as_ref().and_then(|value| value.backend_pid),
                    info.as_ref()
                        .and_then(|value| value.backend_process.as_ref()),
                    LegacyProcessKind::Backend,
                    info.as_ref(),
                    port,
                    FORCED_SHUTDOWN_TIMEOUT,
                ));
    }
    let mut tunnel_closed = true;
    if let Some(pid) = info.as_ref().and_then(|value| value.cloudflared_pid) {
        stop_recorded_process(
            pid,
            info.as_ref()
                .and_then(|value| value.cloudflared_process.as_ref()),
            LegacyProcessKind::Cloudflared,
            info.as_ref(),
            port,
        )?;
        tunnel_closed = wait_for_recorded_process_exit(
            Some(pid),
            info.as_ref()
                .and_then(|value| value.cloudflared_process.as_ref()),
            LegacyProcessKind::Cloudflared,
            info.as_ref(),
            port,
            FORCED_SHUTDOWN_TIMEOUT,
        );
    }

    if (shutdown_accepted || info.is_some()) && gateway_closed && backend_closed && tunnel_closed {
        remove_recorded_serve_state(port, info.as_ref())?;
        if print_status {
            println!("OmniInfer service stopped on port {port}");
        }
        Ok(())
    } else if !shutdown_accepted && info.is_none() {
        if print_status {
            println!("OmniInfer service is not running on port {port}");
        }
        Ok(())
    } else {
        anyhow::bail!("failed to stop OmniInfer service on port {port} within the shutdown timeout")
    }
}

#[derive(Clone, Copy)]
enum LegacyProcessKind {
    Gateway,
    Cloudflared,
    Backend,
}

#[derive(Debug, Clone, Copy)]
struct BackendShutdownTarget {
    process_owned: bool,
    port: Option<u16>,
}

fn live_backend_shutdown_target(config: &config::AppConfig) -> Option<BackendShutdownTarget> {
    let url = format!("{}/health?deep=true", config.service_base_url());
    let response = http_client::get_json(&url, Duration::from_secs(2)).ok()?;
    if response.status >= 400 {
        return None;
    }
    let state = response.body.get("omni").unwrap_or(&response.body);
    let process_owned = json_bool(state, "process_owned")?;
    let port = json_u64(state, "backend_port").and_then(|value| u16::try_from(value).ok());
    Some(BackendShutdownTarget {
        process_owned,
        port,
    })
}

fn recorded_backend_process_owned(
    info: Option<&serve_state::ServePidInfo>,
    live_backend: Option<&BackendShutdownTarget>,
) -> Option<bool> {
    info.and_then(|value| value.backend_process_owned)
        .or_else(|| info.and_then(|value| value.backend_pid).map(|_| true))
        .or_else(|| live_backend.map(|value| value.process_owned))
}

fn managed_backend_port(
    info: Option<&serve_state::ServePidInfo>,
    live_backend: Option<&BackendShutdownTarget>,
    process_owned: Option<bool>,
) -> Option<u16> {
    if process_owned == Some(false) {
        return None;
    }
    info.and_then(|value| value.backend_port).or_else(|| {
        live_backend
            .filter(|value| value.process_owned)
            .and_then(|value| value.port)
    })
}

fn validate_recorded_processes(
    info: &serve_state::ServePidInfo,
    port: u16,
    backend_process_owned: Option<bool>,
) -> Result<()> {
    let mut recorded = vec![
        (
            info.pid,
            info.gateway_process.as_ref(),
            LegacyProcessKind::Gateway,
            "gateway",
        ),
        (
            info.cloudflared_pid,
            info.cloudflared_process.as_ref(),
            LegacyProcessKind::Cloudflared,
            "cloudflared",
        ),
    ];
    if backend_process_owned != Some(false) {
        recorded.push((
            info.backend_pid,
            info.backend_process.as_ref(),
            LegacyProcessKind::Backend,
            "backend",
        ));
    }
    for (pid, identity, kind, label) in recorded {
        let Some(pid) = pid else {
            continue;
        };
        if let Some(identity) = identity {
            if identity.pid != pid
                || serve_state::process_identity_status(identity)
                    == serve_state::ProcessIdentityStatus::Mismatched
            {
                anyhow::bail!(
                    "refusing to stop {label} PID {pid}: process identity does not match serve state"
                );
            }
        } else if legacy_process_status(pid, kind, info, port) == LegacyProcessStatus::Mismatched {
            anyhow::bail!(
                "refusing to stop legacy {label} PID {pid}: process ownership could not be verified"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegacyProcessStatus {
    Running,
    Exited,
    Mismatched,
}

fn legacy_process_status(
    pid: u32,
    kind: LegacyProcessKind,
    info: &serve_state::ServePidInfo,
    port: u16,
) -> LegacyProcessStatus {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

    let pid = sysinfo::Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing()
            .with_cmd(UpdateKind::Always)
            .with_exe(UpdateKind::Always)
            .without_tasks(),
    );
    let Some(process) = system.process(pid) else {
        return LegacyProcessStatus::Exited;
    };
    let args = process
        .cmd()
        .iter()
        .map(|value| value.to_string_lossy())
        .collect::<Vec<_>>();
    let joined = args.join(" ");
    let joined_lower = joined.to_ascii_lowercase();
    let port_text = port.to_string();
    let matches = match kind {
        LegacyProcessKind::Gateway => {
            joined_lower.contains("gateway")
                && args
                    .windows(2)
                    .any(|pair| pair[0] == "--port" && pair[1] == port_text)
        }
        LegacyProcessKind::Cloudflared => {
            joined_lower.contains("cloudflared")
                && joined_lower.contains("tunnel")
                && joined.contains(&format!("127.0.0.1:{port}"))
        }
        LegacyProcessKind::Backend => {
            let backend_port = info.backend_port.map(|value| value.to_string());
            let model = info
                .model
                .as_deref()
                .filter(|value| !value.trim().is_empty());
            backend_port
                .as_ref()
                .is_some_and(|value| joined.contains(value))
                && model.is_some_and(|value| joined.contains(value))
        }
    };
    if matches {
        LegacyProcessStatus::Running
    } else {
        LegacyProcessStatus::Mismatched
    }
}

fn recorded_process_exited(
    pid: Option<u32>,
    identity: Option<&serve_state::ProcessIdentity>,
    kind: LegacyProcessKind,
    info: Option<&serve_state::ServePidInfo>,
    port: u16,
) -> bool {
    let Some(pid) = pid else {
        return true;
    };
    if let Some(identity) = identity {
        return serve_state::process_identity_status(identity)
            != serve_state::ProcessIdentityStatus::Running;
    }
    info.is_some_and(|info| {
        legacy_process_status(pid, kind, info, port) != LegacyProcessStatus::Running
    })
}

fn wait_for_recorded_process_exit(
    pid: Option<u32>,
    identity: Option<&serve_state::ProcessIdentity>,
    kind: LegacyProcessKind,
    info: Option<&serve_state::ServePidInfo>,
    port: u16,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    while !recorded_process_exited(pid, identity, kind, info, port) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    recorded_process_exited(pid, identity, kind, info, port)
}

fn stop_recorded_process(
    pid: u32,
    identity: Option<&serve_state::ProcessIdentity>,
    kind: LegacyProcessKind,
    info: Option<&serve_state::ServePidInfo>,
    port: u16,
) -> Result<()> {
    if !recorded_process_exited(Some(pid), identity, kind, info, port) {
        stop_process(pid);
    }
    Ok(())
}

fn remove_recorded_serve_state(port: u16, info: Option<&serve_state::ServePidInfo>) -> Result<()> {
    if let Some(run_id) = info.and_then(|value| value.run_id.as_deref()) {
        serve_state::remove_serve_pid_info_if_run_id(port, run_id)?;
    } else {
        serve_state::remove_serve_pid_info(port)?;
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

pub(super) fn ensure_serve_port_available(config: &config::AppConfig) -> Result<()> {
    #[cfg(debug_assertions)]
    if env_flag("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT") {
        return Ok(());
    }
    match TcpListener::bind((config.host.as_str(), config.port)) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(error) => anyhow::bail!(
            "cannot start OmniInfer service: {}:{} is already in use ({error}). Stop the existing service or choose another --port.",
            config.host,
            config.port
        ),
    }
}

pub(super) fn cleanup_failed_serve(
    gateway: &mut std::process::Child,
    cloudflared: Option<&mut std::process::Child>,
    port: u16,
    run_id: &str,
) {
    if let Some(tunnel) = cloudflared {
        stop_process(tunnel.id());
        let _ = wait_for_child_exit(tunnel, FORCED_SHUTDOWN_TIMEOUT);
    }

    let mut config = config::load_app_config().unwrap_or_default();
    config.port = port;
    let info = serve_state::load_serve_pid_info(port).ok().flatten();
    let url = format!("{}/omni/shutdown", config.service_base_url());
    let shutdown_accepted =
        http_client::post_json(&url, &serde_json::json!({}), SHUTDOWN_REQUEST_TIMEOUT)
            .is_ok_and(|response| response.status < 400);
    let mut gateway_exited = wait_for_child_exit(
        gateway,
        if shutdown_accepted {
            GRACEFUL_SHUTDOWN_TIMEOUT
        } else {
            Duration::ZERO
        },
    );
    let mut gateway_closed = wait_for_local_port_closed(
        port,
        if shutdown_accepted {
            GRACEFUL_SHUTDOWN_TIMEOUT
        } else {
            Duration::ZERO
        },
    );
    if !shutdown_accepted || !gateway_exited || !gateway_closed {
        if let Some(pid) = info.as_ref().and_then(|value| value.backend_pid) {
            stop_process(pid);
        }
        if !gateway_exited {
            stop_process(gateway.id());
        }
        gateway_exited = gateway_exited || wait_for_child_exit(gateway, FORCED_SHUTDOWN_TIMEOUT);
        gateway_closed =
            gateway_closed || wait_for_local_port_closed(port, FORCED_SHUTDOWN_TIMEOUT);
    }

    let backend_closed = info
        .as_ref()
        .and_then(|value| value.backend_port)
        .is_none_or(|backend_port| {
            wait_for_local_port_closed(backend_port, FORCED_SHUTDOWN_TIMEOUT)
        });
    if gateway_exited && gateway_closed && backend_closed {
        let _ = serve_state::remove_serve_pid_info_if_run_id(port, run_id);
    } else {
        eprintln!(
            "OmniInfer: failed startup cleanup did not fully stop the gateway on port {port}; run `omniinfer serve stop --port {port}`"
        );
    }
}

pub(super) fn cleanup_smoke_serve(
    gateway: &mut std::process::Child,
    cloudflared: Option<&mut std::process::Child>,
    port: u16,
    backend_pid: Option<u32>,
    backend_port: Option<u16>,
    run_id: &str,
) -> Result<()> {
    let mut cleanup_errors = Vec::new();
    if let Some(tunnel) = cloudflared
        && !wait_for_child_exit(tunnel, Duration::ZERO)
    {
        stop_process(tunnel.id());
        if !wait_for_child_exit(tunnel, FORCED_SHUTDOWN_TIMEOUT) {
            cleanup_errors.push("cloudflared did not exit".to_string());
        }
    }

    let mut config = config::load_app_config().unwrap_or_default();
    config.port = port;
    let url = format!("{}/omni/shutdown", config.service_base_url());
    let shutdown_accepted =
        http_client::post_json(&url, &serde_json::json!({}), SHUTDOWN_REQUEST_TIMEOUT)
            .is_ok_and(|response| response.status < 400);

    let gateway_exited = wait_for_child_exit(gateway, GRACEFUL_SHUTDOWN_TIMEOUT);
    let gateway_closed = wait_for_local_port_closed(port, GRACEFUL_SHUTDOWN_TIMEOUT);
    let backend_closed = backend_port
        .is_none_or(|value| wait_for_local_port_closed(value, GRACEFUL_SHUTDOWN_TIMEOUT));
    if !shutdown_accepted || !gateway_exited || !gateway_closed || !backend_closed {
        if let Some(pid) = backend_pid {
            stop_process(pid);
        }
        if !gateway_exited {
            stop_process(gateway.id());
        }
    }

    if !wait_for_child_exit(gateway, FORCED_SHUTDOWN_TIMEOUT) {
        cleanup_errors.push("gateway did not exit".to_string());
    }
    if !wait_for_local_port_closed(port, FORCED_SHUTDOWN_TIMEOUT) {
        cleanup_errors.push(format!("gateway port {port} is still in use"));
    }
    if let Some(backend_port) = backend_port
        && !wait_for_local_port_closed(backend_port, FORCED_SHUTDOWN_TIMEOUT)
    {
        cleanup_errors.push(format!("backend port {backend_port} is still in use"));
    }
    if cleanup_errors.is_empty() {
        serve_state::remove_serve_pid_info_if_run_id(port, run_id)?;
        println!("Smoke test cleanup complete; gateway/backend stopped and port {port} released");
        Ok(())
    } else {
        anyhow::bail!(
            "smoke test cleanup failed: {}; serve metadata was retained for `omniinfer serve stop --port {port}`",
            cleanup_errors.join("; ")
        )
    }
}

fn wait_for_child_exit(child: &mut std::process::Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let _ = child.wait();
                return true;
            }
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50));
            }
            Ok(None) | Err(_) => return false,
        }
    }
}

pub(crate) fn stop_process(pid: u32) {
    #[cfg(unix)]
    {
        signal_process_group_or_pid(pid, "-TERM");
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if !process_group_or_pid_exists(pid) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        signal_process_group_or_pid(pid, "-KILL");
    }
    #[cfg(windows)]
    {
        let mut command = ProcessCommand::new("taskkill");
        command
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        hide_child_window(&mut command);
        let _ = command.status();
    }
}

#[cfg(unix)]
fn signal_process_group_or_pid(pid: u32, signal: &str) {
    let group_signalled = ProcessCommand::new("kill")
        .args([signal, "--", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !group_signalled {
        let _ = ProcessCommand::new("kill")
            .args([signal, &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

#[cfg(unix)]
fn process_group_or_pid_exists(pid: u32) -> bool {
    ProcessCommand::new("kill")
        .args(["-0", "--", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
        || ProcessCommand::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
}
