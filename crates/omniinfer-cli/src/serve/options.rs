use super::*;

#[derive(Debug, Clone)]
pub(super) struct ServeModelRequest {
    pub(super) model: String,
    pub(super) mmproj: Option<String>,
    pub(super) no_mmproj: bool,
    pub(super) ctx_size: Option<u32>,
    pub(super) backend_port: Option<u16>,
    pub(super) resource_budget_bytes: Option<u64>,
    pub(super) request_defaults: Option<serde_json::Map<String, serde_json::Value>>,
    pub(super) restored: bool,
}

pub(super) fn resolve_serve_listen_host(args: &ServeArgs) -> String {
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

pub(super) fn resolve_serve_restore_model(args: &ServeArgs) -> Option<ServeModelRequest> {
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
        mmproj: (!args.no_mmproj && !selected.no_mmproj)
            .then_some(selected.mmproj)
            .flatten(),
        no_mmproj: args.no_mmproj || selected.no_mmproj,
        ctx_size: selected.ctx_size,
        backend_port: args.backend_port,
        resource_budget_bytes: args.resource_budget_bytes,
        request_defaults: Some(selected.request_defaults),
        restored: true,
    })
}

pub(super) fn wait_for_foreground_service(
    mut rust_gateway: std::process::Child,
    cloudflared_child: Option<std::process::Child>,
    port: u16,
    run_id: &str,
) -> Result<std::process::ExitStatus> {
    let status = rust_gateway.wait()?;
    if let Some(mut tunnel) = cloudflared_child {
        let _ = tunnel.kill();
        let _ = tunnel.wait();
    }
    let backend_closed = serve_state::load_serve_pid_info(port)
        .ok()
        .flatten()
        .and_then(|value| value.backend_port)
        .is_none_or(|backend_port| TcpStream::connect(("127.0.0.1", backend_port)).is_err());
    if backend_closed {
        let _ = serve_state::remove_serve_pid_info_if_run_id(port, run_id);
    }
    Ok(status)
}

pub(super) fn reject_embedded_serve_backend(args: &ServeArgs) -> Result<()> {
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

pub(super) fn resolve_serve_start_backend(args: &ServeArgs) -> Result<Option<String>> {
    if let Some(backend) = args
        .backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(Some(
            backend_registry::BackendRegistry::load_current()
                .resolve(backend)?
                .id
                .clone(),
        ));
    }
    if let Some(default_backend) = args
        .default_backend
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(Some(
            backend_registry::BackendRegistry::load_current()
                .resolve(default_backend)?
                .id
                .clone(),
        ));
    }
    if let Some(selected_backend) = local_state::load_state()
        .ok()
        .and_then(|state| state.selected_backend)
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(Some(
            backend_registry::BackendRegistry::load_current()
                .resolve(&selected_backend)?
                .id
                .clone(),
        ));
    }
    Ok(backend_registry::BackendRegistry::load_current()
        .api_payload(backend_registry::BackendScope::Installed)
        .get("recommended")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string))
}

pub(super) fn validate_serve_remote_access_args(args: &ServeArgs) -> Result<()> {
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

pub(super) fn is_loopback_host(host: &str) -> bool {
    matches!(host.trim(), "" | "127.0.0.1" | "localhost" | "::1")
}

pub(super) fn resolve_serve_api_key(
    args: &ServeArgs,
    generate_session_key: bool,
) -> Result<Option<String>> {
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

pub(super) fn resolve_serve_admin_api_key(args: &ServeArgs) -> Result<Option<String>> {
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

pub(super) fn resolve_serve_admin_api_keys(
    args: &ServeArgs,
) -> Result<Vec<gateway_auth::GatewayAdminApiKey>> {
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

pub(super) fn admin_keys_file_has_entries() -> bool {
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

pub(super) fn lan_base_urls(config: &config::AppConfig, lan_enabled: bool) -> Vec<String> {
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

pub(super) fn detect_primary_lan_ipv4() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    let ip = addr.ip();
    if ip.is_loopback() || !ip.is_ipv4() {
        return None;
    }
    Some(ip.to_string())
}

pub(super) fn generate_session_api_key() -> Result<String> {
    let token: String = rand::rng()
        .sample_iter(Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    Ok(format!("oi_{token}"))
}
