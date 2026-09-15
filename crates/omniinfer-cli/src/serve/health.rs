use super::*;

pub(super) fn get_serve_health_state(config: &config::AppConfig) -> Result<serde_json::Value> {
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

pub(super) fn print_serve_ready(
    port: u16,
    state: &serde_json::Value,
    public_url: Option<&str>,
    lan_base_urls: &[String],
    api_key: Option<&str>,
    admin_api_key: Option<&str>,
    public_model_root: Option<&std::path::Path>,
    remote_management: bool,
    print_api_key: bool,
    persistent: bool,
    log_path: &std::path::Path,
    smoke_text: Option<&str>,
) {
    println!();
    println!("OmniInfer service is ready");
    let openai_compatible = json_bool(state, "openai_compatible").unwrap_or(true);
    if let Some(public_url) = public_url {
        if openai_compatible {
            println!("OpenAI Base URL: {}/v1", public_url.trim_end_matches('/'));
        } else {
            println!("Public Gateway URL: {}", public_url.trim_end_matches('/'));
        }
        println!("Health URL: {}/health", public_url.trim_end_matches('/'));
    }
    for lan_base_url in lan_base_urls {
        if openai_compatible {
            println!("LAN Base URL: {lan_base_url}");
        } else {
            println!("LAN Gateway URL: {}", lan_base_url.trim_end_matches("/v1"));
        }
    }
    if openai_compatible {
        println!("Local Base URL: http://127.0.0.1:{port}/v1");
    } else {
        println!("Local Gateway URL: http://127.0.0.1:{port}");
        if let Some(endpoint) = json_str(state, "client_endpoint") {
            println!("VLA Client Endpoint: {endpoint}");
        }
    }
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
    println!(
        "Backend: {}",
        omniinfer_core::backend::names::selector(json_str(state, "backend").unwrap_or("-"))
    );
    if let Some(protocol) = json_str(state, "external_server_protocol") {
        println!("Backend protocol: {protocol}");
    }
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
    if !persistent {
        return;
    }
    println!("Stop: ./omniinfer serve stop --port {port}");
    let remote_base_url = if openai_compatible {
        public_url
            .map(|url| format!("{}/v1", url.trim_end_matches('/')))
            .or_else(|| lan_base_urls.first().cloned())
    } else {
        None
    };
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

pub(super) fn serve_smoke(base_url: &str, api_key: Option<&str>) -> Result<String> {
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

pub(super) fn serve_smoke_with_retry(base_url: &str, api_key: Option<&str>) -> Result<String> {
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

pub(super) fn public_smoke_retry_duration() -> Duration {
    env::var("OMNIINFER_RUST_PUBLIC_SMOKE_RETRY_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(45))
}

pub(super) fn is_transient_public_smoke_error(error: &anyhow::Error) -> bool {
    let text = error.to_string().to_ascii_lowercase();
    text.contains("failed to lookup address")
        || text.contains("name or service not known")
        || text.contains("unknown host")
        || text.contains("no such host")
        || text.contains("os error 11001")
        || text.contains("temporary failure in name resolution")
        || text.contains("connection refused")
        || text.contains("connection reset")
        || text.contains("unexpected end of file")
        || text.contains("unexpected eof")
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
            println!(
                "Backend: {}",
                omniinfer_core::backend::names::selector(json_str(state, "backend").unwrap_or("-"))
            );
            if let Some(protocol) = json_str(state, "external_server_protocol") {
                println!("Backend protocol: {protocol}");
            }
            if let Some(endpoint) = json_str(state, "client_endpoint") {
                println!("Client endpoint: {endpoint}");
            }
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
