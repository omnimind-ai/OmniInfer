use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use omniinfer_core::{config, http_client, model_load, paths};

use crate::json_str;

pub(crate) fn get_local_json(endpoint: &str, timeout: Duration) -> Result<serde_json::Value> {
    let config = config::load_app_config().unwrap_or_default();
    get_local_json_for_config(endpoint, timeout, &config)
}

pub(crate) fn get_local_json_for_config(
    endpoint: &str,
    timeout: Duration,
    config: &config::AppConfig,
) -> Result<serde_json::Value> {
    ensure_local_gateway_running(config)?;
    let url = format!("{}{}", config.service_base_url(), endpoint);
    let response = http_client::get_json(&url, timeout)?;
    if response.status >= 400 {
        anyhow::bail!("GET {endpoint} failed with status {}", response.status);
    }
    Ok(response.body)
}

pub(crate) fn post_local_json(
    endpoint: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let config = config::load_app_config().unwrap_or_default();
    post_local_json_for_config(endpoint, body, timeout, &config)
}

pub(crate) fn post_local_json_for_config(
    endpoint: &str,
    body: &serde_json::Value,
    timeout: Duration,
    config: &config::AppConfig,
) -> Result<serde_json::Value> {
    post_local_json_for_config_with_autostart(endpoint, body, timeout, config, true)
}

pub(crate) fn post_local_json_for_config_with_autostart(
    endpoint: &str,
    body: &serde_json::Value,
    timeout: Duration,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<serde_json::Value> {
    if autostart {
        ensure_local_gateway_running(config)?;
    }
    let url = format!("{}{}", config.service_base_url(), endpoint);
    let response = http_client::post_json(&url, body, timeout)?;
    if response.status >= 400 {
        anyhow::bail!("POST {endpoint} failed with status {}", response.status);
    }
    Ok(response.body)
}

pub(crate) fn post_local_model_load_for_config_with_autostart(
    body: &serde_json::Value,
    verbose: bool,
    timeout: Duration,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<serde_json::Value> {
    if autostart {
        ensure_local_gateway_running(config)?;
    }
    let url = format!("{}/omni/model/select", config.service_base_url());
    let response = http_client::post_streaming_lines(
        &url,
        body,
        "text/event-stream, application/json",
        timeout,
        |line| print_model_load_progress_line(line, verbose),
    )?;
    if response.status >= 400 {
        let message = parse_http_error_body(&response.body);
        anyhow::bail!(
            "POST /omni/model/select failed with status {}: {}",
            response.status,
            message
        );
    }
    let (result, _events) =
        model_load::parse_model_load_response(response.content_type.as_deref(), &response.body)?;
    Ok(result)
}

fn print_model_load_progress_line(line: &str, verbose: bool) {
    let Some(data) = line.trim().strip_prefix("data:") else {
        return;
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return;
    }
    let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    let event_type = json_str(&event, "type").unwrap_or("");
    let message = json_str(&event, "message").unwrap_or("");
    match event_type {
        "log" if verbose && !message.is_empty() => println!("  {message}"),
        "done" | "error" | "log" => {}
        _ if !message.is_empty() => println!("{message}"),
        _ => {}
    }
}

pub(crate) fn parse_http_error_body(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body.trim())
        .ok()
        .and_then(|payload| {
            payload
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .or_else(|| json_str(&payload, "message").map(str::to_string))
        })
        .unwrap_or_else(|| body.trim().to_string())
}

pub(crate) fn ensure_local_gateway_running(config: &config::AppConfig) -> Result<()> {
    if is_gateway_running(config) {
        return Ok(());
    }
    anyhow::bail!(
        "local OmniInfer service is not running at {}. Start it with `omniinfer serve`.",
        config.service_base_url()
    )
}

fn is_gateway_running(config: &config::AppConfig) -> bool {
    let url = format!("{}/health", config.service_base_url());
    match http_client::get_json(&url, Duration::from_secs(2)) {
        Ok(response) if response.status == 200 => {
            response
                .body
                .get("status")
                .and_then(serde_json::Value::as_str)
                == Some("ok")
        }
        _ => false,
    }
}

pub(crate) fn wait_for_gateway_ready(config: &config::AppConfig) -> Result<()> {
    let timeout = Duration::from_secs(config.startup_timeout.max(10.0).round() as u64);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if is_gateway_running(config) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(500));
    }
    let log_path = paths::local_logs_dir().join("gateway.log");
    let tail = std::fs::read_to_string(&log_path)
        .map(|text| tail_chars(&text, 4000))
        .unwrap_or_default();
    anyhow::bail!("failed to start local OmniInfer service in time\n{tail}");
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars().rev().take(max_chars).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}
