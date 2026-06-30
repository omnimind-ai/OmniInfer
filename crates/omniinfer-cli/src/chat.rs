use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use omniinfer_core::{chat_stream, config, http_client};
use serde_json::{Map, Value, json};

use crate::{
    ChatArgs, ThinkingMode, ensure_local_gateway_running, get_local_json, json_str, json_u64,
    parse_http_error_body, post_local_json,
};

pub(crate) fn print_chat(args: &ChatArgs) -> Result<()> {
    let mut payload = build_chat_payload(args)?;
    if args.no_stream {
        payload.insert("stream".to_string(), json!(false));
        let response = post_local_json(
            "/v1/chat/completions",
            &Value::Object(payload),
            Duration::from_secs(600),
        )?;
        print_chat_response(&response);
        return Ok(());
    }
    payload.insert("stream".to_string(), json!(true));
    payload.insert(
        "stream_options".to_string(),
        json!({ "include_usage": true }),
    );
    print_chat_stream(&Value::Object(payload))
}

fn build_chat_payload(args: &ChatArgs) -> Result<Map<String, Value>> {
    if args.message.is_some() && args.prompt.is_some() {
        anyhow::bail!("Use either positional prompt or --message, not both.");
    }
    let message = args
        .message
        .as_deref()
        .or(args.prompt.as_deref())
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("Please provide a message, for example: omniinfer chat \"Hello\".")
        })?;
    let state = get_local_json("/omni/state", Duration::from_secs(10))?;
    if json_str(&state, "model").is_none() {
        anyhow::bail!("No model is currently loaded.\nRun `omniinfer load -m <model>` first.");
    }

    let mut payload = state
        .get("request_defaults")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    payload.insert("messages".to_string(), build_chat_messages(message, args)?);
    payload.insert(
        "temperature".to_string(),
        json!(args.temperature.unwrap_or_else(|| {
            payload
                .get("temperature")
                .and_then(Value::as_f64)
                .unwrap_or(0.2) as f32
        })),
    );
    payload.insert(
        "max_tokens".to_string(),
        json!(args.max_tokens.unwrap_or_else(|| {
            payload
                .get("max_tokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(2048)
        })),
    );
    payload.insert("stream".to_string(), json!(false));
    if let Some(think) = &args.think {
        payload.insert(
            "think".to_string(),
            json!(matches!(think, ThinkingMode::On)),
        );
    }
    Ok(payload)
}

fn build_chat_messages(message: &str, args: &ChatArgs) -> Result<Value> {
    let Some(image) = args
        .image
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(json!([{ "role": "user", "content": message }]));
    };
    let image_path = PathBuf::from(image);
    if !image_path.is_file() {
        anyhow::bail!("image file does not exist: {}", image_path.display());
    }
    let bytes = std::fs::read(&image_path)?;
    let image_b64 = BASE64_STANDARD.encode(bytes);
    let mime = image_mime_type(&image_path);
    Ok(json!([
        {
            "role": "user",
            "content": [
                { "type": "text", "text": message },
                { "type": "image_url", "image_url": { "url": format!("data:{mime};base64,{image_b64}") } }
            ]
        }
    ]))
}

fn image_mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

fn print_chat_stream(payload: &Value) -> Result<()> {
    let config = config::load_app_config().unwrap_or_default();
    ensure_local_gateway_running(&config)?;
    let url = format!("{}/v1/chat/completions", config.service_base_url());
    println!("Response");
    let mut filter = chat_stream::StreamPrefixFilter::new();
    let mut final_payload = None;
    let response = http_client::post_streaming_lines(
        &url,
        payload,
        "text/event-stream, application/json",
        Duration::from_secs(3600),
        |line| {
            let chunks = chat_stream::parse_chat_stream_line(line).unwrap_or_default();
            for chunk in chunks {
                match chunk {
                    chat_stream::ChatStreamChunk::Text(text) => {
                        if let Some(text) = filter.push(&text)
                            && !text.is_empty()
                        {
                            print!("{text}");
                            let _ = Write::flush(&mut std::io::stdout());
                        }
                    }
                    chat_stream::ChatStreamChunk::Reasoning(_) => {}
                    chat_stream::ChatStreamChunk::Final(payload) => {
                        final_payload = Some(payload);
                    }
                }
            }
        },
    )?;
    if response.status >= 400 {
        let message = parse_http_error_body(&response.body);
        anyhow::bail!(
            "Streaming inference failed with status {}: {}",
            response.status,
            message
        );
    }
    if let Some(text) = filter.finish()
        && !text.is_empty()
    {
        print!("{text}");
    }
    println!();
    if let Some(final_payload) = final_payload {
        print_chat_performance(&final_payload);
    }
    Ok(())
}

fn print_chat_response(response: &Value) {
    let text = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"));
    println!("Response");
    match text {
        Some(Value::String(content)) => println!("{content}"),
        Some(other) => println!("{other}"),
        None => println!("{response}"),
    }
    if response.get("usage").is_some() || response.get("timings").is_some() {
        println!();
        print_chat_performance(response);
    }
}

pub(crate) fn print_chat_performance(response: &Value) {
    println!("Performance");
    println!("  Model: {}", json_str(response, "model").unwrap_or("-"));
    if let Some(usage) = response.get("usage") {
        let prompt = json_u64(usage, "prompt_tokens")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let completion = json_u64(usage, "completion_tokens")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let total = json_u64(usage, "total_tokens")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!("  Tokens: prompt={prompt}, completion={completion}, total={total}");
    }
}
