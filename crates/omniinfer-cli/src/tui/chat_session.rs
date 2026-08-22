use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamSection {
    Assistant,
    Reasoning,
}

fn enter_stream_section(active: &mut Option<StreamSection>, next: StreamSection) -> bool {
    if *active == Some(next) {
        return false;
    }
    *active = Some(next);
    true
}

fn print_stream_section(active: &mut Option<StreamSection>, next: StreamSection) {
    let had_active_section = active.is_some();
    if !enter_stream_section(active, next) {
        return;
    }
    if had_active_section {
        println!();
    }
    match next {
        StreamSection::Assistant => print_message_header("Assistant", MessageKind::Assistant),
        StreamSection::Reasoning => {
            print_message_header("Reasoning", MessageKind::Reasoning);
            print!("  ");
        }
    }
}

pub(super) fn chat_loop(config: &config::AppConfig, backend: String) -> Result<()> {
    let mut session = ChatSession {
        backend,
        reasoning_visible: local_state::load_state()
            .map(|state| state.tui_show_reasoning)
            .unwrap_or(false),
        messages: Vec::new(),
        last_usage: None,
    };
    print_chat_header(&session);
    loop {
        let message = prompt_default("You", "")?;
        let message = message.trim();
        if message.is_empty() {
            continue;
        }
        match message {
            "/exit" => return Ok(()),
            "/backend" => {
                if let Some(backend) = choose_backend()? {
                    activate_backend(config, &backend)?;
                    session.messages.clear();
                    if let Some(model) = choose_model(config, true, Some(&backend))? {
                        load_model_for_chat(
                            config,
                            &mut session,
                            model.to_string_lossy().as_ref(),
                        )?;
                    } else {
                        session.backend = backend;
                        print_chat_header(&session);
                    }
                }
            }
            "/model" => {
                if let Some(model) = choose_model(config, true, None)? {
                    load_model_for_chat(config, &mut session, model.to_string_lossy().as_ref())?;
                }
            }
            "/clear" => {
                clear_screen();
                print_header("OmniInfer", "Local inference console");
                print_chat_header(&session);
            }
            "/status" => print_status(config, &session)?,
            "/help" => print_help(),
            "/think" | "/thinking" => toggle_thinking(config)?,
            "/reasoning" => toggle_reasoning(&mut session)?,
            _ if message.starts_with("/think ") || message.starts_with("/thinking ") => {
                set_thinking(config, message.split_whitespace().nth(1))?;
            }
            _ if message.starts_with("/reasoning ") => {
                set_reasoning(&mut session, message.split_whitespace().nth(1))?;
            }
            _ => send_chat_message(config, &mut session, message)?,
        }
    }
}

fn load_model_for_chat(
    config: &config::AppConfig,
    session: &mut ChatSession,
    model: &str,
) -> Result<()> {
    match load_model_interactive(config, model) {
        Ok(loaded) => {
            session.backend = loaded;
            session.messages.clear();
        }
        Err(error) => {
            notice(&format!("Model load failed: {error}"), NoticeKind::Warning);
            notice(
                "Still in chat. Use /model to pick another model.",
                NoticeKind::Warning,
            );
        }
    }
    print_chat_header(session);
    Ok(())
}

fn send_chat_message(
    config: &config::AppConfig,
    session: &mut ChatSession,
    message: &str,
) -> Result<()> {
    let state = get_local_json_for_config("/omni/state", Duration::from_secs(10), config)?;
    if json_str(&state, "model").is_none() {
        anyhow::bail!("No model is currently loaded. Use /model first.");
    }
    let mut payload = state
        .get("request_defaults")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut messages = session.messages.clone();
    messages.push(serde_json::json!({ "role": "user", "content": message }));
    payload.insert("messages".to_string(), Value::Array(messages));
    payload.insert("stream".to_string(), serde_json::json!(true));
    payload.insert(
        "stream_options".to_string(),
        serde_json::json!({ "include_usage": true }),
    );
    payload
        .entry("temperature")
        .or_insert(serde_json::json!(0.2));
    payload
        .entry("max_tokens")
        .or_insert(serde_json::json!(2048));
    let assistant_text = stream_chat_response(config, &Value::Object(payload), session)?;
    if !assistant_text.trim().is_empty() {
        session
            .messages
            .push(serde_json::json!({ "role": "user", "content": message }));
        session
            .messages
            .push(serde_json::json!({ "role": "assistant", "content": assistant_text }));
    }
    Ok(())
}

fn stream_chat_response(
    config: &config::AppConfig,
    payload: &Value,
    session: &mut ChatSession,
) -> Result<String> {
    let url = format!("{}/v1/chat/completions", config.service_base_url());
    let mut filter = chat_stream::StreamPrefixFilter::new();
    let mut final_payload = None;
    let mut assistant_text = String::new();
    let mut active_section = None;
    let response = http_client::post_streaming_lines(
        &url,
        payload,
        "text/event-stream, application/json",
        Duration::from_secs(3600),
        |line| {
            for chunk in chat_stream::parse_chat_stream_line(line).unwrap_or_default() {
                match chunk {
                    chat_stream::ChatStreamChunk::Text(text) => {
                        if let Some(text) = filter.push(&text)
                            && !text.is_empty()
                        {
                            assistant_text.push_str(&text);
                            print_stream_section(&mut active_section, StreamSection::Assistant);
                            print!("{text}");
                            let _ = io::stdout().flush();
                        }
                    }
                    chat_stream::ChatStreamChunk::Reasoning(text) => {
                        if session.reasoning_visible && !text.trim().is_empty() {
                            print_stream_section(&mut active_section, StreamSection::Reasoning);
                            print!("{text}");
                            let _ = io::stdout().flush();
                        }
                    }
                    chat_stream::ChatStreamChunk::Final(payload) => {
                        final_payload = Some(payload);
                    }
                }
            }
        },
    )?;
    if response.status >= 400 {
        anyhow::bail!("Streaming inference failed with status {}", response.status);
    }
    if let Some(text) = filter.finish()
        && !text.is_empty()
    {
        assistant_text.push_str(&text);
        print_stream_section(&mut active_section, StreamSection::Assistant);
        print!("{text}");
    }
    if active_section.is_some() {
        println!();
    }
    if let Some(payload) = final_payload {
        if let Some(usage) = payload.get("usage") {
            session.last_usage = Some(usage.clone());
        }
        print_tui_performance(&payload);
    }
    Ok(assistant_text)
}

fn print_status(config: &config::AppConfig, session: &ChatSession) -> Result<()> {
    let state = get_local_json_for_config("/omni/state", Duration::from_secs(10), config)?;
    print_section("Status", "Current OmniInfer session");
    print_kv(
        "Backend",
        json_str(&state, "backend").unwrap_or(&session.backend),
    );
    print_health_kv(
        "State",
        if json_bool(&state, "backend_ready").unwrap_or(false) {
            "ready"
        } else {
            "not ready"
        },
        json_bool(&state, "backend_ready").unwrap_or(false),
    );
    print_kv("Model", json_str(&state, "model").unwrap_or("-"));
    print_kv(
        "Context size",
        &json_u64(&state, "ctx_size")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    print_kv(
        "Reasoning display",
        if session.reasoning_visible {
            "show"
        } else {
            "hide"
        },
    );
    if let Some(usage) = &session.last_usage {
        print_kv(
            "Last usage",
            &format!(
                "prompt={}, completion={}, total={}",
                json_u64(usage, "prompt_tokens")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                json_u64(usage, "completion_tokens")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                json_u64(usage, "total_tokens")
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string())
            ),
        );
    }
    println!();
    Ok(())
}

fn toggle_thinking(config: &config::AppConfig) -> Result<()> {
    let current = get_thinking(config).unwrap_or(false);
    set_thinking_value(config, !current)
}

fn set_thinking(config: &config::AppConfig, value: Option<&str>) -> Result<()> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        Some("on") => set_thinking_value(config, true),
        Some("off") => set_thinking_value(config, false),
        _ => {
            notice(
                "Usage: /think, /think on, or /think off",
                NoticeKind::Warning,
            );
            Ok(())
        }
    }
}

fn get_thinking(config: &config::AppConfig) -> Result<bool> {
    let payload = get_local_json_for_config("/omni/thinking", Duration::from_secs(10), config)?;
    Ok(json_bool(&payload, "default_enabled").unwrap_or(false))
}

fn set_thinking_value(config: &config::AppConfig, enabled: bool) -> Result<()> {
    let payload = post_local_json_for_config(
        "/omni/thinking/select",
        &serde_json::json!({ "enabled": enabled }),
        Duration::from_secs(10),
        config,
    )?;
    notice(
        &format!(
            "Thinking mode: {}",
            if json_bool(&payload, "default_enabled").unwrap_or(false) {
                "on"
            } else {
                "off"
            }
        ),
        NoticeKind::Success,
    );
    Ok(())
}

fn toggle_reasoning(session: &mut ChatSession) -> Result<()> {
    set_reasoning_value(session, !session.reasoning_visible)
}

fn set_reasoning(session: &mut ChatSession, value: Option<&str>) -> Result<()> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        Some("on" | "show") => set_reasoning_value(session, true),
        Some("off" | "hide") => set_reasoning_value(session, false),
        _ => {
            notice(
                "Usage: /reasoning, /reasoning on, or /reasoning off",
                NoticeKind::Warning,
            );
            Ok(())
        }
    }
}

fn set_reasoning_value(session: &mut ChatSession, enabled: bool) -> Result<()> {
    session.reasoning_visible = enabled;
    local_state::save_tui_show_reasoning(enabled)?;
    notice(
        &format!(
            "Reasoning display: {}",
            if enabled { "show" } else { "hide" }
        ),
        NoticeKind::Success,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_sections_group_contiguous_reasoning_chunks() {
        let mut active = None;
        assert!(enter_stream_section(&mut active, StreamSection::Reasoning));
        assert!(!enter_stream_section(&mut active, StreamSection::Reasoning));
        assert!(!enter_stream_section(&mut active, StreamSection::Reasoning));
        assert!(enter_stream_section(&mut active, StreamSection::Assistant));
        assert!(!enter_stream_section(&mut active, StreamSection::Assistant));
    }
}
