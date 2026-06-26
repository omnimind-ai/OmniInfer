use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn anthropic_request_to_openai(body: &Value) -> Value {
    let mut output = serde_json::Map::new();
    if let Some(model) = body.get("model") {
        output.insert("model".to_string(), model.clone());
    }

    let mut messages = Vec::new();
    if let Some(system) = normalize_system(body.get("system")) {
        messages.push(json!({"role": "system", "content": system}));
    }

    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .to_string();
        let Some(content) = message.get("content") else {
            messages.push(json!({"role": role, "content": ""}));
            continue;
        };
        if let Some(text) = content.as_str() {
            messages.push(json!({"role": role, "content": text}));
            continue;
        }
        let Some(blocks) = content.as_array() else {
            messages.push(json!({"role": role, "content": ""}));
            continue;
        };

        let text_blocks = blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .collect::<Vec<_>>();
        let tool_use_blocks = blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            .collect::<Vec<_>>();
        let tool_result_blocks = blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            .collect::<Vec<_>>();

        if role == "assistant" && !tool_use_blocks.is_empty() {
            let text = text_blocks
                .first()
                .and_then(|block| block.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let tool_calls = tool_use_blocks
                .iter()
                .enumerate()
                .map(|(index, block)| {
                    json!({
                        "id": block
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| format!("call_{index}")),
                        "type": "function",
                        "function": {
                            "name": block.get("name").and_then(Value::as_str).unwrap_or(""),
                            "arguments": serde_json::to_string(block.get("input").unwrap_or(&json!({})))
                                .unwrap_or_else(|_| "{}".to_string()),
                        }
                    })
                })
                .collect::<Vec<_>>();
            messages.push(json!({"role": "assistant", "content": text, "tool_calls": tool_calls}));
        } else if role == "user" && !tool_result_blocks.is_empty() {
            for block in tool_result_blocks {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                    "content": tool_result_content(block.get("content")),
                }));
            }
            if let Some(text) = text_blocks
                .first()
                .and_then(|block| block.get("text"))
                .and_then(Value::as_str)
            {
                messages.push(json!({"role": "user", "content": text}));
            }
        } else {
            let mut parts = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        parts.push(json!({"type": "text", "text": block.get("text").and_then(Value::as_str).unwrap_or("")}));
                    }
                    Some("image") => {
                        let source = block.get("source").unwrap_or(&Value::Null);
                        match source.get("type").and_then(Value::as_str) {
                            Some("base64") => {
                                let media_type = source
                                    .get("media_type")
                                    .and_then(Value::as_str)
                                    .unwrap_or("image/jpeg");
                                let data = source.get("data").and_then(Value::as_str).unwrap_or("");
                                parts.push(json!({"type": "image_url", "image_url": {"url": format!("data:{media_type};base64,{data}")}}));
                            }
                            Some("url") => {
                                parts.push(json!({"type": "image_url", "image_url": {"url": source.get("url").and_then(Value::as_str).unwrap_or("")}}));
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            if parts.len() == 1 && parts[0].get("type").and_then(Value::as_str) == Some("text") {
                messages.push(json!({"role": role, "content": parts[0]["text"]}));
            } else if parts.is_empty() {
                messages.push(json!({"role": role, "content": ""}));
            } else {
                messages.push(json!({"role": role, "content": parts}));
            }
        }
    }
    output.insert("messages".to_string(), Value::Array(messages));

    copy_field(body, &mut output, "max_tokens", "max_tokens");
    copy_field(body, &mut output, "stop_sequences", "stop");
    for field in ["temperature", "top_p", "top_k", "stream"] {
        copy_field(body, &mut output, field, field);
    }
    if let Some(thinking) = body.get("thinking").and_then(Value::as_object) {
        match thinking.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                output.insert("think".to_string(), Value::Bool(true));
                if let Some(budget) = thinking.get("budget_tokens") {
                    output.insert("thinking_budget".to_string(), budget.clone());
                }
            }
            Some("disabled") => {
                output.insert("think".to_string(), Value::Bool(false));
            }
            _ => {}
        }
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let tools = tools
            .iter()
            .filter_map(|tool| {
                tool.as_object().map(|_| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.get("name").and_then(Value::as_str).unwrap_or(""),
                            "description": tool.get("description").and_then(Value::as_str).unwrap_or(""),
                            "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                        }
                    })
                })
            })
            .collect::<Vec<_>>();
        if !tools.is_empty() {
            output.insert("tools".to_string(), Value::Array(tools));
        }
    }
    if let Some(choice) = body.get("tool_choice").and_then(Value::as_object) {
        match choice.get("type").and_then(Value::as_str) {
            Some("auto" | "any") => {
                output.insert("tool_choice".to_string(), Value::String("auto".to_string()));
            }
            Some("none") => {
                output.insert("tool_choice".to_string(), Value::String("none".to_string()));
            }
            Some("tool") => {
                output.insert(
                    "tool_choice".to_string(),
                    json!({"type": "function", "function": {"name": choice.get("name").and_then(Value::as_str).unwrap_or("")}}),
                );
            }
            _ => {}
        }
    }

    Value::Object(output)
}

pub fn openai_response_to_anthropic(response: &Value, model: &str) -> Value {
    let choice = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .unwrap_or(&Value::Null);
    let message = choice.get("message").unwrap_or(&Value::Null);
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let mut content = Vec::new();
    if let Some(text) = message.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        content.push(json!({"type": "text", "text": text}));
    }
    for tool_call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let function = tool_call.get("function").unwrap_or(&Value::Null);
        let input = function
            .get("arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .unwrap_or_else(|| json!({}));
        content.push(json!({
            "type": "tool_use",
            "id": tool_call
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(make_tool_id),
            "name": function.get("name").and_then(Value::as_str).unwrap_or(""),
            "input": input,
        }));
    }
    let usage = response.get("usage").unwrap_or(&Value::Null);
    json!({
        "id": make_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": finish_reason_to_anthropic(finish_reason),
        "stop_sequence": null,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
        },
    })
}

pub fn openai_sse_to_anthropic_sse(raw: &[u8], model: &str) -> Vec<u8> {
    let mut converter = AnthropicStreamConverter::new(model);
    let mut output = converter.preamble();
    for data in parse_openai_sse_events(raw) {
        if let Ok(chunk) = serde_json::from_str::<Value>(&data) {
            output.extend(converter.process_chunk(&chunk));
        }
    }
    output.extend(converter.epilogue());
    output.concat().into_bytes()
}

pub fn parse_openai_sse_events(raw: &[u8]) -> Vec<String> {
    let mut events = Vec::new();
    for frame in String::from_utf8_lossy(raw).split("\n\n") {
        let mut data_lines = Vec::new();
        for line in frame.lines() {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if !data.is_empty() {
                data_lines.push(data);
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        let data = data_lines.join("\n");
        if data != "[DONE]" {
            events.push(data);
        }
    }
    events
}

pub struct AnthropicStreamConverter {
    model: String,
    msg_id: String,
    tool_blocks: BTreeMap<u64, u64>,
    text_block_index: Option<u64>,
    next_block_index: u64,
    finish_reason: String,
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl AnthropicStreamConverter {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            msg_id: make_message_id(),
            tool_blocks: BTreeMap::new(),
            text_block_index: None,
            next_block_index: 0,
            finish_reason: "end_turn".to_string(),
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }

    pub fn preamble(&self) -> Vec<String> {
        vec![
            sse(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }),
            ),
            sse("ping", json!({"type": "ping"})),
        ]
    }

    pub fn process_chunk(&mut self, chunk: &Value) -> Vec<String> {
        let mut frames = Vec::new();
        let choice = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .unwrap_or(&Value::Null);
        let delta = choice.get("delta").unwrap_or(&Value::Null);

        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            let index = match self.text_block_index {
                Some(index) => index,
                None => {
                    let index = self.next_block_index;
                    self.next_block_index += 1;
                    self.text_block_index = Some(index);
                    frames.push(sse(
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    index
                }
            };
            frames.push(sse(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                }),
            ));
        }

        for tool_call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let tool_index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            let block_index = if let Some(index) = self.tool_blocks.get(&tool_index).copied() {
                index
            } else {
                let index = self.next_block_index;
                self.next_block_index += 1;
                self.tool_blocks.insert(tool_index, index);
                frames.push(sse(
                    "content_block_start",
                    json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call
                                .get("id")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                                .unwrap_or_else(make_tool_id),
                            "name": function.get("name").and_then(Value::as_str).unwrap_or(""),
                            "input": {},
                        },
                    }),
                ));
                index
            };
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                && !arguments.is_empty()
            {
                frames.push(sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                    }),
                ));
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = finish_reason_to_anthropic(reason).to_string();
        }
        let usage = chunk.get("usage").unwrap_or(&Value::Null);
        self.prompt_tokens = usage
            .get("prompt_tokens")
            .or_else(|| {
                chunk
                    .get("timings")
                    .and_then(|timings| timings.get("prompt_n"))
            })
            .and_then(Value::as_u64)
            .unwrap_or(self.prompt_tokens);
        self.completion_tokens = usage
            .get("completion_tokens")
            .or_else(|| {
                chunk
                    .get("timings")
                    .and_then(|timings| timings.get("predicted_n"))
            })
            .and_then(Value::as_u64)
            .unwrap_or(self.completion_tokens);
        frames
    }

    pub fn epilogue(&mut self) -> Vec<String> {
        let mut frames = Vec::new();
        if self.text_block_index.is_none() && self.tool_blocks.is_empty() {
            self.text_block_index = Some(self.next_block_index);
            self.next_block_index += 1;
            frames.push(sse(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": self.text_block_index.unwrap_or(0),
                    "content_block": {"type": "text", "text": ""},
                }),
            ));
        }
        if let Some(index) = self.text_block_index {
            frames.push(sse(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
        for index in self.tool_blocks.values() {
            frames.push(sse(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
        frames.push(sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": self.finish_reason, "stop_sequence": null},
                "usage": {"output_tokens": self.completion_tokens},
            }),
        ));
        frames.push(sse("message_stop", json!({"type": "message_stop"})));
        frames
    }
}

fn normalize_system(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
        Some(Value::Array(blocks)) => {
            let parts = blocks
                .iter()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        _ => None,
    }
}

fn tool_result_content(value: Option<&Value>) -> String {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" "),
        Some(Value::String(text)) => text.clone(),
        Some(value) => value.to_string(),
        None => String::new(),
    }
}

fn copy_field(
    input: &Value,
    output: &mut serde_json::Map<String, Value>,
    input_key: &str,
    output_key: &str,
) {
    if let Some(value) = input.get(input_key) {
        output.insert(output_key.to_string(), value.clone());
    }
}

fn finish_reason_to_anthropic(reason: &str) -> &'static str {
    match reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

fn sse(event_type: &str, data: Value) -> String {
    format!(
        "event: {event_type}\ndata: {}\n\n",
        serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
    )
}

fn make_message_id() -> String {
    format!("msg_{:024x}", rand::random::<u128>() >> 32)
}

fn make_tool_id() -> String {
    format!("toolu_{:024x}", rand::random::<u128>() >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_simple_request() {
        let request = anthropic_request_to_openai(&json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }));
        assert_eq!(request["model"], "test-model");
        assert_eq!(request["max_tokens"], 100);
        assert_eq!(
            request["messages"][0],
            json!({"role": "user", "content": "hello"})
        );
    }

    #[test]
    fn converts_system_image_tools_and_choice() {
        let request = anthropic_request_to_openai(&json!({
            "system": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]}],
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        }));
        assert_eq!(request["messages"][0]["content"], "Part 1\nPart 2");
        assert_eq!(
            request["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
        assert_eq!(request["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(request["tool_choice"]["function"]["name"], "get_weather");
        assert_eq!(request["think"], true);
        assert_eq!(request["thinking_budget"], 1024);
    }

    #[test]
    fn converts_tool_use_and_tool_result_messages() {
        let tool_use = anthropic_request_to_openai(&json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Tokyo"}}
            ]}]
        }));
        assert_eq!(tool_use["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            tool_use["messages"][0]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Tokyo\"}"
        );

        let tool_result = anthropic_request_to_openai(&json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny"}
            ]}]
        }));
        assert_eq!(tool_result["messages"][0]["role"], "tool");
        assert_eq!(tool_result["messages"][0]["tool_call_id"], "call_1");
    }

    #[test]
    fn converts_openai_response() {
        let response = openai_response_to_anthropic(
            &json!({
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }),
            "test-model",
        );
        assert_eq!(response["type"], "message");
        assert_eq!(
            response["content"][0],
            json!({"type": "text", "text": "Hello!"})
        );
        assert_eq!(response["stop_reason"], "end_turn");
        assert_eq!(response["usage"]["input_tokens"], 10);
        assert_eq!(response["usage"]["output_tokens"], 5);
    }

    #[test]
    fn converts_openai_sse_text_stream() {
        let raw = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\n\
data: [DONE]\n\n";
        let output = String::from_utf8(openai_sse_to_anthropic_sse(raw, "test-model")).unwrap();
        assert!(output.contains("event: message_start"));
        assert!(output.contains("event: content_block_start"));
        assert!(output.contains("\"text\":\"Hel\""));
        assert!(output.contains("\"text\":\"lo\""));
        assert!(output.contains("\"stop_reason\":\"end_turn\""));
        assert!(output.contains("\"output_tokens\":2"));
        assert!(output.contains("event: message_stop"));
    }
}
