use serde_json::{Map, Value, json};
use thiserror::Error;

const DISABLED_REASONING_EFFORTS: &[&str] = &["none", "off", "disabled", "false", "0"];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RequestNormalizationError {
    #[error("cannot parse boolean value from {0}")]
    Bool(String),
    #[error("omni_stream must be an object")]
    OmniStreamNotObject,
    #[error("omni_stream.format must be 'lines'")]
    OmniStreamFormat,
    #[error("omni_stream.max_line_chars must be a positive integer")]
    OmniStreamMaxLineChars,
    #[error("omni_stream.include_reasoning must be a boolean")]
    OmniStreamIncludeReasoning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineStreamOptions {
    pub enabled: bool,
    pub max_line_chars: u64,
    pub include_reasoning: bool,
}

impl Default for LineStreamOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            max_line_chars: 240,
            include_reasoning: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedChatRequest {
    pub payload: Value,
    pub request_defaults: Option<Map<String, Value>>,
    pub line_stream: LineStreamOptions,
}

pub fn normalize_chat_request(
    mut payload: Value,
    default_thinking: bool,
) -> Result<NormalizedChatRequest, RequestNormalizationError> {
    let Some(map) = payload.as_object_mut() else {
        return Ok(NormalizedChatRequest {
            payload,
            request_defaults: None,
            line_stream: LineStreamOptions::default(),
        });
    };
    let line_stream = resolve_line_stream_options(map)?;
    let request_defaults = map
        .remove("request_defaults")
        .and_then(|value| value.as_object().cloned());
    apply_thinking_mode(map, default_thinking)?;
    normalize_legacy_function_tools(map);
    Ok(NormalizedChatRequest {
        payload,
        request_defaults,
        line_stream,
    })
}

pub fn parse_boolish(value: &Value) -> Result<bool, RequestNormalizationError> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Number(value) => Ok(value.as_f64().unwrap_or(0.0) != 0.0),
        Value::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "enable" | "enabled" => Ok(true),
            "0" | "false" | "no" | "off" | "disable" | "disabled" => Ok(false),
            _ => Err(RequestNormalizationError::Bool(value.clone())),
        },
        other => Err(RequestNormalizationError::Bool(other.to_string())),
    }
}

fn resolve_line_stream_options(
    payload: &mut Map<String, Value>,
) -> Result<LineStreamOptions, RequestNormalizationError> {
    let raw = payload.remove("omni_stream");
    let legacy_format = payload.remove("stream_format");
    if raw.is_none() && legacy_format.is_none() {
        return Ok(LineStreamOptions::default());
    }

    let raw = match raw {
        Some(Value::Object(map)) => map,
        Some(_) => return Err(RequestNormalizationError::OmniStreamNotObject),
        None => {
            let mut map = Map::new();
            if let Some(format) = legacy_format.clone() {
                map.insert("format".to_string(), format);
            }
            map
        }
    };

    let raw_format = raw
        .get("format")
        .or(legacy_format.as_ref())
        .and_then(Value::as_str)
        .unwrap_or("");
    if raw_format.is_empty() || matches!(raw_format, "tokens" | "openai") {
        return Ok(LineStreamOptions::default());
    }
    if raw_format != "lines" {
        return Err(RequestNormalizationError::OmniStreamFormat);
    }

    let max_line_chars = raw
        .get("max_line_chars")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or(RequestNormalizationError::OmniStreamMaxLineChars)?;
    let include_reasoning = match raw.get("include_reasoning") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => return Err(RequestNormalizationError::OmniStreamIncludeReasoning),
    };

    Ok(LineStreamOptions {
        enabled: true,
        max_line_chars,
        include_reasoning,
    })
}

fn reasoning_effort_enabled(payload: &mut Map<String, Value>) -> Option<bool> {
    let effort = payload.remove("reasoning_effort");
    if let Some(effort) = effort.filter(|value| !is_empty(value)) {
        return Some(!is_disabled_reasoning_effort(&value_to_text(&effort)));
    }

    let reasoning = payload.remove("reasoning");
    let Value::Object(reasoning) = reasoning? else {
        return None;
    };
    let effort = reasoning.get("effort").filter(|value| !is_empty(value))?;
    Some(!is_disabled_reasoning_effort(&value_to_text(effort)))
}

fn apply_thinking_mode(
    payload: &mut Map<String, Value>,
    default_enabled: bool,
) -> Result<(), RequestNormalizationError> {
    let requested = payload.remove("think");
    let reasoning_enabled = reasoning_effort_enabled(payload);

    if !payload
        .get("chat_template_kwargs")
        .is_some_and(Value::is_object)
    {
        payload.insert("chat_template_kwargs".to_string(), json!({}));
    }

    let Some(chat_template_kwargs) = payload
        .get_mut("chat_template_kwargs")
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };

    let enabled = match requested.as_ref() {
        Some(value) => Some(parse_boolish(value)?),
        None if !chat_template_kwargs.contains_key("enable_thinking") => reasoning_enabled,
        None => None,
    };

    if let Some(enabled) = enabled {
        chat_template_kwargs.insert("enable_thinking".to_string(), Value::Bool(enabled));
    } else if !chat_template_kwargs.contains_key("enable_thinking") {
        chat_template_kwargs.insert("enable_thinking".to_string(), Value::Bool(default_enabled));
    }

    let final_enabled = chat_template_kwargs
        .get("enable_thinking")
        .and_then(Value::as_bool);
    if final_enabled == Some(false) && !payload.contains_key("reasoning_format") {
        payload.insert(
            "reasoning_format".to_string(),
            Value::String("none".to_string()),
        );
    }
    Ok(())
}

fn normalize_legacy_function_tools(payload: &mut Map<String, Value>) {
    let functions = payload.remove("functions");
    if let Some(Value::Array(functions)) = functions {
        if !payload.contains_key("tools") {
            let tools = functions
                .into_iter()
                .filter_map(|function| match function {
                    Value::Object(function) => Some(json!({
                        "type": "function",
                        "function": Value::Object(function),
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if !tools.is_empty() {
                payload.insert("tools".to_string(), Value::Array(tools));
            }
        }
    }

    let function_call = payload.remove("function_call");
    if payload.contains_key("tool_choice") {
        return;
    }
    match function_call {
        Some(Value::String(value)) if matches!(value.as_str(), "auto" | "none") => {
            payload.insert("tool_choice".to_string(), Value::String(value));
        }
        Some(Value::Object(function_call)) => {
            let name = function_call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                payload.insert(
                    "tool_choice".to_string(),
                    json!({"type": "function", "function": {"name": name}}),
                );
            }
        }
        _ => {}
    }
}

fn is_disabled_reasoning_effort(value: &str) -> bool {
    DISABLED_REASONING_EFFORTS.contains(&value.trim().to_ascii_lowercase().as_str())
}

fn is_empty(value: &Value) -> bool {
    matches!(value, Value::Null) || value.as_str().is_some_and(|text| text.trim().is_empty())
}

fn value_to_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_reasoning_effort_and_default_thinking() {
        let request = normalize_chat_request(json!({"reasoning": {"effort": "none"}}), true)
            .expect("normalize");
        assert_eq!(
            request.payload["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert_eq!(request.payload["reasoning_format"], "none");
        assert!(request.payload.get("reasoning").is_none());

        let request = normalize_chat_request(json!({}), true).expect("normalize");
        assert_eq!(
            request.payload["chat_template_kwargs"]["enable_thinking"],
            true
        );
    }

    #[test]
    fn think_overrides_reasoning_effort() {
        let request = normalize_chat_request(
            json!({"think": false, "reasoning": {"effort": "high"}}),
            true,
        )
        .expect("normalize");
        assert_eq!(
            request.payload["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert_eq!(request.payload["reasoning_format"], "none");
        assert!(request.payload.get("think").is_none());
    }

    #[test]
    fn converts_legacy_functions() {
        let request = normalize_chat_request(
            json!({
                "functions": [{"name": "context_time_now", "parameters": {"type": "object"}}],
                "function_call": {"name": "context_time_now"}
            }),
            false,
        )
        .expect("normalize");
        assert!(request.payload.get("functions").is_none());
        assert_eq!(
            request.payload["tools"][0]["function"]["name"],
            "context_time_now"
        );
        assert_eq!(
            request.payload["tool_choice"],
            json!({"type": "function", "function": {"name": "context_time_now"}})
        );
    }

    #[test]
    fn extracts_request_defaults_and_line_stream() {
        let request = normalize_chat_request(
            json!({
                "request_defaults": {"temperature": 0.2},
                "omni_stream": {"format": "lines", "max_line_chars": 80, "include_reasoning": true}
            }),
            false,
        )
        .expect("normalize");
        assert_eq!(request.request_defaults.unwrap()["temperature"], json!(0.2));
        assert!(request.payload.get("request_defaults").is_none());
        assert!(request.line_stream.enabled);
        assert_eq!(request.line_stream.max_line_chars, 80);
        assert!(request.line_stream.include_reasoning);
    }
}
