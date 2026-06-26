use thiserror::Error;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedLoadArgs {
    pub ctx_size: Option<u32>,
    pub launch_args: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedChatArgs {
    pub ctx_size: Option<u32>,
    pub message: Option<String>,
    pub image: Option<String>,
    pub request_overrides: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BackendArgError {
    #[error("{0} is a basic OmniInfer parameter and should be passed with the main CLI options")]
    ReservedBasic(String),
    #[error("{0} is managed by OmniInfer and should not be passed as a backend-native extra arg")]
    ReservedManaged(String),
    #[error("{flag} requires a value")]
    MissingValue { flag: String },
    #[error("{flag} must be an integer: {value}")]
    InvalidInteger { flag: String, value: String },
    #[error(
        "mlx-mac does not currently expose backend-native load extra args in OmniInfer; use the stable CLI options only"
    )]
    MlxLoadUnsupported,
}

pub fn parse_backend_load_extra_args(
    backend_id: &str,
    family: &str,
    tokens: &[String],
) -> Result<ParsedLoadArgs, BackendArgError> {
    if tokens.is_empty() {
        return Ok(ParsedLoadArgs::default());
    }
    match family {
        "llama.cpp" | "turboquant" => parse_llama_cpp_load_args(backend_id, tokens),
        "vllm" => parse_vllm_load_args(tokens),
        "mlx-lm" => Err(BackendArgError::MlxLoadUnsupported),
        _ => Ok(ParsedLoadArgs {
            ctx_size: None,
            launch_args: tokens.to_vec(),
        }),
    }
}

pub fn parse_backend_chat_extra_args(
    family: &str,
    tokens: &[String],
) -> Result<ParsedChatArgs, BackendArgError> {
    if tokens.is_empty() {
        return Ok(ParsedChatArgs::default());
    }
    match family {
        "llama.cpp" | "turboquant" => parse_llama_cpp_chat_args(tokens),
        "mlx-lm" => parse_mlx_chat_args(tokens),
        _ => parse_generic_chat_args(tokens),
    }
}

fn parse_llama_cpp_load_args(
    backend_id: &str,
    tokens: &[String],
) -> Result<ParsedLoadArgs, BackendArgError> {
    let mut parsed = ParsedLoadArgs::default();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        let (flag, inline_value) = split_flag_value(token);
        if matches!(
            flag,
            "-m" | "--model" | "-mm" | "--mmproj" | "--message" | "-p" | "--prompt" | "--image"
        ) {
            return Err(BackendArgError::ReservedBasic(flag.to_string()));
        }
        if matches!(flag, "-c" | "--ctx-size") {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.ctx_size = Some(parse_u32(flag, value)?);
            index += consumed;
            continue;
        }
        if backend_id.starts_with("ik_llama.cpp") && flag == "--no-cache-prompt" {
            parsed
                .launch_args
                .extend(["-cram".to_string(), "0".to_string()]);
            index += 1;
            continue;
        }
        parsed.launch_args.push(token.clone());
        index += 1;
    }
    Ok(parsed)
}

fn parse_vllm_load_args(tokens: &[String]) -> Result<ParsedLoadArgs, BackendArgError> {
    let mut parsed = ParsedLoadArgs::default();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        let (flag, inline_value) = split_flag_value(token);
        if matches!(flag, "--model" | "--host" | "--port" | "--api-key") {
            return Err(BackendArgError::ReservedManaged(flag.to_string()));
        }
        if matches!(flag, "-c" | "--ctx-size" | "--max-model-len") {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.ctx_size = Some(parse_u32(flag, value)?);
            index += consumed;
            continue;
        }
        parsed.launch_args.push(token.clone());
        index += 1;
    }
    Ok(parsed)
}

fn parse_llama_cpp_chat_args(tokens: &[String]) -> Result<ParsedChatArgs, BackendArgError> {
    let mut parsed = ParsedChatArgs::default();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        let (flag, inline_value) = split_flag_value(token);
        if matches!(flag, "-m" | "--model" | "-mm" | "--mmproj") {
            return Err(BackendArgError::ReservedBasic(flag.to_string()));
        }
        if matches!(flag, "-p" | "--prompt" | "--message") {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.message = Some(value.to_string());
            index += consumed;
            continue;
        }
        if flag == "--image" {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.image = Some(value.to_string());
            index += consumed;
            continue;
        }
        if matches!(flag, "-c" | "--ctx-size") {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.ctx_size = Some(parse_u32(flag, value)?);
            index += consumed;
            continue;
        }
        if let Some((key, value_kind)) = llama_cpp_chat_alias(flag) {
            let (value, consumed) = if value_kind == ValueKind::Bool {
                take_bool_or_true(tokens, index, inline_value, flag)?
            } else {
                take_option_value(tokens, index, inline_value, flag)?
            };
            parsed
                .request_overrides
                .insert(key.to_string(), coerce_value(value, value_kind));
            index += consumed;
            continue;
        }
        if let Some(key) = llama_cpp_list_alias(flag) {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            push_list_override(&mut parsed.request_overrides, key, value);
            index += consumed;
            continue;
        }
        if let Some(key) = flag.strip_prefix("--no-") {
            parsed
                .request_overrides
                .insert(key.replace('-', "_"), serde_json::Value::Bool(false));
            index += 1;
            continue;
        }
        if let Some(key) = flag.strip_prefix("--") {
            let key = key.replace('-', "_");
            if matches!(
                key.as_str(),
                "messages" | "model" | "backend" | "mmproj" | "stream_options"
            ) {
                return Err(BackendArgError::ReservedManaged(flag.to_string()));
            }
            if let Some(value) = inline_value {
                parsed.request_overrides.insert(key, auto_scalar(value));
                index += 1;
                continue;
            }
            if next_is_value(tokens, index) {
                parsed
                    .request_overrides
                    .insert(key, auto_scalar(&tokens[index + 1]));
                index += 2;
                continue;
            }
            parsed
                .request_overrides
                .insert(key, serde_json::Value::Bool(true));
            index += 1;
            continue;
        }
        return Err(BackendArgError::ReservedManaged(flag.to_string()));
    }
    Ok(parsed)
}

fn parse_mlx_chat_args(tokens: &[String]) -> Result<ParsedChatArgs, BackendArgError> {
    let mut parsed = ParsedChatArgs::default();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        let (flag, inline_value) = split_flag_value(token);
        if matches!(flag, "-p" | "--prompt" | "--message") {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.message = Some(value.to_string());
            index += consumed;
            continue;
        }
        if flag == "--image" {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.image = Some(value.to_string());
            index += consumed;
            continue;
        }
        if let Some((key, value_kind)) = mlx_chat_alias(flag) {
            let (value, consumed) = if value_kind == ValueKind::Bool {
                take_bool_or_true(tokens, index, inline_value, flag)?
            } else {
                take_option_value(tokens, index, inline_value, flag)?
            };
            parsed
                .request_overrides
                .insert(key.to_string(), coerce_value(value, value_kind));
            index += consumed;
            continue;
        }
        if flag == "--stop" {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            push_list_override(&mut parsed.request_overrides, "stop", value);
            index += consumed;
            continue;
        }
        return Err(BackendArgError::ReservedManaged(flag.to_string()));
    }
    Ok(parsed)
}

fn parse_generic_chat_args(tokens: &[String]) -> Result<ParsedChatArgs, BackendArgError> {
    let mut parsed = ParsedChatArgs::default();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        let (flag, inline_value) = split_flag_value(token);
        if matches!(flag, "-p" | "--prompt" | "--message") {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.message = Some(value.to_string());
            index += consumed;
            continue;
        }
        if flag == "--image" {
            let (value, consumed) = take_option_value(tokens, index, inline_value, flag)?;
            parsed.image = Some(value.to_string());
            index += consumed;
            continue;
        }
        if let Some(key) = flag.strip_prefix("--") {
            let key = key.replace('-', "_");
            if let Some(value) = inline_value {
                parsed.request_overrides.insert(key, auto_scalar(value));
                index += 1;
            } else if next_is_value(tokens, index) {
                parsed
                    .request_overrides
                    .insert(key, auto_scalar(&tokens[index + 1]));
                index += 2;
            } else {
                parsed
                    .request_overrides
                    .insert(key, serde_json::Value::Bool(true));
                index += 1;
            }
            continue;
        }
        return Err(BackendArgError::ReservedManaged(flag.to_string()));
    }
    Ok(parsed)
}

fn split_flag_value(token: &str) -> (&str, Option<&str>) {
    match token.split_once('=') {
        Some((flag, value)) if flag.starts_with('-') => (flag, Some(value)),
        _ => (token, None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    Bool,
    Float,
    Int,
    Str,
}

fn llama_cpp_chat_alias(flag: &str) -> Option<(&'static str, ValueKind)> {
    match flag {
        "-n" | "--n-predict" | "--max-tokens" => Some(("max_tokens", ValueKind::Int)),
        "--temp" | "--temperature" => Some(("temperature", ValueKind::Float)),
        "--top-k" => Some(("top_k", ValueKind::Int)),
        "--top-p" => Some(("top_p", ValueKind::Float)),
        "--min-p" => Some(("min_p", ValueKind::Float)),
        "--typical-p" => Some(("typical_p", ValueKind::Float)),
        "--seed" => Some(("seed", ValueKind::Int)),
        "--repeat-penalty" => Some(("repeat_penalty", ValueKind::Float)),
        "--repeat-last-n" => Some(("repeat_last_n", ValueKind::Int)),
        "--presence-penalty" => Some(("presence_penalty", ValueKind::Float)),
        "--frequency-penalty" => Some(("frequency_penalty", ValueKind::Float)),
        "--min-keep" => Some(("min_keep", ValueKind::Int)),
        "--mirostat" => Some(("mirostat", ValueKind::Int)),
        "--mirostat-tau" | "--mirostat-ent" => Some(("mirostat_tau", ValueKind::Float)),
        "--mirostat-eta" | "--mirostat-lr" => Some(("mirostat_eta", ValueKind::Float)),
        "--dynatemp-range" => Some(("dynatemp_range", ValueKind::Float)),
        "--dynatemp-exp" => Some(("dynatemp_exponent", ValueKind::Float)),
        "--json-schema" => Some(("json_schema", ValueKind::Str)),
        "--grammar" => Some(("grammar", ValueKind::Str)),
        "--samplers" => Some(("samplers", ValueKind::Str)),
        "--cache-prompt" => Some(("cache_prompt", ValueKind::Bool)),
        "--ignore-eos" | "-e" => Some(("ignore_eos", ValueKind::Bool)),
        "--stream" => Some(("stream", ValueKind::Bool)),
        "--think" => Some(("think", ValueKind::Bool)),
        _ => None,
    }
}

fn mlx_chat_alias(flag: &str) -> Option<(&'static str, ValueKind)> {
    match flag {
        "-n" | "--max-tokens" => Some(("max_tokens", ValueKind::Int)),
        "--temp" | "--temperature" => Some(("temperature", ValueKind::Float)),
        "--top-k" => Some(("top_k", ValueKind::Int)),
        "--top-p" => Some(("top_p", ValueKind::Float)),
        "--min-p" => Some(("min_p", ValueKind::Float)),
        "--seed" => Some(("seed", ValueKind::Int)),
        "--stream" => Some(("stream", ValueKind::Bool)),
        "--think" => Some(("think", ValueKind::Bool)),
        _ => None,
    }
}

fn llama_cpp_list_alias(flag: &str) -> Option<&'static str> {
    match flag {
        "--stop" => Some("stop"),
        "--dry-sequence-breaker" => Some("dry_sequence_breaker"),
        _ => None,
    }
}

fn take_option_value<'a>(
    tokens: &'a [String],
    index: usize,
    inline_value: Option<&'a str>,
    flag: &str,
) -> Result<(&'a str, usize), BackendArgError> {
    if let Some(value) = inline_value {
        return Ok((value, 1));
    }
    let value =
        tokens
            .get(index + 1)
            .map(String::as_str)
            .ok_or_else(|| BackendArgError::MissingValue {
                flag: flag.to_string(),
            })?;
    Ok((value, 2))
}

fn take_bool_or_true<'a>(
    tokens: &'a [String],
    index: usize,
    inline_value: Option<&'a str>,
    _flag: &str,
) -> Result<(&'a str, usize), BackendArgError> {
    if let Some(value) = inline_value {
        return Ok((value, 1));
    }
    if next_is_value(tokens, index) {
        return Ok((tokens[index + 1].as_str(), 2));
    }
    Ok(("true", 1))
}

fn next_is_value(tokens: &[String], index: usize) -> bool {
    tokens
        .get(index + 1)
        .map(|value| !value.starts_with('-'))
        .unwrap_or(false)
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, BackendArgError> {
    value
        .parse::<u32>()
        .map_err(|_| BackendArgError::InvalidInteger {
            flag: flag.to_string(),
            value: value.to_string(),
        })
}

fn coerce_value(value: &str, value_kind: ValueKind) -> serde_json::Value {
    match value_kind {
        ValueKind::Bool => serde_json::Value::Bool(parse_boolish(value).unwrap_or(true)),
        ValueKind::Float => value
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(serde_json::Value::Number)
            .unwrap_or_else(|| serde_json::Value::String(value.to_string())),
        ValueKind::Int => value
            .parse::<i64>()
            .map(|value| serde_json::Value::Number(value.into()))
            .unwrap_or_else(|_| serde_json::Value::String(value.to_string())),
        ValueKind::Str => serde_json::Value::String(value.to_string()),
    }
}

fn auto_scalar(value: &str) -> serde_json::Value {
    if let Some(value) = parse_auto_boolish(value) {
        return serde_json::Value::Bool(value);
    }
    if let Ok(value) = value.parse::<i64>() {
        return serde_json::Value::Number(value.into());
    }
    if let Ok(value) = value.parse::<f64>() {
        if let Some(number) = serde_json::Number::from_f64(value) {
            return serde_json::Value::Number(number);
        }
    }
    serde_json::Value::String(value.to_string())
}

fn parse_boolish(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enable" | "enabled" => Some(true),
        "0" | "false" | "no" | "off" | "disable" | "disabled" => Some(false),
        _ => None,
    }
}

fn parse_auto_boolish(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" => Some(true),
        "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn push_list_override(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: &str,
) {
    let entry = map
        .entry(key.to_string())
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    if !entry.is_array() {
        *entry = serde_json::Value::Array(vec![entry.clone()]);
    }
    if let Some(values) = entry.as_array_mut() {
        values.push(serde_json::Value::String(value.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|token| token.to_string()).collect()
    }

    #[test]
    fn parses_llama_ctx_and_keeps_launch_args() {
        let parsed = parse_backend_load_extra_args(
            "llama.cpp-linux-cuda",
            "llama.cpp",
            &args(&["-c", "8192", "-ngl", "999"]),
        )
        .unwrap();
        assert_eq!(parsed.ctx_size, Some(8192));
        assert_eq!(parsed.launch_args, args(&["-ngl", "999"]));
    }

    #[test]
    fn parses_inline_ctx_value() {
        let parsed = parse_backend_load_extra_args(
            "llama.cpp-linux-cuda",
            "llama.cpp",
            &args(&["--ctx-size=4096"]),
        )
        .unwrap();
        assert_eq!(parsed.ctx_size, Some(4096));
        assert!(parsed.launch_args.is_empty());
    }

    #[test]
    fn maps_ik_no_cache_prompt() {
        let parsed = parse_backend_load_extra_args(
            "ik_llama.cpp-linux-cuda",
            "llama.cpp",
            &args(&["--no-cache-prompt"]),
        )
        .unwrap();
        assert_eq!(parsed.launch_args, args(&["-cram", "0"]));
    }

    #[test]
    fn rejects_reserved_model_flag_for_llama() {
        let error = parse_backend_load_extra_args(
            "llama.cpp-linux-cuda",
            "llama.cpp",
            &args(&["--model", "x.gguf"]),
        )
        .unwrap_err();
        assert_eq!(error, BackendArgError::ReservedBasic("--model".to_string()));
    }

    #[test]
    fn parses_vllm_max_model_len() {
        let parsed = parse_backend_load_extra_args(
            "vllm-linux-cuda",
            "vllm",
            &args(&["--max-model-len", "2048"]),
        )
        .unwrap();
        assert_eq!(parsed.ctx_size, Some(2048));
    }

    #[test]
    fn rejects_mlx_load_extras() {
        let error = parse_backend_load_extra_args("mlx-mac", "mlx-lm", &args(&["--temp", "0.1"]))
            .unwrap_err();
        assert_eq!(error, BackendArgError::MlxLoadUnsupported);
    }

    #[test]
    fn generic_backend_keeps_tokens() {
        let parsed =
            parse_backend_load_extra_args("other", "other", &args(&["--foo", "bar"])).unwrap();
        assert_eq!(parsed.launch_args, args(&["--foo", "bar"]));
    }

    #[test]
    fn parses_llama_chat_request_overrides() {
        let parsed = parse_backend_chat_extra_args(
            "llama.cpp",
            &args(&[
                "--temp",
                "0.2",
                "--stream",
                "false",
                "--stop",
                "END",
                "--cache-prompt",
            ]),
        )
        .unwrap();
        assert_eq!(
            parsed.request_overrides["temperature"],
            serde_json::json!(0.2)
        );
        assert_eq!(parsed.request_overrides["stream"], serde_json::json!(false));
        assert_eq!(
            parsed.request_overrides["cache_prompt"],
            serde_json::json!(true)
        );
        assert_eq!(parsed.request_overrides["stop"], serde_json::json!(["END"]));
    }

    #[test]
    fn parses_generic_chat_request_overrides() {
        let parsed =
            parse_backend_chat_extra_args("other", &args(&["--foo", "1", "--bar"])).unwrap();
        assert_eq!(parsed.request_overrides["foo"], serde_json::json!(1));
        assert_eq!(parsed.request_overrides["bar"], serde_json::json!(true));
    }
}
