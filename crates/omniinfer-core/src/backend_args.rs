use thiserror::Error;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedLoadArgs {
    pub ctx_size: Option<u32>,
    pub launch_args: Vec<String>,
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

fn split_flag_value(token: &str) -> (&str, Option<&str>) {
    match token.split_once('=') {
        Some((flag, value)) if flag.starts_with('-') => (flag, Some(value)),
        _ => (token, None),
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

fn parse_u32(flag: &str, value: &str) -> Result<u32, BackendArgError> {
    value
        .parse::<u32>()
        .map_err(|_| BackendArgError::InvalidInteger {
            flag: flag.to_string(),
            value: value.to_string(),
        })
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
}
