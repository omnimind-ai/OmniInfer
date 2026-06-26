use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamChunk {
    Text(String),
    Reasoning(String),
    Final(Value),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamPrefixFilter {
    buffer: String,
    flushed: bool,
}

impl StreamPrefixFilter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            flushed: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Option<String> {
        if self.flushed {
            return Some(text.to_string());
        }
        self.buffer.push_str(text);
        let trimmed = self.buffer.trim_start();
        if !trimmed.starts_with("<think>") {
            self.flushed = true;
            return Some(std::mem::take(&mut self.buffer));
        }
        let Some(end) = self.buffer.find("</think>") else {
            return None;
        };
        let before = &self.buffer[..end];
        let after_start = end + "</think>".len();
        if before.trim() == "<think>" {
            let after = self.buffer[after_start..].trim_start().to_string();
            self.buffer.clear();
            self.flushed = true;
            return Some(after);
        }
        self.flushed = true;
        Some(std::mem::take(&mut self.buffer))
    }

    pub fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        self.flushed = true;
        Some(std::mem::take(&mut self.buffer))
    }
}

impl Default for StreamPrefixFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error)]
pub enum ChatStreamError {
    #[error("chat stream event JSON parse failed: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn parse_chat_stream_line(line: &str) -> Result<Vec<ChatStreamChunk>, ChatStreamError> {
    let Some(data) = line.trim().strip_prefix("data:") else {
        return Ok(Vec::new());
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(Vec::new());
    }
    let event: Value = serde_json::from_str(data)?;
    Ok(parse_chat_stream_event(&event))
}

pub fn parse_chat_stream_event(event: &Value) -> Vec<ChatStreamChunk> {
    let mut chunks = Vec::new();
    if let Some(delta) = event
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(Value::as_object)
    {
        if let Some(content) = delta.get("content").and_then(Value::as_str) {
            chunks.push(ChatStreamChunk::Text(content.to_string()));
        }
        if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
            chunks.push(ChatStreamChunk::Reasoning(reasoning.to_string()));
        }
    }
    if event.get("usage").is_some() || event.get("timings").is_some() {
        chunks.push(ChatStreamChunk::Final(event.clone()));
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_reasoning_and_usage() {
        let content =
            parse_chat_stream_line(r#"data: {"choices":[{"delta":{"content":"answer"}}]}"#)
                .unwrap();
        assert_eq!(content, vec![ChatStreamChunk::Text("answer".to_string())]);

        let reasoning =
            parse_chat_stream_line(r#"data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}"#)
                .unwrap();
        assert_eq!(
            reasoning,
            vec![ChatStreamChunk::Reasoning("plan".to_string())]
        );

        let usage = parse_chat_stream_line(r#"data: {"usage":{"total_tokens":3}}"#).unwrap();
        assert!(matches!(usage.as_slice(), [ChatStreamChunk::Final(_)]));
    }

    #[test]
    fn ignores_done_and_non_data_lines() {
        assert!(parse_chat_stream_line("event: ping").unwrap().is_empty());
        assert!(parse_chat_stream_line("data: [DONE]").unwrap().is_empty());
    }

    #[test]
    fn strips_empty_think_prefix() {
        let mut filter = StreamPrefixFilter::new();
        assert_eq!(filter.push("<think>"), None);
        assert_eq!(filter.push("</think>Hello"), Some("Hello".to_string()));
        assert_eq!(filter.push(" world"), Some(" world".to_string()));
    }

    #[test]
    fn preserves_nonempty_think_blocks() {
        let mut filter = StreamPrefixFilter::new();
        assert_eq!(
            filter.push("<think>plan</think>Hello"),
            Some("<think>plan</think>Hello".to_string())
        );
    }
}
