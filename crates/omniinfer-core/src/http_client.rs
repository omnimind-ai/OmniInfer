use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HttpError {
    #[error("unsupported URL: {0}")]
    UnsupportedUrl(String),
    #[error("request failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("response was not valid UTF-8")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("response JSON parse failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("malformed HTTP response")]
    MalformedResponse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JsonResponse {
    pub status: u16,
    pub body: Value,
}

pub fn get_json(url: &str, timeout: Duration) -> Result<JsonResponse, HttpError> {
    let parsed = parse_http_url(url)?;
    let mut stream = TcpStream::connect((parsed.host.as_str(), parsed.port))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}:{}\r\nAccept: application/json\r\nConnection: close\r\n\r\n",
        parsed.path, parsed.host, parsed.port
    );
    stream.write_all(request.as_bytes())?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8(raw)?;
    let (head, body) = text.split_once("\r\n\r\n").ok_or(HttpError::MalformedResponse)?;
    let status = parse_status(head)?;
    Ok(JsonResponse {
        status,
        body: serde_json::from_str(body.trim())?,
    })
}

struct ParsedUrl {
    host: String,
    port: u16,
    path: String,
}

fn parse_http_url(url: &str) -> Result<ParsedUrl, HttpError> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| HttpError::UnsupportedUrl(url.to_string()))?;
    let (authority, path) = match rest.split_once('/') {
        Some((authority, path)) => (authority, format!("/{path}")),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => {
            let parsed_port = port
                .parse::<u16>()
                .map_err(|_| HttpError::UnsupportedUrl(url.to_string()))?;
            (host.to_string(), parsed_port)
        }
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        return Err(HttpError::UnsupportedUrl(url.to_string()));
    }
    Ok(ParsedUrl { host, port, path })
}

fn parse_status(head: &str) -> Result<u16, HttpError> {
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or(HttpError::MalformedResponse)?;
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_local_http_urls() {
        let parsed = parse_http_url("http://127.0.0.1:9000/omni/state").unwrap();
        assert_eq!(parsed.host, "127.0.0.1");
        assert_eq!(parsed.port, 9000);
        assert_eq!(parsed.path, "/omni/state");
    }
}
