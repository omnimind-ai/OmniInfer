use std::io::{BufRead, BufReader, Read, Write};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub body: String,
}

pub fn get_json(url: &str, timeout: Duration) -> Result<JsonResponse, HttpError> {
    request_json("GET", url, None, timeout)
}

pub fn post_json(url: &str, body: &Value, timeout: Duration) -> Result<JsonResponse, HttpError> {
    request_json("POST", url, Some(body), timeout)
}

pub fn post(
    url: &str,
    body: &Value,
    accept: &str,
    timeout: Duration,
) -> Result<RawResponse, HttpError> {
    request("POST", url, Some(body), accept, timeout)
}

pub fn post_streaming_lines<F>(
    url: &str,
    body: &Value,
    accept: &str,
    timeout: Duration,
    on_line: F,
) -> Result<RawResponse, HttpError>
where
    F: FnMut(&str),
{
    request_streaming_lines("POST", url, Some(body), accept, timeout, on_line)
}

fn request_json(
    method: &str,
    url: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<JsonResponse, HttpError> {
    let response = request(method, url, body, "application/json", timeout)?;
    Ok(JsonResponse {
        status: response.status,
        body: serde_json::from_str(response.body.trim())?,
    })
}

fn request(
    method: &str,
    url: &str,
    body: Option<&Value>,
    accept: &str,
    timeout: Duration,
) -> Result<RawResponse, HttpError> {
    request_streaming_lines(method, url, body, accept, timeout, |_| {})
}

fn request_streaming_lines<F>(
    method: &str,
    url: &str,
    body: Option<&Value>,
    accept: &str,
    timeout: Duration,
    mut on_line: F,
) -> Result<RawResponse, HttpError>
where
    F: FnMut(&str),
{
    let parsed = parse_http_url(url)?;
    let mut stream = TcpStream::connect((parsed.host.as_str(), parsed.port))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let body_bytes = match body {
        Some(body) => serde_json::to_vec(body)?,
        None => Vec::new(),
    };
    let request = format!(
        "{method} {} HTTP/1.1\r\nHost: {}:{}\r\nAccept: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        parsed.path,
        parsed.host,
        parsed.port,
        accept,
        body_bytes.len()
    );
    stream.write_all(request.as_bytes())?;
    if !body_bytes.is_empty() {
        stream.write_all(&body_bytes)?;
    }

    let mut reader = BufReader::new(stream);
    let head = read_response_head(&mut reader)?;
    let status = parse_status(&head)?;
    let content_type = header_value(&head, "Content-Type");
    let body = read_response_body(reader, content_type.as_deref(), &mut on_line)?;
    Ok(RawResponse {
        status,
        content_type,
        body,
    })
}

fn read_response_head(reader: &mut impl BufRead) -> Result<String, HttpError> {
    let mut head = String::new();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        head.push_str(&line);
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    if head.trim().is_empty() {
        return Err(HttpError::MalformedResponse);
    }
    Ok(head)
}

fn read_response_body<F>(
    mut reader: BufReader<TcpStream>,
    content_type: Option<&str>,
    on_line: &mut F,
) -> Result<String, HttpError>
where
    F: FnMut(&str),
{
    let mut body = String::new();
    if content_type
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        loop {
            let mut line = String::new();
            let bytes = reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            on_line(line.trim_end_matches(['\r', '\n']));
            body.push_str(&line);
        }
    } else {
        reader.read_to_string(&mut body)?;
    }
    Ok(body)
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

fn header_value(head: &str, key: &str) -> Option<String> {
    head.lines().skip(1).find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(key)
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
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

    #[test]
    fn parses_header_values_case_insensitively() {
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\n";
        assert_eq!(
            header_value(head, "Content-Type").as_deref(),
            Some("text/event-stream; charset=utf-8")
        );
    }
}
