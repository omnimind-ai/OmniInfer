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
    #[error("HTTPS request failed: {0}")]
    Https(String),
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
    if url.starts_with("https://") {
        return request_json_https("GET", url, None, None, timeout);
    }
    request_json("GET", url, None, timeout)
}

pub fn post_json(url: &str, body: &Value, timeout: Duration) -> Result<JsonResponse, HttpError> {
    if url.starts_with("https://") {
        return request_json_https("POST", url, Some(body), None, timeout);
    }
    request_json("POST", url, Some(body), timeout)
}

pub fn post_json_with_bearer(
    url: &str,
    body: &Value,
    bearer: Option<&str>,
    timeout: Duration,
) -> Result<JsonResponse, HttpError> {
    if url.starts_with("https://") {
        return request_json_https("POST", url, Some(body), bearer, timeout);
    }
    request_json_with_bearer_http("POST", url, Some(body), bearer, timeout)
}

fn request_json_with_bearer_http(
    method: &str,
    url: &str,
    body: Option<&Value>,
    bearer: Option<&str>,
    timeout: Duration,
) -> Result<JsonResponse, HttpError> {
    let response = request_with_bearer(method, url, body, "application/json", bearer, timeout)?;
    Ok(JsonResponse {
        status: response.status,
        body: serde_json::from_str(response.body.trim())?,
    })
}

fn request_json_https(
    method: &str,
    url: &str,
    body: Option<&Value>,
    bearer: Option<&str>,
    timeout: Duration,
) -> Result<JsonResponse, HttpError> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .new_agent();
    let response = match method {
        "GET" => {
            let mut request = agent.get(url).header("Accept", "application/json");
            if let Some(token) = bearer.filter(|value| !value.trim().is_empty()) {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
            request
                .call()
                .map_err(|error| HttpError::Https(error.to_string()))?
        }
        "POST" => {
            let mut request = agent.post(url).header("Accept", "application/json");
            if let Some(token) = bearer.filter(|value| !value.trim().is_empty()) {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
            request
                .send_json(body.unwrap_or(&Value::Null))
                .map_err(|error| HttpError::Https(error.to_string()))?
        }
        _ => return Err(HttpError::UnsupportedUrl(url.to_string())),
    };
    let status = response.status().as_u16();
    let mut body = response.into_body();
    let value: Value = body
        .read_json()
        .map_err(|error| HttpError::Https(error.to_string()))?;
    Ok(JsonResponse {
        status,
        body: value,
    })
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
    request_with_bearer(method, url, body, accept, None, timeout)
}

fn request_with_bearer(
    method: &str,
    url: &str,
    body: Option<&Value>,
    accept: &str,
    bearer: Option<&str>,
    timeout: Duration,
) -> Result<RawResponse, HttpError> {
    request_streaming_lines_with_bearer(method, url, body, accept, bearer, timeout, |_| {})
}

fn request_streaming_lines<F>(
    method: &str,
    url: &str,
    body: Option<&Value>,
    accept: &str,
    timeout: Duration,
    on_line: F,
) -> Result<RawResponse, HttpError>
where
    F: FnMut(&str),
{
    request_streaming_lines_with_bearer(method, url, body, accept, None, timeout, on_line)
}

fn request_streaming_lines_with_bearer<F>(
    method: &str,
    url: &str,
    body: Option<&Value>,
    accept: &str,
    bearer: Option<&str>,
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
    let auth = bearer
        .filter(|value| !value.trim().is_empty())
        .map(|token| format!("Authorization: Bearer {token}\r\n"))
        .unwrap_or_default();
    let request = format!(
        "{method} {} HTTP/1.1\r\nHost: {}:{}\r\nAccept: {}\r\n{}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        parsed.path,
        parsed.host,
        parsed.port,
        accept,
        auth,
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

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

    #[test]
    fn post_json_with_bearer_sends_authorization_header() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut raw = Vec::new();
            let mut buffer = [0_u8; 1024];
            let mut expected_len = None;
            loop {
                let bytes = stream.read(&mut buffer).unwrap();
                if bytes == 0 {
                    break;
                }
                raw.extend_from_slice(&buffer[..bytes]);
                if expected_len.is_none()
                    && let Some(header_end) =
                        raw.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let header = String::from_utf8_lossy(&raw[..header_end]);
                    let content_length = header
                        .lines()
                        .find_map(|line| line.strip_prefix("Content-Length: "))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    expected_len = Some(header_end + 4 + content_length);
                }
                if let Some(length) = expected_len
                    && raw.len() >= length
                {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&raw).to_string();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
                )
                .unwrap();
            request
        });
        let response = post_json_with_bearer(
            &format!("http://127.0.0.1:{port}/v1/chat/completions"),
            &serde_json::json!({"stream": false}),
            Some("test-key"),
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body["ok"], true);
        let request = handle.join().unwrap();
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(request.contains("Authorization: Bearer test-key\r\n"));
    }
}
