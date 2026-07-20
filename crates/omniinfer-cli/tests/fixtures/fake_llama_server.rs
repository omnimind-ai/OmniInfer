use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

fn main() {
    let mut port = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--port" {
            port = args.next();
        }
    }
    let port = port.expect("--port is required");
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).expect("bind fake runtime");
    for stream in listener.incoming().flatten() {
        handle(stream);
    }
}

fn handle(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut content_length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.is_empty() {
            return;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let response = if request_line.starts_with("GET /health") {
        r#"{"status":"ok"}"#
    } else if request_line.starts_with("POST /v1/chat/completions") {
        r#"{"choices":[{"message":{"content":"fake backend"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}"#
    } else {
        r#"{"ok":true}"#
    };
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(response.as_bytes());
}
