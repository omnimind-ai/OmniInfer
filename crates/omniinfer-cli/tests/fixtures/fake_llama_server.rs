use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn main() {
    let mut port = None;
    let mut model = "test-model".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--port" {
            port = args.next();
        } else if matches!(arg.as_str(), "-m" | "--model") {
            model = args.next().expect("model value is required");
        }
    }
    let port = port.expect("--port is required");
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).expect("bind fake runtime");
    if let Ok(path) = std::env::var("OMNIINFER_TEST_RUNTIME_STARTED_FILE") {
        let mut marker = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open runtime started marker");
        writeln!(marker, "{}", std::process::id()).expect("write runtime started marker");
    }
    if std::env::var_os("OMNIINFER_TEST_RUNTIME_EXIT_AFTER_BIND").is_some() {
        if let Ok(path) = std::env::var("OMNIINFER_TEST_RUNTIME_EXITED_FILE") {
            std::fs::write(path, std::process::id().to_string())
                .expect("write runtime exit marker");
        }
        return;
    }
    if let Ok(path) = std::env::var("OMNIINFER_TEST_RUNTIME_READY_FILE") {
        std::fs::write(path, "ready").expect("write runtime ready marker");
    }
    if let Ok(path) = std::env::var("OMNIINFER_TEST_RUNTIME_DELAY_FILE") {
        while !std::path::Path::new(&path).exists() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    for stream in listener.incoming().flatten() {
        handle(stream, &model);
    }
}

fn handle(mut stream: TcpStream, model: &str) {
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
        r#"{"status":"ok"}"#.to_string()
    } else if request_line.starts_with("GET /v1/models") {
        format!(r#"{{"object":"list","data":[{{"id":"{model}"}}]}}"#)
    } else if request_line.starts_with("POST /v1/chat/completions") {
        r#"{"choices":[{"message":{"content":"fake backend"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}"#.to_string()
    } else {
        r#"{"ok":true}"#.to_string()
    };
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(response.as_bytes());
}
