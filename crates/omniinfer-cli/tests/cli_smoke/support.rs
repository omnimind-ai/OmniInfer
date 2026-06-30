pub(super) use assert_cmd::Command;
pub(super) use omniinfer_core::http_client;
pub(super) use predicates::prelude::*;
pub(super) use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
pub(super) use std::process::{Command as StdCommand, Stdio};
use std::sync::mpsc;
use std::thread;
pub(super) use std::time::Duration;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(super) struct TestGateway {
    pub(super) port: u16,
    request_rx: mpsc::Receiver<String>,
    handle: thread::JoinHandle<()>,
}

impl TestGateway {
    pub(super) fn start(responses: Vec<Response>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test gateway");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking test gateway");
        let port = listener.local_addr().expect("local addr").port();
        let (request_tx, request_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            for response_body in responses {
                let mut stream = accept_test_request(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("set request read timeout");
                let request = read_http_request(&mut stream);
                request_tx.send(request).expect("send request");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.content_type,
                    response_body.content_len()
                );
                stream.write_all(response.as_bytes()).expect("write header");
                response_body.write_body(&mut stream);
                stream.flush().expect("flush response");
            }
        });
        Self {
            port,
            request_rx,
            handle,
        }
    }

    pub(super) fn request(&self) -> String {
        self.request_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("receive request")
    }

    pub(super) fn join(self) {
        self.handle.join().expect("server thread");
    }
}

fn accept_test_request(listener: &TcpListener) -> std::net::TcpStream {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return stream,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    panic!("timed out waiting for test gateway request");
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("accept request: {error}"),
        }
    }
}

pub(super) struct Response {
    body: ResponseBody,
    content_type: String,
}

enum ResponseBody {
    Text(String),
    Chunks(Vec<String>),
}

impl Response {
    pub(super) fn new(body: &str) -> Self {
        Self::with_content_type(body, "application/json")
    }

    pub(super) fn with_content_type(body: &str, content_type: &str) -> Self {
        Self {
            body: ResponseBody::Text(body.to_string()),
            content_type: content_type.to_string(),
        }
    }

    pub(super) fn chunks(chunks: &[&str], content_type: &str) -> Self {
        Self {
            body: ResponseBody::Chunks(chunks.iter().map(|chunk| (*chunk).to_string()).collect()),
            content_type: content_type.to_string(),
        }
    }

    fn content_len(&self) -> usize {
        match &self.body {
            ResponseBody::Text(body) => body.len(),
            ResponseBody::Chunks(chunks) => chunks.iter().map(String::len).sum(),
        }
    }

    fn write_body(&self, stream: &mut impl Write) {
        match &self.body {
            ResponseBody::Text(body) => {
                stream.write_all(body.as_bytes()).expect("write body");
            }
            ResponseBody::Chunks(chunks) => {
                for chunk in chunks {
                    stream.write_all(chunk.as_bytes()).expect("write chunk");
                    stream.flush().expect("flush chunk");
                    thread::sleep(Duration::from_millis(15));
                }
            }
        }
    }
}

pub(super) fn temp_repo_root(test_name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-rs-{test_name}-{nanos}"))
}

pub(super) fn install_fake_backend(root: &std::path::Path, backend_id: &str) {
    let binary_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join(binary_name);
    fs::create_dir_all(launcher.parent().unwrap()).expect("create fake backend dir");
    fs::write(&launcher, "#!/usr/bin/env bash\nexit 0\n").expect("write fake backend");
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(&launcher)
            .expect("fake backend metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).expect("chmod fake backend");
    }
}

pub(super) fn test_external_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-cpu"
    } else {
        "llama.cpp-linux"
    }
}

pub(super) fn test_secondary_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac-intel"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-windows-arm64"
    } else {
        "llama.cpp-linux"
    }
}

pub(super) fn test_runtime_platform_dir() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

#[cfg(unix)]
pub(super) fn install_fake_runtime_server(root: &std::path::Path, backend_id: &str) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("llama-server");
    fs::create_dir_all(launcher.parent().unwrap()).expect("create fake runtime dir");
    fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$port" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def _json(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok"})
        else:
            self._json({"ok": True})
    def do_POST(self):
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#,
    )
    .expect("write fake runtime");
    let mut permissions = fs::metadata(&launcher)
        .expect("fake runtime metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod fake runtime");
}

pub(super) fn fake_cloudflared_launcher(root: &std::path::Path) -> std::path::PathBuf {
    fake_cloudflared_launcher_with_url(root, "https://example-test.trycloudflare.com")
}

pub(super) fn fake_cloudflared_launcher_with_url(
    root: &std::path::Path,
    url: &str,
) -> std::path::PathBuf {
    #[cfg(unix)]
    {
        fake_cloudflared_launcher_unix(root, url)
    }
    #[cfg(windows)]
    {
        fake_cloudflared_launcher_windows(root, url)
    }
}

#[cfg(unix)]
pub(super) fn fake_cloudflared_launcher_unix(
    root: &std::path::Path,
    url: &str,
) -> std::path::PathBuf {
    let launcher = root.join("fake-cloudflared.sh");
    let output = root.join("cloudflared.args");
    fs::write(
        &launcher,
        format!(
            "#!/usr/bin/env bash\nprintf '%s ' \"$@\" > '{}'\necho 'Your quick Tunnel has been created! Visit it at {}'\nsleep 30\n",
            output.display(),
            url
        ),
    )
    .expect("write fake cloudflared");
    let mut permissions = fs::metadata(&launcher)
        .expect("cloudflared metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod cloudflared");
    launcher
}

#[cfg(windows)]
pub(super) fn fake_cloudflared_launcher_windows(
    root: &std::path::Path,
    url: &str,
) -> std::path::PathBuf {
    let launcher = root.join("fake-cloudflared.cmd");
    let output = root.join("cloudflared.args");
    fs::write(
        &launcher,
        format!(
            "@echo off\r\necho %* > \"{}\"\r\necho Your quick Tunnel has been created! Visit it at {}\r\npowershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command \"Start-Sleep -Seconds 30\" > nul\r\n",
            output.display(),
            url
        ),
    )
    .expect("write fake cloudflared");
    launcher
}

fn read_http_request(stream: &mut impl Read) -> String {
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut expected_len = None;
    loop {
        let bytes = stream.read(&mut buffer).expect("read request");
        if bytes == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..bytes]);
        if expected_len.is_none()
            && let Some(header_end) = find_header_end(&raw)
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
    String::from_utf8_lossy(&raw).to_string()
}

fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|window| window == b"\r\n\r\n")
}

pub(super) fn request_body_json(request: &str) -> serde_json::Value {
    let body = request
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("request body separator");
    serde_json::from_str(body).expect("request body json")
}

pub(super) fn wait_for_file(path: std::path::PathBuf) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(text) = fs::read_to_string(&path) {
            return text;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {}", path.display());
}

pub(super) fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    listener.local_addr().expect("local addr").port()
}

pub(super) fn wait_for_http_json(port: u16, path: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    let url = format!("http://127.0.0.1:{port}{path}");
    while Instant::now() < deadline {
        if let Ok(response) = http_client::get_json(&url, Duration::from_secs(1)) {
            if response.status < 400 {
                return response.body;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for {url}");
}

pub(super) fn stop_rust_serve(
    source_root: &std::path::Path,
    state_root: &std::path::Path,
    port: u16,
) {
    let mut stop = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", state_root)
        .args(["serve", "stop", "--port"])
        .arg(port.to_string())
        .assert()
        .success();
    assert!(wait_for_port_closed(port));
}

pub(super) fn wait_for_port_closed(port: u16) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}
