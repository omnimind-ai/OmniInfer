use super::runtime_manager::{LoadedRuntimeSummary, pick_runtime_port};
use super::*;

static TEST_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
use axum::Json;
use axum::extract::Query;
use axum::routing::{get, post};
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

mod gpu;
mod history_auth;
mod models_proxy;
mod routing;
mod runtime;
struct TestServer {
    port: u16,
    stop: Option<oneshot::Sender<()>>,
    stopped: Arc<AtomicBool>,
}

impl TestServer {
    async fn stop(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn wait_stopped(&self) -> bool {
        for _ in 0..40 {
            if self.stopped.load(Ordering::SeqCst) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }
}

async fn spawn_test_upstream() -> TestServer {
    let (tx, rx) = oneshot::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_for_task = Arc::clone(&stopped);
    let app = axum::Router::new()
        .route(
            "/health",
            get(|| async { axum::Json(json!({"status": "ok"})) }),
        )
        .route("/v1/chat/completions", post(echo_chat_completion))
        .route("/omni/model/select", post(echo_model_select))
        .route(
            "/omni/shutdown",
            post(|| async { axum::Json(json!({"ok": true})) }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
        stopped_for_task.store(true, Ordering::SeqCst);
    });
    TestServer {
        port,
        stop: Some(tx),
        stopped,
    }
}

async fn echo_chat_completion(
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(json!({
        "trace": query.get("trace").cloned().unwrap_or_default(),
        "auth": header_text(&headers, "authorization").unwrap_or_default(),
        "body": body,
    }))
}

async fn echo_model_select(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(json!({
        "ok": true,
        "delegated": true,
        "selected_backend": body.get("backend").cloned().unwrap_or(Value::Null),
        "selected_model": body.get("model").cloned().unwrap_or(Value::Null),
        "body": body,
    }))
}

async fn spawn_test_gateway(_unused_port: u16, access_policy: GatewayAccessPolicy) -> TestServer {
    spawn_test_gateway_with_public_root(access_policy, None).await
}

async fn spawn_test_gateway_with_public_root(
    access_policy: GatewayAccessPolicy,
    public_model_root: Option<PathBuf>,
) -> TestServer {
    spawn_test_gateway_with_options(access_policy, public_model_root).await
}

async fn spawn_test_gateway_with_options(
    access_policy: GatewayAccessPolicy,
    public_model_root: Option<PathBuf>,
) -> TestServer {
    spawn_test_gateway_with_runtime_options(
        access_policy,
        public_model_root,
        Duration::from_secs(120),
    )
    .await
}

async fn spawn_test_gateway_with_runtime_timeout(
    access_policy: GatewayAccessPolicy,
    runtime_startup_timeout: Duration,
) -> TestServer {
    spawn_test_gateway_with_runtime_options(access_policy, None, runtime_startup_timeout).await
}

async fn spawn_test_gateway_with_runtime_options(
    access_policy: GatewayAccessPolicy,
    public_model_root: Option<PathBuf>,
    runtime_startup_timeout: Duration,
) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = oneshot::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_for_task = Arc::clone(&stopped);
    tokio::spawn(async move {
        tokio::select! {
            result = run_gateway_with_listener(GatewayConfig {
                listen_host: "127.0.0.1".to_string(),
                listen_port: port,
                runtime_startup_timeout,
                access_policy,
                public_model_root,
            }, listener) => {
                let _ = result;
            }
            _ = rx => {}
        }
        stopped_for_task.store(true, Ordering::SeqCst);
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("test gateway must become ready");
    TestServer {
        port,
        stop: Some(tx),
        stopped,
    }
}

async fn remote_admin_post(
    port: u16,
    path: &'static str,
    key: &'static str,
    payload: Value,
) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}{path}"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .send_json(payload)
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

async fn gateway_state(port: u16) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::get(format!("http://127.0.0.1:{port}/omni/state"))
            .call()
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

fn resource_total(state: &Value, field: &str) -> u64 {
    state["resource_ledger"][field]
        .as_object()
        .into_iter()
        .flat_map(|domains| domains.values())
        .filter_map(Value::as_u64)
        .sum()
}

async fn remote_chat(port: u16, key: &'static str, model: &'static str) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .send_json(json!({
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

async fn remote_chat_without_model(port: u16, key: &'static str) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .send_json(json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

async fn wait_for_history(port: u16, key: &'static str, model: &'static str) -> Value {
    for _ in 0..20 {
        let value = tokio::task::spawn_blocking(move || {
            let response = ureq::get(format!(
                "http://127.0.0.1:{port}/omni/request-history?limit=5&model={model}"
            ))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .call()
            .unwrap();
            response.into_body().read_json::<Value>().unwrap()
        })
        .await
        .unwrap();
        if value
            .get("data")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        {
            return value;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out waiting for request history");
}

fn temp_root(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}"))
}

fn external_test_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-cuda"
    } else {
        "llama.cpp-linux-cuda"
    }
}

fn embedded_test_backend_id() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("mlx-mac")
    } else if cfg!(target_os = "linux") {
        Some("mnn-linux")
    } else {
        None
    }
}

fn test_runtime_platform_dir() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

fn write_public_model_manifest(root: &std::path::Path, id: &str) -> PathBuf {
    let dir = root.join(id);
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    std::fs::write(&model, b"gguf").unwrap();
    std::fs::write(
        dir.join("omni-model.json"),
        format!(
            r#"{{
                    "id": "{id}",
                    "display_name": "Qwen3.5 4B Q4_K_M",
                    "backend": "{}",
                    "model": "model.gguf",
                    "ctx_size": 512,
                    "modalities": ["text"],
                    "quant": "Q4_K_M",
                    "launch_args": ["-ngl", "999"]
                }}"#,
            external_test_backend_id()
        ),
    )
    .unwrap();
    model
}

#[cfg(target_os = "linux")]
fn write_vllm_public_model_manifest(root: &std::path::Path, id: &str) -> PathBuf {
    let dir = root.join(id);
    let model = dir.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(
        model.join("config.json"),
        r#"{"max_position_embeddings":32768}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("omni-model.json"),
        format!(
            r#"{{
                    "id": "{id}",
                    "display_name": "GELab Zero 4B Preview",
                    "backend": "vllm-linux-cuda",
                    "model": "model",
                    "ctx_size": 32768,
                    "modalities": ["text", "vision"],
                    "quant": "BF16",
                    "launch_args": ["--served-model-name", "local"]
                }}"#
        ),
    )
    .unwrap();
    model
}

fn install_fake_llama_server(root: &std::path::Path, backend_id: &str) {
    let launcher_name = if cfg!(target_os = "windows") {
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
        .join(launcher_name);
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    #[cfg(any(windows, target_os = "macos"))]
    {
        install_fake_llama_server_native(&launcher);
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        let script = r#"#!/usr/bin/env bash
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
delay_file="$(dirname "$0")/startup-delay-ms"
delay_ms="$(cat "$delay_file" 2>/dev/null || printf 0)"
placement_mode="$(cat "$(dirname "$0")/placement-mode" 2>/dev/null || printf partial)"
if [ "$placement_mode" = "oversized" ]; then
  printf '%s\n' \
    'load_tensors: offloaded 2/4 layers to GPU' \
    'load_tensors: CPU_Mapped model buffer size = 1000000.00 GiB' \
    'load_tensors: CUDA0 model buffer size = 1000000.00 GiB'
else
  printf '%s\n' \
    'load_tensors: offloaded 2/4 layers to GPU' \
    'load_tensors: CPU_Mapped model buffer size = 8.00 MiB' \
    'load_tensors: CUDA0 model buffer size = 16.00 MiB' \
    'llama_kv_cache: CPU KV buffer size = 2.00 MiB' \
    'llama_kv_cache: CUDA0 KV buffer size = 4.00 MiB' \
    'sched_reserve: CPU compute buffer size = 2.00 MiB' \
    'sched_reserve: CUDA0 compute buffer size = 4.00 MiB'
fi
exec python3 - "$port" "$delay_ms" <<'PY'
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])
time.sleep(int(sys.argv[2]) / 1000)

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
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()
    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()
    def do_PUT(self):
        self._json({"ok": True})
    def do_DELETE(self):
        self._json({"ok": True})
    def do_PATCH(self):
        self._json({"ok": True})
    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok"})
        elif self.path.startswith("/props"):
            self._json({"n_ctx": 512, "total_slots": 2})
        else:
            self._json({"ok": True})
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode() or "{}")
        if self.path.startswith("/tokenize"):
            self._json({"tokens": [1, 2, 3], "echo": payload})
            return
        if self.path.startswith("/detokenize"):
            self._json({"content": "hello", "echo": payload})
            return
        if self.path.startswith("/slots/"):
            self._json({"ok": True})
            return
        if self.path.startswith("/v1/chat/completions") and payload.get("stream") is True:
            frames = [
                'data: {"choices":[{"delta":{"content":"fake"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" backend"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                'data: [DONE]\n\n',
            ]
            raw = "".join(frames).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "model_echo": payload.get("model"),
            "enable_thinking_echo": payload.get("chat_template_kwargs", {}).get("enable_thinking"),
            "max_tokens_echo": payload.get("max_tokens"),
            "temperature_echo": payload.get("temperature"),
            "top_p_echo": payload.get("top_p"),
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#;
        std::fs::write(&launcher, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&launcher, permissions).unwrap();
        }
    }
}

#[cfg(target_os = "linux")]
fn install_fake_vla_server(root: &std::path::Path, backend_id: &str) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("vla-server");
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    std::fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
set -euo pipefail
bind=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bind) bind="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'vla-server: bound to %s. ready.\n' "$bind"
exec python3 - "$bind" <<'PY'
import socket
import sys

endpoint = sys.argv[1].removeprefix("tcp://")
host, port = endpoint.rsplit(":", 1)
with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen()
    while True:
        connection, _ = server.accept()
        connection.close()
PY
"#,
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&launcher, permissions).unwrap();
}

#[cfg(target_os = "linux")]
fn install_fake_stable_diffusion_server(root: &std::path::Path) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join("stable-diffusion.cpp-linux-vulkan")
        .join("bin")
        .join("sd-server");
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    std::fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
set -euo pipefail
host=""
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --listen-ip) host="$2"; shift 2 ;;
    --listen-port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'listening on: http://%s:%s\n' "$host" "$port"
exec python3 - "$host" "$port" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

host, port = sys.argv[1], int(sys.argv[2])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def _json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Sdcpp-Test", "passthrough")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path.startswith("/sdcpp/v1/capabilities"):
            self._json({"backend": "fake-sdcpp", "path": self.path})
        elif self.path.startswith("/sdcpp/v1/jobs/job-42"):
            self._json({"id": "job-42", "status": "completed"})
        else:
            self._json({"error": "not found"}, 404)
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode() or "{}")
        if self.path == "/sdcpp/v1/vid_gen":
            self._json({"id": "job-42", "status": "queued", "request": payload}, 202)
        elif self.path == "/sdcpp/v1/jobs/job-42/cancel":
            self._json({"id": "job-42", "status": "cancelled"})
        else:
            self._json({"error": "not found"}, 404)

HTTPServer((host, port), Handler).serve_forever()
PY
"#,
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&launcher, permissions).unwrap();
}

#[cfg(target_os = "linux")]
fn install_fake_vllm_server(root: &std::path::Path) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join("vllm-linux-cuda")
        .join("bin")
        .join("vllm");
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    std::fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
exec python3 - "$port" <<'PY'
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
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._json({"ok": True})
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode() or "{}")
        if self.path.startswith("/v1/chat/completions") and payload.get("stream") is True:
            assert payload.get("stream_options", {}).get("include_usage") is True
            frames = [
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"local","choices":[{"index":0,"delta":{"role":"assistant","content":"fake"},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"local","choices":[{"index":0,"delta":{"content":" backend"},"finish_reason":"stop"}]}\n\n',
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"local","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                'data: [DONE]\n\n',
            ]
            raw = "".join(frames).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "model_echo": payload.get("model"),
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&launcher, permissions).unwrap();
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn install_fake_llama_server_native(launcher: &std::path::Path) {
    let source = launcher.with_file_name("fake-llama-server.rs");
    let code = r##"
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

fn main() {
    let mut port = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--port" {
            port = args.next().unwrap_or_default();
        }
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Ok(raw) = std::fs::read_to_string(executable.with_file_name("startup-delay-ms")) {
            if let Ok(delay_ms) = raw.trim().parse::<u64>() {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
        let oversized = std::fs::read_to_string(executable.with_file_name("placement-mode"))
            .is_ok_and(|value| value.trim() == "oversized");
        println!("load_tensors: offloaded 2/4 layers to GPU");
        if oversized {
            println!("load_tensors: CPU_Mapped model buffer size = 1000000.00 GiB");
            println!("load_tensors: CUDA0 model buffer size = 1000000.00 GiB");
        } else {
            println!("load_tensors: CPU_Mapped model buffer size = 8.00 MiB");
            println!("load_tensors: CUDA0 model buffer size = 16.00 MiB");
            println!("llama_kv_cache: CPU KV buffer size = 2.00 MiB");
            println!("llama_kv_cache: CUDA0 KV buffer size = 4.00 MiB");
            println!("sched_reserve: CPU compute buffer size = 2.00 MiB");
            println!("sched_reserve: CUDA0 compute buffer size = 4.00 MiB");
        }
    }
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).unwrap();
    for stream in listener.incoming().flatten() {
        handle(stream);
    }
}

fn handle(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let body = String::from_utf8_lossy(&body);
    let payload = response_payload(&request_line, &body);
    write_response(&mut stream, &payload.0, payload.1);
}

fn response_payload(request_line: &str, body: &str) -> (String, &'static str) {
    if request_line.starts_with("GET /health") {
        return (r#"{"status":"ok"}"#.to_string(), "application/json");
    }
    if request_line.starts_with("GET /props") {
        return (r#"{"n_ctx":512,"total_slots":2}"#.to_string(), "application/json");
    }
    if request_line.starts_with("POST /tokenize") {
        return (
            r#"{"tokens":[1,2,3],"echo":{"content":"hello"}}"#.to_string(),
            "application/json",
        );
    }
    if request_line.starts_with("POST /detokenize") {
        return (
            r#"{"content":"hello","echo":{"tokens":[1,2,3]}}"#.to_string(),
            "application/json",
        );
    }
    if request_line.starts_with("POST /slots/") {
        return (r#"{"ok":true}"#.to_string(), "application/json");
    }
    if request_line.starts_with("POST /v1/chat/completions") && wants_stream(body) {
        return (
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"fake\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" backend\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n"
            )
            .to_string(),
            "text/event-stream",
        );
    }
    if request_line.starts_with("POST /v1/chat/completions") {
        let model = extract_json_string(body, "model")
            .map(|value| format!(r#""{value}""#))
            .unwrap_or_else(|| "null".to_string());
        let max_tokens = extract_json_number(body, "max_tokens")
            .unwrap_or_else(|| "null".to_string());
        let temperature = extract_json_number(body, "temperature")
            .unwrap_or_else(|| "null".to_string());
        let top_p = extract_json_number(body, "top_p")
            .unwrap_or_else(|| "null".to_string());
        let enable_thinking = if body
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>()
            .contains(r#""enable_thinking":true"#)
        {
            "true"
        } else {
            "false"
        };
        return (
            format!(
                r#"{{"choices":[{{"message":{{"content":"fake backend"}},"finish_reason":"stop"}}],"model_echo":{model},"enable_thinking_echo":{enable_thinking},"max_tokens_echo":{max_tokens},"temperature_echo":{temperature},"top_p_echo":{top_p},"usage":{{"prompt_tokens":3,"completion_tokens":2}}}}"#
            ),
            "application/json",
        );
    }
    (r#"{"ok":true}"#.to_string(), "application/json")
}

fn wants_stream(body: &str) -> bool {
    let compact: String = body.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact.contains(r#""stream":true"#)
}

fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!(r#""{key}""#);
    let start = body.find(&needle)?;
    let after_key = &body[start + needle.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let value = after_colon.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

fn extract_json_number(body: &str, key: &str) -> Option<String> {
    let needle = format!(r#""{key}""#);
    let start = body.find(&needle)?;
    let after_key = &body[start + needle.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    let end = value
        .find(|ch: char| !matches!(ch, '0'..='9' | '-' | '+' | '.' | 'e' | 'E'))
        .unwrap_or(value.len());
    (end > 0).then(|| value[..end].to_string())
}

fn write_response(stream: &mut TcpStream, body: &str, content_type: &str) {
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.as_bytes().len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(body.as_bytes());
}
"##;
    std::fs::write(&source, code).unwrap();
    let status = std::process::Command::new("rustc")
        .arg("--edition=2021")
        .arg(&source)
        .arg("-o")
        .arg(launcher)
        .status()
        .expect("compile native fake llama-server");
    assert!(
        status.success(),
        "failed to compile native fake llama-server"
    );
}

struct EnvGuard {
    key: &'static str,
    old: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: String) -> Self {
        let old = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, old }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(old) = &self.old {
                std::env::set_var(self.key, old);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}
