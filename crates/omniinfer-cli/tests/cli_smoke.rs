use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[test]
fn help_lists_core_commands() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("advisor"))
        .stdout(predicate::str::contains("serve"));
}

#[test]
fn completion_generates_bash_script() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.args(["completion", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_omniinfer-rs"));
}

#[test]
fn strict_mode_reports_unported_commands_without_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .args(["advisor", "system"])
        .assert()
        .success()
        .stdout(predicate::str::contains("implementation pending"));
}

#[test]
fn model_load_posts_payload_and_persists_state() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"object":"list","data":[{"id":"llama.cpp-linux-cuda","family":"llama.cpp","binary_exists":true}]}"#,
        ),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf","selected_mmproj":null,"selected_ctx_size":8192}"#,
        ),
    ]);
    let root = temp_repo_root("model-load");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");
    let model = root.join("model.gguf");
    fs::write(&model, "").expect("write model");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["model", "load", "-m"])
        .arg(&model)
        .args(["--ctx-size", "8192", "--", "-ngl", "999"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Auto-selected backend: llama.cpp-linux-cuda",
        ))
        .stdout(predicate::str::contains("Model loaded"))
        .stdout(predicate::str::contains("ctx-size: 8192"));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /omni/backends?scope=all HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    assert!(request.contains(&format!(r#""model":"{}""#, model.display())));
    assert!(request.contains(r#""backend":"llama.cpp-linux-cuda""#));
    assert!(request.contains(r#""ctx_size":8192"#));
    assert!(request.contains(r#""launch_args":["-ngl","999"]"#));
    gateway.join();

    let state_raw = fs::read_to_string(root.join(".local").join("config").join("state.json"))
        .expect("state file");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("state json");
    assert_eq!(state["selected_backend"], "llama.cpp-linux-cuda");
    assert_eq!(state["selected_model"], "/tmp/model.gguf");
    assert_eq!(state["selected_ctx_size"], 8192);
    assert!(state.get("selected_mmproj").is_none());
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_stop_posts_to_local_gateway() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"stopped":true}"#),
    ]);

    let root = temp_repo_root("backend-stop");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "stop"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Current backend process stopped"));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/backend/stop HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(root).ok();
}

#[test]
fn shutdown_posts_to_local_gateway() {
    let gateway = TestGateway::start(vec![Response::new(r#"{"ok":true}"#)]);

    let root = temp_repo_root("shutdown");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .arg("shutdown")
        .assert()
        .success()
        .stdout(predicate::str::contains("OmniInfer service stopped"));

    let request = gateway.request();
    assert!(request.starts_with("POST /omni/shutdown HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_stop_starts_gateway_when_needed() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"stopped":true}"#),
    ]);
    let root = temp_repo_root("backend-stop-autostart");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            gateway.port
        ),
    )
    .expect("write config");
    let launcher = fake_python_launcher(&root);

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PYTHON", &launcher)
        .args(["backend", "stop"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Current backend process stopped"));

    let launched = wait_for_file(root.join("started_gateway.args"));
    assert!(launched.contains("omniinfer.py"));
    assert!(launched.contains("serve --host 127.0.0.1"));
    assert!(launched.contains(&format!("--port {}", gateway.port)));
    assert!(launched.contains("--startup-timeout 10"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/backend/stop HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_select_persists_state_and_profile() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"object":"list","data":[{"id":"llama.cpp-linux-cuda","family":"llama.cpp","models_dir":"/tmp/models"}]}"#,
        ),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"ok":true,"selected_backend":"llama.cpp-linux-cuda","models_dir":"/tmp/models"}"#,
        ),
    ]);
    let root = temp_repo_root("backend-select");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");
    fs::create_dir_all(root.join(".local").join("config")).expect("create local config dir");
    fs::write(
        root.join(".local").join("config").join("state.json"),
        r#"{"future":{"keep":true}}"#,
    )
    .expect("write state");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "select", "llama.cpp-linux-cuda"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Selected backend: llama.cpp-linux-cuda",
        ))
        .stdout(predicate::str::contains("Models directory: /tmp/models"))
        .stdout(predicate::str::contains("Backend config:"))
        .stdout(predicate::str::contains("(created)"));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /omni/backends?scope=all HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/backend/select HTTP/1.1"));
    assert!(request.contains(r#"{"backend":"llama.cpp-linux-cuda"}"#));
    gateway.join();

    let state_raw = fs::read_to_string(root.join(".local").join("config").join("state.json"))
        .expect("state file");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("state json");
    assert_eq!(state["selected_backend"], "llama.cpp-linux-cuda");
    assert_eq!(state["future"]["keep"], true);

    let profile_raw = fs::read_to_string(
        root.join(".local")
            .join("config")
            .join("backend_profiles")
            .join("llama.cpp-linux-cuda.json"),
    )
    .expect("profile file");
    let profile: serde_json::Value = serde_json::from_str(&profile_raw).expect("profile json");
    assert_eq!(profile["schema_version"], 2);
    assert_eq!(profile["backend"], "llama.cpp-linux-cuda");
    assert_eq!(profile["family"], "llama.cpp");
    assert_eq!(profile["load"]["extra_args"], serde_json::json!([]));
    assert_eq!(profile["infer"]["extra_args"], serde_json::json!([]));
    fs::remove_dir_all(root).ok();
}

struct TestGateway {
    port: u16,
    request_rx: mpsc::Receiver<String>,
    handle: thread::JoinHandle<()>,
}

impl TestGateway {
    fn start(responses: Vec<Response>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test gateway");
        let port = listener.local_addr().expect("local addr").port();
        let (request_tx, request_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            for response_body in responses {
                let (mut stream, _) = listener.accept().expect("accept request");
                let request = read_http_request(&mut stream);
                request_tx.send(request).expect("send request");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.body.len()
                );
                stream.write_all(response.as_bytes()).expect("write header");
                stream
                    .write_all(response_body.body.as_bytes())
                    .expect("write body");
                stream.flush().expect("flush response");
            }
        });
        Self {
            port,
            request_rx,
            handle,
        }
    }

    fn request(&self) -> String {
        self.request_rx.recv().expect("receive request")
    }

    fn join(self) {
        self.handle.join().expect("server thread");
    }
}

struct Response {
    body: String,
}

impl Response {
    fn new(body: &str) -> Self {
        Self {
            body: body.to_string(),
        }
    }
}

fn temp_repo_root(test_name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-rs-{test_name}-{nanos}"))
}

fn fake_python_launcher(root: &std::path::Path) -> std::path::PathBuf {
    #[cfg(unix)]
    {
        fake_python_launcher_unix(root)
    }
    #[cfg(windows)]
    {
        fake_python_launcher_windows(root)
    }
}

#[cfg(unix)]
fn fake_python_launcher_unix(root: &std::path::Path) -> std::path::PathBuf {
    let launcher = root.join("fake-python.sh");
    let output = root.join("started_gateway.args");
    fs::write(
        &launcher,
        format!(
            "#!/usr/bin/env bash\nprintf '%s ' \"$@\" > '{}'\n",
            output.display()
        ),
    )
    .expect("write fake launcher");
    let mut permissions = fs::metadata(&launcher)
        .expect("launcher metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod launcher");
    launcher
}

#[cfg(windows)]
fn fake_python_launcher_windows(root: &std::path::Path) -> std::path::PathBuf {
    let launcher = root.join("fake-python.cmd");
    let output = root.join("started_gateway.args");
    fs::write(
        &launcher,
        format!("@echo off\r\necho %* > \"{}\"\r\n", output.display()),
    )
    .expect("write fake launcher");
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

fn wait_for_file(path: std::path::PathBuf) -> String {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if let Ok(text) = fs::read_to_string(&path) {
            return text;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {}", path.display());
}
