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
fn serve_launch_options_parse_before_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .args([
            "serve",
            "--cloudflare",
            "--cloudflared-path",
            "/opt/bin/cloudflared",
            "--cloudflare-no-print-key",
            "--backend",
            "llama.cpp-linux-cuda",
            "--model",
            "/tmp/model.gguf",
            "--ctx-size",
            "8192",
            "--api-key",
            "auto",
            "--detach",
            "--smoke-test",
            "--port",
            "19000",
            "--startup-timeout",
            "20",
            "--default-backend",
            "llama.cpp-linux-cuda",
            "--window-mode",
            "hidden",
            "--log-level",
            "warning",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("implementation pending"));
}

#[test]
fn serve_detach_starts_gateway_and_writes_state() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":false,"model":null,"ctx_size":null}}"#,
        ),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-detach-source");
    let state_root = temp_repo_root("serve-detach-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(source_root.join("omniinfer.py"), "").expect("write source script");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10,"default_backend":"llama.cpp-linux-cuda"}}"#,
            port
        ),
    )
    .expect("write config");
    let launcher = fake_python_launcher(&state_root);

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .env("OMNIINFER_PYTHON", &launcher)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("OmniInfer service is ready"))
        .stdout(predicate::str::contains(format!(
            "Local Base URL: http://127.0.0.1:{}/v1",
            port
        )));

    let launched = wait_for_file(state_root.join("started_gateway.args"));
    assert!(launched.contains("serve --host 127.0.0.1"));
    assert!(launched.contains("--default-backend llama.cpp-linux-cuda"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health?deep=true HTTP/1.1"));
    gateway.join();

    let state_raw = fs::read_to_string(
        state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json")),
    )
    .expect("serve state");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("serve state json");
    assert_eq!(state["port"], port);
    assert_eq!(state["backend"], "llama.cpp-linux-cuda");
    assert_eq!(state["backend_ready"], false);
    assert!(state["log"].as_str().unwrap().contains("serve-"));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_loads_model_before_ready() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"object":"list","recommended":"llama.cpp-linux-cuda","data":[{"id":"llama.cpp-linux-cuda","family":"llama.cpp","binary_exists":true}]}"#,
        ),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf","selected_ctx_size":1024}"#,
        ),
        Response::new(
            r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":true,"model":"/tmp/model.gguf","ctx_size":1024}}"#,
        ),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-detach-load-source");
    let state_root = temp_repo_root("serve-detach-load-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(source_root.join("omniinfer.py"), "").expect("write source script");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    let model = state_root.join("model.gguf");
    fs::write(&model, "").expect("write model");
    let launcher = fake_python_launcher(&state_root);

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .env("OMNIINFER_PYTHON", &launcher)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .arg("--model")
        .arg(&model)
        .args(["--ctx-size", "1024"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Model loaded"))
        .stdout(predicate::str::contains("Backend ready: yes"))
        .stdout(predicate::str::contains("ctx-size: 1024"));

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("GET /omni/backends?scope=all HTTP/1.1"));
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    assert!(request.contains(r#""ctx_size":1024"#));
    let request = gateway.request();
    assert!(request.starts_with("GET /health?deep=true HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_runs_smoke_test() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"object":"list","recommended":"llama.cpp-linux-cuda","data":[{"id":"llama.cpp-linux-cuda","family":"llama.cpp","binary_exists":true}]}"#,
        ),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf","selected_ctx_size":1024}"#,
        ),
        Response::new(
            r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":true,"model":"/tmp/model.gguf","ctx_size":1024}}"#,
        ),
        Response::new(r#"{"choices":[{"message":{"content":"hello smoke"}}]}"#),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-detach-smoke-source");
    let state_root = temp_repo_root("serve-detach-smoke-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(source_root.join("omniinfer.py"), "").expect("write source script");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    let model = state_root.join("model.gguf");
    fs::write(&model, "").expect("write model");
    let launcher = fake_python_launcher(&state_root);

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .env("OMNIINFER_PYTHON", &launcher)
        .args(["serve", "--detach", "--smoke-test", "--port"])
        .arg(port.to_string())
        .arg("--model")
        .arg(&model)
        .assert()
        .success()
        .stdout(predicate::str::contains("Smoke: hello smoke"));

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"][0]["content"], "Hello");
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn model_load_posts_payload_and_persists_state() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"object":"list","recommended":"llama.cpp-linux-cuda","data":[{"id":"llama.cpp-linux","family":"llama.cpp","binary_exists":true},{"id":"llama.cpp-linux-cuda","family":"llama.cpp","binary_exists":true}]}"#,
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
fn model_load_handles_sse_progress() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"object":"list","recommended":"llama.cpp-linux-cuda","data":[{"id":"llama.cpp-linux","family":"llama.cpp","binary_exists":true},{"id":"llama.cpp-linux-cuda","family":"llama.cpp","binary_exists":true}]}"#,
        ),
        Response::new(r#"{"status":"ok"}"#),
        Response::chunks(
            &[
                r#"data: {"type":"status","message":"Resolving model files..."}"#,
                "\n\n",
                r#"data: {"type":"log","message":"backend detail"}"#,
                "\n\n",
                r#"data: {"type":"done","selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf","selected_ctx_size":4096}"#,
                "\n\n",
                "data: [DONE]\n\n",
            ],
            "text/event-stream; charset=utf-8",
        ),
    ]);
    let root = temp_repo_root("model-load-sse");
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
        .assert()
        .success()
        .stdout(predicate::str::contains("Resolving model files..."))
        .stdout(predicate::str::contains("Model loaded"))
        .stdout(predicate::str::contains("ctx-size: 4096"))
        .stdout(predicate::str::contains("backend detail").not());

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    assert!(request.contains("Accept: text/event-stream, application/json"));
    gateway.join();
    fs::remove_dir_all(root).ok();
}

#[test]
fn chat_streams_text_and_usage() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"model":"/tmp/model.gguf","request_defaults":{"temperature":0.1}}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::chunks(
            &[
                r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#,
                "\n\n",
                r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#,
                "\n\n",
                r#"data: {"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"model":"omniinfer"}"#,
                "\n\n",
                "data: [DONE]\n\n",
            ],
            "text/event-stream; charset=utf-8",
        ),
    ]);
    let root = temp_repo_root("chat-stream");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["chat", "Hello"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Response\nHello"))
        .stdout(predicate::str::contains("Performance"))
        .stdout(predicate::str::contains(
            "Tokens: prompt=1, completion=2, total=3",
        ));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /omni/state HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(request.contains(r#""stream":true"#));
    assert!(request.contains(r#""stream_options":{"include_usage":true}"#));
    assert!(request.contains(r#""temperature":0.1"#));
    gateway.join();
    fs::remove_dir_all(root).ok();
}

#[test]
fn chat_includes_image_data_url() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"model":"/tmp/model.gguf"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"choices":[{"message":{"content":"looks good"}}],"usage":{"total_tokens":4},"model":"omniinfer"}"#,
        ),
    ]);
    let root = temp_repo_root("chat-image");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");
    let image = root.join("image.png");
    fs::write(&image, b"fake-png-bytes").expect("write image");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["chat", "Describe it", "--image"])
        .arg(&image)
        .arg("--no-stream")
        .assert()
        .success()
        .stdout(predicate::str::contains("Response"))
        .stdout(predicate::str::contains("looks good"));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /omni/state HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["messages"][0]["content"][0]["type"], "text");
    assert_eq!(body["messages"][0]["content"][0]["text"], "Describe it");
    assert_eq!(body["messages"][0]["content"][1]["type"], "image_url");
    assert_eq!(
        body["messages"][0]["content"][1]["image_url"]["url"],
        "data:image/png;base64,ZmFrZS1wbmctYnl0ZXM="
    );
    assert_eq!(body["stream"], false);
    gateway.join();
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
fn gateway_autostart_can_use_separate_state_root() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"stopped":true}"#),
    ]);
    let source_root = temp_repo_root("source-root");
    let state_root = temp_repo_root("state-root");
    fs::create_dir_all(source_root.join(".local").join("logs")).expect("create source logs");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(source_root.join("omniinfer.py"), "").expect("write source script");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");
    let launcher = fake_python_launcher(&state_root);

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .env("OMNIINFER_PYTHON", &launcher)
        .args(["backend", "stop"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Current backend process stopped"));

    let launched = wait_for_file(state_root.join("started_gateway.args"));
    assert!(launched.contains(&source_root.join("omniinfer.py").display().to_string()));
    assert!(
        state_root
            .join(".local")
            .join("logs")
            .join("gateway.log")
            .is_file()
    );
    assert!(
        !source_root
            .join(".local")
            .join("logs")
            .join("gateway.log")
            .exists()
    );
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/backend/stop HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
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

    fn request(&self) -> String {
        self.request_rx.recv().expect("receive request")
    }

    fn join(self) {
        self.handle.join().expect("server thread");
    }
}

struct Response {
    body: ResponseBody,
    content_type: String,
}

enum ResponseBody {
    Text(String),
    Chunks(Vec<String>),
}

impl Response {
    fn new(body: &str) -> Self {
        Self::with_content_type(body, "application/json")
    }

    fn with_content_type(body: &str, content_type: &str) -> Self {
        Self {
            body: ResponseBody::Text(body.to_string()),
            content_type: content_type.to_string(),
        }
    }

    fn chunks(chunks: &[&str], content_type: &str) -> Self {
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

fn request_body_json(request: &str) -> serde_json::Value {
    let body = request
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("request body separator");
    serde_json::from_str(body).expect("request body json")
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
