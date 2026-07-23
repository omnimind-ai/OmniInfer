use super::support::*;

#[cfg(unix)]
#[test]
fn serve_detach_starts_lan_gateway_with_api_key() {
    let port = free_port();
    let source_root = temp_repo_root("serve-lan-source");
    let state_root = temp_repo_root("serve-lan-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args([
            "serve",
            "--detach",
            "--lan",
            "--api-key",
            "lan-key",
            "--port",
        ])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("Local Base URL:"))
        .stdout(predicate::str::contains("API Key: lan-key"))
        .stdout(predicate::str::contains("Curl:"));

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    stop_rust_serve(&source_root, &state_root, port);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_rejects_remote_management_without_key() {
    let source_root = temp_repo_root("serve-reject-management-source");
    let state_root = temp_repo_root("serve-reject-management-state");
    let public_root = state_root.join("public-models");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(&public_root).expect("create public root");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args([
            "serve",
            "--detach",
            "--lan",
            "--allow-insecure-lan",
            "--allow-remote-management",
            "--public-model-root",
            public_root.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--allow-remote-management requires --admin-api-key, --admin-api-keys, OMNIINFER_ADMIN_API_KEY, OMNIINFER_ADMIN_API_KEYS, or .local/config/admin_keys.json",
        ));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_external_backend_runs_without_python_upstream() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("serve-rust-external-source");
    let state_root = temp_repo_root("serve-rust-external-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    let port = free_port();
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10,"default_backend":"{backend_id}"}}"#,
            port,
        ),
    )
    .expect("write config");
    install_fake_backend(&state_root, backend_id);

    let stdout_path = state_root.join("serve-detach.stdout.txt");
    let stderr_path = state_root.join("serve-detach.stderr.txt");
    let status = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"))
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--api-key", "test-key", "--port"])
        .arg(port.to_string())
        .stdout(Stdio::from(
            fs::File::create(&stdout_path).expect("create stdout capture"),
        ))
        .stderr(Stdio::from(
            fs::File::create(&stderr_path).expect("create stderr capture"),
        ))
        .status()
        .expect("run omniinfer serve");
    let stdout = fs::read_to_string(&stdout_path).expect("read stdout capture");
    let stderr = fs::read_to_string(&stderr_path).expect("read stderr capture");
    assert!(
        status.success(),
        "serve failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("OmniInfer service is ready"));
    assert!(stdout.contains(&format!("Local Base URL: http://127.0.0.1:{port}/v1")));

    let health = wait_for_http_json(port, "/health");
    assert_eq!(health["status"], "ok");
    let state_raw = fs::read_to_string(
        state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json")),
    )
    .expect("serve state");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("serve state json");
    assert_eq!(state["port"], port);
    assert!(state["pid"].as_u64().unwrap_or(0) > 0);

    let mut stop = Command::cargo_bin("omniinfer").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "stop", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "OmniInfer service stopped on port {port}"
        )));
    assert!(wait_for_port_closed(port));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_explicit_roots_reach_gateway_model_load_lifecycle() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("serve-explicit-roots-source");
    let state_root = temp_repo_root("serve-explicit-roots-state");
    let runtime_root = temp_repo_root("serve-explicit-roots-runtime");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    let port = free_port();
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10,"default_backend":"{backend_id}"}}"#,
            port,
        ),
    )
    .expect("write config");
    install_fake_runtime_server_in_root(&runtime_root, backend_id);
    let model = state_root.join("model.gguf");
    fs::write(&model, "gguf").expect("write model");

    let stdout_path = state_root.join("serve-explicit-roots.stdout.txt");
    let stderr_path = state_root.join("serve-explicit-roots.stderr.txt");
    let status = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"))
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env_remove("OMNIINFER_STATE_ROOT")
        .env_remove("OMNIINFER_RUNTIME_ROOT")
        .env_remove("OMNIINFER_RUST_STATE_ROOT")
        .args([
            "serve",
            "--detach",
            "--api-key",
            "test-key",
            "--backend",
            backend_id,
            "--model",
        ])
        .arg(&model)
        .arg("--port")
        .arg(port.to_string())
        .arg("--state-root")
        .arg(&state_root)
        .arg("--runtime-root")
        .arg(&runtime_root)
        .stdout(Stdio::from(
            fs::File::create(&stdout_path).expect("create stdout capture"),
        ))
        .stderr(Stdio::from(
            fs::File::create(&stderr_path).expect("create stderr capture"),
        ))
        .status()
        .expect("run serve with explicit roots");
    let stdout = fs::read_to_string(&stdout_path).expect("read stdout capture");
    let stderr = fs::read_to_string(&stderr_path).expect("read stderr capture");
    assert!(
        status.success(),
        "serve failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("Backend ready: yes"), "stdout:\n{stdout}");

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    assert_eq!(health["omni"]["backend"], backend_id);
    assert_eq!(health["omni"]["backend_ready"], true);
    assert_eq!(
        health["omni"]["model"].as_str().unwrap(),
        model.display().to_string()
    );
    let launch_command = health["omni"]["launch_command"]
        .as_array()
        .expect("runtime launch command");
    let backend_port = health["omni"]["backend_port"]
        .as_u64()
        .and_then(|value| u16::try_from(value).ok())
        .expect("runtime backend port");
    assert_eq!(
        std::path::PathBuf::from(launch_command[0].as_str().unwrap()),
        runtime_root
            .join(backend_id)
            .join("bin")
            .join(if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            })
    );
    assert!(!state_root.join(".local").join("runtime").exists());

    let mut shutdown = Command::cargo_bin("omniinfer").expect("binary exists");
    shutdown
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env_remove("OMNIINFER_STATE_ROOT")
        .env_remove("OMNIINFER_RUNTIME_ROOT")
        .env_remove("OMNIINFER_RUST_STATE_ROOT")
        .arg("shutdown")
        .arg("--state-root")
        .arg(&state_root)
        .arg("--runtime-root")
        .arg(&runtime_root)
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "OmniInfer service stopped on port {port}"
        )));
    assert!(wait_for_port_closed(port));
    assert!(wait_for_port_closed(backend_port));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
    fs::remove_dir_all(runtime_root).ok();
}

#[cfg(unix)]
#[test]
fn serve_detach_starts_gateway_and_writes_state() {
    let backend_id = test_external_backend_id();
    let port = free_port();
    let source_root = temp_repo_root("serve-detach-source");
    let state_root = temp_repo_root("serve-detach-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10,"default_backend":"{backend_id}"}}"#,
            port,
        ),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("OmniInfer service is ready"))
        .stdout(predicate::str::contains(format!(
            "Local Base URL: http://127.0.0.1:{}/v1",
            port
        )));

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");

    let state_raw = fs::read_to_string(
        state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json")),
    )
    .expect("serve state");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("serve state json");
    assert_eq!(state["port"], port);
    assert!(state["log"].as_str().unwrap().contains("serve-"));
    stop_rust_serve(&source_root, &state_root, port);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[cfg(unix)]
#[test]
fn serve_detach_ignores_config_host_by_default() {
    let port = free_port();
    let source_root = temp_repo_root("serve-ignore-config-host-source");
    let state_root = temp_repo_root("serve-ignore-config-host-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"0.0.0.0","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "Local Base URL: http://127.0.0.1:{port}/v1"
        )));

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    stop_rust_serve(&source_root, &state_root, port);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[cfg(unix)]
#[test]
fn serve_detach_respects_explicit_host() {
    let port = free_port();
    let source_root = temp_repo_root("serve-explicit-host-source");
    let state_root = temp_repo_root("serve-explicit-host-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args([
            "serve",
            "--detach",
            "--host",
            "0.0.0.0",
            "--api-key",
            "host-key",
            "--port",
        ])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("API Key: host-key"));

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    stop_rust_serve(&source_root, &state_root, port);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_loads_model_before_ready() {
    let backend_id = test_external_backend_id();
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf","selected_ctx_size":1024}"#,
        ),
        Response::new(
            r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":true,"model":"/tmp/model.gguf","ctx_size":1024}}"#,
        ),
    ]);
    let port = gateway.port;
    let backend_port = 50212;
    let source_root = temp_repo_root("serve-detach-load-source");
    let state_root = temp_repo_root("serve-detach-load-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    install_fake_backend(&state_root, backend_id);
    let model = state_root.join("model.gguf");
    fs::write(&model, "").expect("write model");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .arg("--backend-port")
        .arg(backend_port.to_string())
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
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    assert!(request.contains(r#""ctx_size":1024"#));
    assert!(request.contains(&format!(r#""backend_port":{backend_port}"#)));
    let request = gateway.request();
    assert!(request.starts_with("GET /health?deep=true HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_restores_last_model_when_model_is_omitted() {
    let backend_id = test_external_backend_id();
    let load_response = format!(
        r#"{{"selected_backend":"{backend_id}","selected_model":"/tmp/last-model.gguf","selected_mmproj":"/tmp/mmproj-F16.gguf","selected_ctx_size":4096}}"#
    );
    let state_response = format!(
        r#"{{"omni":{{"backend":"{backend_id}","backend_ready":true,"model":"/tmp/last-model.gguf","mmproj":"/tmp/mmproj-F16.gguf","ctx_size":4096}}}}"#
    );
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(&load_response),
        Response::new(&state_response),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-detach-restore-source");
    let state_root = temp_repo_root("serve-detach-restore-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    let model = state_root.join("last-model.gguf");
    let mmproj = state_root.join("mmproj-F16.gguf");
    fs::write(&model, "gguf").expect("write model");
    fs::write(&mmproj, "gguf").expect("write mmproj");
    fs::create_dir_all(state_root.join(".local").join("config")).expect("create local config");
    let state_payload = serde_json::json!({
        "selected_backend": backend_id,
        "selected_model": model.display().to_string(),
        "selected_mmproj": mmproj.display().to_string(),
        "selected_ctx_size": 4096,
    });
    fs::write(
        state_root.join(".local").join("config").join("state.json"),
        serde_json::to_string_pretty(&state_payload).expect("state json"),
    )
    .expect("write state");
    install_fake_backend(&state_root, backend_id);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "Restoring last model: {}",
            model.display()
        )))
        .stdout(predicate::str::contains("Backend ready: yes"))
        .stdout(predicate::str::contains("ctx-size: 4096"));

    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["model"], model.display().to_string());
    assert_eq!(body["mmproj"], mmproj.display().to_string());
    assert_eq!(body["ctx_size"], 4096);
    let request = gateway.request();
    assert!(request.starts_with("GET /health?deep=true HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_can_skip_restoring_last_model() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":false}}"#),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-detach-no-restore-source");
    let state_root = temp_repo_root("serve-detach-no-restore-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    let model = state_root.join("last-model.gguf");
    fs::write(&model, "gguf").expect("write model");
    fs::create_dir_all(state_root.join(".local").join("config")).expect("create local config");
    fs::write(
        state_root.join(".local").join("config").join("state.json"),
        format!(
            r#"{{
  "selected_backend": "{}",
  "selected_model": "{}",
  "selected_ctx_size": 4096
}}"#,
            test_external_backend_id(),
            model.display()
        ),
    )
    .expect("write state");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--no-restore-model", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("Backend ready: no"))
        .stdout(predicate::str::contains("Restoring last model").not());

    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("GET /health?deep=true HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[cfg(unix)]
#[test]
fn serve_detach_restores_last_model_without_python_upstream() {
    let source_root = temp_repo_root("serve-rust-restore-source");
    let state_root = temp_repo_root("serve-rust-restore-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::create_dir_all(state_root.join(".local").join("config")).expect("create local config");
    let port = free_port();
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10,"default_backend":"{}"}}"#,
            port,
            test_external_backend_id()
        ),
    )
    .expect("write config");
    let model = state_root.join("last-model.gguf");
    fs::write(&model, "gguf").expect("write model");
    fs::write(
        state_root.join(".local").join("config").join("state.json"),
        format!(
            r#"{{
  "selected_backend": "{}",
  "selected_model": "{}",
  "selected_ctx_size": 512
}}"#,
            test_external_backend_id(),
            model.display()
        ),
    )
    .expect("write state");
    install_fake_runtime_server(&state_root, test_external_backend_id());

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--api-key", "test-key", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "Restoring last model: {}",
            model.display()
        )))
        .stdout(predicate::str::contains("Backend ready: yes"))
        .stdout(predicate::str::contains("ctx-size: 512"));

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    assert_eq!(
        health["omni"]["model"].as_str().unwrap(),
        model.display().to_string()
    );
    assert_eq!(health["omni"]["restore_status"], "loaded");
    assert_eq!(health["omni"]["restore_completed"], true);
    let backend_pid = health["omni"]["backend_pid"].as_u64().unwrap();

    let repeated = ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
        .send_json(serde_json::json!({
            "backend": test_external_backend_id(),
            "model": model.display().to_string(),
            "ctx_size": 512,
        }))
        .expect("repeat restored model selection");
    let repeated: serde_json::Value = repeated
        .into_body()
        .read_json()
        .expect("repeat response json");
    assert_eq!(repeated["already_loaded"], true);
    assert_eq!(repeated["requires_reload"], false);
    assert_eq!(repeated["backend_pid"], backend_pid);

    let conflict = ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(serde_json::json!({
            "backend": test_external_backend_id(),
            "model": model.display().to_string(),
            "ctx_size": 1024,
        }))
        .expect("select restored model with different settings");
    assert_eq!(conflict.status().as_u16(), 409);
    let conflict: serde_json::Value = conflict
        .into_body()
        .read_json()
        .expect("conflict response json");
    assert_eq!(conflict["requires_reload"], true);
    assert_eq!(conflict["error"]["code"], "model_reload_required");

    let mut stop = Command::cargo_bin("omniinfer").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "stop", "--port"])
        .arg(port.to_string())
        .assert()
        .success();
    assert!(wait_for_port_closed(port));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_runs_smoke_test() {
    let backend_id = test_external_backend_id();
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"selected_backend":"llama.cpp-linux-cuda","selected_model":"/tmp/model.gguf","selected_ctx_size":1024}"#,
        ),
        Response::new(
            r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":true,"model":"/tmp/model.gguf","ctx_size":1024}}"#,
        ),
        Response::new(r#"{"choices":[{"message":{"content":"hello smoke"}}]}"#),
        Response::new(r#"{"ok":true}"#),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-detach-smoke-source");
    let state_root = temp_repo_root("serve-detach-smoke-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    install_fake_backend(&state_root, backend_id);
    let model = state_root.join("model.gguf");
    fs::write(&model, "").expect("write model");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--smoke-test", "--port"])
        .arg(port.to_string())
        .arg("--model")
        .arg(&model)
        .assert()
        .success()
        .stdout(predicate::str::contains("Smoke: hello smoke"))
        .stdout(predicate::str::contains("Smoke test cleanup complete"));

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"][0]["content"], "Hello");
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/shutdown HTTP/1.1"));
    gateway.join();
    assert!(wait_for_port_closed(port));
    assert!(
        !state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json"))
            .exists()
    );
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn successful_smoke_test_stops_gateway_backend_and_releases_ports() {
    let backend_id = test_external_backend_id();
    let runtime_root = temp_repo_root("serve-success-smoke-runtime");
    install_fake_runtime_server_in_root(&runtime_root, backend_id);

    for detach in [false, true] {
        let suffix = if detach { "detached" } else { "foreground" };
        let source_root = temp_repo_root(&format!("serve-success-smoke-{suffix}-source"));
        let state_root = temp_repo_root(&format!("serve-success-smoke-{suffix}-state"));
        fs::create_dir_all(&source_root).expect("create source root");
        fs::create_dir_all(state_root.join("config")).expect("create state config");
        fs::write(
            state_root.join("config").join("omniinfer.json"),
            r#"{"host":"127.0.0.1","startup_timeout":10}"#,
        )
        .expect("write config");
        let model = state_root.join("model.gguf");
        fs::write(&model, "gguf").expect("write model");
        let gateway_port = free_port();
        let mut backend_port = free_port();
        while backend_port == gateway_port {
            backend_port = free_port();
        }
        let stdout_path = state_root.join("smoke.stdout.txt");
        let stderr_path = state_root.join("smoke.stderr.txt");
        let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
        command
            .env("OMNIINFER_RUST_STRICT", "1")
            .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
            .env_remove("OMNIINFER_STATE_ROOT")
            .env_remove("OMNIINFER_RUNTIME_ROOT")
            .env_remove("OMNIINFER_RUST_STATE_ROOT")
            .args(["serve", "--smoke-test", "--backend", backend_id, "--model"])
            .arg(&model)
            .arg("--backend-port")
            .arg(backend_port.to_string())
            .arg("--port")
            .arg(gateway_port.to_string())
            .arg("--state-root")
            .arg(&state_root)
            .arg("--runtime-root")
            .arg(&runtime_root)
            .stdout(Stdio::from(
                fs::File::create(&stdout_path).expect("create stdout capture"),
            ))
            .stderr(Stdio::from(
                fs::File::create(&stderr_path).expect("create stderr capture"),
            ));
        if detach {
            command.arg("--detach");
        }

        let mut child = command.spawn().expect("spawn successful smoke test");
        let Some(status) = wait_for_process_exit(&mut child, Duration::from_secs(30)) else {
            let mut stop = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
            let _ = stop
                .env("OMNIINFER_RUST_STRICT", "1")
                .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
                .args(["serve", "stop", "--port"])
                .arg(gateway_port.to_string())
                .arg("--state-root")
                .arg(&state_root)
                .arg("--runtime-root")
                .arg(&runtime_root)
                .status();
            let _ = child.kill();
            let _ = child.wait();
            panic!("successful smoke test did not exit within 30 seconds (detach={detach})");
        };
        let stdout = fs::read_to_string(&stdout_path).expect("read stdout capture");
        let stderr = fs::read_to_string(&stderr_path).expect("read stderr capture");
        assert_eq!(
            status.code(),
            Some(0),
            "smoke test failed (detach={detach})\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(stdout.contains("Smoke: fake backend"), "stdout:\n{stdout}");
        assert!(
            stdout.contains("Smoke test cleanup complete"),
            "stdout:\n{stdout}"
        );
        assert!(
            wait_for_port_closed(gateway_port),
            "gateway port {gateway_port} remained open (detach={detach})"
        );
        assert!(
            wait_for_port_closed(backend_port),
            "backend port {backend_port} remained open (detach={detach})"
        );
        assert!(
            !state_root
                .join(".local")
                .join("run")
                .join(format!("serve-{gateway_port}.json"))
                .exists(),
            "smoke test must remove serve metadata"
        );
        fs::remove_dir_all(source_root).ok();
        fs::remove_dir_all(state_root).ok();
    }

    fs::remove_dir_all(runtime_root).ok();
}

#[test]
fn serve_rejects_an_occupied_port_before_spawning_gateway() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind occupied port");
    let port = listener.local_addr().expect("occupied port address").port();
    let source_root = temp_repo_root("serve-occupied-source");
    let state_root = temp_repo_root("serve-occupied-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "127.0.0.1:{port} is already in use"
        )));

    drop(listener);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn failed_smoke_test_releases_gateway_port_for_retry() {
    for detach in [false, true] {
        let port = free_port();
        let source_root = temp_repo_root(if detach {
            "serve-smoke-cleanup-detached-source"
        } else {
            "serve-smoke-cleanup-foreground-source"
        });
        let state_root = temp_repo_root(if detach {
            "serve-smoke-cleanup-detached-state"
        } else {
            "serve-smoke-cleanup-foreground-state"
        });
        fs::create_dir_all(&source_root).expect("create source root");
        fs::create_dir_all(state_root.join("config")).expect("create state config");
        fs::write(
            state_root.join("config").join("omniinfer.json"),
            format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
        )
        .expect("write config");

        for _ in 0..2 {
            let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
            cmd.env("OMNIINFER_RUST_STRICT", "1")
                .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
                .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
                .args(["serve", "--smoke-test", "--no-restore-model", "--port"])
                .arg(port.to_string());
            if detach {
                cmd.arg("--detach");
            }
            cmd.assert()
                .code(1)
                .failure()
                .stderr(predicate::str::contains("smoke test failed"))
                .stderr(predicate::str::contains("10048").not());
            assert!(
                wait_for_port_closed(port),
                "failed smoke test must release port {port} (detach={detach})"
            );
        }

        fs::remove_dir_all(source_root).ok();
        fs::remove_dir_all(state_root).ok();
    }
}

#[cfg(unix)]
#[test]
fn serve_detach_starts_cloudflare_tunnel() {
    let port = free_port();
    let source_root = temp_repo_root("serve-cloudflare-source");
    let state_root = temp_repo_root("serve-cloudflare-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    let cloudflared = fake_cloudflared_launcher(&state_root);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--cloudflare", "--cloudflared-path"])
        .arg(&cloudflared)
        .args(["--api-key", "test-key", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "OpenAI Base URL: https://example-test.trycloudflare.com/v1",
        ))
        .stdout(predicate::str::contains("API Key: test-key"))
        .stdout(predicate::str::contains("Curl:"));

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    let tunnel_args = wait_for_file(state_root.join("cloudflared.args"));
    assert!(tunnel_args.contains(&format!("tunnel --url http://127.0.0.1:{port}")));

    let state_raw = fs::read_to_string(
        state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json")),
    )
    .expect("serve state");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("serve state json");
    assert_eq!(
        state["public_url"],
        "https://example-test.trycloudflare.com"
    );
    assert_eq!(
        state["openai_base_url"],
        "https://example-test.trycloudflare.com/v1"
    );
    assert!(state["cloudflared_pid"].as_u64().unwrap() > 0);
    stop_rust_serve(&source_root, &state_root, port);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn serve_detach_warns_on_transient_public_smoke_failure() {
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(
            r#"{"omni":{"backend":"llama.cpp-linux-cuda","backend_ready":true,"model":"test.gguf","ctx_size":512}}"#,
        ),
        Response::new(
            r#"{"choices":[{"message":{"content":"hello local"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        ),
        Response::new(r#"{"ok":true}"#),
    ]);
    let port = gateway.port;
    let source_root = temp_repo_root("serve-cloudflare-smoke-warning-source");
    let state_root = temp_repo_root("serve-cloudflare-smoke-warning-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");
    let cloudflared = fake_cloudflared_launcher_with_url(
        &state_root,
        "https://definitely-missing.invalid.trycloudflare.com",
    );

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT", "1")
        .env("OMNIINFER_RUST_PUBLIC_SMOKE_RETRY_SECONDS", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args([
            "serve",
            "--detach",
            "--cloudflare",
            "--smoke-test",
            "--cloudflared-path",
        ])
        .arg(&cloudflared)
        .args(["--api-key", "test-key", "--port"])
        .arg(port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("Smoke: local ok: hello local"))
        .stdout(predicate::str::contains("public warning:"))
        .stdout(predicate::str::contains("Smoke test cleanup complete"));

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/shutdown HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn ps_lists_detached_services_from_pid_files() {
    let gateway = TestGateway::start(vec![Response::new(
        r#"{"backend":"llama.cpp-linux-cuda","backend_ready":true,"model":"/tmp/model.gguf","ctx_size":512}"#,
    )]);
    let state_root = temp_repo_root("ps-state");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::create_dir_all(state_root.join(".local").join("run")).expect("create run dir");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");
    fs::write(
        state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{}.json", gateway.port)),
        format!(
            r#"{{
  "pid": 123,
  "cloudflared_pid": 456,
  "port": {},
  "log": "/tmp/serve.log",
  "public_url": "https://example-test.trycloudflare.com",
  "openai_base_url": "https://example-test.trycloudflare.com/v1",
  "backend": "unknown",
  "backend_ready": false
}}"#,
            gateway.port
        ),
    )
    .expect("write serve state");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .arg("ps")
        .assert()
        .success()
        .stdout(predicate::str::contains("Running OmniInfer Services:"))
        .stdout(predicate::str::contains(format!("Port {}:", gateway.port)))
        .stdout(predicate::str::contains(
            "OpenAI Base URL: https://example-test.trycloudflare.com/v1",
        ))
        .stdout(predicate::str::contains("Backend: llama.cpp-linux-cuda"))
        .stdout(predicate::str::contains("Backend Ready: yes"))
        .stdout(predicate::str::contains("Context Size: 512"));

    let request = gateway.request();
    assert!(request.starts_with("GET /omni/state HTTP/1.1"));
    gateway.join();
    fs::remove_dir_all(state_root).ok();
}
