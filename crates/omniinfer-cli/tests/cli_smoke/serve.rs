use super::support::*;

#[cfg(unix)]
fn foreground_serve_command(
    source_root: &std::path::Path,
    state_root: &std::path::Path,
    runtime_root: &std::path::Path,
    port: u16,
    backend_port: u16,
    backend_id: &str,
    model: &std::path::Path,
) -> StdCommand {
    let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
    command
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", source_root)
        .arg("serve")
        .args(["--backend", backend_id, "--model"])
        .arg(model)
        .args(["--port", &port.to_string(), "--backend-port"])
        .arg(backend_port.to_string())
        .args(["--state-root"])
        .arg(state_root)
        .args(["--runtime-root"])
        .arg(runtime_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

#[cfg(unix)]
fn assert_interrupted_serve_cleaned(
    child: &mut std::process::Child,
    state_root: &std::path::Path,
    port: u16,
    backend_port: u16,
) {
    send_sigint(child);
    let status = wait_for_process_exit(child, Duration::from_secs(15))
        .expect("foreground serve exits after SIGINT");
    assert_eq!(status.code(), Some(130));
    assert!(
        wait_for_port_closed(port),
        "gateway port {port} remains open"
    );
    assert!(
        wait_for_port_closed(backend_port),
        "backend port {backend_port} remains open"
    );
    assert!(
        !state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json"))
            .exists(),
        "serve state was not removed"
    );
}

#[cfg(unix)]
#[test]
fn serve_sigint_before_gateway_ready_cleans_up_gateway_and_state() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("serve-sigint-pre-ready-source");
    let state_root = temp_repo_root("serve-sigint-pre-ready-state");
    let runtime_root = temp_repo_root("serve-sigint-pre-ready-runtime");
    let port = free_port();
    let backend_port = free_port();
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create config root");
    install_fake_runtime_server_in_root(&runtime_root, backend_id);
    let model = state_root.join("model.gguf");
    fs::write(&model, "gguf").expect("write model");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
    )
    .expect("write config");

    let mut child = foreground_serve_command(
        &source_root,
        &state_root,
        &runtime_root,
        port,
        backend_port,
        backend_id,
        &model,
    )
    .spawn()
    .expect("start foreground serve");
    let state_path = state_root
        .join(".local")
        .join("run")
        .join(format!("serve-{port}.json"));
    wait_for_file(state_path);
    assert_interrupted_serve_cleaned(&mut child, &state_root, port, backend_port);
}

#[cfg(unix)]
#[test]
fn serve_sigint_during_model_load_cleans_gateway_backend_and_state() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("serve-sigint-load-source");
    let state_root = temp_repo_root("serve-sigint-load-state");
    let runtime_root = temp_repo_root("serve-sigint-load-runtime");
    let port = free_port();
    let backend_port = free_port();
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create config root");
    install_fake_runtime_server_in_root(&runtime_root, backend_id);
    let model = state_root.join("model.gguf");
    let ready_file = state_root.join("backend-ready");
    let release_file = state_root.join("backend-release");
    fs::write(&model, "gguf").expect("write model");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
    )
    .expect("write config");

    let mut command = foreground_serve_command(
        &source_root,
        &state_root,
        &runtime_root,
        port,
        backend_port,
        backend_id,
        &model,
    );
    command
        .env("OMNIINFER_TEST_RUNTIME_READY_FILE", &ready_file)
        .env("OMNIINFER_TEST_RUNTIME_DELAY_FILE", &release_file);
    let mut child = command.spawn().expect("start foreground serve");
    wait_for_file(ready_file);
    assert_interrupted_serve_cleaned(&mut child, &state_root, port, backend_port);
}

#[cfg(unix)]
#[test]
fn serve_sigint_after_ready_cleans_gateway_backend_and_state() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("serve-sigint-ready-source");
    let state_root = temp_repo_root("serve-sigint-ready-state");
    let runtime_root = temp_repo_root("serve-sigint-ready-runtime");
    let port = free_port();
    let backend_port = free_port();
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create config root");
    install_fake_runtime_server_in_root(&runtime_root, backend_id);
    let model = state_root.join("model.gguf");
    fs::write(&model, "gguf").expect("write model");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
    )
    .expect("write config");

    let mut child = foreground_serve_command(
        &source_root,
        &state_root,
        &runtime_root,
        port,
        backend_port,
        backend_id,
        &model,
    )
    .spawn()
    .expect("start foreground serve");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let health = wait_for_http_json(port, "/health?deep=true");
        if health["omni"]["backend_port"] == backend_port {
            break;
        }
        assert!(Instant::now() < deadline, "backend did not become ready");
        thread::sleep(Duration::from_millis(50));
    }
    assert_interrupted_serve_cleaned(&mut child, &state_root, port, backend_port);
}

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
        .stdout(predicate::str::contains("Local Gateway URL:"))
        .stdout(predicate::str::contains("API Key: lan-key"));

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
    assert!(stdout.contains(&format!("Local Gateway URL: http://127.0.0.1:{port}")));

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
fn serve_stop_detaches_external_runtime_and_restarts_immediately() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("serve-stop-external-source");
    let state_root = temp_repo_root("serve-stop-external-state");
    let runtime_root = temp_repo_root("serve-stop-external-runtime");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    let gateway_port = free_port();
    let upstream_port = free_port();
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{gateway_port},"startup_timeout":10,"default_backend":"{backend_id}"}}"#,
        ),
    )
    .expect("write config");
    install_fake_backend(&state_root, backend_id);
    install_fake_runtime_server_in_root(&runtime_root, backend_id);
    let runtime_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let runtime = runtime_root.join(backend_id).join("bin").join(runtime_name);
    let mut upstream = StdCommand::new(runtime)
        .args(["--host", "127.0.0.1", "--port"])
        .arg(upstream_port.to_string())
        .args(["--model", "external-test-model"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start external runtime");
    let upstream_health = wait_for_http_json(upstream_port, "/health");
    assert_eq!(upstream_health["status"], "ok");

    let start_serve = || {
        let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
        command
            .env("OMNIINFER_RUST_STRICT", "1")
            .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
            .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
            .args(["serve", "--detach", "--port"])
            .arg(gateway_port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("start detached serve")
    };
    assert!(start_serve().success());
    let health = wait_for_http_json(gateway_port, "/health");
    assert_eq!(health["status"], "ok");

    let attach = http_client::post_json(
        &format!("http://127.0.0.1:{gateway_port}/omni/runtime/attach"),
        &serde_json::json!({
            "client_endpoint": format!("http://127.0.0.1:{upstream_port}"),
            "external_server_protocol": "llama.cpp-server",
            "model": "external-test-model",
            "backend": backend_id,
        }),
        Duration::from_secs(5),
    )
    .expect("attach external runtime");
    assert_eq!(attach.status, 200, "attach: {:?}", attach.body);
    assert_eq!(attach.body["process_owned"], false);

    let state_path = state_root
        .join(".local")
        .join("run")
        .join(format!("serve-{gateway_port}.json"));
    let mut recorded: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&state_path).expect("read serve state"))
            .expect("parse serve state");
    recorded["backend_ready"] = serde_json::json!(true);
    recorded["backend"] = serde_json::json!(backend_id);
    recorded["model"] = serde_json::json!("external-test-model");
    recorded["backend_pid"] = serde_json::Value::Null;
    recorded["backend_port"] = serde_json::json!(upstream_port);
    recorded
        .as_object_mut()
        .expect("serve state object")
        .remove("backend_process_owned");
    fs::write(
        &state_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&recorded).expect("serialize serve state")
        ),
    )
    .expect("write v0.3.29-compatible serve state");

    let mut stop = Command::cargo_bin("omniinfer").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "stop", "--port"])
        .arg(gateway_port.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "OmniInfer service stopped on port {gateway_port}"
        )));
    assert!(wait_for_port_closed(gateway_port));
    assert!(!state_path.exists());
    assert_eq!(wait_for_http_json(upstream_port, "/health")["status"], "ok");

    assert!(start_serve().success());
    assert_eq!(wait_for_http_json(gateway_port, "/health")["status"], "ok");
    stop_rust_serve(&source_root, &state_root, gateway_port);
    assert_eq!(wait_for_http_json(upstream_port, "/health")["status"], "ok");

    upstream.kill().expect("stop external runtime");
    upstream.wait().expect("wait external runtime");
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
    fs::remove_dir_all(runtime_root).ok();
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
            "Local Gateway URL: http://127.0.0.1:{}",
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
            "Local Gateway URL: http://127.0.0.1:{port}"
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
        "selected_request_defaults": {"max_tokens": 64, "temperature": 0.2},
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
    assert_eq!(body["request_defaults"]["max_tokens"], 64);
    assert_eq!(body["request_defaults"]["temperature"], 0.2);
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
fn serve_restore_preserves_persisted_no_mmproj_over_discoverable_sibling() {
    let backend_id = test_external_backend_id();
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"starting"}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(&format!(
            r#"{{"selected_backend":"{backend_id}","selected_model":"/tmp/model.gguf","selected_mmproj":null,"selected_ctx_size":512}}"#
        )),
        Response::new(
            r#"{"omni":{"backend":"test","backend_ready":true,"model":"/tmp/model.gguf","ctx_size":512}}"#,
        ),
    ]);
    let source_root = temp_repo_root("serve-no-mmproj-source");
    let state_root = temp_repo_root("serve-no-mmproj-state");
    fs::create_dir_all(&source_root).expect("source root");
    fs::create_dir_all(state_root.join("config")).expect("config root");
    fs::create_dir_all(state_root.join(".local").join("config")).expect("state config");
    let port = gateway.port;
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
    )
    .expect("config");
    let model = state_root.join("model.gguf");
    fs::write(&model, "gguf").expect("model");
    fs::write(state_root.join("mmproj-F16.gguf"), "mmproj").expect("sibling mmproj");
    fs::write(
        state_root.join(".local").join("config").join("state.json"),
        serde_json::to_string(&serde_json::json!({
            "selected_backend": backend_id,
            "selected_model": model.display().to_string(),
            "selected_no_mmproj": true,
            "selected_ctx_size": 512,
        }))
        .unwrap(),
    )
    .expect("state");
    install_fake_backend(&state_root, backend_id);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_TEST_ALLOW_OCCUPIED_SERVE_PORT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .success();

    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["no_mmproj"], true);
    assert!(body.get("mmproj").is_none());
    gateway.join();
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
fn failed_model_load_stops_gateway_and_releases_port_for_retry() {
    let backend_id = test_external_backend_id();
    let runtime_root = temp_repo_root("serve-failed-load-runtime");
    install_failing_runtime_in_root(&runtime_root, backend_id);

    for detach in [false, true] {
        let suffix = if detach { "detached" } else { "foreground" };
        let source_root = temp_repo_root(&format!("serve-failed-load-{suffix}-source"));
        let state_root = temp_repo_root(&format!("serve-failed-load-{suffix}-state"));
        fs::create_dir_all(&source_root).expect("create source root");
        fs::create_dir_all(state_root.join("config")).expect("create state config");
        fs::write(
            state_root.join("config").join("omniinfer.json"),
            r#"{"host":"127.0.0.1","startup_timeout":2}"#,
        )
        .expect("write config");
        let model = state_root.join("model.gguf");
        fs::write(&model, "gguf").expect("write model");
        let gateway_port = free_port();

        for attempt in 1..=2 {
            let stdout_path = state_root.join(format!("attempt-{attempt}.stdout.txt"));
            let stderr_path = state_root.join(format!("attempt-{attempt}.stderr.txt"));
            let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
            command
                .env("OMNIINFER_RUST_STRICT", "1")
                .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
                .env_remove("OMNIINFER_STATE_ROOT")
                .env_remove("OMNIINFER_RUNTIME_ROOT")
                .env_remove("OMNIINFER_RUST_STATE_ROOT")
                .args(["serve", "--smoke-test", "--backend", backend_id, "--model"])
                .arg(&model)
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

            let mut child = command.spawn().expect("spawn failed model load");
            let Some(status) = wait_for_process_exit(&mut child, Duration::from_secs(20)) else {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "failed model load did not exit within 20 seconds (attempt={attempt}, detach={detach})"
                );
            };
            let stdout = fs::read_to_string(&stdout_path).expect("read stdout capture");
            let stderr = fs::read_to_string(&stderr_path).expect("read stderr capture");
            assert_eq!(
                status.code(),
                Some(1),
                "failed model load returned the wrong exit code (attempt={attempt}, detach={detach})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            assert!(stdout.contains("Loading model..."), "stdout:\n{stdout}");
            assert!(
                stderr.contains("POST /omni/model/select failed with status 502"),
                "stderr:\n{stderr}"
            );
            assert!(
                wait_for_port_closed(gateway_port),
                "failed model load must release gateway port {gateway_port} (attempt={attempt}, detach={detach})"
            );
            assert!(
                !state_root
                    .join(".local")
                    .join("run")
                    .join(format!("serve-{gateway_port}.json"))
                    .exists(),
                "failed model load must not retain stale serve metadata"
            );
        }

        fs::remove_dir_all(source_root).ok();
        fs::remove_dir_all(state_root).ok();
    }

    fs::remove_dir_all(runtime_root).ok();
}

#[cfg(target_os = "linux")]
#[test]
fn speculative_load_failure_rolls_back_real_backend_and_cuda_reservation() {
    let backend_id = "llama.cpp-linux-cuda";
    let source_root = temp_repo_root("serve-speculative-failure-source");
    let state_root = temp_repo_root("serve-speculative-failure-state");
    let runtime_root = temp_repo_root("serve-speculative-failure-runtime");
    let fake_bin_root = temp_repo_root("serve-speculative-failure-tools");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        r#"{"host":"127.0.0.1","startup_timeout":3}"#,
    )
    .expect("write config");
    install_fake_runtime_server_in_root(&runtime_root, backend_id);
    let fake_nvidia_smi = install_fake_nvidia_smi(&fake_bin_root, 2900);
    let model = state_root.join("model.gguf");
    fs::File::create(&model)
        .expect("create model")
        .set_len(2 * 1024 * 1024 * 1024)
        .expect("size model");
    let gateway_port = free_port();
    let backend_port = free_port();
    let started_marker = state_root.join("backend-started");
    let exited_marker = state_root.join("backend-exited");
    let gateway_path = assert_cmd::cargo::cargo_bin("omniinfer");
    let existing_path = std::env::var_os("PATH").unwrap_or_default();
    let path = format!(
        "{}:{}",
        fake_nvidia_smi.parent().unwrap().display(),
        existing_path.to_string_lossy()
    );
    let mut gateway = StdCommand::new(gateway_path)
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .env("OMNIINFER_RUNTIME_ROOT", &runtime_root)
        .env("OMNIINFER_CUDA_VISIBLE_DEVICES", "0")
        .env("OMNIINFER_TEST_RUNTIME_STARTED_FILE", &started_marker)
        .env("OMNIINFER_TEST_RUNTIME_EXITED_FILE", &exited_marker)
        .env("OMNIINFER_TEST_RUNTIME_EXIT_AFTER_BIND", "1")
        .env("PATH", path)
        .args(["gateway", "--host", "127.0.0.1", "--port"])
        .arg(gateway_port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start gateway");

    let health = wait_for_http_json(gateway_port, "/health");
    assert_eq!(health["status"], "ok");
    let first = http_client::post_json(
        &format!("http://127.0.0.1:{gateway_port}/omni/model/select"),
        &serde_json::json!({
            "backend": backend_id,
            "model": model.display().to_string(),
            "backend_port": backend_port,
        }),
        Duration::from_secs(5),
    )
    .expect("speculative model-select response");
    assert_eq!(first.status, 502, "first response: {:?}", first.body);
    assert!(
        first.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("runtime exited before becoming ready")
    );
    let started = wait_for_file(started_marker);
    let exited = wait_for_file(exited_marker);
    assert!(!started.trim().is_empty());
    assert!(!exited.trim().is_empty());
    assert!(wait_for_port_closed(backend_port));

    let state = http_client::get_json(
        &format!("http://127.0.0.1:{gateway_port}/omni/state"),
        Duration::from_secs(5),
    )
    .expect("state after failed speculative load");
    assert_eq!(state.status, 200);
    assert!(!state.body["backend_ready"].as_bool().unwrap_or(false));
    assert!(state.body["model"].is_null());
    assert_eq!(
        state.body["resource_ledger"]["reserved_bytes"],
        serde_json::json!({})
    );
    assert_eq!(
        state.body["resource_ledger"]["committed_bytes"],
        serde_json::json!({})
    );

    let second = http_client::post_json(
        &format!("http://127.0.0.1:{gateway_port}/omni/model/select"),
        &serde_json::json!({
            "backend": backend_id,
            "model": model.display().to_string(),
            "backend_port": backend_port,
        }),
        Duration::from_secs(5),
    )
    .expect("follow-on model-select response");
    assert_eq!(second.status, 502, "second response: {:?}", second.body);
    assert!(
        !second.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("exclusively held by a speculative runtime")
    );
    assert!(wait_for_port_closed(backend_port));

    let shutdown = http_client::post_json(
        &format!("http://127.0.0.1:{gateway_port}/omni/shutdown"),
        &serde_json::json!({}),
        Duration::from_secs(5),
    )
    .expect("gateway shutdown response");
    assert_eq!(shutdown.status, 200);
    let status = gateway.wait().expect("wait gateway");
    assert!(status.success());
    assert!(wait_for_port_closed(gateway_port));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
    fs::remove_dir_all(runtime_root).ok();
    fs::remove_dir_all(fake_bin_root).ok();
}

#[cfg(windows)]
#[test]
fn windows_vllm_wsl2_install_and_smoke_cover_managed_lifecycle() {
    let source_root = temp_repo_root("serve-vllm-wsl2-source");
    let state_root = temp_repo_root("serve-vllm-wsl2-state");
    let runtime_root = temp_repo_root("serve-vllm-wsl2-runtime");
    let fake_root = temp_repo_root("serve-vllm-wsl2-fake");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        r#"{"host":"127.0.0.1","startup_timeout":10}"#,
    )
    .expect("write state config");
    let fake_wsl = compile_fake_wsl(&state_root.join("tools"));
    let catalog = write_wsl_python_runtime_fixture(&state_root);

    let mut install = Command::cargo_bin("omniinfer").expect("binary exists");
    install
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_PREBUILT_CATALOG", &catalog)
        .env("OMNIINFER_WSL_EXE", &fake_wsl)
        .env("OMNIINFER_VLLM_NVIDIA_SMI", &fake_wsl)
        .env("OMNIINFER_FAKE_WSL_ROOT", &fake_root)
        .args([
            "backend",
            "install",
            "vllm-wsl2-cuda",
            "--wsl-distro",
            "Ubuntu-24.04",
            "--state-root",
        ])
        .arg(&state_root)
        .arg("--runtime-root")
        .arg(&runtime_root)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"event\":\"completed\""));

    let launcher_manifest = runtime_root
        .join("vllm-wsl2-cuda")
        .join("bin")
        .join("vllm-wsl2.json");
    assert!(launcher_manifest.is_file());
    let local_model = state_root.join("models").join("Local Model");
    fs::create_dir_all(&local_model).expect("create local model directory");
    fs::write(local_model.join("config.json"), "{}").expect("write local model config");
    let models = [
        "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        local_model.display().to_string(),
    ];

    for (index, model) in models.iter().enumerate() {
        let gateway_port = free_port();
        let mut backend_port = free_port();
        while backend_port == gateway_port {
            backend_port = free_port();
        }
        let stdout_path = state_root.join(format!("vllm-smoke-{index}.stdout.txt"));
        let stderr_path = state_root.join(format!("vllm-smoke-{index}.stderr.txt"));
        let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
        command
            .env("OMNIINFER_RUST_STRICT", "1")
            .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
            .env("OMNIINFER_WSL_EXE", &fake_wsl)
            .env("OMNIINFER_VLLM_NVIDIA_SMI", &fake_wsl)
            .env("OMNIINFER_FAKE_WSL_ROOT", &fake_root)
            .env("OMNIINFER_CUDA_VISIBLE_DEVICES", "0")
            .args([
                "serve",
                "--smoke-test",
                "--backend",
                "vllm-wsl2-cuda",
                "--model",
            ])
            .arg(model)
            .args(["--resource-budget-bytes", "1073741824"])
            .arg("--backend-port")
            .arg(backend_port.to_string())
            .arg("--port")
            .arg(gateway_port.to_string())
            .arg("--state-root")
            .arg(&state_root)
            .arg("--runtime-root")
            .arg(&runtime_root)
            .stdout(Stdio::from(
                fs::File::create(&stdout_path).expect("create smoke stdout"),
            ))
            .stderr(Stdio::from(
                fs::File::create(&stderr_path).expect("create smoke stderr"),
            ));
        let mut child = command.spawn().expect("spawn vLLM WSL2 smoke test");
        let Some(status) = wait_for_process_exit(&mut child, Duration::from_secs(30)) else {
            let _ = child.kill();
            let _ = child.wait();
            panic!("vLLM WSL2 smoke test did not exit");
        };
        let stdout = fs::read_to_string(&stdout_path).expect("read smoke stdout");
        let stderr = fs::read_to_string(&stderr_path).expect("read smoke stderr");
        assert_eq!(
            status.code(),
            Some(0),
            "vLLM WSL2 smoke failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("Smoke: fake vLLM WSL2"),
            "stdout:\n{stdout}"
        );
        assert!(stdout.contains("Smoke test cleanup complete"));
        assert!(wait_for_port_closed(gateway_port));
        assert!(wait_for_port_closed(backend_port));
        assert!(
            !state_root
                .join(".local")
                .join("run")
                .join(format!("serve-{gateway_port}.json"))
                .exists()
        );
    }

    let invocations =
        fs::read_to_string(fake_root.join("invocations.log")).expect("read fake WSL invocations");
    assert!(invocations.contains("Qwen/Qwen2.5-0.5B-Instruct"));
    let local_model_wsl = local_model.display().to_string().replace('\\', "/");
    let drive = local_model_wsl
        .chars()
        .next()
        .expect("Windows model drive")
        .to_ascii_lowercase();
    assert!(
        invocations.contains(&format!(
            "/mnt/{drive}/{}",
            local_model_wsl[3..].trim_start_matches('/')
        )),
        "local Windows model path was not translated:\n{invocations}"
    );
    assert!(
        invocations
            .lines()
            .filter(|line| line.contains("/omniinfer-vllm-stop"))
            .count()
            >= 2
    );
    assert!(!invocations.contains("--terminate"));
    assert!(launcher_manifest.is_file());

    let status = StdCommand::new(&fake_wsl)
        .env("OMNIINFER_FAKE_WSL_ROOT", &fake_root)
        .args(["--list", "--quiet"])
        .status()
        .expect("query fake WSL after managed shutdown");
    assert!(status.success(), "managed shutdown must not terminate WSL");

    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
    fs::remove_dir_all(runtime_root).ok();
    fs::remove_dir_all(fake_root).ok();
}

#[cfg(windows)]
#[test]
fn windows_vllm_wsl2_rocm_smoke_stops_process_tree_and_releases_ports() {
    let source_root = temp_repo_root("serve-vllm-wsl2-rocm-source");
    let state_root = temp_repo_root("serve-vllm-wsl2-rocm-state");
    let runtime_root = temp_repo_root("serve-vllm-wsl2-rocm-runtime");
    let fake_root = temp_repo_root("serve-vllm-wsl2-rocm-fake");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        r#"{"host":"127.0.0.1","startup_timeout":10}"#,
    )
    .expect("write state config");
    let fake_wsl = compile_fake_wsl(&state_root.join("tools"));
    let catalog = write_wsl_rocm_runtime_fixture(&state_root);

    let mut install = Command::cargo_bin("omniinfer").expect("binary exists");
    install
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_PREBUILT_CATALOG", &catalog)
        .env("OMNIINFER_WSL_EXE", &fake_wsl)
        .env("OMNIINFER_FAKE_WSL_ROOT", &fake_root)
        .args([
            "backend",
            "install",
            "vllm-wsl2-rocm",
            "--wsl-distro",
            "Ubuntu-24.04",
            "--state-root",
        ])
        .arg(&state_root)
        .arg("--runtime-root")
        .arg(&runtime_root)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"event\":\"completed\""));

    let gateway_port = free_port();
    let mut backend_port = free_port();
    while backend_port == gateway_port {
        backend_port = free_port();
    }
    let stdout_path = state_root.join("vllm-rocm-smoke.stdout.txt");
    let stderr_path = state_root.join("vllm-rocm-smoke.stderr.txt");
    let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer"));
    command
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_WSL_EXE", &fake_wsl)
        .env("OMNIINFER_FAKE_WSL_ROOT", &fake_root)
        .args([
            "serve",
            "--smoke-test",
            "--backend",
            "vllm-wsl2-rocm",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--resource-budget-bytes",
            "1073741824",
            "--backend-port",
        ])
        .arg(backend_port.to_string())
        .arg("--port")
        .arg(gateway_port.to_string())
        .arg("--state-root")
        .arg(&state_root)
        .arg("--runtime-root")
        .arg(&runtime_root)
        .stdout(Stdio::from(
            fs::File::create(&stdout_path).expect("create smoke stdout"),
        ))
        .stderr(Stdio::from(
            fs::File::create(&stderr_path).expect("create smoke stderr"),
        ));
    let mut child = command.spawn().expect("spawn ROCm WSL2 smoke test");
    let Some(status) = wait_for_process_exit(&mut child, Duration::from_secs(30)) else {
        let _ = child.kill();
        let _ = child.wait();
        panic!("ROCm WSL2 smoke test did not exit");
    };
    let stdout = fs::read_to_string(&stdout_path).expect("read smoke stdout");
    let stderr = fs::read_to_string(&stderr_path).expect("read smoke stderr");
    assert_eq!(
        status.code(),
        Some(0),
        "ROCm WSL2 smoke failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("Smoke: fake vLLM WSL2"));
    assert!(stdout.contains("Smoke test cleanup complete"));
    assert!(wait_for_port_closed(gateway_port));
    assert!(wait_for_port_closed(backend_port));
    assert!(
        !state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{gateway_port}.json"))
            .exists()
    );
    let invocations =
        fs::read_to_string(fake_root.join("invocations.log")).expect("read fake WSL invocations");
    assert!(invocations.contains("HSA_ENABLE_DXG_DETECTION=1"));
    assert!(invocations.contains("/omniinfer-vllm-stop"));
    assert!(!invocations.contains("--terminate"));

    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
    fs::remove_dir_all(runtime_root).ok();
    fs::remove_dir_all(fake_root).ok();
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

#[test]
fn missing_explicit_cloudflared_does_not_start_gateway() {
    let port = free_port();
    let source_root = temp_repo_root("serve-cloudflare-missing-helper-source");
    let state_root = temp_repo_root("serve-cloudflare-missing-helper-state");
    let missing = state_root.join("missing-cloudflared");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(&state_root).expect("create state root");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--cloudflare", "--cloudflared-path"])
        .arg(&missing)
        .args(["--api-key", "test-key", "--port"])
        .arg(port.to_string())
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cloudflared was not found or is not executable at",
        ));

    assert!(
        wait_for_port_closed(port),
        "missing helper must fail before the gateway starts"
    );
    assert!(
        !state_root
            .join(".local")
            .join("run")
            .join(format!("serve-{port}.json"))
            .exists(),
        "failed startup must not publish a serve state file"
    );
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
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
            "Public Gateway URL: https://example-test.trycloudflare.com",
        ))
        .stdout(predicate::str::contains("API Key: test-key"));

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
    let cloudflared_pid = state["cloudflared_pid"].as_u64().unwrap();
    assert!(cloudflared_pid > 0);
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        StdCommand::new("kill")
            .args(["-0", &cloudflared_pid.to_string()])
            .status()
            .expect("check cloudflared process")
            .success(),
        "detached cloudflared must survive continued log writes"
    );
    stop_rust_serve(&source_root, &state_root, port);
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[cfg(unix)]
#[test]
fn serve_replacement_cleans_orphaned_cloudflare_tunnel() {
    let port = free_port();
    let source_root = temp_repo_root("serve-cloudflare-replace-source");
    let state_root = temp_repo_root("serve-cloudflare-replace-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port},"startup_timeout":10}}"#),
    )
    .expect("write config");
    let cloudflared = fake_cloudflared_launcher(&state_root);

    let mut first = Command::cargo_bin("omniinfer").expect("binary exists");
    first
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--cloudflare", "--cloudflared-path"])
        .arg(&cloudflared)
        .args(["--api-key", "test-key", "--port"])
        .arg(port.to_string())
        .assert()
        .success();

    let state_path = state_root
        .join(".local")
        .join("run")
        .join(format!("serve-{port}.json"));
    let old_state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&state_path).expect("old state"))
            .expect("old state JSON");
    let old_run_id = old_state["run_id"]
        .as_str()
        .expect("old run ID")
        .to_string();
    let old_tunnel_pid = old_state["cloudflared_pid"]
        .as_u64()
        .expect("old tunnel PID") as u32;

    let shutdown = omniinfer_core::http_client::post_json(
        &format!("http://127.0.0.1:{port}/omni/shutdown"),
        &serde_json::json!({}),
        Duration::from_secs(3),
    )
    .expect("shutdown old gateway");
    assert!(shutdown.status < 400);
    assert!(wait_for_port_closed(port));
    assert!(
        StdCommand::new("kill")
            .args(["-0", &old_tunnel_pid.to_string()])
            .status()
            .expect("check old tunnel")
            .success(),
        "the reproduction requires an orphaned tunnel before replacement"
    );

    let mut replacement = Command::cargo_bin("omniinfer").expect("binary exists");
    replacement
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args([
            "serve",
            "--detach",
            "--lan",
            "--api-key",
            "test-key",
            "--port",
        ])
        .arg(port.to_string())
        .assert()
        .success();

    let new_state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&state_path).expect("replacement state"))
            .expect("replacement state JSON");
    assert_ne!(new_state["run_id"], old_run_id);
    assert_eq!(new_state["phase"], "ready");
    assert!(new_state["cloudflared_pid"].is_null());
    assert!(
        !StdCommand::new("kill")
            .args(["-0", &old_tunnel_pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .expect("check cleaned tunnel")
            .success(),
        "replacement must stop the old tunnel before publishing new state"
    );

    stop_rust_serve(&source_root, &state_root, port);
    assert!(!state_path.exists());
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn concurrent_serve_operation_fails_while_port_lock_is_held() {
    let port = free_port();
    let source_root = temp_repo_root("serve-port-lock-source");
    let state_root = temp_repo_root("serve-port-lock-state");
    let run_dir = state_root.join(".local").join("run");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(&run_dir).expect("create run directory");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(run_dir.join(format!("serve-{port}.lock")))
        .expect("open port lock");
    lock.try_lock().expect("hold port lock");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--port"])
        .arg(port.to_string())
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "another serve operation already owns port",
        ));
    drop(lock);
    assert!(wait_for_port_closed(port));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[cfg(unix)]
#[test]
fn serve_stop_rejects_mismatched_process_identity() {
    let port = free_port();
    let source_root = temp_repo_root("serve-identity-source");
    let state_root = temp_repo_root("serve-identity-state");
    let run_dir = state_root.join(".local").join("run");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(&run_dir).expect("create run directory");
    let mut unrelated = StdCommand::new("sleep")
        .arg("30")
        .spawn()
        .expect("start unrelated process");
    let mut identity = omniinfer_core::serve_state::capture_process_identity(unrelated.id())
        .expect("capture unrelated identity");
    identity.start_time = identity.start_time.saturating_add(1);
    fs::write(
        run_dir.join(format!("serve-{port}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "run_id": "mismatched-test",
            "phase": "starting",
            "pid": unrelated.id(),
            "gateway_process": identity,
            "port": port
        }))
        .expect("encode mismatched state"),
    )
    .expect("write mismatched state");

    let mut stop = Command::cargo_bin("omniinfer").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "stop", "--port"])
        .arg(port.to_string())
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "process identity does not match serve state",
        ));
    assert!(
        unrelated
            .try_wait()
            .expect("check unrelated process")
            .is_none()
    );
    unrelated.kill().expect("stop unrelated process");
    unrelated.wait().expect("reap unrelated process");
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
