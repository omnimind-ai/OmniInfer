use super::support::*;

#[test]
fn backend_list_installed_empty_succeeds() {
    let root = temp_repo_root("backend-list-installed-empty");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "list", "--scope", "installed"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Installed backends"))
        .stdout(predicate::str::contains("(none)"));
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

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
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
    let root = temp_repo_root("backend-stop-autostart");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            free_port()
        ),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "stop"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Start it with `omniinfer serve`"));
    fs::remove_dir_all(root).ok();
}

#[test]
fn gateway_autostart_can_use_separate_state_root() {
    let source_root = temp_repo_root("source-root");
    let state_root = temp_repo_root("state-root");
    fs::create_dir_all(source_root.join(".local").join("logs")).expect("create source logs");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, free_port()),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["backend", "stop"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Start it with `omniinfer serve`"));

    assert!(
        !source_root
            .join(".local")
            .join("logs")
            .join("gateway.log")
            .exists()
    );
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn backend_select_persists_state_and_profile() {
    let backend_id = test_external_backend_id();
    let models_dir = if cfg!(windows) {
        "C:/tmp/models"
    } else {
        "/tmp/models"
    };
    let select_response =
        format!(r#"{{"ok":true,"selected_backend":"{backend_id}","models_dir":"{models_dir}"}}"#);
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(&select_response),
    ]);
    let root = temp_repo_root("backend-select");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"backends":{{"{backend_id}":{{"models_dir":"{models_dir}"}}}}}}"#,
            gateway.port,
        ),
    )
    .expect("write config");
    install_fake_backend(&root, backend_id);
    fs::create_dir_all(root.join(".local").join("config")).expect("create local config dir");
    fs::write(
        root.join(".local").join("config").join("state.json"),
        r#"{"future":{"keep":true}}"#,
    )
    .expect("write state");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "select", backend_id])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "Selected backend: {backend_id}"
        )))
        .stdout(predicate::str::contains(format!(
            "Models directory: {models_dir}"
        )))
        .stdout(predicate::str::contains("Backend config:"))
        .stdout(predicate::str::contains("(created)"));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/backend/select HTTP/1.1"));
    assert!(request.contains(&format!(r#"{{"backend":"{backend_id}"}}"#)));
    gateway.join();

    let state_raw = fs::read_to_string(root.join(".local").join("config").join("state.json"))
        .expect("state file");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("state json");
    assert_eq!(state["selected_backend"], backend_id);
    assert_eq!(state["future"]["keep"], true);

    let profile_raw = fs::read_to_string(
        root.join(".local")
            .join("config")
            .join("backend_profiles")
            .join(format!("{backend_id}.json")),
    )
    .expect("profile file");
    let profile: serde_json::Value = serde_json::from_str(&profile_raw).expect("profile json");
    assert_eq!(profile["schema_version"], 2);
    assert_eq!(profile["backend"], backend_id);
    assert_eq!(profile["family"], "llama.cpp");
    assert_eq!(profile["load"]["extra_args"], serde_json::json!([]));
    assert_eq!(profile["infer"]["extra_args"], serde_json::json!([]));
    fs::remove_dir_all(root).ok();
}
