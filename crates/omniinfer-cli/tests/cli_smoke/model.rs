use super::support::*;

#[test]
fn model_load_posts_payload_and_persists_state() {
    let backend_id = test_external_backend_id();
    let secondary_backend_id = test_secondary_backend_id();
    let load_response = format!(
        r#"{{"selected_backend":"{backend_id}","selected_model":"/tmp/model.gguf","selected_mmproj":null,"selected_ctx_size":8192}}"#
    );
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(&load_response),
    ]);
    let root = temp_repo_root("model-load");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");
    install_fake_backend(&root, secondary_backend_id);
    install_fake_backend(&root, backend_id);
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
        .stdout(predicate::str::contains(format!(
            "Auto-selected backend: {backend_id}"
        )))
        .stdout(predicate::str::contains("Model loaded"))
        .stdout(predicate::str::contains("ctx-size: 8192"));

    let request = gateway.request();
    assert!(request.starts_with("GET /health HTTP/1.1"));
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["model"], model.display().to_string());
    assert_eq!(body["backend"], backend_id);
    assert_eq!(body["ctx_size"], 8192);
    assert_eq!(body["launch_args"], serde_json::json!(["-ngl", "999"]));
    gateway.join();

    let state_raw = fs::read_to_string(root.join(".local").join("config").join("state.json"))
        .expect("state file");
    let state: serde_json::Value = serde_json::from_str(&state_raw).expect("state json");
    assert_eq!(state["selected_backend"], backend_id);
    assert_eq!(state["selected_model"], "/tmp/model.gguf");
    assert_eq!(state["selected_ctx_size"], 8192);
    assert!(state.get("selected_mmproj").is_none());
    fs::remove_dir_all(root).ok();
}

#[test]
fn model_load_handles_sse_progress() {
    let backend_id = test_external_backend_id();
    let secondary_backend_id = test_secondary_backend_id();
    let gateway = TestGateway::start(vec![
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
    install_fake_backend(&root, secondary_backend_id);
    install_fake_backend(&root, backend_id);
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
    let request = gateway.request();
    assert!(request.starts_with("POST /omni/model/select HTTP/1.1"));
    assert!(request.contains("Accept: text/event-stream, application/json"));
    gateway.join();
    fs::remove_dir_all(root).ok();
}
