use super::support::*;

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

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
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
