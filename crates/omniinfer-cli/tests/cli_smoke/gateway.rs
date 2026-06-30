use super::support::*;

#[test]
fn gateway_foreground_serves_health_and_shutdown() {
    let port = free_port();
    let source_root = temp_repo_root("serve-foreground-source");
    let state_root = temp_repo_root("serve-foreground-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::write(source_root.join("Cargo.toml"), "[workspace]\n").expect("write source manifest");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":10}}"#,
            port
        ),
    )
    .expect("write config");

    let stdout_path = state_root.join("serve-foreground.stdout.txt");
    let stderr_path = state_root.join("serve-foreground.stderr.txt");
    let mut child = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer-rs"))
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["gateway", "--host", "127.0.0.1", "--port"])
        .arg(port.to_string())
        .stdout(Stdio::from(
            fs::File::create(&stdout_path).expect("create stdout capture"),
        ))
        .stderr(Stdio::from(
            fs::File::create(&stderr_path).expect("create stderr capture"),
        ))
        .spawn()
        .expect("spawn foreground serve");

    let health = wait_for_http_json(port, "/health?deep=true");
    assert_eq!(health["status"], "ok");
    let _ = http_client::post_json(
        &format!("http://127.0.0.1:{port}/omni/shutdown"),
        &serde_json::json!({}),
        Duration::from_secs(2),
    );
    let status = child.wait().expect("wait foreground serve");
    let stdout = fs::read_to_string(&stdout_path).expect("read stdout capture");
    let stderr = fs::read_to_string(&stderr_path).expect("read stderr capture");
    assert!(
        status.success(),
        "foreground gateway failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
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
