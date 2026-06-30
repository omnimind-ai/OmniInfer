use super::support::*;

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
fn thinking_set_updates_local_state_without_gateway() {
    let root = temp_repo_root("thinking-set-local");
    let port = free_port();
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(
            r#"{{"host":"127.0.0.1","port":{},"startup_timeout":1}}"#,
            port
        ),
    )
    .expect("write isolated config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_STATE_ROOT", &root)
        .args(["thinking", "set", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Default thinking set to: off"));

    let raw = fs::read_to_string(root.join(".local").join("config").join("state.json"))
        .expect("read state");
    let state: serde_json::Value = serde_json::from_str(&raw).expect("state json");
    assert_eq!(state["default_thinking"], "off");
    fs::remove_dir_all(root).ok();
}

#[test]
fn tui_requires_interactive_terminal_without_python_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "OmniInfer TUI requires an interactive terminal.",
        ));
}

#[test]
fn strict_mode_reports_unported_commands_without_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .args(["build", "llama.cpp-linux"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Python control-plane fallback has been removed",
        ));
}
