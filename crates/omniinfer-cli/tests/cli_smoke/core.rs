use super::support::*;

#[test]
fn help_lists_core_commands() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("advisor"))
        .stdout(predicate::str::contains("serve"))
        .stdout(predicate::str::contains("thinking").not());
}

#[test]
fn completion_generates_bash_script() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.args(["completion", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_omniinfer"));
}

#[test]
fn chat_help_keeps_request_level_thinking_switch() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .args(["chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--think"));
}

#[test]
fn tui_requires_interactive_terminal_without_python_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "OmniInfer TUI requires an interactive terminal.",
        ));
}

#[test]
fn strict_mode_reports_unported_commands_without_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .args(["build", "llama.cpp-linux", "--from-source"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Python control-plane fallback has been removed",
        ));
}

#[test]
fn packaged_build_reports_source_checkout_requirement() {
    let root = temp_repo_root("packaged-build");
    fs::create_dir_all(&root).expect("create package root");
    fs::write(root.join("VERSION"), "0.3.2").expect("write version marker");
    fs::write(root.join("omniinfer"), "").expect("write launcher marker");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["build", "llama.cpp-linux", "--from-source"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Source backend builds are only available from a source checkout, not packaged releases.",
        ));
    fs::remove_dir_all(root).ok();
}
