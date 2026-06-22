use assert_cmd::Command;
use predicates::prelude::*;

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
