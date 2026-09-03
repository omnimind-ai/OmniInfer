use super::support::*;

#[test]
fn help_lists_core_commands() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("advisor"))
        .stdout(predicate::str::contains("bench"))
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

#[cfg(unix)]
#[test]
fn source_build_dispatches_to_platform_script() {
    use std::os::unix::fs::PermissionsExt;

    let root = temp_repo_root("source-build-dispatch");
    let script = if cfg!(target_os = "macos") {
        root.join("scripts/platforms/macos/llama.cpp-mac/build.sh")
    } else {
        root.join("scripts/platforms/linux/vla.cpp-linux/build.sh")
    };
    let backend = if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else {
        "vla.cpp-linux"
    };
    fs::create_dir_all(script.parent().unwrap()).expect("create source build fixture");
    fs::write(
        &script,
        "#!/usr/bin/env bash\nset -eu\nprintf '%s\\n' \"$@\" > \"$OMNIINFER_TEST_BUILD_ARGS\"\n",
    )
    .expect("write source build fixture");
    let mut permissions = fs::metadata(&script).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&script, permissions).unwrap();
    let recorded_args = root.join("build-args.txt");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_TEST_BUILD_ARGS", &recorded_args)
        .args(["build", backend, "--from-source", "--", "--jobs", "2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Building backend from source"));
    assert_eq!(
        fs::read_to_string(recorded_args).unwrap(),
        "--from-source\n--jobs\n2\n"
    );
    fs::remove_dir_all(root).ok();
}

#[cfg(windows)]
#[test]
fn source_build_dispatches_to_platform_script() {
    let root = temp_repo_root("source-build-dispatch");
    let script = root.join("scripts/platforms/windows/llama.cpp-cpu/build.ps1");
    fs::create_dir_all(script.parent().unwrap()).expect("create source build fixture");
    fs::write(
        &script,
        "param([string]$BuildType)\n[System.IO.File]::WriteAllText($env:OMNIINFER_TEST_BUILD_MARKER, $BuildType)\n",
    )
    .expect("write source build fixture");
    let marker = root.join("build-ran.txt");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_TEST_BUILD_MARKER", &marker)
        .args([
            "build",
            "llama.cpp-cpu",
            "--from-source",
            "--",
            "-BuildType",
            "Debug",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Building backend from source"));
    assert_eq!(fs::read_to_string(marker).unwrap(), "Debug");
    fs::remove_dir_all(root).ok();
}

#[test]
fn build_rejects_conflicting_source_and_prebuilt_modes() {
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.args(["build", "llama.cpp-linux", "--prebuilt", "--from-source"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cannot be used with '--from-source'",
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
