use assert_cmd::Command;
use omniinfer_core::http_client;
use predicates::prelude::*;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::process::{Command as StdCommand, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

#[test]
fn advisor_system_json_uses_rust_path() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("advisor-system-source");
    let state_root = temp_repo_root("advisor-system-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    install_fake_backend(&state_root, backend_id);
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        r#"{"host":"127.0.0.1","port":1,"startup_timeout":10}"#,
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    let output = cmd
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["advisor", "system", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""object": "advisor.system""#))
        .stdout(predicate::str::contains(format!(
            r#""recommended_installed_backend": "{backend_id}""#
        )))
        .stdout(predicate::str::contains(r#""installed": true"#))
        .get_output()
        .stdout
        .clone();
    let payload: serde_json::Value = serde_json::from_slice(&output).expect("advisor system json");
    assert_eq!(payload["object"], "advisor.system");
    assert_eq!(
        payload["summary"]["recommended_installed_backend"],
        backend_id
    );
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn advisor_system_text_prints_usable_backends() {
    let backend_id = test_external_backend_id();
    let source_root = temp_repo_root("advisor-system-text-source");
    let state_root = temp_repo_root("advisor-system-text-state");
    fs::create_dir_all(&source_root).expect("create source root");
    fs::create_dir_all(state_root.join("config")).expect("create state config");
    install_fake_backend(&state_root, backend_id);
    fs::write(
        state_root.join("config").join("omniinfer.json"),
        r#"{"host":"127.0.0.1","port":1,"startup_timeout":10}"#,
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["advisor", "system"])
        .assert()
        .success()
        .stdout(predicate::str::contains("OmniInfer Advisor System"))
        .stdout(predicate::str::contains("Usable backends:"))
        .stdout(predicate::str::contains(backend_id))
        .stdout(predicate::str::contains("Hidden backends:"));
    fs::remove_dir_all(source_root).ok();
    fs::remove_dir_all(state_root).ok();
}

#[test]
fn advisor_inspect_json_estimates_local_model() {
    let root = temp_repo_root("advisor-inspect");
    fs::create_dir_all(&root).expect("create root");
    let model = root.join("Qwen3.5-4B-Q4_K_M.gguf");
    fs::write(&model, vec![0_u8; 1024]).expect("write model");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    let output = cmd
        .env("OMNIINFER_RUST_STRICT", "1")
        .args(["advisor", "inspect"])
        .arg(&model)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""object": "advisor.model""#))
        .stdout(predicate::str::contains(r#""quantization": "Q4_K_M""#))
        .stdout(predicate::str::contains(r#""params_b": 4.0"#))
        .get_output()
        .stdout
        .clone();
    let payload: serde_json::Value = serde_json::from_slice(&output).expect("inspect json");
    assert_eq!(payload["format"], "gguf");
    assert_eq!(payload["artifact_kind"], "file");
    assert_eq!(payload["exists"], true);
    assert_eq!(payload["estimate"]["breakdown"]["context_size"], 8192);
    fs::remove_dir_all(root).ok();
}

#[test]
fn advisor_fit_json_ranks_installed_backend() {
    let backend_id = test_external_backend_id();
    let root = temp_repo_root("advisor-fit");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    install_fake_backend(&root, backend_id);
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let model = root.join("Qwen3.5-4B-Q4_K_M.gguf");
    fs::write(&model, vec![0_u8; 1024]).expect("write model");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    let output = cmd
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["advisor", "fit"])
        .arg(&model)
        .args(["--ctx-size", "512", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""object": "advisor.fit""#))
        .stdout(predicate::str::contains(format!(
            r#""backend": "{backend_id}""#
        )))
        .stdout(predicate::str::contains(
            r#""recommendation_confidence": "high""#,
        ))
        .get_output()
        .stdout
        .clone();
    let payload: serde_json::Value = serde_json::from_slice(&output).expect("fit json");
    assert!(
        payload["recommended"]["backend"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
    assert!(matches!(
        payload["recommended"]["evidence"]["level"].as_str(),
        Some("direct" | "variant")
    ));
    assert!(
        payload["next_command"]
            .as_str()
            .unwrap()
            .contains("omniinfer backend select ")
    );

    fs::remove_dir_all(root).ok();
}

#[test]
fn advisor_plan_text_shows_simulated_hardware() {
    let backend_id = test_external_backend_id();
    let root = temp_repo_root("advisor-plan");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    install_fake_backend(&root, backend_id);
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let model = root.join("Qwen3.5-4B-Q4_K_M.gguf");
    fs::write(&model, vec![0_u8; 1024]).expect("write model");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["advisor", "plan"])
        .arg(&model)
        .args([
            "--ctx-size",
            "512",
            "--gpu-vram",
            "2",
            "--ram",
            "4",
            "--cpu-cores",
            "8",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("OmniInfer Advisor Plan"))
        .stdout(predicate::str::contains(
            "Planning hardware: free_vram=2.0 GiB",
        ))
        .stdout(predicate::str::contains("Run paths:"));

    fs::remove_dir_all(root).ok();
}

#[test]
fn advisor_recommend_json_scans_managed_models() {
    let backend_id = test_external_backend_id();
    let root = temp_repo_root("advisor-recommend");
    let models_dir = root.join("models");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::create_dir_all(&models_dir).expect("create models dir");
    install_fake_backend(&root, backend_id);
    let model = models_dir.join("Qwen3.5-Coder-4B-Q4_K_M.gguf");
    fs::write(&model, vec![0_u8; 1024]).expect("write model");
    let config_payload = serde_json::json!({
        "port": 1,
        "backends": {
            backend_id: {
                "models_dir": models_dir.display().to_string(),
            },
        },
    });
    fs::write(
        root.join("config").join("omniinfer.json"),
        serde_json::to_string(&config_payload).expect("config json"),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    let output = cmd
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args([
            "advisor",
            "recommend",
            "--task",
            "coding",
            "-n",
            "1",
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""object": "advisor.recommend""#))
        .stdout(predicate::str::contains(r#""returned": 1"#))
        .get_output()
        .stdout
        .clone();
    let payload: serde_json::Value = serde_json::from_slice(&output).expect("recommend json");
    assert_eq!(payload["models_scanned"], 1);
    assert!(
        payload["recommendations"][0]["recommended"]["backend"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
    assert!(matches!(
        payload["recommendations"][0]["evidence"]["level"].as_str(),
        Some("direct" | "variant")
    ));

    fs::remove_dir_all(root).ok();
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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
    let status = StdCommand::new(assert_cmd::cargo::cargo_bin("omniinfer-rs"))
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
        .expect("run omniinfer-rs serve");
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

    let mut stop = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut stop = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", &state_root)
        .args(["serve", "--detach", "--smoke-test", "--port"])
        .arg(port.to_string())
        .arg("--model")
        .arg(&model)
        .assert()
        .success()
        .stdout(predicate::str::contains("Smoke: hello smoke"));

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    let body = request_body_json(&request);
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"][0]["content"], "Hello");
    gateway.join();
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
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
        .stdout(predicate::str::contains("public warning:"));

    let _ = gateway.request();
    let _ = gateway.request();
    let _ = gateway.request();
    let request = gateway.request();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
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

struct TestGateway {
    port: u16,
    request_rx: mpsc::Receiver<String>,
    handle: thread::JoinHandle<()>,
}

impl TestGateway {
    fn start(responses: Vec<Response>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test gateway");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking test gateway");
        let port = listener.local_addr().expect("local addr").port();
        let (request_tx, request_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            for response_body in responses {
                let mut stream = accept_test_request(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("set request read timeout");
                let request = read_http_request(&mut stream);
                request_tx.send(request).expect("send request");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.content_type,
                    response_body.content_len()
                );
                stream.write_all(response.as_bytes()).expect("write header");
                response_body.write_body(&mut stream);
                stream.flush().expect("flush response");
            }
        });
        Self {
            port,
            request_rx,
            handle,
        }
    }

    fn request(&self) -> String {
        self.request_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("receive request")
    }

    fn join(self) {
        self.handle.join().expect("server thread");
    }
}

fn accept_test_request(listener: &TcpListener) -> std::net::TcpStream {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return stream,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    panic!("timed out waiting for test gateway request");
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("accept request: {error}"),
        }
    }
}

struct Response {
    body: ResponseBody,
    content_type: String,
}

enum ResponseBody {
    Text(String),
    Chunks(Vec<String>),
}

impl Response {
    fn new(body: &str) -> Self {
        Self::with_content_type(body, "application/json")
    }

    fn with_content_type(body: &str, content_type: &str) -> Self {
        Self {
            body: ResponseBody::Text(body.to_string()),
            content_type: content_type.to_string(),
        }
    }

    fn chunks(chunks: &[&str], content_type: &str) -> Self {
        Self {
            body: ResponseBody::Chunks(chunks.iter().map(|chunk| (*chunk).to_string()).collect()),
            content_type: content_type.to_string(),
        }
    }

    fn content_len(&self) -> usize {
        match &self.body {
            ResponseBody::Text(body) => body.len(),
            ResponseBody::Chunks(chunks) => chunks.iter().map(String::len).sum(),
        }
    }

    fn write_body(&self, stream: &mut impl Write) {
        match &self.body {
            ResponseBody::Text(body) => {
                stream.write_all(body.as_bytes()).expect("write body");
            }
            ResponseBody::Chunks(chunks) => {
                for chunk in chunks {
                    stream.write_all(chunk.as_bytes()).expect("write chunk");
                    stream.flush().expect("flush chunk");
                    thread::sleep(Duration::from_millis(15));
                }
            }
        }
    }
}

fn temp_repo_root(test_name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-rs-{test_name}-{nanos}"))
}

fn install_fake_backend(root: &std::path::Path, backend_id: &str) {
    let binary_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join(binary_name);
    fs::create_dir_all(launcher.parent().unwrap()).expect("create fake backend dir");
    fs::write(&launcher, "#!/usr/bin/env bash\nexit 0\n").expect("write fake backend");
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(&launcher)
            .expect("fake backend metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).expect("chmod fake backend");
    }
}

fn test_external_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-cpu"
    } else {
        "llama.cpp-linux"
    }
}

fn test_secondary_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac-intel"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-windows-arm64"
    } else {
        "llama.cpp-linux"
    }
}

fn test_runtime_platform_dir() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

#[cfg(unix)]
fn install_fake_runtime_server(root: &std::path::Path, backend_id: &str) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("llama-server");
    fs::create_dir_all(launcher.parent().unwrap()).expect("create fake runtime dir");
    fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$port" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def _json(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok"})
        else:
            self._json({"ok": True})
    def do_POST(self):
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#,
    )
    .expect("write fake runtime");
    let mut permissions = fs::metadata(&launcher)
        .expect("fake runtime metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod fake runtime");
}

fn fake_cloudflared_launcher(root: &std::path::Path) -> std::path::PathBuf {
    fake_cloudflared_launcher_with_url(root, "https://example-test.trycloudflare.com")
}

fn fake_cloudflared_launcher_with_url(root: &std::path::Path, url: &str) -> std::path::PathBuf {
    #[cfg(unix)]
    {
        fake_cloudflared_launcher_unix(root, url)
    }
    #[cfg(windows)]
    {
        fake_cloudflared_launcher_windows(root, url)
    }
}

#[cfg(unix)]
fn fake_cloudflared_launcher_unix(root: &std::path::Path, url: &str) -> std::path::PathBuf {
    let launcher = root.join("fake-cloudflared.sh");
    let output = root.join("cloudflared.args");
    fs::write(
        &launcher,
        format!(
            "#!/usr/bin/env bash\nprintf '%s ' \"$@\" > '{}'\necho 'Your quick Tunnel has been created! Visit it at {}'\nsleep 30\n",
            output.display(),
            url
        ),
    )
    .expect("write fake cloudflared");
    let mut permissions = fs::metadata(&launcher)
        .expect("cloudflared metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod cloudflared");
    launcher
}

#[cfg(windows)]
fn fake_cloudflared_launcher_windows(root: &std::path::Path, url: &str) -> std::path::PathBuf {
    let launcher = root.join("fake-cloudflared.cmd");
    let output = root.join("cloudflared.args");
    fs::write(
        &launcher,
        format!(
            "@echo off\r\necho %* > \"{}\"\r\necho Your quick Tunnel has been created! Visit it at {}\r\npowershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command \"Start-Sleep -Seconds 30\" > nul\r\n",
            output.display(),
            url
        ),
    )
    .expect("write fake cloudflared");
    launcher
}

fn read_http_request(stream: &mut impl Read) -> String {
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut expected_len = None;
    loop {
        let bytes = stream.read(&mut buffer).expect("read request");
        if bytes == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..bytes]);
        if expected_len.is_none()
            && let Some(header_end) = find_header_end(&raw)
        {
            let header = String::from_utf8_lossy(&raw[..header_end]);
            let content_length = header
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            expected_len = Some(header_end + 4 + content_length);
        }
        if let Some(length) = expected_len
            && raw.len() >= length
        {
            break;
        }
    }
    String::from_utf8_lossy(&raw).to_string()
}

fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|window| window == b"\r\n\r\n")
}

fn request_body_json(request: &str) -> serde_json::Value {
    let body = request
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("request body separator");
    serde_json::from_str(body).expect("request body json")
}

fn wait_for_file(path: std::path::PathBuf) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(text) = fs::read_to_string(&path) {
            return text;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {}", path.display());
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    listener.local_addr().expect("local addr").port()
}

fn wait_for_http_json(port: u16, path: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    let url = format!("http://127.0.0.1:{port}{path}");
    while Instant::now() < deadline {
        if let Ok(response) = http_client::get_json(&url, Duration::from_secs(1)) {
            if response.status < 400 {
                return response.body;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for {url}");
}

fn stop_rust_serve(source_root: &std::path::Path, state_root: &std::path::Path, port: u16) {
    let mut stop = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", state_root)
        .args(["serve", "stop", "--port"])
        .arg(port.to_string())
        .assert()
        .success();
    assert!(wait_for_port_closed(port));
}

fn wait_for_port_closed(port: u16) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}
