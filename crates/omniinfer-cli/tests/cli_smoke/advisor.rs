use super::support::*;

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
