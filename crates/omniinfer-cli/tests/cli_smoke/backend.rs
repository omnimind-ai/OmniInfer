use super::support::*;
use sha2::{Digest, Sha256};

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
fn backend_list_marks_missing_runtime() {
    let root = temp_repo_root("backend-list-compatible-missing");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Compatible backends"))
        .stdout(predicate::str::contains("Runtime"))
        .stdout(predicate::str::contains("missing"))
        .stdout(predicate::str::contains(
            "Install a runtime: omniinfer backend install <backend>",
        ));
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_install_prebuilt_from_local_catalog() {
    let root = temp_repo_root("backend-install-prebuilt");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let backend_id = test_external_backend_id();
    let fixture = write_prebuilt_fixture(&root, backend_id, false);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "Prebuilt backend: {}/{}",
            test_runtime_platform_dir(),
            backend_id
        )))
        .stdout(predicate::str::contains("Prebuilt backend installed:"));

    let launcher = installed_launcher(&root, backend_id);
    assert!(launcher.is_file(), "launcher should be installed");
    let helper = launcher.parent().unwrap().join("helper.txt");
    assert!(helper.is_file(), "sibling runtime file should be copied");
    let manifest_raw = fs::read_to_string(
        root.join(".local")
            .join("runtime")
            .join(test_runtime_platform_dir())
            .join(backend_id)
            .join("prebuilt.json"),
    )
    .expect("prebuilt manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_raw).expect("manifest json");
    assert_eq!(manifest["schema_version"], 3);
    assert_eq!(manifest["backend"], backend_id);
    assert_eq!(manifest["catalog_sha256"], fixture.sha256);
    assert_eq!(manifest["assets"].as_array().unwrap().len(), 1);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("Backend already installed"));

    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_install_prebuilt_rejects_checksum_mismatch() {
    let root = temp_repo_root("backend-install-checksum");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let backend_id = test_external_backend_id();
    let fixture = write_prebuilt_fixture(&root, backend_id, true);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "failed to download prebuilt archive",
        ));

    assert!(
        !installed_launcher(&root, backend_id).exists(),
        "checksum failure must not install launcher"
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_install_prebuilt_merges_companion_assets() {
    let root = temp_repo_root("backend-install-companion");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let backend_id = test_external_backend_id();
    let fixture = write_companion_prebuilt_fixture(&root, backend_id, false);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("Prebuilt backend installed:"));

    let bin_dir = installed_launcher(&root, backend_id)
        .parent()
        .unwrap()
        .to_path_buf();
    assert!(bin_dir.join("runtime-dependency.dll").is_file());
    let manifest_raw = fs::read_to_string(
        root.join(".local")
            .join("runtime")
            .join(test_runtime_platform_dir())
            .join(backend_id)
            .join("prebuilt.json"),
    )
    .expect("prebuilt manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_raw).expect("manifest json");
    assert_eq!(manifest["schema_version"], 3);
    assert_eq!(manifest["assets"].as_array().unwrap().len(), 2);
    assert_eq!(
        manifest["required_files"],
        serde_json::json!(["runtime-dependency.dll"])
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_install_incomplete_companion_preserves_existing_runtime() {
    let root = temp_repo_root("backend-install-companion-incomplete");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let backend_id = test_external_backend_id();
    let launcher = installed_launcher(&root, backend_id);
    fs::create_dir_all(launcher.parent().unwrap()).expect("create existing runtime");
    fs::write(&launcher, "existing launcher").expect("write existing launcher");
    fs::write(launcher.parent().unwrap().join("existing.txt"), "keep")
        .expect("write existing marker");
    let fixture = write_companion_prebuilt_fixture(&root, backend_id, true);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .failure()
        .stdout(predicate::str::contains(
            "Existing backend is incomplete; reinstalling",
        ))
        .stderr(predicate::str::contains(
            "required companion file runtime-dependency.dll was not found",
        ));

    assert_eq!(fs::read_to_string(&launcher).unwrap(), "existing launcher");
    assert!(launcher.parent().unwrap().join("existing.txt").is_file());
    assert!(
        !launcher
            .parent()
            .unwrap()
            .join("runtime-dependency.dll")
            .exists()
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_install_without_catalog_entry_explains_from_source() {
    let root = temp_repo_root("backend-install-no-catalog-entry");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let catalog = root.join("empty-prebuilt.json");
    fs::write(
        &catalog,
        serde_json::json!({
            "schema_version": 2,
            "platforms": {
                test_runtime_platform_dir(): {}
            }
        })
        .to_string(),
    )
    .expect("write empty catalog");

    let backend_id = test_external_backend_id();
    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "no prebuilt archive is configured for {}/{}",
            test_runtime_platform_dir(),
            backend_id
        )))
        .stderr(predicate::str::contains(format!(
            "omniinfer build {backend_id} --from-source"
        )));
    fs::remove_dir_all(root).ok();
}

#[cfg(target_os = "linux")]
#[test]
fn linux_cuda_prebuilt_install_explains_from_source() {
    let root = temp_repo_root("linux-cuda-no-prebuilt");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "install", "llama.cpp-linux-cuda"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "no prebuilt archive is configured for linux/llama.cpp-linux-cuda",
        ))
        .stderr(predicate::str::contains(
            "omniinfer build llama.cpp-linux-cuda --from-source",
        ));
    fs::remove_dir_all(root).ok();
}

#[test]
fn packaged_backend_install_uses_rust_prebuilt_path() {
    let root = temp_repo_root("packaged-backend-install");
    fs::create_dir_all(&root).expect("create package root");
    fs::create_dir_all(root.join("config")).expect("create config");
    fs::write(root.join("VERSION"), "0.3.2").expect("write version marker");
    fs::write(root.join("omniinfer"), "").expect("write launcher marker");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let backend_id = test_external_backend_id();
    let fixture = write_prebuilt_fixture(&root, backend_id, false);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["backend", "install", backend_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("Prebuilt backend installed:"));

    assert!(installed_launcher(&root, backend_id).is_file());
    fs::remove_dir_all(root).ok();
}

#[test]
fn legacy_build_command_defaults_to_prebuilt_install() {
    let root = temp_repo_root("legacy-build-prebuilt");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");
    let backend_id = test_external_backend_id();
    let fixture = write_prebuilt_fixture(&root, backend_id, false);

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .env("OMNIINFER_PREBUILT_CATALOG", &fixture.catalog)
        .args(["build", backend_id])
        .assert()
        .success()
        .stdout(predicate::str::contains("Prebuilt backend installed:"));

    assert!(installed_launcher(&root, backend_id).is_file());
    fs::remove_dir_all(root).ok();
}

#[test]
fn backend_install_from_source_is_explicitly_not_prebuilt() {
    let root = temp_repo_root("backend-install-from-source");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(root.join("config").join("omniinfer.json"), r#"{"port":1}"#).expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args([
            "backend",
            "install",
            test_external_backend_id(),
            "--from-source",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Source builds require a source checkout",
        ));
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

struct PrebuiltFixture {
    catalog: std::path::PathBuf,
    sha256: String,
}

fn write_prebuilt_fixture(
    root: &std::path::Path,
    backend_id: &str,
    wrong_checksum: bool,
) -> PrebuiltFixture {
    let fixture = root.join("prebuilt-fixture");
    let payload_root = fixture.join("payload").join("runtime").join("bin");
    fs::create_dir_all(&payload_root).expect("create payload");
    let launcher_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let launcher = payload_root.join(launcher_name);
    fs::write(&launcher, "#!/usr/bin/env bash\nexit 0\n").expect("write launcher");
    fs::write(payload_root.join("helper.txt"), "runtime helper").expect("write helper");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).unwrap();
    }

    let archive = fixture.join(if cfg!(windows) {
        "runtime.zip"
    } else {
        "runtime.tar.gz"
    });
    write_runtime_archive(&fixture.join("payload"), &archive);
    let bytes = fs::read(&archive).expect("read archive");
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let catalog_sha = if wrong_checksum {
        "0".repeat(64)
    } else {
        sha256.clone()
    };
    let catalog = fixture.join("prebuilt.json");
    fs::write(
        &catalog,
        serde_json::json!({
            "schema_version": 2,
            "platforms": {
                test_runtime_platform_dir(): {
                    backend_id: {
                        "source": "fixture",
                        "tag": "test",
                        "url": format!("file://{}", archive.display()),
                        "archive": if cfg!(windows) { "zip" } else { "tar.gz" },
                        "launcher": launcher_name,
                        "sha256": catalog_sha,
                        "submodule_path": "framework/llama.cpp",
                        "submodule_commit": "fixture"
                    }
                }
            }
        })
        .to_string(),
    )
    .expect("write catalog");
    PrebuiltFixture { catalog, sha256 }
}

fn write_companion_prebuilt_fixture(
    root: &std::path::Path,
    backend_id: &str,
    omit_required_file: bool,
) -> PrebuiltFixture {
    let fixture = root.join("prebuilt-companion-fixture");
    let launcher_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let primary_payload = fixture.join("primary-payload").join("runtime").join("bin");
    fs::create_dir_all(&primary_payload).expect("create primary payload");
    let launcher = primary_payload.join(launcher_name);
    fs::write(&launcher, "#!/usr/bin/env bash\nexit 0\n").expect("write launcher");
    fs::write(primary_payload.join("helper.txt"), "runtime helper").expect("write helper");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).unwrap();
    }

    let companion_payload = fixture.join("companion-payload").join("dependencies");
    fs::create_dir_all(&companion_payload).expect("create companion payload");
    if !omit_required_file {
        fs::write(
            companion_payload.join("runtime-dependency.dll"),
            "runtime dependency",
        )
        .expect("write dependency");
    }

    let extension = if cfg!(windows) { "zip" } else { "tar.gz" };
    let primary_archive = fixture.join(format!("runtime.{extension}"));
    let companion_archive = fixture.join(format!("companion.{extension}"));
    write_runtime_archive(&fixture.join("primary-payload"), &primary_archive);
    write_runtime_archive(&fixture.join("companion-payload"), &companion_archive);
    let primary_sha = format!(
        "{:x}",
        Sha256::digest(fs::read(&primary_archive).expect("read primary archive"))
    );
    let companion_sha = format!(
        "{:x}",
        Sha256::digest(fs::read(&companion_archive).expect("read companion archive"))
    );
    let catalog = fixture.join("prebuilt.json");
    fs::write(
        &catalog,
        serde_json::json!({
            "schema_version": 2,
            "platforms": {
                test_runtime_platform_dir(): {
                    backend_id: {
                        "source": "fixture",
                        "tag": "test",
                        "url": format!("file://{}", primary_archive.display()),
                        "archive": extension,
                        "launcher": launcher_name,
                        "sha256": primary_sha,
                        "companion_assets": [{
                            "url": format!("file://{}", companion_archive.display()),
                            "archive": extension,
                            "sha256": companion_sha,
                            "files": ["runtime-dependency.dll"]
                        }],
                        "required_files": ["runtime-dependency.dll"],
                        "submodule_path": "framework/llama.cpp",
                        "submodule_commit": "fixture"
                    }
                }
            }
        })
        .to_string(),
    )
    .expect("write companion catalog");
    PrebuiltFixture {
        catalog,
        sha256: primary_sha,
    }
}

fn write_runtime_archive(source_root: &std::path::Path, archive_path: &std::path::Path) {
    #[cfg(windows)]
    {
        let file = fs::File::create(archive_path).expect("create zip");
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        add_zip_tree(&mut zip, source_root, source_root, options);
        zip.finish().expect("finish zip");
    }
    #[cfg(not(windows))]
    {
        let file = fs::File::create(archive_path).expect("create tar");
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut tar = tar::Builder::new(encoder);
        tar.append_dir_all(".", source_root).expect("append tar");
        tar.finish().expect("finish tar");
    }
}

#[cfg(windows)]
fn add_zip_tree(
    zip: &mut zip::ZipWriter<fs::File>,
    source_root: &std::path::Path,
    current: &std::path::Path,
    options: zip::write::SimpleFileOptions,
) {
    for entry in fs::read_dir(current).expect("read zip source") {
        let entry = entry.expect("zip source entry");
        let path = entry.path();
        let name = path
            .strip_prefix(source_root)
            .expect("relative zip path")
            .to_string_lossy()
            .replace('\\', "/");
        if path.is_dir() {
            zip.add_directory(format!("{name}/"), options)
                .expect("add zip dir");
            add_zip_tree(zip, source_root, &path, options);
        } else {
            zip.start_file(name, options).expect("start zip file");
            let mut file = fs::File::open(&path).expect("open zip source");
            std::io::copy(&mut file, zip).expect("copy zip file");
        }
    }
}

fn installed_launcher(root: &std::path::Path, backend_id: &str) -> std::path::PathBuf {
    root.join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        })
}
