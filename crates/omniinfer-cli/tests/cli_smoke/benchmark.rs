use super::support::*;

#[test]
fn bench_archives_submission_compatible_json() {
    let root = temp_repo_root("bench-run");
    let runtime_dir = root.join("runtime");
    fs::create_dir_all(&runtime_dir).expect("create runtime dir");
    fs::write(
        runtime_dir.join("prebuilt.json"),
        r#"{"backend":"llama.cpp-linux-cuda","tag":"b10280"}"#,
    )
    .expect("write prebuilt manifest");
    let state = serde_json::json!({
        "backend_ready": true,
        "model": "/models/qwen.gguf",
        "backend": "llama.cpp-linux-cuda",
        "ctx_size": 128,
        "mmproj": null,
        "launch_command": [
            "llama-server", "-m", "/models/qwen.gguf", "-b", "64",
            "--cache-ram", "0", "--no-cache-idle-slots", "--no-cache-prompt",
            "--slot-prompt-similarity", "0", "--slot-save-path", "/tmp/benchmark-slots",
            "--api-key", "runtime-secret"
        ],
        "available_backends": {"data": [{
            "id": "llama.cpp-linux-cuda",
            "runtime_dir": runtime_dir,
        }]},
    })
    .to_string();
    let measurement = r#"{
        "usage": {"prompt_tokens": 64, "completion_tokens": 16},
        "timings": {"prompt_ms": 400.0, "predicted_ms": 800.0}
    }"#;
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(&state),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
    ]);
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    let benchmark_id = "contract-test-omniinfer-bench";
    let mut command = Command::cargo_bin("omniinfer").expect("binary exists");
    let assert = command
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args([
            "bench",
            "run",
            "--benchmark-id",
            benchmark_id,
            "--catalog-model-id",
            "qwen3-5-2b",
            "--format",
            "GGUF",
            "--quantization",
            "Q4_K_M",
            "--model-url",
            "https://example.com/qwen.gguf",
            "--device-name",
            "NVIDIA GeForce RTX 5090",
            "--soc",
            "rtx-5090",
            "--baseline",
            "--runs",
            "3",
            "--warmup-runs",
            "0",
            "--submitter-name",
            "OmniInfer Test",
            "--json",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Run 3/3"))
        .stderr(predicate::str::contains("Schema: 1.4.0"));
    let printed_payload: serde_json::Value = serde_json::from_slice(&assert.get_output().stdout)
        .expect("--json stdout is one JSON value");
    assert_eq!(printed_payload["benchmark_id"], benchmark_id);

    assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
    assert!(gateway.request().starts_with("GET /omni/state HTTP/1.1"));
    for _ in 0..3 {
        assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
        assert!(
            gateway
                .request()
                .starts_with("POST /omni/cache/clear HTTP/1.1")
        );
        assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
        let request = gateway.request();
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        let body = request_body_json(&request);
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["cache_prompt"], false);
        assert!(body.get("ignore_eos").is_none());
    }
    gateway.join();

    let result = root
        .join(".local")
        .join("benchmarks")
        .join("results")
        .join(format!("{benchmark_id}.json"));
    let payload: serde_json::Value =
        serde_json::from_slice(&fs::read(&result).expect("read benchmark result"))
            .expect("parse benchmark result");
    assert_eq!(payload["schema_version"], "1.4.0");
    assert_eq!(payload["producer"]["name"], "OmniInfer CLI");
    assert_eq!(payload["backend"]["version"], "b10280");
    assert_eq!(
        payload["runtime"]["build_command"],
        "omniinfer backend install llama.cpp-linux-cuda"
    );
    assert_eq!(payload["protocol"]["profile"], "text-pp-tg-standard-v1");
    assert_eq!(payload["protocol"]["cache_policy"], "cleared_each_run");
    assert_eq!(payload["workload"]["pp"], 64);
    assert_eq!(payload["workload"]["tg"], 16);
    assert_eq!(payload["workload"]["scored_tokens"]["prefill"], 64);
    assert_eq!(payload["workload"]["scored_tokens"]["decode"], 16);
    assert_eq!(payload["execution"]["compute_mode"], "single");
    assert_eq!(payload["execution"]["prefill_accelerator"], "gpu");
    assert_eq!(payload["execution"]["decode_accelerator"], "gpu");
    assert_eq!(payload["execution"]["privilege_level"], "standard");
    assert_eq!(payload["workload"]["batch_size"], 64);
    assert_eq!(
        payload["runs"]["prefill_tps"],
        serde_json::json!([160.0, 160.0, 160.0])
    );
    assert_eq!(
        payload["runs"]["decode_tps"],
        serde_json::json!([20.0, 20.0, 20.0])
    );
    assert_eq!(payload["optimization"]["mode"], "baseline");
    assert_eq!(payload["optimization"]["methods"], serde_json::json!([]));
    assert!(payload["protocol"].get("notes").is_none());
    let run_command = payload["runtime"]["run_command"]
        .as_str()
        .expect("runtime command is a string");
    assert!(run_command.contains("llama-server -m /models/qwen.gguf -b 64"));
    assert!(run_command.contains("--api-key"));
    assert!(run_command.contains("<redacted>"));
    assert!(!run_command.contains("runtime-secret"));

    let mut list = Command::cargo_bin("omniinfer").expect("binary exists");
    list.env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["bench", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains(benchmark_id))
        .stdout(predicate::str::contains("qwen3-5-2b"));
    fs::remove_dir_all(root).ok();
}

#[test]
fn bench_includes_ignore_eos_when_requested() {
    let state = r#"{
        "backend_ready": true,
        "model": "/models/qwen.gguf",
        "backend": "llama.cpp-linux-cuda",
        "ctx_size": 128,
        "launch_command": [
            "llama-server", "-m", "/models/qwen.gguf", "-b", "64",
            "--cache-ram", "0", "--no-cache-idle-slots", "--no-cache-prompt",
            "--slot-prompt-similarity", "0", "--slot-save-path", "/tmp/benchmark-slots"
        ]
    }"#;
    let measurement = r#"{
        "usage": {"prompt_tokens": 64, "completion_tokens": 16},
        "timings": {"prompt_ms": 400.0, "predicted_ms": 800.0}
    }"#;
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(state),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(measurement),
    ]);
    let root = temp_repo_root("bench-ignore-eos");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    Command::cargo_bin("omniinfer")
        .expect("binary exists")
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args([
            "bench",
            "run",
            "--benchmark-id",
            "fixed-length-generation",
            "--catalog-model-id",
            "qwen3-5-2b",
            "--format",
            "GGUF",
            "--quantization",
            "Q4_K_M",
            "--model-url",
            "https://example.com/qwen.gguf",
            "--device-name",
            "NVIDIA GeForce RTX 5090",
            "--soc",
            "rtx-5090",
            "--backend-version",
            "test-runtime-1",
            "--build-command",
            "bash build.sh",
            "--baseline",
            "--runs",
            "3",
            "--warmup-runs",
            "0",
            "--max-tokens",
            "16",
            "--submitter-name",
            "OmniInfer Test",
            "--notes",
            "device note",
            "--ignore-eos",
        ])
        .assert()
        .success();

    assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
    assert!(gateway.request().starts_with("GET /omni/state HTTP/1.1"));
    for _ in 0..3 {
        assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
        assert!(
            gateway
                .request()
                .starts_with("POST /omni/cache/clear HTTP/1.1")
        );
        assert!(gateway.request().starts_with("GET /health HTTP/1.1"));
        let request = gateway.request();
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        let body = request_body_json(&request);
        assert_eq!(body["ignore_eos"], true);
    }
    gateway.join();

    let result = root
        .join(".local")
        .join("benchmarks")
        .join("results")
        .join("fixed-length-generation.json");
    let payload: serde_json::Value =
        serde_json::from_slice(&fs::read(&result).expect("read benchmark result"))
            .expect("parse benchmark result");
    assert_eq!(payload["workload"]["tg"], 16);
    assert_eq!(
        payload["protocol"]["notes"],
        "device note; fixed_length_generation=true; ignore_eos=true; completion_tokens=max_tokens"
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn bench_ignore_eos_rejects_short_measured_response() {
    let state = r#"{
        "backend_ready": true,
        "model": "/models/qwen.gguf",
        "backend": "llama.cpp-linux-cuda",
        "ctx_size": 128,
        "launch_command": [
            "llama-server", "-m", "/models/qwen.gguf", "-b", "64",
            "--cache-ram", "0", "--no-cache-idle-slots", "--no-cache-prompt",
            "--slot-prompt-similarity", "0", "--slot-save-path", "/tmp/benchmark-slots"
        ]
    }"#;
    let full_measurement = r#"{
        "usage": {"prompt_tokens": 64, "completion_tokens": 16},
        "timings": {"prompt_ms": 400.0, "predicted_ms": 800.0}
    }"#;
    let short_measurement = r#"{
        "usage": {"prompt_tokens": 64, "completion_tokens": 7},
        "timings": {"prompt_ms": 400.0, "predicted_ms": 350.0}
    }"#;
    let gateway = TestGateway::start(vec![
        Response::new(r#"{"status":"ok"}"#),
        Response::new(state),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(full_measurement),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(r#"{"ok":true,"cleared_slots":[0]}"#),
        Response::new(r#"{"status":"ok"}"#),
        Response::new(short_measurement),
    ]);
    let root = temp_repo_root("bench-ignore-eos-failure");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{}}}"#, gateway.port),
    )
    .expect("write config");

    let _assert = Command::cargo_bin("omniinfer")
        .expect("binary exists")
        .env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args([
            "bench",
            "run",
            "--benchmark-id",
            "fixed-length-failure",
            "--catalog-model-id",
            "qwen3-5-2b",
            "--format",
            "GGUF",
            "--quantization",
            "Q4_K_M",
            "--model-url",
            "https://example.com/qwen.gguf",
            "--device-name",
            "NVIDIA GeForce RTX 5090",
            "--soc",
            "rtx-5090",
            "--backend-version",
            "test-runtime-1",
            "--build-command",
            "bash build.sh",
            "--baseline",
            "--runs",
            "3",
            "--warmup-runs",
            "0",
            "--max-tokens",
            "16",
            "--submitter-name",
            "OmniInfer Test",
            "--ignore-eos",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "measured run 2 with --ignore-eos returned completion_tokens=7; expected max_tokens=16",
        ));
    assert!(
        !root
            .join(".local")
            .join("benchmarks")
            .join("results")
            .join("fixed-length-failure.json")
            .exists()
    );
    gateway.join();
    fs::remove_dir_all(root).ok();
}

#[test]
fn bench_requires_an_explicit_optimization_declaration() {
    let mut command = Command::cargo_bin("omniinfer").expect("binary exists");
    command
        .args([
            "bench",
            "run",
            "--catalog-model-id",
            "qwen3-5-2b",
            "--format",
            "GGUF",
            "--quantization",
            "Q4_K_M",
            "--model-url",
            "https://example.com/qwen.gguf",
            "--device-name",
            "NVIDIA GeForce RTX 5090",
            "--soc",
            "rtx-5090",
            "--backend-version",
            "test-runtime-1",
            "--build-command",
            "bash build.sh",
            "--submitter-name",
            "OmniInfer Test",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Explicit optimization declaration is required",
        ));
}
