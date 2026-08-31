use super::*;

#[tokio::test]
async fn rust_gateway_loads_external_runtime_and_forwards_chat() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-runtime");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, "").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();

    let load_response = tokio::task::spawn_blocking({
        let model = model.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(json!({
                    "backend": backend_id,
                    "model": model.display().to_string(),
                    "ctx_size": 512,
                    "backend_port": backend_port,
                    "launch_args": ["-np", "5", "--cache-ram", "2048"]
                }))
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(load_response.status().as_u16(), 200);
    let load_body: Value = load_response.into_body().read_json().unwrap();
    assert_eq!(load_body["selected_backend"], backend_id);
    assert_eq!(load_body["selected_ctx_size"], 512);
    assert_eq!(load_body["backend_port"], backend_port);
    assert!(load_body["backend_pid"].as_u64().unwrap() > 0);
    assert_eq!(load_body["route_state"], "ready");
    if backend_id.contains("cuda") {
        assert_eq!(load_body["runtime_placement"]["policy"], "auto");
        assert_eq!(load_body["runtime_placement"]["mode"], "partial");
        assert_eq!(load_body["runtime_placement"]["offloaded_layers"], 2);
        assert_eq!(load_body["runtime_placement"]["total_layers"], 4);
        assert!(
            load_body["runtime_placement"]["reconciled_budget"]["domains_bytes"]["host"]
                .as_u64()
                .is_some_and(|bytes| bytes > 0)
        );
        assert!(
            load_body["runtime_placement"]["reconciled_budget"]["domains_bytes"]
                .as_object()
                .is_some_and(|domains| domains.iter().any(|(domain, bytes)| {
                    domain.starts_with("cuda:") && bytes.as_u64().is_some_and(|bytes| bytes > 0)
                }))
        );
    }
    let first_generation = load_body["generation"].as_u64().unwrap();
    assert!(first_generation > 0);
    assert!(load_body["allocation_id"].as_u64().unwrap() > 0);
    assert!(
        load_body["resource_budget"]["domains_bytes"]
            .as_object()
            .is_some_and(|domains| !domains.is_empty())
    );
    let launch_command = load_body["launch_command"]
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    assert!(
        launch_command
            .windows(2)
            .any(|args| args == ["--slot-prompt-similarity", "0"])
    );
    assert!(launch_command.contains(&"--cache-idle-slots"));
    assert!(launch_command.windows(2).any(|args| args == ["-np", "5"]));
    if backend_id.contains("cuda") {
        assert!(!launch_command.iter().any(|arg| {
            matches!(*arg, "-ngl" | "--gpu-layers")
                || arg.starts_with("-ngl=")
                || arg.starts_with("--gpu-layers=")
        }));
        assert!(launch_command.windows(2).any(|args| args == ["-lv", "4"]));
    }
    let cache_ram_values = launch_command
        .windows(2)
        .filter(|args| args[0] == "--cache-ram")
        .map(|args| args[1])
        .collect::<Vec<_>>();
    assert_eq!(cache_ram_values, vec!["8192", "2048"]);

    let thinking_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/thinking/select"))
            .send_json(json!({"enabled": true}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(thinking_response.status().as_u16(), 200);

    let chat_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .send_json(json!({
                "model": "local",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(chat_response.status().as_u16(), 200);
    let chat_body: Value = chat_response.into_body().read_json().unwrap();
    assert_eq!(
        chat_body["choices"][0]["message"]["content"],
        "fake backend"
    );
    assert_eq!(chat_body["enable_thinking_echo"], true);

    let anthropic_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/messages"))
            .send_json(json!({
                "model": "claude-compatible",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(anthropic_response.status().as_u16(), 200);
    let anthropic_body: Value = anthropic_response.into_body().read_json().unwrap();
    assert_eq!(anthropic_body["type"], "message");
    assert_eq!(anthropic_body["model"], "claude-compatible");
    assert_eq!(
        anthropic_body["content"][0],
        json!({"type": "text", "text": "fake backend"})
    );

    let tokenize_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/tokenize"))
            .send_json(json!({"content": "hello", "add_special": true}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(tokenize_response.status().as_u16(), 200);
    let tokenize_body: Value = tokenize_response.into_body().read_json().unwrap();
    assert_eq!(tokenize_body["tokens"], json!([1, 2, 3]));
    assert_eq!(tokenize_body["echo"]["content"], "hello");

    let detokenize_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/detokenize"))
            .send_json(json!({"tokens": [1, 2, 3]}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(detokenize_response.status().as_u16(), 200);
    let detokenize_body: Value = detokenize_response.into_body().read_json().unwrap();
    assert_eq!(detokenize_body["content"], "hello");

    let cache_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/cache/clear"))
            .send_empty()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(cache_response.status().as_u16(), 200);
    let cache_body: Value = cache_response.into_body().read_json().unwrap();
    assert_eq!(cache_body["ok"], true);
    assert_eq!(cache_body["cache_policy"], "cleared_each_run");
    assert_eq!(cache_body["cleared_slots"], json!([0, 1]));

    let props_response = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/backend/props"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(props_response.status().as_u16(), 200);
    let props_body: Value = props_response.into_body().read_json().unwrap();
    assert_eq!(props_body["n_ctx"], 512);

    let committed_before_unload = gateway_state(port).await;
    assert!(resource_total(&committed_before_unload, "committed_bytes") > 0);

    let model_text = model.display().to_string();
    let unload_body = tokio::task::spawn_blocking({
        let model_text = model_text.clone();
        move || {
            let response = ureq::post(format!("http://127.0.0.1:{port}/omni/model/unload"))
                .send_json(json!({"model": model_text}))
                .unwrap();
            response.into_body().read_json::<Value>().unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(unload_body["invalidated_generation"], first_generation);
    assert_eq!(unload_body["resources_released"], true);

    let released = gateway_state(port).await;
    assert_eq!(resource_total(&released, "reserved_bytes"), 0);
    assert_eq!(resource_total(&released, "committed_bytes"), 0);
    assert!(released["loaded_models"].as_array().unwrap().is_empty());

    let second_backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let reload_body = tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/omni/model/load"))
            .send_json(json!({
                "backend": backend_id,
                "model": model_text,
                "ctx_size": 512,
                "backend_port": second_backend_port,
                "launch_args": ["-np", "5", "--cache-ram", "2048"]
            }))
            .unwrap();
        response.into_body().read_json::<Value>().unwrap()
    })
    .await
    .unwrap();
    assert!(reload_body["generation"].as_u64().unwrap() > first_generation);
    assert_eq!(reload_body["route_state"], "ready");

    #[cfg(unix)]
    {
        let pid = reload_body["backend_pid"].as_u64().unwrap().to_string();
        assert!(
            std::process::Command::new("kill")
                .args(["-TERM", &pid])
                .status()
                .unwrap()
                .success()
        );
        let mut reaped = false;
        for _ in 0..30 {
            let state = gateway_state(port).await;
            if state["loaded_models"].as_array().unwrap().is_empty()
                && resource_total(&state, "committed_bytes") == 0
            {
                reaped = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(reaped, "exited runtime generation should be invalidated");
    }

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::test]
async fn partial_offload_reconciliation_failure_cleans_runtime_and_ledger() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("partial-offload-reconciliation-rollback");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let placement_mode = temp
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("placement-mode");
    std::fs::write(placement_mode, "oversized").unwrap();
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/load"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "backend": backend_id,
                "model": model.display().to_string(),
                "ctx_size": 512,
                "backend_port": backend_port
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 502);
    let body: Value = response.into_body().read_json().unwrap();
    assert!(
        body.to_string()
            .contains("placement exceeds safe reconciled capacity"),
        "unexpected failure response: {body}"
    );
    assert!(body.to_string().contains("log:"));

    let state = gateway_state(port).await;
    assert_eq!(resource_total(&state, "reserved_bytes"), 0);
    assert_eq!(resource_total(&state, "committed_bytes"), 0);
    assert!(state["loaded_models"].as_array().unwrap().is_empty());
    assert_eq!(state["backend_ready"], false);
    for _ in 0..40 {
        if std::net::TcpStream::connect(("127.0.0.1", backend_port)).is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(std::net::TcpStream::connect(("127.0.0.1", backend_port)).is_err());

    gateway.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::test]
async fn explicit_full_offload_keeps_strict_cuda_admission() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("explicit-full-offload-admission");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/load"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "backend": backend_id,
                "model": model.display().to_string(),
                "ctx_size": 512,
                "backend_port": backend_port,
                "launch_args": ["-ngl", "999"],
                "resource_budget_bytes": 2_u64 * 1024 * 1024 * 1024 * 1024
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 502);

    let state = gateway_state(port).await;
    assert_eq!(resource_total(&state, "reserved_bytes"), 0);
    assert_eq!(resource_total(&state, "committed_bytes"), 0);
    assert!(state["loaded_models"].as_array().unwrap().is_empty());
    assert!(std::net::TcpStream::connect(("127.0.0.1", backend_port)).is_err());

    gateway.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::test]
async fn partial_offload_rejects_disabled_startup_logs_before_launch() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("partial-offload-disabled-logs");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/load"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "backend": backend_id,
                "model": model.display().to_string(),
                "ctx_size": 512,
                "backend_port": backend_port,
                "launch_args": ["--log-disable"]
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 502);
    let body: Value = response.into_body().read_json().unwrap();
    assert!(body.to_string().contains("remove --log-disable"));

    let state = gateway_state(port).await;
    assert_eq!(resource_total(&state, "reserved_bytes"), 0);
    assert_eq!(resource_total(&state, "committed_bytes"), 0);
    assert!(state["loaded_models"].as_array().unwrap().is_empty());
    assert!(std::net::TcpStream::connect(("127.0.0.1", backend_port)).is_err());

    gateway.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn runtime_request_defaults_merge_without_restarting_backend() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("runtime-request-defaults");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let load_request = json!({
        "backend": backend_id,
        "model": model.display().to_string(),
        "ctx_size": 512,
        "backend_port": backend_port,
        "request_defaults": {
            "max_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9
        }
    });

    let loaded = tokio::task::spawn_blocking({
        let load_request = load_request.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(load_request)
                .unwrap()
                .into_body()
                .read_json::<Value>()
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(loaded["request_defaults"]["max_tokens"], 64);
    assert_eq!(loaded["request_defaults"]["temperature"], 0.2);
    let backend_pid = loaded["backend_pid"].as_u64().unwrap();
    let generation = loaded["generation"].as_u64().unwrap();

    let state = gateway_state(port).await;
    assert_eq!(state["request_defaults"]["max_tokens"], 64);
    assert_eq!(state["loaded_models"][0]["request_defaults"]["top_p"], 0.9);
    assert_eq!(
        state["restore_selection"]["request_defaults"]["max_tokens"],
        64
    );
    assert_eq!(state["restore_status"], "loaded");

    let merged = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .send_json(json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "request_defaults": {"max_tokens": 128, "temperature": 0.4},
                "temperature": 0.7,
                "stream": false
            }))
            .unwrap()
            .into_body()
            .read_json::<Value>()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(merged["max_tokens_echo"], 128);
    assert_eq!(merged["temperature_echo"], 0.7);
    assert_eq!(merged["top_p_echo"], 0.9);

    let defaults_only = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .send_json(json!({
                "messages": [{"role": "user", "content": "Hello again"}],
                "stream": false
            }))
            .unwrap()
            .into_body()
            .read_json::<Value>()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(defaults_only["max_tokens_echo"], 64);
    assert_eq!(defaults_only["temperature_echo"], 0.2);
    assert_eq!(defaults_only["top_p_echo"], 0.9);

    let updated = tokio::task::spawn_blocking({
        let mut load_request = load_request.clone();
        load_request["request_defaults"] = json!({"max_tokens": 32});
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(load_request)
                .unwrap()
                .into_body()
                .read_json::<Value>()
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(updated["already_loaded"], true);
    assert_eq!(updated["backend_pid"], backend_pid);
    assert_eq!(updated["backend_port"], backend_port);
    assert_eq!(updated["generation"], generation);
    assert_eq!(updated["request_defaults"], json!({"max_tokens": 32}));

    let invalid_load = tokio::task::spawn_blocking(move || {
        let mut invalid = load_request;
        invalid["request_defaults"] = json!(true);
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(invalid)
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(invalid_load.status().as_u16(), 400);

    let invalid_chat = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "request_defaults": true
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(invalid_chat.status().as_u16(), 400);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn direct_gateway_no_mmproj_load_persists_and_reports_restore_choice() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("runtime-no-mmproj-persistence");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let request = json!({
        "backend": backend_id,
        "model": model.display().to_string(),
        "ctx_size": 512,
        "backend_port": backend_port,
        "no_mmproj": true,
    });
    let load = tokio::task::spawn_blocking({
        let request = request.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(request)
                .unwrap()
                .into_body()
                .read_json::<Value>()
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(load["selected_mmproj"], Value::Null);

    let state = gateway_state(port).await;
    assert_eq!(state["restore_selection"]["no_mmproj"], true);
    assert_eq!(state["restore_selection"]["mmproj"], Value::Null);
    let persisted = omniinfer_core::local_state::load_state().unwrap();
    assert!(persisted.selected_model.unwrap().no_mmproj);

    let repeat = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .send_json(json!({
                "backend": backend_id,
                "model": model.display().to_string(),
                "ctx_size": 512,
                "no_mmproj": true,
            }))
            .unwrap()
            .into_body()
            .read_json::<Value>()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(repeat["already_loaded"], true);
    assert!(
        omniinfer_core::local_state::load_state()
            .unwrap()
            .selected_model
            .unwrap()
            .no_mmproj
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn runtime_load_rolls_back_when_local_state_commit_fails() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-state-rollback");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, b"gguf").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    std::fs::write(temp.join(".local").join("config"), b"not-a-directory").unwrap();
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();

    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/load"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "backend": backend_id,
                "model": model.display().to_string(),
                "ctx_size": 512,
                "backend_port": backend_port
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 502);

    let state = gateway_state(port).await;
    assert_eq!(resource_total(&state, "reserved_bytes"), 0);
    assert_eq!(resource_total(&state, "committed_bytes"), 0);
    assert!(state["loaded_models"].as_array().unwrap().is_empty());
    assert!(std::net::TcpStream::connect(("127.0.0.1", backend_port)).is_err());

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn vla_runtime_exposes_zmq_contract_and_rejects_openai_proxying() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("vla-runtime-protocol-contract");
    let model = temp.join("smolvla.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, "GGUF").unwrap();
    install_fake_vla_server(&temp, "vla.cpp-linux");
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();

    let load = tokio::task::spawn_blocking({
        let model = model.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(json!({
                    "backend": "vla.cpp-linux",
                    "model": model.display().to_string(),
                    "backend_port": backend_port,
                }))
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(load.status().as_u16(), 200);
    let load: Value = load.into_body().read_json().unwrap();
    assert_eq!(load["external_server_protocol"], "vla.cpp-zmq-server");
    assert_eq!(
        load["client_endpoint"],
        format!("tcp://127.0.0.1:{backend_port}")
    );
    assert_eq!(load["openai_compatible"], false);

    let state = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/state"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let state: Value = state.into_body().read_json().unwrap();
    assert_eq!(state["external_server_protocol"], "vla.cpp-zmq-server");
    assert_eq!(state["openai_compatible"], false);
    assert_eq!(
        state["runtime"]["client_endpoint"],
        format!("tcp://127.0.0.1:{backend_port}")
    );

    for endpoint in ["/v1/chat/completions", "/v1/messages"] {
        let response = tokio::task::spawn_blocking(move || {
            ureq::post(format!("http://127.0.0.1:{port}{endpoint}"))
                .config()
                .http_status_as_error(false)
                .build()
                .send_json(json!({
                    "model": "omniinfer",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Hello"}],
                }))
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(response.status().as_u16(), 422);
        let body: Value = response.into_body().read_json().unwrap();
        assert_eq!(body["error"]["code"], "backend_protocol_not_supported");
        assert_eq!(
            body["error"]["external_server_protocol"],
            "vla.cpp-zmq-server"
        );
        assert_eq!(
            body["error"]["client_endpoint"],
            format!("tcp://127.0.0.1:{backend_port}")
        );
    }

    let diffusion_response = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/sdcpp/v1/capabilities"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(diffusion_response.status().as_u16(), 422);
    let body: Value = diffusion_response.into_body().read_json().unwrap();
    assert_eq!(body["error"]["code"], "backend_protocol_not_supported");

    gateway.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn diffusion_runtime_proxies_native_async_api_and_rejects_chat() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("diffusion-runtime-protocol-contract");
    std::fs::create_dir_all(&temp).unwrap();
    let model = temp.join("minimax_h3_fl2va_pruned-Q4_K.gguf");
    let llm = temp.join("qwen3vl_32b_minimax_h3-Q4_K_M.gguf");
    let vae = temp.join("minimax_h3_video_vae_fp16.safetensors");
    let audio_vae = temp.join("minimax_h3_audio_vae_fp32.safetensors");
    for path in [&model, &llm, &vae, &audio_vae] {
        std::fs::write(path, b"test").unwrap();
    }
    install_fake_stable_diffusion_server(&temp);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let load = tokio::task::spawn_blocking({
        let model = model.clone();
        let llm = llm.clone();
        let vae = vae.clone();
        let audio_vae = audio_vae.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(json!({
                    "backend": "stable-diffusion.cpp-linux-vulkan",
                    "model": model.display().to_string(),
                    "backend_port": backend_port,
                    "launch_args": [
                        "--llm", llm.display().to_string(),
                        "--vae", vae.display().to_string(),
                        "--audio-vae", audio_vae.display().to_string(),
                        "--cfg-scale", "1.0",
                        "--diffusion-fa",
                        "--backend", "te=cpu"
                    ]
                }))
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(load.status().as_u16(), 200);
    let load: Value = load.into_body().read_json().unwrap();
    assert_eq!(
        load["external_server_protocol"],
        "stable-diffusion.cpp-server"
    );
    assert_eq!(load["openai_compatible"], false);
    assert_eq!(
        load["client_endpoint"],
        format!("http://127.0.0.1:{backend_port}")
    );

    let capabilities = tokio::task::spawn_blocking(move || {
        ureq::get(format!(
            "http://127.0.0.1:{port}/sdcpp/v1/capabilities?detail=1"
        ))
        .call()
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(capabilities.status().as_u16(), 200);
    assert_eq!(
        capabilities
            .headers()
            .get("x-sdcpp-test")
            .and_then(|value| value.to_str().ok()),
        Some("passthrough")
    );
    let capabilities: Value = capabilities.into_body().read_json().unwrap();
    assert_eq!(capabilities["backend"], "fake-sdcpp");
    assert_eq!(capabilities["path"], "/sdcpp/v1/capabilities?detail=1");

    let submit = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/sdcpp/v1/vid_gen"))
            .send_json(json!({
                "prompt": "a silver tabby surfing",
                "width": 640,
                "height": 384,
                "video_frames": 25,
                "steps": 4
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(submit.status().as_u16(), 202);
    let submit: Value = submit.into_body().read_json().unwrap();
    assert_eq!(submit["id"], "job-42");
    assert_eq!(submit["request"]["steps"], 4);

    let poll = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/sdcpp/v1/jobs/job-42"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(poll.status().as_u16(), 200);
    let poll: Value = poll.into_body().read_json().unwrap();
    assert_eq!(poll["status"], "completed");

    let cancel = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/sdcpp/v1/jobs/job-42/cancel"
        ))
        .send_json(json!({}))
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(cancel.status().as_u16(), 200);
    let cancel: Value = cancel.into_body().read_json().unwrap();
    assert_eq!(cancel["status"], "cancelled");

    for endpoint in ["/v1/chat/completions", "/v1/messages"] {
        let chat = tokio::task::spawn_blocking(move || {
            ureq::post(format!("http://127.0.0.1:{port}{endpoint}"))
                .config()
                .http_status_as_error(false)
                .build()
                .send_json(json!({
                    "model": "omniinfer",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Hello"}]
                }))
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(chat.status().as_u16(), 422);
        let chat: Value = chat.into_body().read_json().unwrap();
        assert_eq!(chat["error"]["code"], "backend_protocol_not_supported");
        assert_eq!(
            chat["error"]["external_server_protocol"],
            "stable-diffusion.cpp-server"
        );
    }

    let too_large = tokio::task::spawn_blocking(move || {
        let payload = json!({"prompt": "x".repeat(16 * 1024 * 1024)});
        ureq::post(format!("http://127.0.0.1:{port}/sdcpp/v1/vid_gen"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(payload)
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(too_large.status().as_u16(), 413);
    let too_large: Value = too_large.into_body().read_json().unwrap();
    assert_eq!(
        too_large["error"]["message"],
        "diffusion request body exceeds 16 MiB"
    );

    gateway.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn gateway_uses_configured_runtime_startup_timeout() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-runtime-timeout");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, "").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let delay_file = temp
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("startup-delay-ms");
    std::fs::write(delay_file, "3000").unwrap();
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let gateway = spawn_test_gateway_with_runtime_timeout(
        GatewayAccessPolicy::default(),
        Duration::from_millis(100),
    )
    .await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let error = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .send_json(json!({
                "backend": backend_id,
                "model": model.display().to_string(),
                "backend_port": backend_port
            }))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(
        error.to_string().contains("502"),
        "unexpected load response: {error}"
    );

    gateway.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn model_selection_is_idempotent_and_restore_state_is_explicit() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("model-restore-contract");
    let model = temp.join("model.gguf");
    std::fs::create_dir_all(&temp).unwrap();
    std::fs::write(&model, "").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let backend_port = pick_runtime_port("127.0.0.1").unwrap();
    let request = json!({
        "backend": backend_id,
        "model": model.display().to_string(),
        "ctx_size": 512,
        "backend_port": backend_port
    });

    let first = tokio::task::spawn_blocking({
        let request = request.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(request)
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(first.status().as_u16(), 200);
    let first: Value = first.into_body().read_json().unwrap();
    assert_eq!(first["already_loaded"], false);
    assert_eq!(first["requires_reload"], false);
    let backend_pid = first["backend_pid"].as_u64().unwrap();

    let repeated = tokio::task::spawn_blocking({
        let mut request = request.clone();
        request["public_model_id"] = json!("public-model");
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(request)
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(repeated.status().as_u16(), 200);
    let repeated: Value = repeated.into_body().read_json().unwrap();
    assert_eq!(repeated["already_loaded"], true);
    assert_eq!(repeated["requires_reload"], false);
    assert_eq!(repeated["backend_pid"], backend_pid);
    assert_eq!(repeated["model"], "public-model");
    assert_eq!(repeated["selected_public_model_id"], "public-model");

    let models = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/v1/models"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let models: Value = models.into_body().read_json().unwrap();
    assert_eq!(models["data"][0]["id"], "public-model");

    let restored_state = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/state"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let restored_state: Value = restored_state.into_body().read_json().unwrap();
    assert_eq!(restored_state["restore_status"], "loaded");
    assert_eq!(restored_state["restore_completed"], true);

    let conflict = tokio::task::spawn_blocking({
        let mut request = request.clone();
        request["ctx_size"] = json!(1024);
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .config()
                .http_status_as_error(false)
                .build()
                .send_json(request)
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(conflict.status().as_u16(), 409);
    let conflict: Value = conflict.into_body().read_json().unwrap();
    assert_eq!(conflict["requires_reload"], true);
    assert_eq!(conflict["error"]["code"], "model_reload_required");
    assert_eq!(conflict["current"]["ctx_size"], 512);
    assert_eq!(conflict["requested"]["ctx_size"], 1024);

    let clear = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/omni/model/clear-selection"
        ))
        .send_json(json!({}))
        .unwrap()
    })
    .await
    .unwrap();
    let clear: Value = clear.into_body().read_json().unwrap();
    assert_eq!(clear["selection_cleared"], true);
    assert_eq!(clear["backend_ready"], true);
    assert_eq!(clear["current_model"], "public-model");
    assert_eq!(clear["restore_selection"], Value::Null);
    assert_eq!(clear["restore_status"], "not_configured");
    assert_eq!(clear["restore_completed"], false);

    let selected_again = tokio::task::spawn_blocking({
        let request = request.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(request)
                .unwrap()
        }
    })
    .await
    .unwrap();
    let selected_again: Value = selected_again.into_body().read_json().unwrap();
    assert_eq!(selected_again["already_loaded"], true);

    let stop = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/backend/stop"))
            .send_json(json!({}))
            .unwrap()
    })
    .await
    .unwrap();
    let stop: Value = stop.into_body().read_json().unwrap();
    assert_eq!(stop["selected_model_preserved"], true);
    assert_eq!(stop["restore_status"], "pending");

    let state = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/state"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let state: Value = state.into_body().read_json().unwrap();
    assert_eq!(state["backend_ready"], false);
    assert_eq!(state["restore_status"], "pending");
    assert_eq!(state["restore_completed"], false);
    assert_eq!(
        state["restore_selection"]["model"],
        model.display().to_string()
    );

    let clear_again = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/omni/model/clear-selection"
        ))
        .send_json(json!({}))
        .unwrap()
    })
    .await
    .unwrap();
    let clear_again: Value = clear_again.into_body().read_json().unwrap();
    assert_eq!(clear_again["selection_cleared"], true);
    assert_eq!(clear_again["restore_status"], "not_configured");

    let persisted: Value = serde_json::from_str(
        &std::fs::read_to_string(temp.join(".local/config/state.json")).unwrap(),
    )
    .unwrap();
    assert!(persisted.get("selected_model").is_none());
    assert!(persisted.get("selected_mmproj").is_none());
    assert!(persisted.get("selected_ctx_size").is_none());

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}
