use super::runtime_manager::{LoadedRuntimeSummary, pick_runtime_port};
use super::*;

static TEST_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
use axum::Json;
use axum::extract::Query;
use axum::routing::{get, post};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[tokio::test]
async fn proxy_forwards_public_request_with_auth() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(
        upstream.port,
        GatewayAccessPolicy {
            api_key: "secret".to_string(),
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
    )
    .await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/health"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer secret")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let value: serde_json::Value = response.into_body().read_json().unwrap();
    assert_eq!(value["status"], "ok");
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn proxy_rejects_remote_management_request() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(
        upstream.port,
        GatewayAccessPolicy {
            api_key: "secret".to_string(),
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
    )
    .await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/shutdown"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer secret")
            .send_json(serde_json::json!({}))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(response.to_string().contains("403"));
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn chat_without_loaded_model_returns_rust_error() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(
        upstream.port,
        GatewayAccessPolicy {
            api_key: "secret".to_string(),
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
    )
    .await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/v1/chat/completions?trace=1"
        ))
        .config()
        .http_status_as_error(false)
        .build()
        .header("CF-Connecting-IP", "203.0.113.10")
        .header("Authorization", "Bearer secret")
        .send_json(serde_json::json!({
            "model": "omniinfer",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 404);
    let body: Value = response.into_body().read_json().unwrap();
    assert_eq!(body["error"]["message"], "model is not loaded: omniinfer");
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn unknown_endpoint_returns_rust_gateway_error() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/unknown"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(serde_json::json!({}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 404);
    let body: Value = response.into_body().read_json().unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("endpoint is not implemented")
    );
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn proxy_answers_options_with_cors_headers() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::options(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 204);
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .and_then(|value| value.to_str().ok()),
        Some("*")
    );
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn proxy_serves_model_catalog_without_upstream() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::get(format!(
            "http://127.0.0.1:{port}/omni/supported-models/best?system=linux"
        ))
        .call()
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let value: Value = response.into_body().read_json().unwrap();
    assert!(value.is_object());
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn proxy_serves_empty_openai_models_without_loaded_runtime() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/v1/models"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let value: Value = response.into_body().read_json().unwrap();
    assert_eq!(value["object"], "list");
    assert!(value["data"].as_array().unwrap().is_empty());
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn pure_rust_gateway_rejects_chat_without_loaded_runtime() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(serde_json::json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap()
    })
    .await
    .unwrap();
    let status = response.status();
    let body: Value = response.into_body().read_json().unwrap();
    assert_eq!(status.as_u16(), 503);
    assert_eq!(body["error"]["message"], "no model is loaded");
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn pure_rust_gateway_rejects_unloaded_chat_model() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_options(GatewayAccessPolicy::default(), None).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(serde_json::json!({
                "model": "not-loaded",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap()
    })
    .await
    .unwrap();
    let status = response.status();
    let body: Value = response.into_body().read_json().unwrap();
    assert_eq!(status.as_u16(), 404);
    assert_eq!(body["error"]["message"], "model is not loaded: not-loaded");
    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn rust_gateway_serves_small_management_endpoints() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-small-management");
    std::fs::create_dir_all(&temp).unwrap();
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;

    let thinking = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/thinking"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let thinking_body: Value = thinking.into_body().read_json().unwrap();
    assert_eq!(thinking_body["default_enabled"], false);

    let selected = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/thinking/select"))
            .send_json(json!({"enabled": true}))
            .unwrap()
    })
    .await
    .unwrap();
    let selected_body: Value = selected.into_body().read_json().unwrap();
    assert_eq!(selected_body["default_enabled"], true);
    assert_eq!(
        local_state::load_state()
            .unwrap()
            .default_thinking
            .unwrap_or(false),
        true
    );

    let props = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/backend/props"))
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let props_body: Value = props.into_body().read_json().unwrap();
    assert_eq!(props_body, json!({}));

    let deprecated = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/models"))
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(deprecated.to_string().contains("410"));

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn proxy_shutdown_stops_gateway_after_upstream_success() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/shutdown"))
            .send_json(serde_json::json!({}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(gateway.wait_stopped().await);
    upstream.stop().await;
}

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
    let cache_ram_values = launch_command
        .windows(2)
        .filter(|args| args[0] == "--cache-ram")
        .map(|args| args[1])
        .collect::<Vec<_>>();
    assert_eq!(cache_ram_values, vec!["8192", "2048"]);

    let chat_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .send_json(json!({
                "model": "local",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false,
                "think": false
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

#[test]
fn cuda_picker_prefers_lowest_idle_index() {
    let mut devices = parse_cuda_gpu_rows(
        "\
0, GPU-a, 128
1, GPU-b, 64
2, GPU-c, 0
",
    );
    apply_cuda_process_rows(&mut devices, "GPU-a, 1001\n");

    let choice = select_cuda_device_from_usage(&devices).unwrap();

    assert_eq!(choice.index, "1");
    assert_eq!(choice.warning, None);
}

#[test]
fn cuda_picker_warns_and_uses_least_loaded_when_all_busy() {
    let mut devices = parse_cuda_gpu_rows(
        "\
0, GPU-a, 900
1, GPU-b, 256
2, GPU-c, 512
",
    );
    apply_cuda_process_rows(&mut devices, "GPU-a, 1001\nGPU-b, 1002\nGPU-c, 1003\n");

    let choice = select_cuda_device_from_usage(&devices).unwrap();

    assert_eq!(choice.index, "1");
    assert!(
        choice
            .warning
            .as_deref()
            .unwrap()
            .contains("all CUDA GPUs appear to be in use")
    );
}

#[test]
fn cuda_picker_allows_driver_memory_when_no_compute_process() {
    let mut devices = parse_cuda_gpu_rows(
        "\
0, GPU-a, 512
1, GPU-b, 128
",
    );
    apply_cuda_process_rows(&mut devices, "GPU-a, 1001\n");

    let choice = select_cuda_device_from_usage(&devices).unwrap();

    assert_eq!(choice.index, "1");
    assert_eq!(choice.warning, None);
}

#[test]
fn cuda_picker_detects_explicit_multi_gpu_args() {
    assert!(uses_explicit_cuda_device_args(&[
        "--tensor-split".to_string(),
        "1,1".to_string()
    ]));
    assert!(uses_explicit_cuda_device_args(&[
        "--main-gpu=1".to_string(),
        "-ngl".to_string(),
        "999".to_string()
    ]));
    assert!(!uses_explicit_cuda_device_args(&[
        "-ngl".to_string(),
        "999".to_string()
    ]));
}

#[test]
fn gpu_status_parses_devices_and_owners() {
    let mut devices = parse_gpu_status_rows(
        "\
0, GPU-a, NVIDIA GeForce RTX 3090, 24576, 12000, 12576, 91
1, GPU-b, NVIDIA GeForce RTX 3090, 24576, 1, 24258, 0
",
    );
    let loaded = vec![LoadedRuntimeSummary {
        id: "qwen3.5-35b-a3b-q4_k_m".to_string(),
        owner_admin_id: Some("adminA".to_string()),
        backend_pid: 4242,
    }];
    let processes = parse_gpu_process_rows(
        "\
GPU-a, 4242, llama-server, 11998
GPU-a, 5151, python, 256
",
        &loaded,
    );

    apply_gpu_process_rows(&mut devices, processes);

    assert_eq!(devices.len(), 2);
    assert_eq!(devices[0].memory_total_mib, 24576);
    assert_eq!(devices[0].utilization_gpu_percent, Some(91));
    assert_eq!(devices[0].processes.len(), 2);
    assert_eq!(
        devices[0].processes[0].owner_model.as_deref(),
        Some("qwen3.5-35b-a3b-q4_k_m")
    );
    assert_eq!(
        devices[0].processes[0].owner_admin_id.as_deref(),
        Some("adminA")
    );
    assert_eq!(devices[0].processes[0].owner_type, "admin");
    assert_eq!(
        devices[0].processes[0].owner_name.as_deref(),
        Some("adminA")
    );
    assert_eq!(devices[0].processes[0].display_name, "llama-server");
    assert_eq!(devices[0].processes[1].owner_model, None);
    assert_eq!(devices[0].processes[1].owner_type, "user");
    assert!(devices[1].processes.is_empty());
}

#[test]
fn gpu_status_payload_uses_numeric_indexes() {
    let device = GpuStatusDevice {
        index: "7".to_string(),
        uuid: "GPU-x".to_string(),
        name: "NVIDIA GeForce RTX 3090".to_string(),
        memory_total_mib: 24576,
        memory_used_mib: 1,
        memory_free_mib: 24258,
        utilization_gpu_percent: Some(0),
        processes: Vec::new(),
    };

    let payload = gpu_status_device_payload(&device);

    assert_eq!(payload["index"], json!(7));
    assert_eq!(payload["memory_total_mb"], json!(24576));
    assert!(payload["processes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn rust_gateway_rejects_embedded_model_loads() {
    let Some(backend_id) = embedded_test_backend_id() else {
        return;
    };
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-embedded");
    std::fs::create_dir_all(&temp).unwrap();
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;

    let load_response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(json!({
                "backend": backend_id,
                "model": "embedded-demo",
                "ctx_size": 512
            }))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(load_response.status().as_u16(), 400);
    let load_body: Value = load_response.into_body().read_json().unwrap();
    assert!(
        load_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("embedded backend")
    );
    assert!(
        load_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Python control-plane fallback has been removed")
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn anthropic_stream_response_emits_before_backend_finishes() {
    let (tx, rx) = mpsc::channel::<Result<HyperBytes, std::io::Error>>(4);
    let backend_body = Body::from_stream(ReceiverStream::new(rx));
    let response = anthropic_stream_response(backend_body, "claude-compatible".to_string());
    let mut body = response.into_body();

    tx.send(Ok(HyperBytes::from_static(
        b"data: {\"choices\":[{\"delta\":{\"content\":\"fake\"}}]}\n\n",
    )))
    .await
    .unwrap();

    let first = tokio::time::timeout(Duration::from_millis(300), body.frame())
        .await
        .expect("first Anthropic stream frame should arrive before backend completes")
        .expect("body frame")
        .expect("body frame ok")
        .into_data()
        .expect("data frame");
    let first_text = String::from_utf8(first.to_vec()).unwrap();
    assert!(first_text.contains("event: message_start"));

    tx.send(Ok(HyperBytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\" backend\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
        )))
        .await
        .unwrap();
    tx.send(Ok(HyperBytes::from_static(b"data: [DONE]\n\n")))
        .await
        .unwrap();
    drop(tx);
    let rest = body.collect().await.unwrap().to_bytes();
    let rest_text = String::from_utf8(rest.to_vec()).unwrap();
    assert!(rest_text.contains("\"text\":\"fake\""));
    assert!(rest_text.contains("\"text\":\" backend\""));
    assert!(rest_text.contains("event: message_stop"));
}

#[tokio::test]
async fn rust_gateway_discovers_model_directory_artifacts() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-artifacts");
    let model_dir = temp.join("models").join("vision-model");
    let nested = model_dir.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    let model = nested.join("model.gguf");
    let mmproj = model_dir.join("mmproj-F16.gguf");
    std::fs::write(&model, "").unwrap();
    std::fs::write(&mmproj, "").unwrap();
    let backend_id = external_test_backend_id();
    install_fake_llama_server(&temp, backend_id);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;

    let load_response = tokio::task::spawn_blocking({
        let model_dir = model_dir.clone();
        move || {
            ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                .send_json(json!({
                    "backend": backend_id,
                    "model": model_dir.display().to_string(),
                    "ctx_size": 512
                }))
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert_eq!(load_response.status().as_u16(), 200);
    let load_body: Value = load_response.into_body().read_json().unwrap();
    assert_eq!(
        load_body["selected_model"].as_str().unwrap(),
        model.display().to_string()
    );
    assert_eq!(
        load_body["selected_mmproj"].as_str().unwrap(),
        mmproj.display().to_string()
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn public_models_requires_admin_key_for_remote_clients() {
    let temp = temp_root("rust-gateway-public-models-auth");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let denied = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("401"));

    let allowed = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(allowed.status().as_u16(), 200);
    let body: Value = allowed.into_body().read_json().unwrap();
    assert_eq!(body["data"][0]["id"], "qwen3.5-4b-q4_k_m");

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn remote_backend_management_requires_admin_key_and_validates_backend() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(
        upstream.port,
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
    )
    .await;
    let port = gateway.port;

    let denied = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/advisor/system"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("401"));

    let advisor = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/advisor/system"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(advisor.status().as_u16(), 200);
    let advisor_body: Value = advisor.into_body().read_json().unwrap();
    assert_eq!(advisor_body["object"], "advisor.system");

    let invalid = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/backend/install"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .send_json(json!({"backend": "not-a-real-backend"}))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(invalid.to_string().contains("400"));

    let status = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/backend/install"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let status_body: Value = status.into_body().read_json().unwrap();
    assert_eq!(status_body["status"], "idle");

    gateway.stop().await;
    upstream.stop().await;
}

#[tokio::test]
async fn backend_install_endpoint_completes_verified_prebuilt_install() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-backend-install");
    let backend = install_test_backend_id();
    let catalog = write_gateway_prebuilt_fixture(&temp, backend);
    let _root_guard = EnvGuard::set("OMNIINFER_RUST_REPO_ROOT", temp.display().to_string());
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());
    let _catalog_guard = EnvGuard::set("OMNIINFER_PREBUILT_CATALOG", catalog.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(
        upstream.port,
        GatewayAccessPolicy {
            admin_api_keys: vec![
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminA".to_string(),
                    key: "admin-a".to_string(),
                },
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminB".to_string(),
                    key: "admin-b".to_string(),
                },
            ],
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
    )
    .await;
    let port = gateway.port;
    let backend_for_start = backend.to_string();
    let started = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/backend/install"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin-a")
            .send_json(json!({"backend": backend_for_start}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(started.status().as_u16(), 202);

    let mut final_snapshot = Value::Null;
    for _ in 0..100 {
        final_snapshot = tokio::task::spawn_blocking(move || {
            ureq::get(format!("http://127.0.0.1:{port}/omni/backend/install"))
                .call()
                .unwrap()
                .into_body()
                .read_json::<Value>()
                .unwrap()
        })
        .await
        .unwrap();
        if matches!(
            final_snapshot.get("status").and_then(Value::as_str),
            Some("completed" | "failed" | "cancelled")
        ) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(final_snapshot["status"], "completed", "{final_snapshot}");
    assert_eq!(final_snapshot["latest_event"]["event"], "completed");
    assert!(gateway_installed_launcher(&temp, backend).is_file());

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn backend_install_endpoint_cancels_before_runtime_activation() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-backend-install-cancel");
    std::fs::create_dir_all(&temp).unwrap();
    let backend = install_test_backend_id();
    let archive_server = spawn_slow_archive_server().await;
    let catalog = write_gateway_remote_prebuilt_catalog(&temp, backend, archive_server.port);
    let _root_guard = EnvGuard::set("OMNIINFER_RUST_REPO_ROOT", temp.display().to_string());
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());
    let _catalog_guard = EnvGuard::set("OMNIINFER_PREBUILT_CATALOG", catalog.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(
        upstream.port,
        GatewayAccessPolicy {
            admin_api_keys: vec![
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminA".to_string(),
                    key: "admin-a".to_string(),
                },
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminB".to_string(),
                    key: "admin-b".to_string(),
                },
            ],
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
    )
    .await;
    let port = gateway.port;
    let backend_for_start = backend.to_string();
    let started: Value = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/backend/install"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin-a")
            .send_json(json!({"backend": backend_for_start}))
            .unwrap()
            .into_body()
            .read_json()
            .unwrap()
    })
    .await
    .unwrap();
    let task_id = started["task_id"].as_str().unwrap().to_string();

    let mut observed_download = false;
    for _ in 0..100 {
        let snapshot = backend_install_status(port).await;
        if snapshot
            .get("latest_event")
            .and_then(|event| event.get("event"))
            .and_then(Value::as_str)
            .is_some_and(|event| matches!(event, "download_started" | "download_progress"))
        {
            observed_download = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(observed_download, "install never entered download state");
    let task_id_for_denied = task_id.clone();
    let denied = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/omni/backend/install/cancel"
        ))
        .header("CF-Connecting-IP", "203.0.113.10")
        .header("Authorization", "Bearer admin-b")
        .send_json(json!({"task_id": task_id_for_denied}))
        .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("403"));

    let stale_task = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/omni/backend/install/cancel"
        ))
        .header("CF-Connecting-IP", "203.0.113.10")
        .header("Authorization", "Bearer admin-a")
        .send_json(json!({"task_id": "stale-task-id"}))
        .unwrap_err()
    })
    .await
    .unwrap();
    assert!(stale_task.to_string().contains("409"));

    let cancel = tokio::task::spawn_blocking(move || {
        ureq::post(format!(
            "http://127.0.0.1:{port}/omni/backend/install/cancel"
        ))
        .header("CF-Connecting-IP", "203.0.113.10")
        .header("Authorization", "Bearer admin-a")
        .send_json(json!({"task_id": task_id}))
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(cancel.status().as_u16(), 202);

    let mut final_snapshot = Value::Null;
    for _ in 0..200 {
        final_snapshot = backend_install_status(port).await;
        if final_snapshot.get("status").and_then(Value::as_str) == Some("cancelled") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(final_snapshot["status"], "cancelled", "{final_snapshot}");
    assert!(!gateway_installed_launcher(&temp, backend).is_file());

    gateway.stop().await;
    upstream.stop().await;
    archive_server.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn remote_public_model_select_resolves_model_id() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-public-model-select");
    let root = temp.join("public_models");
    let model_path = write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let response = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin")
            .send_json(json!({"model": "qwen3.5-4b-q4_k_m"}))
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let body: Value = response.into_body().read_json().unwrap();
    assert_eq!(body["selected_model"], model_path.display().to_string());
    assert_eq!(body["selected_backend"], external_test_backend_id());
    assert_eq!(body["selected_ctx_size"], 512);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn vllm_public_model_rewrites_to_served_model_name() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-vllm-public-model");
    let root = temp.join("public_models");
    write_vllm_public_model_manifest(&root, "gelab-zero-4b-preview");
    install_fake_vllm_server(&temp);
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());
    let _stream_guard = EnvGuard::set("OMNIINFER_VLLM_NONSTREAM_VIA_STREAM", "0".to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let load = remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "gelab-zero-4b-preview"}),
    )
    .await;
    assert_eq!(load["selected_backend"], "vllm-linux-cuda");
    assert_eq!(load["model"], "gelab-zero-4b-preview");

    let chat = remote_chat(port, "inference", "gelab-zero-4b-preview").await;
    assert_eq!(chat["model_echo"], "local");
    assert_eq!(chat["usage"]["prompt_tokens"], 3);
    assert_eq!(chat["usage"]["completion_tokens"], 2);
    assert_eq!(chat["usage"]["total_tokens"], 5);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn vllm_nonstream_can_be_aggregated_from_stream() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-vllm-nonstream-via-stream");
    let root = temp.join("public_models");
    write_vllm_public_model_manifest(&root, "gelab-zero-4b-preview");
    install_fake_vllm_server(&temp);
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let load = remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "gelab-zero-4b-preview"}),
    )
    .await;
    assert_eq!(load["selected_backend"], "vllm-linux-cuda");

    let chat = remote_chat(port, "inference", "gelab-zero-4b-preview").await;
    assert_eq!(chat["object"], "chat.completion");
    assert_eq!(chat["model"], "local");
    assert_eq!(chat["choices"][0]["message"]["content"], "fake backend");
    assert_eq!(chat["choices"][0]["finish_reason"], "stop");
    assert_eq!(chat["usage"]["prompt_tokens"], 3);
    assert_eq!(chat["usage"]["completion_tokens"], 2);
    assert_eq!(chat["usage"]["total_tokens"], 5);
    assert_eq!(chat["omniinfer_metrics"]["mode"], "nonstream_via_stream");
    assert!(chat["omniinfer_metrics"]["latency_ms"].as_u64().is_some());
    assert!(
        chat["omniinfer_metrics"]
            .get("observed_decode_tps")
            .is_some()
    );

    let denied = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/request-history"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("401") || denied.to_string().contains("403"));

    let history = wait_for_history(port, "admin", "gelab-zero-4b-preview").await;
    assert_eq!(history["data"][0]["model"], "gelab-zero-4b-preview");
    assert_eq!(history["data"][0]["backend"], "vllm-linux-cuda");
    assert_eq!(history["data"][0]["status"], 200);
    assert_eq!(
        history["data"][0]["response"]["choices"][0]["message"]["content"],
        "fake backend"
    );
    assert_eq!(
        history["data"][0]["response"]["omniinfer_metrics"]["mode"],
        "nonstream_via_stream"
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn request_history_summarizes_image_payloads() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-request-history-images");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "vision-test-model");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "vision-test-model"}),
    )
    .await;

    tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .send_json(json!({
                "model": "vision-test-model",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ]
                }],
                "stream": false
            }))
            .unwrap();
    })
    .await
    .unwrap();

    let history = wait_for_history(port, "admin", "vision-test-model").await;
    assert_eq!(
        history["data"][0]["request"]["messages"][0]["content"][1]["image_url"]["url"]["omitted"],
        "data_url"
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn request_history_captures_streaming_chat_response() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-request-history-stream");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "qwen3.5-4b-q4_k_m"}),
    )
    .await;

    let stream_text = tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .send_json(json!({
                "model": "qwen3.5-4b-q4_k_m",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }))
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let mut text = String::new();
        response
            .into_body()
            .into_reader()
            .read_to_string(&mut text)
            .unwrap();
        text
    })
    .await
    .unwrap();
    assert!(stream_text.contains("fake"));
    assert!(stream_text.contains("[DONE]"));

    let history = wait_for_history(port, "admin", "qwen3.5-4b-q4_k_m").await;
    assert_eq!(history["data"][0]["status"], 200);
    assert_eq!(
        history["data"][0]["response"]["choices"][0]["message"]["content"],
        "fake backend"
    );
    assert_eq!(history["data"][0]["usage"]["prompt_tokens"], 3);
    assert_eq!(history["data"][0]["usage"]["completion_tokens"], 2);
    assert_eq!(
        history["data"][0]["response"]["omniinfer_metrics"]["mode"],
        "stream_passthrough"
    );
    assert_eq!(
        history["data"][0]["response"]["omniinfer_metrics"]["response_truncated"],
        false
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[test]
fn normalizes_openai_usage_total_tokens() {
    let mut payload = json!({
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7}
    });
    normalize_openai_usage(&mut payload);
    assert_eq!(payload["usage"]["total_tokens"], 18);
}

#[test]
fn keeps_existing_openai_usage_total_tokens() {
    let mut payload = json!({
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 99}
    });
    normalize_openai_usage(&mut payload);
    assert_eq!(payload["usage"]["total_tokens"], 99);
}

#[tokio::test]
async fn remote_admins_can_load_multiple_models_with_owner_unload_policy() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-multi-model-owner");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-35b-a3b-q4_k_m");
    write_public_model_manifest(&root, "gemma-4-e4b-it-q4_k_m");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_keys: vec![
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminA".to_string(),
                    key: "admin-a".to_string(),
                },
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminB".to_string(),
                    key: "admin-b".to_string(),
                },
            ],
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let qwen_load = remote_admin_post(
        port,
        "/omni/model/load",
        "admin-a",
        json!({"model": "qwen3.5-35b-a3b-q4_k_m"}),
    )
    .await;
    assert_eq!(qwen_load["model"], "qwen3.5-35b-a3b-q4_k_m");
    assert_eq!(qwen_load["owner_admin_id"], "adminA");

    let gemma_load = remote_admin_post(
        port,
        "/omni/model/load",
        "admin-b",
        json!({"model": "gemma-4-e4b-it-q4_k_m"}),
    )
    .await;
    assert_eq!(gemma_load["model"], "gemma-4-e4b-it-q4_k_m");
    assert_eq!(gemma_load["owner_admin_id"], "adminB");
    assert_ne!(qwen_load["backend_port"], gemma_load["backend_port"]);

    let denied = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/unload"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin-b")
            .send_json(json!({"model": "qwen3.5-35b-a3b-q4_k_m"}))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("403"));

    let qwen_chat = remote_chat(port, "inference", "qwen3.5-35b-a3b-q4_k_m").await;
    assert_eq!(qwen_chat["model_echo"], "qwen3.5-35b-a3b-q4_k_m");
    let gemma_chat = remote_chat(port, "inference", "gemma-4-e4b-it-q4_k_m").await;
    assert_eq!(gemma_chat["model_echo"], "gemma-4-e4b-it-q4_k_m");

    let missing_chat = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .send_json(json!({
                "model": "not-loaded-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(missing_chat.to_string().contains("404"));

    let default_chat = remote_chat_without_model(port, "inference").await;
    assert_eq!(default_chat["model_echo"], Value::Null);

    let loaded = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/loaded-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin-a")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let loaded_body: Value = loaded.into_body().read_json().unwrap();
    assert_eq!(loaded_body["data"].as_array().unwrap().len(), 2);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn gateway_hot_reloads_admin_keys_file() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-admin-keys-file");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "old-admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let before = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer alice-key")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(before.to_string().contains("401"));

    let config_dir = temp.join(".local").join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("admin_keys.json"),
        r#"{"keys":{"alice":"alice-key","bob":"bob-key"}}"#,
    )
    .unwrap();

    let after = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer alice-key")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(after.status().as_u16(), 200);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

struct TestServer {
    port: u16,
    stop: Option<oneshot::Sender<()>>,
    stopped: Arc<AtomicBool>,
}

impl TestServer {
    async fn stop(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn wait_stopped(&self) -> bool {
        for _ in 0..40 {
            if self.stopped.load(Ordering::SeqCst) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }
}

async fn spawn_test_upstream() -> TestServer {
    let (tx, rx) = oneshot::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_for_task = Arc::clone(&stopped);
    let app = axum::Router::new()
        .route(
            "/health",
            get(|| async { axum::Json(json!({"status": "ok"})) }),
        )
        .route("/v1/chat/completions", post(echo_chat_completion))
        .route("/omni/model/select", post(echo_model_select))
        .route(
            "/omni/shutdown",
            post(|| async { axum::Json(json!({"ok": true})) }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
        stopped_for_task.store(true, Ordering::SeqCst);
    });
    TestServer {
        port,
        stop: Some(tx),
        stopped,
    }
}

async fn spawn_slow_archive_server() -> TestServer {
    let (tx, rx) = oneshot::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_for_task = Arc::clone(&stopped);
    let app = axum::Router::new().route(
        "/runtime.zip",
        get(|| async {
            let (chunk_tx, chunk_rx) = mpsc::channel::<Result<HyperBytes, std::io::Error>>(1);
            tokio::spawn(async move {
                for _ in 0..1024 {
                    if chunk_tx
                        .send(Ok(HyperBytes::from(vec![0_u8; 64 * 1024])))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            });
            Response::builder()
                .header(CONTENT_LENGTH, (64_u64 * 1024 * 1024).to_string())
                .body(Body::from_stream(ReceiverStream::new(chunk_rx)))
                .unwrap()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
        stopped_for_task.store(true, Ordering::SeqCst);
    });
    TestServer {
        port,
        stop: Some(tx),
        stopped,
    }
}

async fn echo_chat_completion(
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(json!({
        "trace": query.get("trace").cloned().unwrap_or_default(),
        "auth": header_text(&headers, "authorization").unwrap_or_default(),
        "body": body,
    }))
}

async fn echo_model_select(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(json!({
        "ok": true,
        "delegated": true,
        "selected_backend": body.get("backend").cloned().unwrap_or(Value::Null),
        "selected_model": body.get("model").cloned().unwrap_or(Value::Null),
        "body": body,
    }))
}

async fn spawn_test_gateway(_unused_port: u16, access_policy: GatewayAccessPolicy) -> TestServer {
    spawn_test_gateway_with_public_root(access_policy, None).await
}

async fn spawn_test_gateway_with_public_root(
    access_policy: GatewayAccessPolicy,
    public_model_root: Option<PathBuf>,
) -> TestServer {
    spawn_test_gateway_with_options(access_policy, public_model_root).await
}

async fn spawn_test_gateway_with_options(
    access_policy: GatewayAccessPolicy,
    public_model_root: Option<PathBuf>,
) -> TestServer {
    spawn_test_gateway_with_runtime_options(
        access_policy,
        public_model_root,
        Duration::from_secs(120),
    )
    .await
}

async fn spawn_test_gateway_with_runtime_timeout(
    access_policy: GatewayAccessPolicy,
    runtime_startup_timeout: Duration,
) -> TestServer {
    spawn_test_gateway_with_runtime_options(access_policy, None, runtime_startup_timeout).await
}

async fn spawn_test_gateway_with_runtime_options(
    access_policy: GatewayAccessPolicy,
    public_model_root: Option<PathBuf>,
    runtime_startup_timeout: Duration,
) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let (tx, rx) = oneshot::channel();
    let stopped = Arc::new(AtomicBool::new(false));
    let stopped_for_task = Arc::clone(&stopped);
    tokio::spawn(async move {
        tokio::select! {
            result = run_gateway(GatewayConfig {
                listen_host: "127.0.0.1".to_string(),
                listen_port: port,
                runtime_startup_timeout,
                access_policy,
                public_model_root,
            }) => {
                let _ = result;
            }
            _ = rx => {}
        }
        stopped_for_task.store(true, Ordering::SeqCst);
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    TestServer {
        port,
        stop: Some(tx),
        stopped,
    }
}

async fn remote_admin_post(
    port: u16,
    path: &'static str,
    key: &'static str,
    payload: Value,
) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}{path}"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .send_json(payload)
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

async fn backend_install_status(port: u16) -> Value {
    tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/backend/install"))
            .call()
            .unwrap()
            .into_body()
            .read_json::<Value>()
            .unwrap()
    })
    .await
    .unwrap()
}

async fn remote_chat(port: u16, key: &'static str, model: &'static str) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .send_json(json!({
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

async fn remote_chat_without_model(port: u16, key: &'static str) -> Value {
    tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .send_json(json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap();
        response.into_body().read_json().unwrap()
    })
    .await
    .unwrap()
}

async fn wait_for_history(port: u16, key: &'static str, model: &'static str) -> Value {
    for _ in 0..20 {
        let value = tokio::task::spawn_blocking(move || {
            let response = ureq::get(format!(
                "http://127.0.0.1:{port}/omni/request-history?limit=5&model={model}"
            ))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", &format!("Bearer {key}"))
            .call()
            .unwrap();
            response.into_body().read_json::<Value>().unwrap()
        })
        .await
        .unwrap();
        if value
            .get("data")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        {
            return value;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out waiting for request history");
}

fn temp_root(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}"))
}

fn external_test_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-cpu"
    } else {
        "llama.cpp-linux-cuda"
    }
}

fn install_test_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-cpu"
    } else {
        "llama.cpp-linux"
    }
}

fn gateway_runtime_platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

fn write_gateway_prebuilt_fixture(root: &Path, backend: &str) -> PathBuf {
    let fixture = root.join("gateway-prebuilt-fixture");
    let payload = fixture.join("payload").join("runtime").join("bin");
    std::fs::create_dir_all(&payload).unwrap();
    let launcher_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let launcher = payload.join(launcher_name);
    std::fs::write(&launcher, "#!/usr/bin/env bash\nexit 0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&launcher, permissions).unwrap();
    }

    let archive = fixture.join(if cfg!(windows) {
        "runtime.zip"
    } else {
        "runtime.tar.gz"
    });
    #[cfg(windows)]
    {
        let file = std::fs::File::create(&archive).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file(format!("runtime/bin/{launcher_name}"), options)
            .unwrap();
        std::io::Write::write_all(&mut zip, b"test launcher").unwrap();
        zip.finish().unwrap();
    }
    #[cfg(not(windows))]
    {
        let file = std::fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut tar = tar::Builder::new(encoder);
        tar.append_dir_all(".", fixture.join("payload")).unwrap();
        tar.finish().unwrap();
    }
    let archive_sha = format!("{:x}", Sha256::digest(std::fs::read(&archive).unwrap()));
    let catalog = fixture.join("catalog.json");
    std::fs::write(
        &catalog,
        json!({
            "schema_version": 2,
            "platforms": {
                gateway_runtime_platform(): {
                    backend: {
                        "source": "gateway-test",
                        "tag": "test",
                        "url": format!("file://{}", archive.display()),
                        "archive": if cfg!(windows) { "zip" } else { "tar.gz" },
                        "launcher": launcher_name,
                        "sha256": archive_sha,
                        "submodule_path": "framework/llama.cpp",
                        "submodule_commit": "fixture"
                    }
                }
            }
        })
        .to_string(),
    )
    .unwrap();
    catalog
}

fn write_gateway_remote_prebuilt_catalog(root: &Path, backend: &str, port: u16) -> PathBuf {
    let catalog = root.join("slow-prebuilt-catalog.json");
    let launcher_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    std::fs::write(
        &catalog,
        json!({
            "schema_version": 2,
            "platforms": {
                gateway_runtime_platform(): {
                    backend: {
                        "source": "gateway-cancel-test",
                        "tag": "test",
                        "url": format!("http://127.0.0.1:{port}/runtime.zip"),
                        "archive": "zip",
                        "launcher": launcher_name,
                        "sha256": "0".repeat(64),
                        "submodule_path": "framework/llama.cpp",
                        "submodule_commit": "fixture"
                    }
                }
            }
        })
        .to_string(),
    )
    .unwrap();
    catalog
}

fn gateway_installed_launcher(root: &Path, backend: &str) -> PathBuf {
    root.join(".local")
        .join("runtime")
        .join(gateway_runtime_platform())
        .join(backend)
        .join("bin")
        .join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        })
}

fn embedded_test_backend_id() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("mlx-mac")
    } else if cfg!(target_os = "linux") {
        Some("mnn-linux")
    } else {
        None
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

fn write_public_model_manifest(root: &std::path::Path, id: &str) -> PathBuf {
    let dir = root.join(id);
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    std::fs::write(&model, b"gguf").unwrap();
    std::fs::write(
        dir.join("omni-model.json"),
        format!(
            r#"{{
                    "id": "{id}",
                    "display_name": "Qwen3.5 4B Q4_K_M",
                    "backend": "{}",
                    "model": "model.gguf",
                    "ctx_size": 512,
                    "modalities": ["text"],
                    "quant": "Q4_K_M",
                    "launch_args": ["-ngl", "999"]
                }}"#,
            external_test_backend_id()
        ),
    )
    .unwrap();
    model
}

#[cfg(target_os = "linux")]
fn write_vllm_public_model_manifest(root: &std::path::Path, id: &str) -> PathBuf {
    let dir = root.join(id);
    let model = dir.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(
        model.join("config.json"),
        r#"{"max_position_embeddings":32768}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("omni-model.json"),
        format!(
            r#"{{
                    "id": "{id}",
                    "display_name": "GELab Zero 4B Preview",
                    "backend": "vllm-linux-cuda",
                    "model": "model",
                    "ctx_size": 32768,
                    "modalities": ["text", "vision"],
                    "quant": "BF16",
                    "launch_args": ["--served-model-name", "local"]
                }}"#
        ),
    )
    .unwrap();
    model
}

fn install_fake_llama_server(root: &std::path::Path, backend_id: &str) {
    let launcher_name = if cfg!(target_os = "windows") {
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
        .join(launcher_name);
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    #[cfg(windows)]
    {
        install_fake_llama_server_windows(&launcher);
    }
    #[cfg(not(windows))]
    {
        let script = r#"#!/usr/bin/env bash
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
delay_file="$(dirname "$0")/startup-delay-ms"
delay_ms="$(cat "$delay_file" 2>/dev/null || printf 0)"
python3 - "$port" "$delay_ms" <<'PY'
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])
time.sleep(int(sys.argv[2]) / 1000)

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
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()
    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()
    def do_PUT(self):
        self._json({"ok": True})
    def do_DELETE(self):
        self._json({"ok": True})
    def do_PATCH(self):
        self._json({"ok": True})
    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok"})
        elif self.path.startswith("/props"):
            self._json({"n_ctx": 512, "slots": 1})
        else:
            self._json({"ok": True})
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode() or "{}")
        if self.path.startswith("/tokenize"):
            self._json({"tokens": [1, 2, 3], "echo": payload})
            return
        if self.path.startswith("/detokenize"):
            self._json({"content": "hello", "echo": payload})
            return
        if self.path.startswith("/slots/0"):
            self._json({"ok": True})
            return
        if self.path.startswith("/v1/chat/completions") and payload.get("stream") is True:
            frames = [
                'data: {"choices":[{"delta":{"content":"fake"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" backend"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                'data: [DONE]\n\n',
            ]
            raw = "".join(frames).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "model_echo": payload.get("model"),
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#;
        std::fs::write(&launcher, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&launcher, permissions).unwrap();
        }
    }
}

#[cfg(target_os = "linux")]
fn install_fake_vla_server(root: &std::path::Path, backend_id: &str) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("vla-server");
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    std::fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
set -euo pipefail
bind=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bind) bind="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'vla-server: bound to %s. ready.\n' "$bind"
python3 - "$bind" <<'PY'
import socket
import sys

endpoint = sys.argv[1].removeprefix("tcp://")
host, port = endpoint.rsplit(":", 1)
with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen()
    while True:
        connection, _ = server.accept()
        connection.close()
PY
"#,
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&launcher, permissions).unwrap();
}

#[cfg(target_os = "linux")]
fn install_fake_vllm_server(root: &std::path::Path) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join("vllm-linux-cuda")
        .join("bin")
        .join("vllm");
    std::fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    std::fs::write(
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
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._json({"ok": True})
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode() or "{}")
        if self.path.startswith("/v1/chat/completions") and payload.get("stream") is True:
            assert payload.get("stream_options", {}).get("include_usage") is True
            frames = [
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"local","choices":[{"index":0,"delta":{"role":"assistant","content":"fake"},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"local","choices":[{"index":0,"delta":{"content":" backend"},"finish_reason":"stop"}]}\n\n',
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"local","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                'data: [DONE]\n\n',
            ]
            raw = "".join(frames).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "model_echo": payload.get("model"),
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&launcher, permissions).unwrap();
    }
}

#[cfg(windows)]
fn install_fake_llama_server_windows(launcher: &std::path::Path) {
    let source = launcher.with_file_name("fake-llama-server.rs");
    let code = r##"
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

fn main() {
    let mut port = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--port" {
            port = args.next().unwrap_or_default();
        }
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Ok(raw) = std::fs::read_to_string(executable.with_file_name("startup-delay-ms")) {
            if let Ok(delay_ms) = raw.trim().parse::<u64>() {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
    }
    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).unwrap();
    for stream in listener.incoming().flatten() {
        handle(stream);
    }
}

fn handle(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let body = String::from_utf8_lossy(&body);
    let payload = response_payload(&request_line, &body);
    write_response(&mut stream, &payload.0, payload.1);
}

fn response_payload(request_line: &str, body: &str) -> (String, &'static str) {
    if request_line.starts_with("GET /health") {
        return (r#"{"status":"ok"}"#.to_string(), "application/json");
    }
    if request_line.starts_with("GET /props") {
        return (r#"{"n_ctx":512,"slots":1}"#.to_string(), "application/json");
    }
    if request_line.starts_with("POST /tokenize") {
        return (
            r#"{"tokens":[1,2,3],"echo":{"content":"hello"}}"#.to_string(),
            "application/json",
        );
    }
    if request_line.starts_with("POST /detokenize") {
        return (
            r#"{"content":"hello","echo":{"tokens":[1,2,3]}}"#.to_string(),
            "application/json",
        );
    }
    if request_line.starts_with("POST /slots/0") {
        return (r#"{"ok":true}"#.to_string(), "application/json");
    }
    if request_line.starts_with("POST /v1/chat/completions") && wants_stream(body) {
        return (
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"fake\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" backend\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n"
            )
            .to_string(),
            "text/event-stream",
        );
    }
    if request_line.starts_with("POST /v1/chat/completions") {
        let model = extract_json_string(body, "model")
            .map(|value| format!(r#""{value}""#))
            .unwrap_or_else(|| "null".to_string());
        return (
            format!(
                r#"{{"choices":[{{"message":{{"content":"fake backend"}},"finish_reason":"stop"}}],"model_echo":{model},"usage":{{"prompt_tokens":3,"completion_tokens":2}}}}"#
            ),
            "application/json",
        );
    }
    (r#"{"ok":true}"#.to_string(), "application/json")
}

fn wants_stream(body: &str) -> bool {
    let compact: String = body.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact.contains(r#""stream":true"#)
}

fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!(r#""{key}""#);
    let start = body.find(&needle)?;
    let after_key = &body[start + needle.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let value = after_colon.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

fn write_response(stream: &mut TcpStream, body: &str, content_type: &str) {
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.as_bytes().len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(body.as_bytes());
}
"##;
    std::fs::write(&source, code).unwrap();
    let status = std::process::Command::new("rustc")
        .arg("--edition=2021")
        .arg(&source)
        .arg("-o")
        .arg(launcher)
        .status()
        .expect("compile fake llama-server.exe");
    assert!(status.success(), "failed to compile fake llama-server.exe");
}

struct EnvGuard {
    key: &'static str,
    old: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: String) -> Self {
        let old = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, old }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(old) = &self.old {
                std::env::set_var(self.key, old);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}
