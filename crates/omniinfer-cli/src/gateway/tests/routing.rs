use super::*;

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
async fn proxy_serves_exact_catalog_bytes_without_upstream() {
    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
    let port = gateway.port;
    let response = tokio::task::spawn_blocking(move || {
        ureq::get(format!(
            "http://127.0.0.1:{port}/omni/supported-models?system=mac"
        ))
        .call()
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let value: Value = response.into_body().read_json().unwrap();
    let model = &value["llama.cpp-mac"]["Qwen3.5"]["Qwen3.5-4B"];
    let quant = &model["quantization"]["Q4_K_M"];
    assert_eq!(quant["size_bytes"], serde_json::json!(2_740_937_888_u64));
    assert_eq!(
        model["vision"]["size_bytes"],
        serde_json::json!(672_423_616_u64)
    );
    assert_eq!(
        quant["bundle_size_bytes"],
        serde_json::json!(3_413_361_504_u64)
    );
    assert_eq!(quant["required_memory_gib"], serde_json::json!(3.18));
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
