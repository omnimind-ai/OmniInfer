use super::*;

pub(super) async fn should_handle_rust_endpoint(
    state: &GatewayState,
    method: &Method,
    path: &str,
) -> bool {
    match (method, path) {
        (
            &Method::GET,
            "/health"
            | "/omni/state"
            | "/omni/backends"
            | "/omni/thinking"
            | "/omni/models"
            | "/omni/gpus"
            | "/omni/loaded-models",
        ) => true,
        (&Method::GET, "/omni/backend/props") => true,
        (&Method::GET, "/omni/public-models") => true,
        (&Method::GET, path) if request_history_path(path) => true,
        (&Method::GET, "/omni/supported-models" | "/omni/supported-models/best" | "/v1/models") => {
            true
        }
        (&Method::POST, "/omni/shutdown") => true,
        (
            &Method::POST,
            "/omni/backend/select"
            | "/omni/backend/stop"
            | "/omni/model/clear-selection"
            | "/omni/model/select"
            | "/omni/model/load"
            | "/omni/model/unload",
        ) => true,
        (&Method::POST, "/omni/thinking/select") => true,
        (&Method::POST, "/v1/chat/completions" | "/v1/messages") => true,
        (&Method::GET | &Method::POST, path) if path.starts_with("/sdcpp/v1/") => true,
        (
            &Method::POST,
            "/tokenize" | "/detokenize" | "/omni/tokenize" | "/omni/detokenize"
            | "/omni/cache/clear",
        ) => state.runtime.lock().await.has_loaded_runtime(),
        _ => false,
    }
}

pub(super) async fn try_handle_rust_endpoint(
    state: &GatewayState,
    path: &str,
    auth: GatewayAuthDecision,
    request: Request<Body>,
) -> Result<Option<Response<Body>>> {
    match (request.method(), path) {
        (&Method::GET, "/health") => {
            let deep = request
                .uri()
                .query()
                .map(|query| query.contains("deep=true") || query.contains("deep=1"))
                .unwrap_or(false);
            let snapshot = state.runtime.lock().await.snapshot();
            let mut payload = json!({
                "status": "ok",
                "omni": snapshot,
            });
            if deep {
                payload["backend_health"] = backend_health(&snapshot);
            }
            Ok(Some(json_response(StatusCode::OK, payload)))
        }
        (&Method::GET, "/omni/state") => {
            let mut payload = state.runtime.lock().await.snapshot();
            payload["available_backends"] =
                BackendRegistry::load_current().api_payload(BackendScope::All)["data"].clone();
            Ok(Some(json_response(StatusCode::OK, payload)))
        }
        (&Method::GET, "/omni/backends") => {
            let scope = request
                .uri()
                .query()
                .and_then(|query| {
                    query.split('&').find_map(|part| {
                        let (key, value) = part.split_once('=')?;
                        (key == "scope").then_some(value)
                    })
                })
                .unwrap_or("installed");
            let scope = match scope {
                "installed" => BackendScope::Installed,
                "compatible" => BackendScope::Compatible,
                "all" => BackendScope::All,
                other => {
                    return Ok(Some(json_response(
                        StatusCode::BAD_REQUEST,
                        json!({"error": {"message": format!("invalid scope: {other}. Must be one of: installed, compatible, all")}}),
                    )));
                }
            };
            Ok(Some(json_response(
                StatusCode::OK,
                BackendRegistry::load_current().api_payload(scope),
            )))
        }
        (&Method::GET, "/omni/thinking") => Ok(Some(json_response(
            StatusCode::OK,
            json!({"default_enabled": default_thinking_enabled()}),
        ))),
        (&Method::GET, "/omni/backend/props") => {
            let target = state.runtime.lock().await.proxy_base_for_model(None);
            let Some(target) = target else {
                return Ok(Some(json_response(StatusCode::OK, json!({}))));
            };
            let response = proxy_get_to_runtime(&state.client, &format!("{target}/props")).await?;
            Ok(Some(response))
        }
        (&Method::GET, "/omni/loaded-models") => {
            let payload = state.runtime.lock().await.loaded_models_payload();
            Ok(Some(json_response(StatusCode::OK, payload)))
        }
        (&Method::GET, "/omni/gpus") => {
            let loaded = state.runtime.lock().await.loaded_runtime_summaries();
            match query_nvidia_smi_gpu_status(&loaded) {
                Ok(devices) => Ok(Some(json_response(
                    StatusCode::OK,
                    gpu_status_payload(&devices),
                ))),
                Err(error) => Ok(Some(json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"error": {"message": error.to_string()}}),
                ))),
            }
        }
        (&Method::GET, "/omni/models") => Ok(Some(json_response(
            StatusCode::GONE,
            json!({"error": {"message": "GET /omni/models has been deprecated and is no longer maintained"}}),
        ))),
        (&Method::GET, "/omni/public-models") => {
            match public_models::list_public_models(state.public_model_root.as_deref()) {
                Ok(entries) => Ok(Some(json_response(
                    StatusCode::OK,
                    public_models::public_models_payload(&entries),
                ))),
                Err(error) => Ok(Some(json_response(
                    public_model_error_status(&error),
                    json!({"error": {"message": error.to_string()}}),
                ))),
            }
        }
        (&Method::GET, path) if request_history_path(path) => {
            if auth.admin_id.is_none() {
                return Ok(Some(json_response(
                    StatusCode::FORBIDDEN,
                    json!({"error": {"message": "request history requires an admin key"}}),
                )));
            }
            if let Some(id) = path.strip_prefix("/omni/request-history/") {
                let history_dir = state.request_history_dir.clone();
                let id = id.to_string();
                let lookup_id = id.clone();
                let result = tokio::task::spawn_blocking(move || {
                    request_history::get_record(&history_dir, &lookup_id)
                })
                .await??;
                return Ok(Some(match result {
                    Some(entry) => json_response(StatusCode::OK, entry),
                    None => json_response(
                        StatusCode::NOT_FOUND,
                        json!({"error": {"message": format!("request history entry not found: {id}")}}),
                    ),
                }));
            }
            let query = query_from_pairs(query_pairs(request.uri()));
            let history_dir = state.request_history_dir.clone();
            let payload = tokio::task::spawn_blocking(move || {
                request_history::query_records(&history_dir, query)
            })
            .await??;
            Ok(Some(json_response(StatusCode::OK, payload)))
        }
        (&Method::GET, "/omni/supported-models") => {
            let system = query_value(request.uri(), "system").unwrap_or_else(current_system_name);
            match model_catalog::list_supported_models(&system) {
                Ok(payload) => Ok(Some(json_response(StatusCode::OK, payload))),
                Err(error) => Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"message": error.to_string()}}),
                ))),
            }
        }
        (&Method::GET, "/omni/supported-models/best") => {
            let system = query_value(request.uri(), "system").unwrap_or_else(current_system_name);
            match model_catalog::list_supported_models_best(&system) {
                Ok(payload) => Ok(Some(json_response(StatusCode::OK, payload))),
                Err(error) => Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"message": error.to_string()}}),
                ))),
            }
        }
        (&Method::GET, "/v1/models") => {
            let loaded = state.runtime.lock().await.loaded_models_payload();
            let data = loaded
                .get("data")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|item| {
                    let id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("omniinfer")
                        .to_string();
                    json!({
                        "id": id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "omniinfer",
                        "permission": [],
                        "root": id,
                        "parent": null,
                    })
                })
                .collect::<Vec<_>>();
            Ok(Some(json_response(
                StatusCode::OK,
                json!({"object": "list", "data": data}),
            )))
        }
        (&Method::POST, "/omni/backend/select") => {
            let body = request.into_body().collect().await?.to_bytes();
            let payload: Value = serde_json::from_slice(&body)?;
            let Some(backend_id) = payload
                .get("backend")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
            else {
                return Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"message": "field 'backend' is required"}}),
                )));
            };
            let result = state.runtime.lock().await.select_backend(backend_id);
            Ok(Some(json_response(StatusCode::OK, result?)))
        }
        (&Method::POST, "/omni/backend/stop") => {
            let result = tokio::task::spawn_blocking({
                let runtime = Arc::clone(&state.runtime);
                move || {
                    let handle = tokio::runtime::Handle::current();
                    handle.block_on(async move { runtime.lock().await.stop_runtime() })
                }
            })
            .await??;
            Ok(Some(json_response(StatusCode::OK, result)))
        }
        (&Method::POST, "/omni/model/clear-selection") => {
            let mut runtime = state.runtime.lock().await;
            let selection_cleared = local_state::clear_selected_model()?;
            let snapshot = runtime.snapshot();
            Ok(Some(json_response(
                StatusCode::OK,
                json!({
                    "ok": true,
                    "selection_cleared": selection_cleared,
                    "backend_ready": snapshot["backend_ready"],
                    "current_model": snapshot["model"],
                    "restore_selection": snapshot["restore_selection"],
                    "restore_status": snapshot["restore_status"],
                    "restore_completed": snapshot["restore_completed"],
                }),
            )))
        }
        (&Method::POST, "/omni/shutdown") => {
            state.startup_cancelled.store(true, Ordering::SeqCst);
            let result = tokio::task::spawn_blocking({
                let runtime = Arc::clone(&state.runtime);
                move || {
                    let handle = tokio::runtime::Handle::current();
                    handle.block_on(async move { runtime.lock().await.stop_runtime() })
                }
            })
            .await??;
            if let Some(sender) = state.shutdown.lock().await.take() {
                let _ = sender.send(());
            }
            Ok(Some(json_response(
                StatusCode::OK,
                json!({"ok": true, "runtime": result}),
            )))
        }
        (&Method::POST, "/omni/thinking/select") => {
            let body = request.into_body().collect().await?.to_bytes();
            let payload: Value = serde_json::from_slice(&body)?;
            let raw_enabled = payload.get("enabled").or_else(|| payload.get("think"));
            let Some(raw_enabled) = raw_enabled else {
                return Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"message": "field 'enabled' is required"}}),
                )));
            };
            let enabled = match omniinfer_core::request_normalization::parse_boolish(raw_enabled) {
                Ok(enabled) => enabled,
                Err(error) => {
                    return Ok(Some(json_response(
                        StatusCode::BAD_REQUEST,
                        json!({"error": {"message": error.to_string()}}),
                    )));
                }
            };
            local_state::save_default_thinking(enabled)?;
            Ok(Some(json_response(
                StatusCode::OK,
                json!({"ok": true, "default_enabled": enabled}),
            )))
        }
        (&Method::POST, "/omni/model/select" | "/omni/model/load") => {
            let body = request.into_body().collect().await?.to_bytes();
            let mut payload: Value = serde_json::from_slice(&body)?;
            if payload
                .get("request_defaults")
                .is_some_and(|defaults| !defaults.is_object())
            {
                return Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"message": "request_defaults must be an object"}}),
                )));
            }
            if let Err(error) = normalize_public_model_select(&mut payload, state, auth.remote) {
                return Ok(Some(json_response(
                    public_model_error_status(&error),
                    json!({"error": {"message": error.to_string()}}),
                )));
            }
            {
                let runtime = state.runtime.lock().await;
                let requested_backend = runtime.resolve_requested_backend(&payload)?;
                let registry = BackendRegistry::load_current();
                let backend = registry
                    .get(&requested_backend)
                    .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
                if backend.runtime_mode == "embedded" {
                    return Ok(Some(json_response(
                        StatusCode::BAD_REQUEST,
                        json!({"error": {"message": format!("{} is an embedded backend. Python control-plane fallback has been removed; use an external-server backend or a backend adapter service.", backend.id)}}),
                    )));
                }
            };
            let backend_host = state.backend_host.clone();
            let runtime_startup_timeout = state.runtime_startup_timeout;
            let runtime = Arc::clone(&state.runtime);
            let startup_cancelled = Arc::clone(&state.startup_cancelled);
            let outcome = tokio::task::spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(async move {
                    runtime.lock().await.load_model(
                        payload,
                        backend_host,
                        runtime_startup_timeout,
                        auth.admin_id.clone(),
                        &startup_cancelled,
                    )
                })
            })
            .await??;
            let (status, result) = match outcome {
                LoadModelOutcome::Success(result) => (StatusCode::OK, result),
                LoadModelOutcome::ReloadRequired(result) => (StatusCode::CONFLICT, result),
            };
            Ok(Some(json_response(status, result)))
        }
        (&Method::POST, "/omni/model/unload") => {
            let body = request.into_body().collect().await?.to_bytes();
            let payload: Value = serde_json::from_slice(&body)?;
            let Some(model) = payload
                .get("model")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
            else {
                return Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"message": "field 'model' is required"}}),
                )));
            };
            match state
                .runtime
                .lock()
                .await
                .unload_model(model, auth.admin_id.as_deref())
            {
                Ok(result) => Ok(Some(json_response(StatusCode::OK, result))),
                Err(error) => Ok(Some(json_response(
                    StatusCode::FORBIDDEN,
                    json!({"error": {"message": error.to_string()}}),
                ))),
            }
        }
        (&Method::POST, "/v1/chat/completions") => {
            let body = request.into_body().collect().await?.to_bytes();
            let raw_payload: Value = serde_json::from_slice(&body)?;
            let requested_model = raw_payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
            let target = {
                let mut runtime = state.runtime.lock().await;
                runtime.proxy_target_for_model(requested_model.as_deref())
            };
            let Some(target) = target else {
                let message = requested_model
                    .as_deref()
                    .map(|model| format!("model is not loaded: {model}"))
                    .unwrap_or_else(|| "no model is loaded".to_string());
                return Ok(Some(json_response(
                    if requested_model.is_some() {
                        StatusCode::NOT_FOUND
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE
                    },
                    json!({"error": {"message": message}}),
                )));
            };
            if !target.protocol.supports_chat() {
                return Ok(Some(backend_protocol_not_supported(
                    &target,
                    "/v1/chat/completions",
                )));
            }
            let mut normalized_payload = match normalize_chat_request_with_defaults(
                raw_payload.clone(),
                &target.request_defaults,
                default_thinking_enabled(),
            ) {
                Ok(payload) => payload,
                Err(error) => {
                    return Ok(Some(json_response(
                        StatusCode::BAD_REQUEST,
                        json!({"error": {"message": error.to_string()}}),
                    )));
                }
            };
            let Some(base_url) = target.base_url.as_deref() else {
                return Ok(Some(backend_protocol_not_supported(
                    &target,
                    "/v1/chat/completions",
                )));
            };
            let response_model = requested_model
                .clone()
                .unwrap_or_else(|| "omniinfer".to_string());
            let stream_requested = normalized_payload
                .payload
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            apply_proxy_model(&mut normalized_payload.payload, target.model.as_deref());
            let started_at = Instant::now();
            let history_context = StreamHistoryContext {
                state: state.clone(),
                admin_id: auth.admin_id.clone(),
                auth_kind: auth_kind(&auth),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: requested_model.clone(),
                backend: Some(target.backend_id.clone()),
                request: raw_payload.clone(),
                response_model: response_model.clone(),
                started_at,
            };
            let (response, captured_response, status, history_deferred) =
                if should_proxy_vllm_nonstream_via_stream(&target.backend_id, stream_requested) {
                    let (payload, status) = proxy_openai_nonstream_via_stream(
                        &state.client,
                        &format!("{base_url}/v1/chat/completions"),
                        normalized_payload.payload,
                        &response_model,
                    )
                    .await?;
                    (
                        json_response(status, payload.clone()),
                        Some(payload),
                        status,
                        false,
                    )
                } else {
                    proxy_openai_chat_to_runtime(
                        &state.client,
                        &format!("{base_url}/v1/chat/completions"),
                        HyperBytes::from(serde_json::to_vec(&normalized_payload.payload)?),
                        Some(history_context.clone()),
                    )
                    .await?
                };
            if !history_deferred {
                record_request_history(
                    &state,
                    RequestHistoryRecord {
                        admin_id: history_context.admin_id,
                        auth_kind: history_context.auth_kind,
                        method: history_context.method,
                        path: history_context.path,
                        model: history_context.model,
                        backend: history_context.backend,
                        status: status.as_u16(),
                        latency_ms: duration_ms(started_at.elapsed()),
                        usage: captured_response
                            .as_ref()
                            .and_then(|payload| payload.get("usage").cloned()),
                        metrics: captured_response
                            .as_ref()
                            .and_then(|payload| payload.get("omniinfer_metrics").cloned()),
                        request: raw_payload,
                        response: captured_response,
                        error: (status.as_u16() >= 400)
                            .then(|| format!("HTTP {}", status.as_u16())),
                    },
                );
            }
            Ok(Some(response))
        }
        (&Method::POST, "/tokenize" | "/detokenize" | "/omni/tokenize" | "/omni/detokenize") => {
            let body = request.into_body().collect().await?.to_bytes();
            let operation = if path.ends_with("detokenize") {
                "detokenize"
            } else {
                "tokenize"
            };
            let target = state.runtime.lock().await.proxy_base_for_model(None);
            let Some(target) = target else {
                return Ok(None);
            };
            let response =
                proxy_body_to_runtime(&state.client, &format!("{target}/{operation}"), body)
                    .await?;
            Ok(Some(response))
        }
        (&Method::POST, "/omni/cache/clear") => {
            let target = state.runtime.lock().await.proxy_base_for_model(None);
            let Some(target) = target else {
                return Ok(None);
            };
            let response = clear_runtime_cache(&state.client, &target).await?;
            Ok(Some(response))
        }
        (&Method::POST, "/v1/messages") => {
            let body = request.into_body().collect().await?.to_bytes();
            let payload: Value = serde_json::from_slice(&body)?;
            let messages = payload.get("messages").and_then(Value::as_array);
            if !messages.is_some_and(|messages| !messages.is_empty()) {
                return Ok(Some(json_response(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"type": "invalid_request_error", "message": "messages is required"}}),
                )));
            }
            let response_model = payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
            let openai_payload = anthropic_request_to_openai(&payload);
            let mut target = {
                let mut runtime = state.runtime.lock().await;
                runtime.proxy_target_for_model(response_model.as_deref())
            };
            if target.is_none() && response_model.is_some() {
                target = state.runtime.lock().await.proxy_target_for_model(None);
            }
            let Some(target) = target else {
                return Ok(Some(json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"error": {"message": "no model is loaded"}}),
                )));
            };
            if !target.protocol.supports_chat() {
                return Ok(Some(backend_protocol_not_supported(
                    &target,
                    "/v1/messages",
                )));
            }
            let mut normalized = match normalize_chat_request_with_defaults(
                openai_payload,
                &target.request_defaults,
                default_thinking_enabled(),
            ) {
                Ok(payload) => payload,
                Err(error) => {
                    return Ok(Some(json_response(
                        StatusCode::BAD_REQUEST,
                        json!({"error": {"message": error.to_string()}}),
                    )));
                }
            };
            let Some(base_url) = target.base_url.as_deref() else {
                return Ok(Some(backend_protocol_not_supported(
                    &target,
                    "/v1/messages",
                )));
            };
            let response_model = response_model.unwrap_or_else(|| "omniinfer".to_string());
            apply_proxy_model(&mut normalized.payload, target.model.as_deref());
            let response = proxy_anthropic_to_runtime(
                &state.client,
                &format!("{base_url}/v1/chat/completions"),
                HyperBytes::from(serde_json::to_vec(&normalized.payload)?),
                &response_model,
                normalized
                    .payload
                    .get("stream")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            )
            .await?;
            Ok(Some(response))
        }
        (&Method::GET | &Method::POST, path) if path.starts_with("/sdcpp/v1/") => {
            const MAX_DIFFUSION_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;
            let target = state.runtime.lock().await.proxy_target_for_model(None);
            let Some(target) = target else {
                return Ok(Some(json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"error": {"message": "no diffusion model is loaded"}}),
                )));
            };
            if target.protocol
                != omniinfer_core::runtime_plan::ExternalServerProtocol::StableDiffusionCppServer
            {
                return Ok(Some(backend_protocol_not_supported(&target, path)));
            }
            let method = request.method().clone();
            let content_type = request.headers().get(CONTENT_TYPE).cloned();
            let path_and_query = request
                .uri()
                .path_and_query()
                .map(|value| value.as_str())
                .unwrap_or(path);
            let upstream = format!("{}{}", target.client_endpoint, path_and_query);
            let body =
                match axum::body::to_bytes(request.into_body(), MAX_DIFFUSION_REQUEST_BODY_BYTES)
                    .await
                {
                    Ok(body) => body,
                    Err(_) => {
                        return Ok(Some(json_response(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            json!({"error": {"message": "diffusion request body exceeds 16 MiB"}}),
                        )));
                    }
                };
            let response =
                proxy_passthrough_to_runtime(&state.client, method, &upstream, content_type, body)
                    .await?;
            Ok(Some(response))
        }
        _ => Ok(None),
    }
}
