use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap};
use axum::http::{Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use bytes::Bytes as HyperBytes;
use http_body_util::{BodyExt, Full};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use omniinfer_core::anthropic::{
    AnthropicStreamConverter, anthropic_request_to_openai, openai_response_to_anthropic,
    parse_openai_sse_events,
};
use omniinfer_core::backend_registry::{BackendRegistry, BackendScope};
use omniinfer_core::gateway_auth::{
    GatewayAccessPolicy, GatewayAuthDecision, RequestAuthContext, authorize_request_with_identity,
};
use omniinfer_core::model_catalog;
use omniinfer_core::public_models;
use omniinfer_core::request_normalization::normalize_chat_request;
use omniinfer_core::{local_state, paths};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

mod access_policy;
mod gpu_status;
mod request_history;
mod response;
mod runtime_manager;

use access_policy::DynamicAccessPolicy;
use gpu_status::{gpu_status_payload, query_nvidia_smi_gpu_status};
use request_history::{RequestHistoryRecord, query_from_pairs};
use response::{add_cors_headers, cors_response, json_response, should_forward_response_header};
use runtime_manager::{LoadModelOutcome, RustRuntimeManager};

const MAX_STREAM_HISTORY_CAPTURE_CHARS: usize = 12_000;

#[cfg(test)]
use gpu_status::{
    GpuStatusDevice, apply_cuda_process_rows, apply_gpu_process_rows, gpu_status_device_payload,
    parse_cuda_gpu_rows, parse_gpu_process_rows, parse_gpu_status_rows,
    select_cuda_device_from_usage, uses_explicit_cuda_device_args,
};

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub listen_host: String,
    pub listen_port: u16,
    pub access_policy: GatewayAccessPolicy,
    pub public_model_root: Option<PathBuf>,
}

#[derive(Clone)]
struct GatewayState {
    backend_host: String,
    access_policy: Arc<tokio::sync::Mutex<DynamicAccessPolicy>>,
    public_model_root: Option<PathBuf>,
    request_history_dir: PathBuf,
    client: Client<HttpConnector, Full<HyperBytes>>,
    shutdown: Arc<tokio::sync::Mutex<Option<oneshot::Sender<()>>>>,
    runtime: Arc<tokio::sync::Mutex<RustRuntimeManager>>,
}

pub fn run_gateway_blocking(config: GatewayConfig) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run_gateway(config))
}

pub async fn run_gateway(config: GatewayConfig) -> Result<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = GatewayState {
        backend_host: "127.0.0.1".to_string(),
        access_policy: Arc::new(tokio::sync::Mutex::new(DynamicAccessPolicy::new(
            config.access_policy,
            paths::admin_keys_file(),
        ))),
        public_model_root: config.public_model_root,
        request_history_dir: paths::local_dir().join("request_history"),
        client: Client::builder(TokioExecutor::new()).build_http(),
        shutdown: Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx))),
        runtime: Arc::new(tokio::sync::Mutex::new(RustRuntimeManager::default())),
    };
    let app = axum::Router::new()
        .fallback(proxy_request)
        .with_state(state);
    let addr: SocketAddr = format!("{}:{}", config.listen_host, config.listen_port).parse()?;
    let listener = TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let _ = shutdown_rx.await;
    })
    .await?;
    Ok(())
}

async fn proxy_request(
    State(state): State<GatewayState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request<Body>,
) -> impl IntoResponse {
    match proxy_request_inner(state, peer.ip(), request).await {
        Ok(response) => response,
        Err(error) => json_response(
            StatusCode::BAD_GATEWAY,
            json!({"error": {"message": error.to_string()}}),
        ),
    }
}

async fn proxy_request_inner(
    state: GatewayState,
    peer_ip: IpAddr,
    request: Request<Body>,
) -> Result<Response<Body>> {
    if request.method() == Method::OPTIONS {
        return Ok(cors_response(StatusCode::NO_CONTENT));
    }

    let path = request.uri().path().to_string();
    let auth_context = auth_context(&request, peer_ip);
    let access_policy = state.access_policy.lock().await.effective_policy();
    let auth = match authorize_request_with_identity(&access_policy, &auth_context) {
        Ok(auth) => auth,
        Err(error) => {
            return Ok(json_response(
                StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::FORBIDDEN),
                json!({"error": {"message": error.to_string()}}),
            ));
        }
    };

    let should_shutdown = request.method() == Method::POST && path == "/omni/shutdown";
    if should_handle_rust_endpoint(&state, request.method(), &path).await {
        let Some(response) = try_handle_rust_endpoint(&state, &path, auth, request).await? else {
            return Ok(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"error": {"message": "Rust endpoint handler declined a selected request"}}),
            ));
        };
        if should_shutdown && response.status().is_success() {
            if let Some(sender) = state.shutdown.lock().await.take() {
                let _ = sender.send(());
            }
        }
        return Ok(response);
    }

    Ok(json_response(
        StatusCode::NOT_FOUND,
        json!({"error": {"message": format!("endpoint is not implemented by the Rust gateway: {} {}", request.method(), path)}}),
    ))
}

async fn should_handle_rust_endpoint(state: &GatewayState, method: &Method, path: &str) -> bool {
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
        (
            &Method::POST,
            "/tokenize" | "/detokenize" | "/omni/tokenize" | "/omni/detokenize"
            | "/omni/cache/clear",
        ) => state.runtime.lock().await.has_loaded_runtime(),
        _ => false,
    }
}

async fn try_handle_rust_endpoint(
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
            let runtime = state.runtime.lock().await;
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
            let runtime = Arc::clone(&state.runtime);
            let outcome = tokio::task::spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(async move {
                    runtime.lock().await.load_model(
                        payload,
                        backend_host,
                        Duration::from_secs(120),
                        auth.admin_id.clone(),
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
            let mut normalized_payload = normalize_chat_request(raw_payload.clone(), false)?;
            let requested_model = normalized_payload
                .payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
            let target = {
                let runtime = state.runtime.lock().await;
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
                        &format!("{}/v1/chat/completions", target.base_url),
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
                        &format!("{}/v1/chat/completions", target.base_url),
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
            let mut normalized = normalize_chat_request(openai_payload, false)?;
            let mut target = {
                let runtime = state.runtime.lock().await;
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
            let response_model = response_model.unwrap_or_else(|| "omniinfer".to_string());
            apply_proxy_model(&mut normalized.payload, target.model.as_deref());
            let response = proxy_anthropic_to_runtime(
                &state.client,
                &format!("{}/v1/chat/completions", target.base_url),
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
        _ => Ok(None),
    }
}

async fn proxy_get_to_runtime(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    uri: &str,
) -> Result<Response<Body>> {
    let request = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Full::new(HyperBytes::new()))?;
    let response = client.request(request).await?;
    response_from_upstream(response).await
}

async fn proxy_body_to_runtime(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    uri: &str,
    body: HyperBytes,
) -> Result<Response<Body>> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(body))?;
    let response = client.request(request).await?;
    response_from_upstream(response).await
}

async fn proxy_openai_chat_to_runtime(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    uri: &str,
    body: HyperBytes,
    stream_history: Option<StreamHistoryContext>,
) -> Result<(Response<Body>, Option<Value>, StatusCode, bool)> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(body))?;
    let upstream = client.request(request).await?;
    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let streaming = content_type.contains("text/event-stream");
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers().iter() {
        if should_forward_response_header(name) {
            builder = builder.header(name, value);
        }
    }
    if streaming {
        if let Some(context) = stream_history.filter(|_| request_history::enabled()) {
            let response =
                stream_openai_chat_with_history(upstream.into_body(), builder, context, status)?;
            return Ok((response, None, status, true));
        }
        let response = builder.body(Body::new(upstream.into_body()))?;
        return Ok((response, None, status, false));
    }
    let mut body = upstream.into_body().collect().await?.to_bytes();
    let captured = if content_type.contains("application/json") {
        body = normalize_upstream_json_body(body)?;
        serde_json::from_slice::<Value>(&body).ok()
    } else {
        None
    };
    builder = builder.header(CONTENT_LENGTH, body.len().to_string());
    let mut response = builder.body(Body::from(body))?;
    add_cors_headers(response.headers_mut());
    Ok((response, captured, status, false))
}

#[derive(Clone)]
struct StreamHistoryContext {
    state: GatewayState,
    admin_id: Option<String>,
    auth_kind: String,
    method: String,
    path: String,
    model: Option<String>,
    backend: Option<String>,
    request: Value,
    response_model: String,
    started_at: Instant,
}

fn stream_openai_chat_with_history(
    upstream_body: hyper::body::Incoming,
    builder: axum::http::response::Builder,
    context: StreamHistoryContext,
    status: StatusCode,
) -> Result<Response<Body>> {
    let (tx, rx) = mpsc::channel::<Result<HyperBytes, std::io::Error>>(16);
    tokio::spawn(async move {
        let mut body = Body::new(upstream_body);
        let mut buffered = Vec::<u8>::new();
        let mut aggregate = OpenAiStreamAggregate::new(
            &context.response_model,
            context.started_at,
            "stream_passthrough",
        );
        let mut error = None::<String>;
        while let Some(frame) = body.frame().await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(frame_error) => {
                    let message = frame_error.to_string();
                    let _ = tx.send(Err(std::io::Error::other(message.clone()))).await;
                    error = Some(format!("upstream stream error: {message}"));
                    break;
                }
            };
            let Some(data) = frame.data_ref() else {
                continue;
            };
            let chunk = HyperBytes::copy_from_slice(data);
            if tx.send(Ok(chunk.clone())).await.is_err() {
                error = Some("client disconnected while streaming response".to_string());
                break;
            }
            buffered.extend_from_slice(&chunk);
            while let Some(index) = buffered.windows(2).position(|window| window == b"\n\n") {
                let event = buffered.drain(..index + 2).collect::<Vec<_>>();
                aggregate.process_sse_bytes(&event);
            }
        }
        if error.is_none() && !buffered.is_empty() {
            aggregate.process_sse_bytes(&buffered);
        }
        let payload = aggregate.finish();
        let record_error =
            error.or_else(|| (status.as_u16() >= 400).then(|| format!("HTTP {}", status.as_u16())));
        record_request_history(
            &context.state,
            RequestHistoryRecord {
                admin_id: context.admin_id,
                auth_kind: context.auth_kind,
                method: context.method,
                path: context.path,
                model: context.model,
                backend: context.backend,
                status: status.as_u16(),
                latency_ms: duration_ms(context.started_at.elapsed()),
                usage: payload.get("usage").cloned(),
                metrics: payload.get("omniinfer_metrics").cloned(),
                request: context.request,
                response: Some(payload),
                error: record_error,
            },
        );
    });
    let mut response = builder.body(Body::from_stream(ReceiverStream::new(rx)))?;
    add_cors_headers(response.headers_mut());
    Ok(response)
}

fn should_proxy_vllm_nonstream_via_stream(backend_id: &str, stream_requested: bool) -> bool {
    !stream_requested
        && backend_id.starts_with("vllm")
        && env_flag_enabled("OMNIINFER_VLLM_NONSTREAM_VIA_STREAM", true)
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

async fn proxy_openai_nonstream_via_stream(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    uri: &str,
    mut payload: Value,
    response_model: &str,
) -> Result<(Value, StatusCode)> {
    payload["stream"] = json!(true);
    ensure_stream_usage(&mut payload);
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(HyperBytes::from(serde_json::to_vec(&payload)?)))?;
    let start = Instant::now();
    let response = client.request(request).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.into_body().collect().await?.to_bytes();
        let payload = serde_json::from_slice::<Value>(&body).unwrap_or_else(
            |_| json!({"error": {"message": String::from_utf8_lossy(&body).trim().to_string()}}),
        );
        return Ok((payload, status));
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !content_type.contains("text/event-stream") {
        let body = response.into_body().collect().await?.to_bytes();
        let mut payload = serde_json::from_slice::<Value>(&body).unwrap_or_else(
            |_| json!({"error": {"message": String::from_utf8_lossy(&body).trim().to_string()}}),
        );
        normalize_openai_usage(&mut payload);
        return Ok((payload, status));
    }
    let mut aggregate = OpenAiStreamAggregate::new(response_model, start, "nonstream_via_stream");
    let mut body = Body::new(response.into_body());
    let mut buffered = Vec::<u8>::new();
    while let Some(frame) = body.frame().await {
        let frame = frame?;
        let Some(data) = frame.data_ref() else {
            continue;
        };
        buffered.extend_from_slice(data);
        while let Some(index) = buffered.windows(2).position(|window| window == b"\n\n") {
            let chunk = buffered.drain(..index + 2).collect::<Vec<_>>();
            aggregate.process_sse_bytes(&chunk);
        }
    }
    if !buffered.is_empty() {
        aggregate.process_sse_bytes(&buffered);
    }
    let mut payload = aggregate.finish();
    normalize_openai_usage(&mut payload);
    Ok((payload, StatusCode::OK))
}

fn ensure_stream_usage(payload: &mut Value) {
    let object = payload
        .as_object_mut()
        .expect("normalized chat payload should be an object");
    let stream_options = object.entry("stream_options").or_insert_with(|| json!({}));
    if !stream_options.is_object() {
        *stream_options = json!({});
    }
    stream_options["include_usage"] = json!(true);
}

#[derive(Default)]
struct AggregatedToolCall {
    id: Option<String>,
    kind: Option<String>,
    name: Option<String>,
    arguments: String,
}

struct OpenAiStreamAggregate {
    metrics_mode: &'static str,
    response_model: String,
    started_at: Instant,
    first_output_at: Option<Instant>,
    id: Option<String>,
    created: Option<u64>,
    upstream_model: Option<String>,
    system_fingerprint: Option<Value>,
    role: Option<String>,
    content: String,
    content_truncated: bool,
    reasoning_content: String,
    reasoning_truncated: bool,
    tool_calls: BTreeMap<u64, AggregatedToolCall>,
    tool_arguments_truncated: bool,
    finish_reason: Option<Value>,
    usage: Option<Value>,
}

impl OpenAiStreamAggregate {
    fn new(response_model: &str, started_at: Instant, metrics_mode: &'static str) -> Self {
        Self {
            metrics_mode,
            response_model: response_model.to_string(),
            started_at,
            first_output_at: None,
            id: None,
            created: None,
            upstream_model: None,
            system_fingerprint: None,
            role: None,
            content: String::new(),
            content_truncated: false,
            reasoning_content: String::new(),
            reasoning_truncated: false,
            tool_calls: BTreeMap::new(),
            tool_arguments_truncated: false,
            finish_reason: None,
            usage: None,
        }
    }

    fn process_sse_bytes(&mut self, bytes: &[u8]) {
        for event in parse_openai_sse_events(bytes) {
            if let Ok(value) = serde_json::from_str::<Value>(&event) {
                self.process_chunk(&value);
            }
        }
    }

    fn process_chunk(&mut self, chunk: &Value) {
        if self.id.is_none() {
            self.id = chunk.get("id").and_then(Value::as_str).map(str::to_string);
        }
        if self.created.is_none() {
            self.created = chunk.get("created").and_then(Value::as_u64);
        }
        if self.upstream_model.is_none() {
            self.upstream_model = chunk
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        if self.system_fingerprint.is_none() {
            self.system_fingerprint = chunk.get("system_fingerprint").cloned();
        }
        if let Some(usage) = chunk.get("usage")
            && !usage.is_null()
        {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return;
        };
        if let Some(reason) = choice.get("finish_reason")
            && !reason.is_null()
        {
            self.finish_reason = Some(reason.clone());
        }
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(role) = delta.get("role").and_then(Value::as_str)
            && self.role.is_none()
        {
            self.role = Some(role.to_string());
        }
        if let Some(content) = delta.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            self.mark_first_output();
            self.content_truncated |= append_limited_stream_capture(&mut self.content, content);
        }
        for key in ["reasoning_content", "reasoning"] {
            if let Some(reasoning) = delta.get(key).and_then(Value::as_str)
                && !reasoning.is_empty()
            {
                self.mark_first_output();
                self.reasoning_truncated |=
                    append_limited_stream_capture(&mut self.reasoning_content, reasoning);
            }
        }
        for tool_call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.mark_first_output();
            let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let entry = self.tool_calls.entry(index).or_default();
            if let Some(id) = tool_call.get("id").and_then(Value::as_str)
                && !id.is_empty()
            {
                entry.id = Some(id.to_string());
            }
            if let Some(kind) = tool_call.get("type").and_then(Value::as_str)
                && !kind.is_empty()
            {
                entry.kind = Some(kind.to_string());
            }
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            if let Some(name) = function.get("name").and_then(Value::as_str)
                && !name.is_empty()
            {
                entry.name = Some(name.to_string());
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                && !arguments.is_empty()
            {
                self.tool_arguments_truncated |=
                    append_limited_stream_capture(&mut entry.arguments, arguments);
            }
        }
    }

    fn finish(self) -> Value {
        let ended_at = Instant::now();
        let latency_ms = duration_ms(ended_at.duration_since(self.started_at));
        let ttft_ms = self
            .first_output_at
            .map(|instant| duration_ms(instant.duration_since(self.started_at)));
        let decode_ms = self
            .first_output_at
            .map(|instant| duration_ms(ended_at.duration_since(instant)));
        let mut usage = self.usage.unwrap_or_else(|| json!({}));
        normalize_openai_usage_object(&mut usage);
        let observed = observed_metrics(&usage, latency_ms, ttft_ms, decode_ms);
        let mut message = json!({
            "role": self.role.unwrap_or_else(|| "assistant".to_string()),
            "content": self.content,
        });
        if !self.reasoning_content.is_empty() {
            message["reasoning_content"] = json!(self.reasoning_content);
        }
        let tool_calls = self
            .tool_calls
            .into_iter()
            .map(|(index, tool)| {
                json!({
                    "index": index,
                    "id": tool.id.unwrap_or_else(|| format!("call_{index}")),
                    "type": tool.kind.unwrap_or_else(|| "function".to_string()),
                    "function": {
                        "name": tool.name.unwrap_or_default(),
                        "arguments": tool.arguments,
                    },
                })
            })
            .collect::<Vec<_>>();
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        let response_truncated =
            self.content_truncated || self.reasoning_truncated || self.tool_arguments_truncated;
        let mut payload = json!({
            "id": self.id.unwrap_or_else(make_chat_completion_id),
            "object": "chat.completion",
            "created": self.created.unwrap_or_else(unix_seconds),
            "model": self.upstream_model.unwrap_or(self.response_model),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason.unwrap_or(Value::String("stop".to_string())),
            }],
            "usage": usage,
            "omniinfer_metrics": {
                "mode": self.metrics_mode,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "decode_ms": decode_ms,
                "observed_prefill_tps": observed.prefill_tps,
                "observed_decode_tps": observed.decode_tps,
                "response_truncated": response_truncated,
            },
        });
        if let Some(fingerprint) = self.system_fingerprint {
            payload["system_fingerprint"] = fingerprint;
        }
        payload
    }

    fn mark_first_output(&mut self) {
        if self.first_output_at.is_none() {
            self.first_output_at = Some(Instant::now());
        }
    }
}

fn append_limited_stream_capture(target: &mut String, text: &str) -> bool {
    let current = target.chars().count();
    if current >= MAX_STREAM_HISTORY_CAPTURE_CHARS {
        return true;
    }
    let remaining = MAX_STREAM_HISTORY_CAPTURE_CHARS - current;
    let incoming = text.chars().count();
    if incoming <= remaining {
        target.push_str(text);
        return false;
    }
    target.extend(text.chars().take(remaining));
    true
}

struct ObservedMetrics {
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
}

fn observed_metrics(
    usage: &Value,
    _latency_ms: u64,
    ttft_ms: Option<u64>,
    decode_ms: Option<u64>,
) -> ObservedMetrics {
    let prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64);
    let completion_tokens = usage.get("completion_tokens").and_then(Value::as_u64);
    ObservedMetrics {
        prefill_tps: tokens_per_second(prompt_tokens, ttft_ms),
        decode_tps: tokens_per_second(completion_tokens, decode_ms),
    }
}

fn tokens_per_second(tokens: Option<u64>, millis: Option<u64>) -> Option<f64> {
    let tokens = tokens?;
    let millis = millis?;
    if tokens == 0 || millis == 0 {
        return None;
    }
    Some((tokens as f64) * 1000.0 / (millis as f64))
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn make_chat_completion_id() -> String {
    format!("chatcmpl-omniinfer-{}", unix_seconds())
}

fn normalize_openai_usage_object(usage: &mut Value) {
    let Some(object) = usage.as_object_mut() else {
        return;
    };
    if object.get("total_tokens").and_then(Value::as_u64).is_some() {
        return;
    }
    let Some(prompt_tokens) = object.get("prompt_tokens").and_then(Value::as_u64) else {
        return;
    };
    let Some(completion_tokens) = object.get("completion_tokens").and_then(Value::as_u64) else {
        return;
    };
    object.insert(
        "total_tokens".to_string(),
        json!(prompt_tokens.saturating_add(completion_tokens)),
    );
}

fn apply_proxy_model(payload: &mut Value, proxy_model: Option<&str>) {
    if let Some(proxy_model) = proxy_model.filter(|value| !value.trim().is_empty()) {
        payload["model"] = json!(proxy_model);
    }
}

async fn clear_runtime_cache(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    runtime_base: &str,
) -> Result<Response<Body>> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("{runtime_base}/slots/0?action=erase"))
        .body(Full::new(HyperBytes::new()))?;
    let response = client.request(request).await?;
    let status = response.status();
    let body = response.into_body().collect().await?.to_bytes();
    if status.is_success() {
        return Ok(json_response(
            StatusCode::OK,
            json!({"ok": true, "message": "KV cache cleared"}),
        ));
    }
    let detail = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| String::from_utf8_lossy(&body).trim().to_string());
    let message = if detail.to_ascii_lowercase().contains("multimodal") {
        "KV cache clear is not supported for multimodal models by llama.cpp; use /omni/backend/stop + /omni/model/select to reload instead".to_string()
    } else if detail.is_empty() {
        format!("backend slot erase failed: HTTP {}", status.as_u16())
    } else {
        format!(
            "backend slot erase failed: HTTP {} - {detail}",
            status.as_u16()
        )
    };
    Ok(json_response(
        StatusCode::CONFLICT,
        json!({"error": {"message": message}}),
    ))
}

async fn proxy_anthropic_to_runtime(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    uri: &str,
    body: HyperBytes,
    response_model: &str,
    stream: bool,
) -> Result<Response<Body>> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(body))?;
    let response = client.request(request).await?;
    let status = response.status();
    if !status.is_success() {
        return response_from_upstream(response).await;
    }
    if stream {
        let converted =
            anthropic_stream_response(Body::new(response.into_body()), response_model.to_string());
        return Ok(converted);
    }
    let body = response.into_body().collect().await?.to_bytes();
    let payload: Value = serde_json::from_slice(&body)?;
    let converted = openai_response_to_anthropic(&payload, response_model);
    Ok(json_response(StatusCode::OK, converted))
}

fn anthropic_stream_response(mut body: Body, response_model: String) -> Response<Body> {
    let (tx, rx) = mpsc::channel::<Result<HyperBytes, std::io::Error>>(16);
    tokio::spawn(async move {
        let mut converter = AnthropicStreamConverter::new(&response_model);
        for frame in converter.preamble() {
            if tx.send(Ok(HyperBytes::from(frame))).await.is_err() {
                return;
            }
        }
        let mut buffered = Vec::<u8>::new();
        while let Some(frame) = body.frame().await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(error) => {
                    let _ = tx.send(Err(std::io::Error::other(error.to_string()))).await;
                    return;
                }
            };
            let Some(data) = frame.data_ref() else {
                continue;
            };
            buffered.extend_from_slice(data);
            while let Some(index) = buffered.windows(2).position(|window| window == b"\n\n") {
                let chunk = buffered.drain(..index + 2).collect::<Vec<_>>();
                for event in parse_openai_sse_events(&chunk) {
                    if let Ok(value) = serde_json::from_str::<Value>(&event) {
                        for frame in converter.process_chunk(&value) {
                            if tx.send(Ok(HyperBytes::from(frame))).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            }
        }
        if !buffered.is_empty() {
            for event in parse_openai_sse_events(&buffered) {
                if let Ok(value) = serde_json::from_str::<Value>(&event) {
                    for frame in converter.process_chunk(&value) {
                        if tx.send(Ok(HyperBytes::from(frame))).await.is_err() {
                            return;
                        }
                    }
                }
            }
        }
        for frame in converter.epilogue() {
            if tx.send(Ok(HyperBytes::from(frame))).await.is_err() {
                return;
            }
        }
    });
    let stream = ReceiverStream::new(rx);
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(stream))
        .expect("response should build");
    add_cors_headers(response.headers_mut());
    response
}

async fn response_from_upstream(
    response: hyper::Response<hyper::body::Incoming>,
) -> Result<Response<Body>> {
    let status = response.status();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let streaming = content_type.contains("text/event-stream");
    let mut builder = Response::builder().status(status);
    for (name, value) in response.headers().iter() {
        if should_forward_response_header(name) {
            builder = builder.header(name, value);
        }
    }
    let mut response = if streaming {
        builder.body(Body::new(response.into_body()))?
    } else {
        let mut body = response.into_body().collect().await?.to_bytes();
        if content_type.contains("application/json") {
            body = normalize_upstream_json_body(body)?;
        }
        builder = builder.header(CONTENT_LENGTH, body.len().to_string());
        builder.body(Body::from(body))?
    };
    add_cors_headers(response.headers_mut());
    Ok(response)
}

fn normalize_upstream_json_body(body: HyperBytes) -> Result<HyperBytes> {
    let Ok(mut payload) = serde_json::from_slice::<Value>(&body) else {
        return Ok(body);
    };
    normalize_openai_usage(&mut payload);
    Ok(HyperBytes::from(serde_json::to_vec(&payload)?))
}

fn normalize_openai_usage(payload: &mut Value) {
    let Some(usage) = payload.get_mut("usage").and_then(Value::as_object_mut) else {
        return;
    };
    if usage.get("total_tokens").and_then(Value::as_u64).is_some() {
        return;
    }
    let Some(prompt_tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) else {
        return;
    };
    let Some(completion_tokens) = usage.get("completion_tokens").and_then(Value::as_u64) else {
        return;
    };
    usage.insert(
        "total_tokens".to_string(),
        json!(prompt_tokens.saturating_add(completion_tokens)),
    );
}

fn default_thinking_enabled() -> bool {
    local_state::load_state()
        .ok()
        .and_then(|state| state.default_thinking)
        .unwrap_or(false)
}

fn backend_health(snapshot: &Value) -> Value {
    if snapshot
        .get("backend_ready")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        json!({"status": "ok"})
    } else {
        json!({"status": "not_loaded"})
    }
}

fn normalize_public_model_select(
    payload: &mut Value,
    state: &GatewayState,
    remote_request: bool,
) -> Result<(), public_models::PublicModelError> {
    let Some(model) = payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let path_like = PathBuf::from(model).is_absolute()
        || model.starts_with("~/")
        || model.contains('/')
        || model.contains('\\');
    if remote_request && path_like {
        return Err(public_models::PublicModelError::ModelNotFound(
            model.to_string(),
        ));
    }
    if path_like || !public_models::looks_like_public_model_id(model) {
        return Ok(());
    }
    if state.public_model_root.is_none() {
        return Ok(());
    }
    let entry = public_models::resolve_public_model(state.public_model_root.as_deref(), model)?;
    let object = payload
        .as_object_mut()
        .expect("serde_json object remains object after field lookup");
    object.insert(
        "model".to_string(),
        Value::String(entry.model_path.display().to_string()),
    );
    object.insert(
        "public_model_id".to_string(),
        Value::String(entry.manifest.id.clone()),
    );
    if let Some(mmproj) = entry.mmproj_path {
        object
            .entry("mmproj".to_string())
            .or_insert_with(|| Value::String(mmproj.display().to_string()));
    }
    if let Some(backend) = entry.manifest.backend {
        object
            .entry("backend".to_string())
            .or_insert_with(|| Value::String(backend));
    }
    if let Some(ctx_size) = entry.manifest.ctx_size {
        object
            .entry("ctx_size".to_string())
            .or_insert_with(|| Value::Number(u64::from(ctx_size).into()));
    }
    if !entry.manifest.launch_args.is_empty() {
        object.entry("launch_args".to_string()).or_insert_with(|| {
            Value::Array(
                entry
                    .manifest
                    .launch_args
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            )
        });
    }
    Ok(())
}

fn public_model_error_status(error: &public_models::PublicModelError) -> StatusCode {
    match error {
        public_models::PublicModelError::RootNotConfigured => StatusCode::NOT_FOUND,
        public_models::PublicModelError::RootMissing(_) => StatusCode::SERVICE_UNAVAILABLE,
        public_models::PublicModelError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        public_models::PublicModelError::InvalidId(_)
        | public_models::PublicModelError::InvalidRelativePath(_)
        | public_models::PublicModelError::ManifestParse { .. }
        | public_models::PublicModelError::DuplicateId(_)
        | public_models::PublicModelError::ModelFileMissing(_)
        | public_models::PublicModelError::MmprojFileMissing(_)
        | public_models::PublicModelError::VisionMmprojMissing(_) => StatusCode::BAD_REQUEST,
        public_models::PublicModelError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn request_history_path(path: &str) -> bool {
    path == "/omni/request-history" || path.starts_with("/omni/request-history/")
}

fn record_request_history(state: &GatewayState, record: RequestHistoryRecord) {
    if !request_history::enabled() {
        return;
    }
    let history_dir = state.request_history_dir.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(error) = request_history::append_record(history_dir, record) {
            eprintln!("warn: failed to append request history: {error}");
        }
    });
}

fn auth_kind(auth: &GatewayAuthDecision) -> String {
    if auth.admin_id.is_some() {
        "admin".to_string()
    } else {
        "api_key".to_string()
    }
}

fn auth_context(request: &Request<Body>, peer_ip: IpAddr) -> RequestAuthContext {
    let headers = request.headers();
    RequestAuthContext {
        method: request.method().as_str().to_string(),
        path: request.uri().path().to_string(),
        client_ip: peer_ip.to_string(),
        authorization: header_text(headers, "authorization"),
        x_api_key: header_text(headers, "x-api-key"),
        cf_connecting_ip: header_text(headers, "cf-connecting-ip"),
        x_forwarded_for: header_text(headers, "x-forwarded-for"),
        x_real_ip: header_text(headers, "x-real-ip"),
    }
}

fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn query_value(uri: &Uri, key: &str) -> Option<String> {
    uri.query()?.split('&').find_map(|part| {
        let (name, value) = part.split_once('=')?;
        (name == key && !value.trim().is_empty()).then(|| value.to_string())
    })
}

fn query_pairs(uri: &Uri) -> BTreeMap<String, String> {
    uri.query()
        .into_iter()
        .flat_map(|query| query.split('&'))
        .filter_map(|part| {
            let (key, value) = part.split_once('=')?;
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

fn current_system_name() -> String {
    match std::env::consts::OS {
        "macos" => "mac".to_string(),
        "windows" => "windows".to_string(),
        _ => "linux".to_string(),
    }
}

#[cfg(test)]
mod tests;
