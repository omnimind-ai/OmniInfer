use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

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
use omniinfer_core::backend_registry;
use omniinfer_core::backend_registry::{BackendRegistry, BackendScope};
use omniinfer_core::gateway_auth::{
    GatewayAccessPolicy, GatewayAuthDecision, RequestAuthContext, authorize_request_with_identity,
};
use omniinfer_core::model_artifacts::{discover_llama_cpp_model_artifacts, maybe_auto_mmproj};
use omniinfer_core::model_catalog;
use omniinfer_core::public_models;
use omniinfer_core::request_normalization::normalize_chat_request;
use omniinfer_core::runtime_plan::{ExternalRuntimeRequest, build_external_runtime_plan};
use omniinfer_core::runtime_process::{RuntimeProcess, RuntimeProcessOptions};
use omniinfer_core::{local_state, paths};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;

mod access_policy;
mod gpu_status;
mod response;

use access_policy::DynamicAccessPolicy;
use gpu_status::{gpu_status_payload, query_nvidia_smi_gpu_status, runtime_env_for_backend};
use response::{add_cors_headers, cors_response, json_response, should_forward_response_header};

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
        (&Method::GET, "/omni/supported-models" | "/omni/supported-models/best" | "/v1/models") => {
            true
        }
        (&Method::POST, "/omni/shutdown") => true,
        (
            &Method::POST,
            "/omni/backend/select"
            | "/omni/backend/stop"
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
            let result = tokio::task::spawn_blocking(move || {
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
            Ok(Some(json_response(StatusCode::OK, result)))
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
            let normalized_payload = normalize_chat_request(serde_json::from_slice(&body)?, false)?;
            let requested_model = normalized_payload
                .payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
            let target = {
                let runtime = state.runtime.lock().await;
                runtime.proxy_base_for_model(requested_model.as_deref())
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
            let response = proxy_body_to_runtime(
                &state.client,
                &format!("{target}/v1/chat/completions"),
                HyperBytes::from(serde_json::to_vec(&normalized_payload.payload)?),
            )
            .await?;
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
            let normalized = normalize_chat_request(openai_payload, false)?;
            let mut target = {
                let runtime = state.runtime.lock().await;
                runtime.proxy_base_for_model(response_model.as_deref())
            };
            if target.is_none() && response_model.is_some() {
                target = state.runtime.lock().await.proxy_base_for_model(None);
            }
            let Some(target) = target else {
                return Ok(Some(json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"error": {"message": "no model is loaded"}}),
                )));
            };
            let response_model = response_model.unwrap_or_else(|| "omniinfer".to_string());
            let response = proxy_anthropic_to_runtime(
                &state.client,
                &format!("{target}/v1/chat/completions"),
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
        let body = response.into_body().collect().await?.to_bytes();
        builder = builder.header(CONTENT_LENGTH, body.len().to_string());
        builder.body(Body::from(body))?
    };
    add_cors_headers(response.headers_mut());
    Ok(response)
}

fn default_thinking_enabled() -> bool {
    local_state::load_state()
        .ok()
        .and_then(|state| state.default_thinking)
        .unwrap_or(false)
}

#[derive(Default)]
struct RustRuntimeManager {
    selected_backend: Option<String>,
    loaded: BTreeMap<String, LoadedRustRuntime>,
    default_model_key: Option<String>,
}

struct LoadedRustRuntime {
    model_key: String,
    owner_admin_id: Option<String>,
    backend_id: String,
    model: String,
    public_model_id: Option<String>,
    mmproj: Option<String>,
    ctx_size: Option<u32>,
    launch_args: Vec<String>,
    cuda_visible_devices: Option<String>,
    cuda_warning: Option<String>,
    process: RuntimeProcess,
    proxy_model_ref: Option<String>,
}

#[derive(Debug, Clone)]
struct LoadedRuntimeSummary {
    id: String,
    owner_admin_id: Option<String>,
    backend_pid: u32,
}

impl RustRuntimeManager {
    fn select_backend(&mut self, backend_id: &str) -> Result<Value> {
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(backend_id)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {backend_id}"))?;
        if self.selected_backend.as_deref() != Some(backend_id) {
            self.stop_runtime()?;
        }
        self.selected_backend = Some(backend_id.to_string());
        local_state::save_selected_backend(backend_id)?;
        Ok(json!({
            "ok": true,
            "selected_backend": backend_id,
            "binary_exists": backend.binary_exists(),
            "models_dir": backend.models_dir,
        }))
    }

    fn stop_runtime(&mut self) -> Result<Value> {
        for (_, mut loaded) in std::mem::take(&mut self.loaded) {
            loaded.process.stop(Duration::from_secs(8))?;
        }
        self.default_model_key = None;
        Ok(json!({
            "ok": true,
            "stopped": true,
            "selected_backend": self.selected_backend,
        }))
    }

    fn has_loaded_runtime(&self) -> bool {
        !self.loaded.is_empty()
    }

    fn load_model(
        &mut self,
        payload: Value,
        backend_host: String,
        startup_timeout: Duration,
        owner_admin_id: Option<String>,
    ) -> Result<Value> {
        let model = json_required_str(&payload, "model")?.to_string();
        let public_model_id = payload
            .get("public_model_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);
        let requested_model_key = public_model_id.clone().unwrap_or_else(|| model.clone());
        if self.loaded.contains_key(&requested_model_key) {
            anyhow::bail!("model is already loaded: {requested_model_key}");
        }
        let requested_backend = self.resolve_requested_backend(&payload)?;
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(&requested_backend)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
        if backend.runtime_mode != "external_server" {
            anyhow::bail!(
                "{} is an embedded backend. Python control-plane fallback has been removed; use an external-server backend or a backend adapter service.",
                backend.id
            );
        }
        if !backend.binary_exists() {
            anyhow::bail!(
                "backend launcher not found: {}",
                backend.launcher_path.as_deref().unwrap_or("(unset)")
            );
        }
        let resolved_model = resolve_model_for_backend(&model, backend)?;
        let explicit_mmproj = payload
            .get("mmproj")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(|value| resolve_path_for_backend(value, backend, "mmproj file"))
            .transpose()?;
        let mmproj_path = explicit_mmproj.or(resolved_model.mmproj_path).or_else(|| {
            maybe_auto_mmproj(backend.models_dir.as_deref(), &resolved_model.model_path)
        });
        if mmproj_path.is_some() && !backend.supports_mmproj {
            anyhow::bail!("{} does not support mmproj inputs", backend.id);
        }
        let ctx_size = payload
            .get("ctx_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        let launch_args = payload
            .get("launch_args")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            });
        let effective_launch_args = launch_args
            .clone()
            .unwrap_or_else(|| backend.default_args.clone());
        let port = payload
            .get("backend_port")
            .and_then(Value::as_u64)
            .filter(|value| (1..=u64::from(u16::MAX)).contains(value))
            .and_then(|value| u16::try_from(value).ok())
            .map(Ok)
            .unwrap_or_else(|| pick_runtime_port(&backend_host))?;
        let backend_payload = serde_json::to_value(backend)?;
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend: backend_payload,
            model_path: resolved_model.model_path.clone(),
            mmproj_path: mmproj_path.clone(),
            host: backend_host.clone(),
            port,
            ctx_size,
            launch_args,
        })?;
        let log_path = PathBuf::from(&backend.runtime_dir)
            .join("logs")
            .join(model_log_file_name(
                &plan.log_file_name,
                &requested_model_key,
            ));
        let (runtime_env, cuda_selection) =
            runtime_env_for_backend(backend, &effective_launch_args);
        let process = RuntimeProcess::start(
            &plan,
            RuntimeProcessOptions {
                log_path,
                env: runtime_env,
                startup_timeout,
                health_host: backend_host.clone(),
            },
        )?;
        let info = process.info().clone();
        self.selected_backend = Some(backend.id.clone());
        local_state::save_selected_backend(&backend.id)?;
        local_state::save_selected_model(
            &resolved_model.model_path,
            mmproj_path.as_deref(),
            plan.ctx_size,
        )?;
        self.loaded.insert(
            requested_model_key.clone(),
            LoadedRustRuntime {
                model_key: requested_model_key.clone(),
                owner_admin_id: owner_admin_id.clone(),
                backend_id: backend.id.clone(),
                model: resolved_model.model_path.clone(),
                public_model_id: public_model_id.clone(),
                mmproj: mmproj_path.clone(),
                ctx_size: plan.ctx_size,
                launch_args: effective_launch_args,
                cuda_visible_devices: cuda_selection
                    .as_ref()
                    .map(|selection| selection.visible_devices.clone()),
                cuda_warning: cuda_selection
                    .as_ref()
                    .and_then(|selection| selection.warning.clone()),
                proxy_model_ref: plan.proxy_model_ref.clone(),
                process,
            },
        );
        self.default_model_key = Some(requested_model_key.clone());
        let mut response = json!({
            "ok": true,
            "model": requested_model_key,
            "owner_admin_id": owner_admin_id,
            "selected_backend": backend.id,
            "selected_model": resolved_model.model_path,
            "selected_public_model_id": public_model_id,
            "selected_mmproj": mmproj_path,
            "selected_ctx_size": plan.ctx_size,
            "backend_pid": info.pid,
            "backend_port": info.port,
            "launch_command": info.command,
            "log_path": info.log_path.display().to_string(),
        });
        if let Some(selection) = cuda_selection {
            response["cuda_visible_devices"] = json!(selection.visible_devices);
            if let Some(warning) = selection.warning {
                response["warning"] = json!(warning);
            }
        }
        Ok(response)
    }

    fn unload_model(&mut self, model: &str, admin_id: Option<&str>) -> Result<Value> {
        let model_key = self
            .resolve_loaded_model_key(model)
            .ok_or_else(|| anyhow::anyhow!("model is not loaded: {model}"))?;
        let owner = self
            .loaded
            .get(&model_key)
            .and_then(|runtime| runtime.owner_admin_id.as_deref())
            .map(str::to_string);
        if let Some(owner) = owner.as_deref()
            && let Some(admin_id) = admin_id
            && owner != admin_id
        {
            anyhow::bail!(
                "model '{model_key}' is owned by admin '{owner}' and cannot be unloaded by admin '{admin_id}'"
            );
        }
        let Some(mut loaded) = self.loaded.remove(&model_key) else {
            anyhow::bail!("model is not loaded: {model}");
        };
        loaded.process.stop(Duration::from_secs(8))?;
        if self.default_model_key.as_deref() == Some(&model_key) {
            self.default_model_key = self.loaded.keys().next_back().cloned();
        }
        Ok(json!({
            "ok": true,
            "unloaded": true,
            "model": model_key,
            "owner_admin_id": owner,
        }))
    }

    fn resolve_requested_backend(&self, payload: &Value) -> Result<String> {
        payload
            .get("backend")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .or_else(|| self.selected_backend.clone())
            .or_else(|| {
                BackendRegistry::load_current()
                    .api_payload(BackendScope::Installed)
                    .get("recommended")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| anyhow::anyhow!("no installed backend available"))
    }

    fn proxy_base_for_model(&self, requested_model: Option<&str>) -> Option<String> {
        let key = match requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some("omniinfer" | "local") => self.default_model_key.clone()?,
            Some(model) => self.resolve_loaded_model_key(model)?,
            None => self.default_model_key.clone()?,
        };
        self.loaded
            .get(&key)
            .map(|loaded| format!("http://127.0.0.1:{}", loaded.process.info().port))
    }

    fn resolve_loaded_model_key(&self, requested: &str) -> Option<String> {
        let requested = requested.trim();
        if requested.is_empty() {
            return None;
        }
        if self.loaded.contains_key(requested) {
            return Some(requested.to_string());
        }
        self.loaded.iter().find_map(|(key, loaded)| {
            (loaded.public_model_id.as_deref() == Some(requested)
                || loaded.model == requested
                || loaded.proxy_model_ref.as_deref() == Some(requested))
            .then(|| key.clone())
        })
    }

    fn loaded_models_payload(&self) -> Value {
        json!({
            "object": "list",
            "data": self.loaded.values().map(loaded_runtime_payload).collect::<Vec<_>>(),
        })
    }

    fn loaded_runtime_summaries(&self) -> Vec<LoadedRuntimeSummary> {
        self.loaded
            .values()
            .map(|loaded| LoadedRuntimeSummary {
                id: loaded.model_key.clone(),
                owner_admin_id: loaded.owner_admin_id.clone(),
                backend_pid: loaded.process.info().pid,
            })
            .collect()
    }

    fn snapshot(&self) -> Value {
        let selected_backend = self.selected_backend.clone().or_else(|| {
            local_state::load_state()
                .ok()
                .and_then(|state| state.selected_backend)
        });
        let loaded_models = self
            .loaded
            .values()
            .map(loaded_runtime_payload)
            .collect::<Vec<_>>();
        let Some(default_key) = self.default_model_key.as_ref() else {
            return json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "public_model_id": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
                "backend_pid": null,
                "backend_port": null,
                "launch_args": [],
                "cuda_visible_devices": null,
                "warning": null,
                "launch_command": [],
                "proxy_model": null,
                "backend_log": null,
                "effective_parameters": {},
                "runtime": null,
                "loaded_models": loaded_models,
                "default_model": null,
            });
        };
        let Some(loaded) = self.loaded.get(default_key) else {
            return json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "public_model_id": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
                "backend_pid": null,
                "backend_port": null,
                "launch_args": [],
                "cuda_visible_devices": null,
                "warning": null,
                "launch_command": [],
                "proxy_model": null,
                "backend_log": null,
                "effective_parameters": {},
                "runtime": null,
                "loaded_models": loaded_models,
                "default_model": null,
            });
        };
        let info = loaded.process.info();
        json!({
            "backend": loaded.backend_id,
            "backend_ready": true,
            "model": loaded.model_key,
            "model_path": loaded.model,
            "public_model_id": loaded.public_model_id,
            "owner_admin_id": loaded.owner_admin_id,
            "mmproj": loaded.mmproj,
            "ctx_size": loaded.ctx_size,
            "request_defaults": {},
            "runtime_mode": "external_server",
            "backend_pid": info.pid,
            "backend_port": info.port,
            "launch_args": loaded.launch_args,
            "cuda_visible_devices": loaded.cuda_visible_devices,
            "warning": loaded.cuda_warning,
            "launch_command": info.command,
            "proxy_model": loaded.proxy_model_ref,
            "backend_log": info.log_path.display().to_string(),
            "effective_parameters": {},
            "runtime": {
                "mode": "external_server",
                "host": "127.0.0.1",
                "port": info.port,
                "pid": info.pid,
                "cuda_visible_devices": loaded.cuda_visible_devices,
                "launch_command": info.command,
                "log_path": info.log_path.display().to_string(),
                "proxy_model_ref": loaded.proxy_model_ref,
            },
            "log_path": info.log_path.display().to_string(),
            "loaded_models": loaded_models,
            "default_model": loaded.model_key,
        })
    }
}

fn loaded_runtime_payload(loaded: &LoadedRustRuntime) -> Value {
    let info = loaded.process.info();
    json!({
        "id": loaded.model_key,
        "owner_admin_id": loaded.owner_admin_id,
        "backend": loaded.backend_id,
        "model": loaded.model_key,
        "model_path": loaded.model,
        "public_model_id": loaded.public_model_id,
        "mmproj": loaded.mmproj,
        "ctx_size": loaded.ctx_size,
        "runtime_mode": "external_server",
        "backend_pid": info.pid,
        "backend_port": info.port,
        "launch_args": loaded.launch_args,
        "cuda_visible_devices": loaded.cuda_visible_devices,
        "warning": loaded.cuda_warning,
        "launch_command": info.command,
        "proxy_model": loaded.proxy_model_ref,
        "backend_log": info.log_path.display().to_string(),
    })
}

fn model_log_file_name(base: &str, model_key: &str) -> String {
    let sanitized = model_key
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    match base.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => {
            format!("{stem}-{sanitized}.{ext}")
        }
        _ => format!("{base}-{sanitized}.log"),
    }
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

fn json_required_str<'a>(payload: &'a Value, key: &'static str) -> Result<&'a str> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("field '{key}' is required"))
}

fn resolve_model_for_backend(
    model: &str,
    backend: &backend_registry::BackendSpec,
) -> Result<omniinfer_core::model_artifacts::ResolvedModelArtifacts> {
    if backend.model_artifact == "reference" {
        return Ok(omniinfer_core::model_artifacts::ResolvedModelArtifacts {
            model_path: model.to_string(),
            mmproj_path: None,
        });
    }
    let path = resolve_path_for_backend(model, backend, "model")?;
    if backend.model_artifact == "file" && PathBuf::from(&path).is_dir() {
        return Ok(discover_llama_cpp_model_artifacts(&PathBuf::from(path))?);
    }
    Ok(omniinfer_core::model_artifacts::ResolvedModelArtifacts {
        model_path: path,
        mmproj_path: None,
    })
}

fn resolve_path_for_backend(
    text: &str,
    backend: &backend_registry::BackendSpec,
    label: &str,
) -> Result<String> {
    let mut path = expand_home(PathBuf::from(text.trim()));
    if !path.is_absolute() {
        let Some(models_dir) = backend.models_dir.as_deref() else {
            anyhow::bail!("relative {label} path requires a configured models_dir");
        };
        path = PathBuf::from(models_dir).join(path);
    }
    if label == "model" && backend.model_artifact == "directory" {
        if !path.is_dir() {
            anyhow::bail!("model directory not found: {}", path.display());
        }
    } else if !path.exists() {
        anyhow::bail!("{label} not found: {}", path.display());
    }
    Ok(path.display().to_string())
}

fn expand_home(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path
}

fn pick_runtime_port(host: &str) -> Result<u16> {
    let listener = std::net::TcpListener::bind((host, 0))?;
    Ok(listener.local_addr()?.port())
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

fn current_system_name() -> String {
    match std::env::consts::OS {
        "macos" => "mac".to_string(),
        "windows" => "windows".to_string(),
        _ => "linux".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    use axum::Json;
    use axum::extract::Query;
    use axum::routing::{get, post};
    use std::collections::HashMap;
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
                        "backend_port": backend_port
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

    async fn spawn_test_gateway(
        _unused_port: u16,
        access_policy: GatewayAccessPolicy,
    ) -> TestServer {
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

    fn temp_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("omniinfer-rs-{name}-{nanos}"))
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
}
