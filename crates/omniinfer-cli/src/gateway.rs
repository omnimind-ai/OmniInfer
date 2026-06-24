use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST, HeaderMap, HeaderName, HeaderValue,
    TRANSFER_ENCODING,
};
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
use omniinfer_core::gateway_auth::{GatewayAccessPolicy, RequestAuthContext, authorize_request};
use omniinfer_core::local_state;
use omniinfer_core::model_artifacts::{discover_llama_cpp_model_artifacts, maybe_auto_mmproj};
use omniinfer_core::model_catalog;
use omniinfer_core::request_normalization::normalize_chat_request;
use omniinfer_core::runtime_plan::{ExternalRuntimeRequest, build_external_runtime_plan};
use omniinfer_core::runtime_process::{RuntimeProcess, RuntimeProcessOptions};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub listen_host: String,
    pub listen_port: u16,
    pub upstream_host: String,
    pub upstream_port: u16,
    pub access_policy: GatewayAccessPolicy,
}

#[derive(Clone)]
struct GatewayState {
    upstream_base: String,
    backend_host: String,
    access_policy: GatewayAccessPolicy,
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
        upstream_base: format!("http://{}:{}", config.upstream_host, config.upstream_port),
        backend_host: "127.0.0.1".to_string(),
        access_policy: config.access_policy,
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
    if let Err(error) = authorize_request(&state.access_policy, &auth_context) {
        return Ok(json_response(
            StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::FORBIDDEN),
            json!({"error": {"message": error.to_string()}}),
        ));
    }

    let should_shutdown = request.method() == Method::POST && path == "/omni/shutdown";
    if should_handle_rust_endpoint(&state, request.method(), &path).await {
        let Some(response) = try_handle_rust_endpoint(&state, &path, request).await? else {
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

    let upstream = upstream_uri(&state.upstream_base, request.uri())?;
    let (parts, body) = request.into_parts();
    let mut body = body.collect().await?.to_bytes();
    if parts.method == Method::POST && path == "/v1/chat/completions" {
        body = normalize_chat_body(body, false)?;
    }
    let mut builder = Request::builder().method(parts.method).uri(upstream);
    for (name, value) in parts.headers.iter() {
        if should_forward_header(name) {
            builder = builder.header(name, value);
        }
    }
    let upstream_request = builder.body(Full::new(body))?;
    let response = state.client.request(upstream_request).await?;
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

    if should_shutdown && status.is_success() {
        if let Some(sender) = state.shutdown.lock().await.take() {
            let _ = sender.send(());
        }
    }
    Ok(response)
}

async fn should_handle_rust_endpoint(state: &GatewayState, method: &Method, path: &str) -> bool {
    match (method, path) {
        (
            &Method::GET,
            "/health" | "/omni/state" | "/omni/backends" | "/omni/thinking" | "/omni/models",
        ) => true,
        (&Method::GET, "/omni/backend/props") => true,
        (&Method::GET, "/omni/supported-models" | "/omni/supported-models/best" | "/v1/models") => {
            true
        }
        (&Method::POST, "/omni/backend/select" | "/omni/backend/stop" | "/omni/model/select") => {
            true
        }
        (&Method::POST, "/omni/thinking/select") => true,
        (&Method::POST, "/v1/chat/completions") => {
            state.runtime.lock().await.current_proxy_base().is_some()
        }
        (&Method::POST, "/v1/messages") => {
            state.runtime.lock().await.current_proxy_base().is_some()
        }
        (
            &Method::POST,
            "/tokenize" | "/detokenize" | "/omni/tokenize" | "/omni/detokenize"
            | "/omni/cache/clear",
        ) => state.runtime.lock().await.current_proxy_base().is_some(),
        _ => false,
    }
}

async fn try_handle_rust_endpoint(
    state: &GatewayState,
    path: &str,
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
            let target = state.runtime.lock().await.current_proxy_base();
            let Some(target) = target else {
                return Ok(Some(json_response(StatusCode::OK, json!({}))));
            };
            let response = proxy_get_to_runtime(&state.client, &format!("{target}/props")).await?;
            Ok(Some(response))
        }
        (&Method::GET, "/omni/models") => Ok(Some(json_response(
            StatusCode::GONE,
            json!({"error": {"message": "GET /omni/models has been deprecated and is no longer maintained"}}),
        ))),
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
            let snapshot = state.runtime.lock().await.snapshot();
            let mut data = Vec::new();
            if snapshot
                .get("backend_ready")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && let Some(model_id) = snapshot.get("model").and_then(Value::as_str)
            {
                data.push(json!({
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "omniinfer",
                    "permission": [],
                    "root": model_id,
                    "parent": null,
                }));
            }
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
        (&Method::POST, "/omni/model/select") => {
            let (parts, body) = request.into_parts();
            let body = body.collect().await?.to_bytes();
            let payload: Value = serde_json::from_slice(&body)?;
            let requested_backend = {
                let mut runtime = state.runtime.lock().await;
                let requested_backend = runtime.resolve_requested_backend(&payload)?;
                let registry = BackendRegistry::load_current();
                let backend = registry
                    .get(&requested_backend)
                    .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
                if backend.runtime_mode == "embedded" {
                    runtime.stop_runtime()?;
                    runtime.selected_backend = Some(backend.id.clone());
                    local_state::save_selected_backend(&backend.id)?;
                    Some(backend.id.clone())
                } else {
                    None
                }
            };
            if requested_backend.is_some() {
                let response = proxy_collected_body_to_upstream(
                    &state.client,
                    &state.upstream_base,
                    &parts.method,
                    &parts.uri,
                    &parts.headers,
                    body,
                )
                .await?;
                return Ok(Some(response));
            }
            let backend_host = state.backend_host.clone();
            let runtime = Arc::clone(&state.runtime);
            let result = tokio::task::spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(async move {
                    runtime
                        .lock()
                        .await
                        .load_model(payload, backend_host, Duration::from_secs(120))
                })
            })
            .await??;
            Ok(Some(json_response(StatusCode::OK, result)))
        }
        (&Method::POST, "/v1/chat/completions") => {
            let body = request.into_body().collect().await?.to_bytes();
            let normalized = normalize_chat_body(body, false)?;
            let target = state.runtime.lock().await.current_proxy_base();
            let Some(target) = target else {
                return Ok(None);
            };
            let response = proxy_body_to_runtime(
                &state.client,
                &format!("{target}/v1/chat/completions"),
                normalized,
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
            let target = state.runtime.lock().await.current_proxy_base();
            let Some(target) = target else {
                return Ok(None);
            };
            let response =
                proxy_body_to_runtime(&state.client, &format!("{target}/{operation}"), body)
                    .await?;
            Ok(Some(response))
        }
        (&Method::POST, "/omni/cache/clear") => {
            let target = state.runtime.lock().await.current_proxy_base();
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
                .unwrap_or("omniinfer")
                .to_string();
            let openai_payload = anthropic_request_to_openai(&payload);
            let normalized = normalize_chat_request(openai_payload, false)?;
            let target = state.runtime.lock().await.current_proxy_base();
            let Some(target) = target else {
                return Ok(None);
            };
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

async fn proxy_collected_body_to_upstream(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    upstream_base: &str,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    body: HyperBytes,
) -> Result<Response<Body>> {
    let upstream = upstream_uri(upstream_base, uri)?;
    let mut builder = Request::builder().method(method).uri(upstream);
    for (name, value) in headers.iter() {
        if should_forward_header(name) {
            builder = builder.header(name, value);
        }
    }
    let upstream_request = builder.body(Full::new(body))?;
    response_from_upstream(client.request(upstream_request).await?).await
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

fn normalize_chat_body(body: HyperBytes, default_thinking: bool) -> Result<HyperBytes> {
    let payload: serde_json::Value = serde_json::from_slice(&body)?;
    let normalized = normalize_chat_request(payload, default_thinking)?;
    Ok(HyperBytes::from(serde_json::to_vec(&normalized.payload)?))
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
    loaded: Option<LoadedRustRuntime>,
}

struct LoadedRustRuntime {
    backend_id: String,
    model: String,
    mmproj: Option<String>,
    ctx_size: Option<u32>,
    launch_args: Vec<String>,
    process: RuntimeProcess,
    proxy_model_ref: Option<String>,
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
        if let Some(mut loaded) = self.loaded.take() {
            loaded.process.stop(Duration::from_secs(8))?;
        }
        Ok(json!({
            "ok": true,
            "stopped": true,
            "selected_backend": self.selected_backend,
        }))
    }

    fn load_model(
        &mut self,
        payload: Value,
        backend_host: String,
        startup_timeout: Duration,
    ) -> Result<Value> {
        let model = json_required_str(&payload, "model")?.to_string();
        let requested_backend = self.resolve_requested_backend(&payload)?;
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(&requested_backend)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
        if backend.runtime_mode != "external_server" {
            anyhow::bail!(
                "{} is an embedded backend; Rust gateway runtime manager currently supports external server backends only",
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
        let port = pick_runtime_port(&backend_host)?;
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
        self.stop_runtime()?;
        let log_path = PathBuf::from(&backend.runtime_dir)
            .join("logs")
            .join(&plan.log_file_name);
        let process = RuntimeProcess::start(
            &plan,
            RuntimeProcessOptions {
                log_path,
                env: runtime_env_for_backend(backend),
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
        self.loaded = Some(LoadedRustRuntime {
            backend_id: backend.id.clone(),
            model: resolved_model.model_path.clone(),
            mmproj: mmproj_path.clone(),
            ctx_size: plan.ctx_size,
            launch_args: effective_launch_args,
            proxy_model_ref: plan.proxy_model_ref.clone(),
            process,
        });
        Ok(json!({
            "ok": true,
            "selected_backend": backend.id,
            "selected_model": resolved_model.model_path,
            "selected_mmproj": mmproj_path,
            "selected_ctx_size": plan.ctx_size,
            "backend_pid": info.pid,
            "backend_port": info.port,
            "launch_command": info.command,
            "log_path": info.log_path.display().to_string(),
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

    fn current_proxy_base(&self) -> Option<String> {
        self.loaded
            .as_ref()
            .map(|loaded| format!("http://127.0.0.1:{}", loaded.process.info().port))
    }

    fn snapshot(&self) -> Value {
        let selected_backend = self.selected_backend.clone().or_else(|| {
            local_state::load_state()
                .ok()
                .and_then(|state| state.selected_backend)
        });
        let Some(loaded) = self.loaded.as_ref() else {
            return json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
                "backend_pid": null,
                "backend_port": null,
                "launch_args": [],
                "launch_command": [],
                "proxy_model": null,
                "backend_log": null,
                "effective_parameters": {},
                "runtime": null,
            });
        };
        let info = loaded.process.info();
        json!({
            "backend": loaded.backend_id,
            "backend_ready": true,
            "model": loaded.model,
            "mmproj": loaded.mmproj,
            "ctx_size": loaded.ctx_size,
            "request_defaults": {},
            "runtime_mode": "external_server",
            "backend_pid": info.pid,
            "backend_port": info.port,
            "launch_args": loaded.launch_args,
            "launch_command": info.command,
            "proxy_model": loaded.proxy_model_ref,
            "backend_log": info.log_path.display().to_string(),
            "effective_parameters": {},
            "runtime": {
                "mode": "external_server",
                "host": "127.0.0.1",
                "port": info.port,
                "pid": info.pid,
                "launch_command": info.command,
                "log_path": info.log_path.display().to_string(),
                "proxy_model_ref": loaded.proxy_model_ref,
            },
            "log_path": info.log_path.display().to_string(),
        })
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

fn runtime_env_for_backend(backend: &backend_registry::BackendSpec) -> Vec<(String, String)> {
    let mut env = Vec::new();
    if let Some(launcher) = backend.launcher_path.as_deref()
        && let Some(parent) = PathBuf::from(launcher).parent()
        && std::env::consts::OS != "windows"
    {
        let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let value = if existing.is_empty() {
            parent.display().to_string()
        } else {
            format!("{}:{existing}", parent.display())
        };
        env.push(("LD_LIBRARY_PATH".to_string(), value));
    }
    if backend.capabilities.iter().any(|cap| cap == "cuda")
        && let Ok(devices) = std::env::var("OMNIINFER_CUDA_VISIBLE_DEVICES")
        && !devices.trim().is_empty()
    {
        env.push(("CUDA_VISIBLE_DEVICES".to_string(), devices));
    }
    env
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

fn upstream_uri(base: &str, uri: &Uri) -> Result<Uri> {
    let path = uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");
    Ok(format!("{}{}", base.trim_end_matches('/'), path).parse()?)
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

fn should_forward_header(name: &HeaderName) -> bool {
    !is_hop_by_hop_header(name) && *name != HOST && *name != CONTENT_LENGTH
}

fn should_forward_response_header(name: &HeaderName) -> bool {
    !is_hop_by_hop_header(name) && *name != CONTENT_LENGTH
}

fn is_hop_by_hop_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) || *name == CONNECTION
        || *name == TRANSFER_ENCODING
}

fn json_response(status: StatusCode, payload: serde_json::Value) -> Response<Body> {
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    let mut response = Response::builder()
        .status(status)
        .header("content-type", "application/json; charset=utf-8")
        .body(Body::from(body))
        .expect("response should build");
    add_cors_headers(response.headers_mut());
    response
}

fn cors_response(status: StatusCode) -> Response<Body> {
    let mut response = Response::builder()
        .status(status)
        .body(Body::empty())
        .expect("response should build");
    add_cors_headers(response.headers_mut());
    response
}

fn add_cors_headers(headers: &mut HeaderMap) {
    headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    headers.insert(
        ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Authorization, anthropic-version, x-api-key"),
    );
    headers.insert(
        ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
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

    #[test]
    fn builds_upstream_uri_with_query() {
        let uri: Uri = "/v1/models?foo=bar".parse().unwrap();
        assert_eq!(
            upstream_uri("http://127.0.0.1:9001", &uri)
                .unwrap()
                .to_string(),
            "http://127.0.0.1:9001/v1/models?foo=bar"
        );
    }

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
    async fn proxy_forwards_openai_body_and_query() {
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
        assert_eq!(response.status().as_u16(), 200);
        let value: serde_json::Value = response.into_body().read_json().unwrap();
        assert_eq!(value["trace"], "1");
        assert_eq!(value["body"]["model"], "omniinfer");
        assert_eq!(value["auth"], "Bearer secret");
        gateway.stop().await;
        upstream.stop().await;
    }

    #[tokio::test]
    async fn proxy_normalizes_chat_request_before_upstream() {
        let upstream = spawn_test_upstream().await;
        let gateway = spawn_test_gateway(upstream.port, GatewayAccessPolicy::default()).await;
        let port = gateway.port;
        let response = tokio::task::spawn_blocking(move || {
            ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
                .send_json(serde_json::json!({
                    "model": "omniinfer",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "think": false,
                    "reasoning": {"effort": "high"},
                    "request_defaults": {"temperature": 0.2},
                    "functions": [{"name": "context_time_now", "parameters": {"type": "object"}}],
                    "function_call": {"name": "context_time_now"}
                }))
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let value: serde_json::Value = response.into_body().read_json().unwrap();
        let body = &value["body"];
        assert!(body.get("think").is_none());
        assert!(body.get("reasoning").is_none());
        assert!(body.get("request_defaults").is_none());
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["reasoning_format"], "none");
        assert_eq!(body["tools"][0]["function"]["name"], "context_time_now");
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "function": {"name": "context_time_now"}})
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

        let load_response = tokio::task::spawn_blocking({
            let model = model.clone();
            move || {
                ureq::post(format!("http://127.0.0.1:{port}/omni/model/select"))
                    .send_json(json!({
                        "backend": backend_id,
                        "model": model.display().to_string(),
                        "ctx_size": 512
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
        assert!(load_body["backend_pid"].as_u64().unwrap() > 0);

        let chat_response = tokio::task::spawn_blocking(move || {
            ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
                .send_json(json!({
                    "model": "omniinfer",
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

    #[tokio::test]
    async fn rust_gateway_delegates_embedded_model_loads_to_upstream() {
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
                .send_json(json!({
                    "backend": backend_id,
                    "model": "embedded-demo",
                    "ctx_size": 512
                }))
                .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(load_response.status().as_u16(), 200);
        let load_body: Value = load_response.into_body().read_json().unwrap();
        assert_eq!(load_body["selected_backend"], backend_id);
        assert_eq!(load_body["delegated"], true);
        assert_eq!(load_body["body"]["model"], "embedded-demo");

        let chat_response = tokio::task::spawn_blocking(move || {
            ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
                .send_json(json!({
                    "model": "omniinfer",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": false
                }))
                .unwrap()
        })
        .await
        .unwrap();
        let chat_body: Value = chat_response.into_body().read_json().unwrap();
        assert_eq!(chat_body["body"]["model"], "omniinfer");

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
        upstream_port: u16,
        access_policy: GatewayAccessPolicy,
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
                    upstream_host: "127.0.0.1".to_string(),
                    upstream_port,
                    access_policy,
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
