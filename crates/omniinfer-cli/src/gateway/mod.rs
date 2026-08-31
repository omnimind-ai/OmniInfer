use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
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
use omniinfer_core::request_normalization::normalize_chat_request_with_defaults;
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
use runtime_manager::{LoadModelOutcome, RuntimeProxyTarget, RustRuntimeManager};

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
    pub runtime_startup_timeout: Duration,
    pub access_policy: GatewayAccessPolicy,
    pub public_model_root: Option<PathBuf>,
}

#[derive(Clone)]
struct GatewayState {
    backend_host: String,
    runtime_startup_timeout: Duration,
    access_policy: Arc<tokio::sync::Mutex<DynamicAccessPolicy>>,
    public_model_root: Option<PathBuf>,
    request_history_dir: PathBuf,
    client: Client<HttpConnector, Full<HyperBytes>>,
    shutdown: Arc<tokio::sync::Mutex<Option<oneshot::Sender<()>>>>,
    startup_cancelled: Arc<AtomicBool>,
    runtime: Arc<tokio::sync::Mutex<RustRuntimeManager>>,
}

pub fn run_gateway_blocking(config: GatewayConfig) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run_gateway(config))
}

pub async fn run_gateway(config: GatewayConfig) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.listen_host, config.listen_port).parse()?;
    let listener = TcpListener::bind(addr).await?;
    run_gateway_with_listener(config, listener).await
}

async fn run_gateway_with_listener(config: GatewayConfig, listener: TcpListener) -> Result<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let state = GatewayState {
        backend_host: "127.0.0.1".to_string(),
        runtime_startup_timeout: config.runtime_startup_timeout,
        access_policy: Arc::new(tokio::sync::Mutex::new(DynamicAccessPolicy::new(
            config.access_policy,
            paths::admin_keys_file(),
        ))),
        public_model_root: config.public_model_root,
        request_history_dir: paths::local_dir().join("request_history"),
        client: Client::builder(TokioExecutor::new()).build_http(),
        shutdown: Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx))),
        startup_cancelled: Arc::new(AtomicBool::new(false)),
        runtime: Arc::new(tokio::sync::Mutex::new(RustRuntimeManager::default())),
    };
    let app = axum::Router::new()
        .fallback(proxy_request)
        .with_state(state);
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

mod management;

use management::{should_handle_rust_endpoint, try_handle_rust_endpoint};
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

async fn proxy_passthrough_to_runtime(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    method: Method,
    uri: &str,
    content_type: Option<axum::http::HeaderValue>,
    body: HyperBytes,
) -> Result<Response<Body>> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(content_type) = content_type {
        builder = builder.header(CONTENT_TYPE, content_type);
    }
    let request = builder.body(Full::new(body))?;
    let response = client.request(request).await?;
    let status = response.status();
    let content_length = response.headers().get(CONTENT_LENGTH).cloned();
    let mut builder = Response::builder().status(status);
    for (name, value) in response.headers().iter() {
        if should_forward_response_header(name) {
            builder = builder.header(name, value);
        }
    }
    if let Some(content_length) = content_length {
        builder = builder.header(CONTENT_LENGTH, content_length);
    }
    let mut response = builder.body(Body::new(response.into_body()))?;
    add_cors_headers(response.headers_mut());
    Ok(response)
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

mod openai_stream;

use openai_stream::{
    StreamHistoryContext, apply_proxy_model, duration_ms, proxy_openai_nonstream_via_stream,
    should_proxy_vllm_nonstream_via_stream, stream_openai_chat_with_history,
};
async fn clear_runtime_cache(
    client: &Client<HttpConnector, Full<HyperBytes>>,
    runtime_base: &str,
) -> Result<Response<Body>> {
    let props_request = Request::builder()
        .method(Method::GET)
        .uri(format!("{runtime_base}/props"))
        .body(Full::new(HyperBytes::new()))?;
    let props_response = client.request(props_request).await?;
    let props_status = props_response.status();
    let props_body = props_response.into_body().collect().await?.to_bytes();
    if !props_status.is_success() {
        return Ok(cache_clear_error_response(props_status, &props_body));
    }
    let props = serde_json::from_slice::<Value>(&props_body).unwrap_or(Value::Null);
    let slot_count = props
        .get("total_slots")
        .or_else(|| props.get("slots"))
        .and_then(Value::as_u64)
        .filter(|count| (1..=256).contains(count));
    let Some(slot_count) = slot_count else {
        return Ok(json_response(
            StatusCode::CONFLICT,
            json!({"error": {"message": "backend did not report a valid slot count; cache erasure cannot be proven"}}),
        ));
    };
    let mut cleared_slots = Vec::with_capacity(slot_count as usize);
    for slot_id in 0..slot_count {
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("{runtime_base}/slots/{slot_id}?action=erase"))
            .body(Full::new(HyperBytes::new()))?;
        let response = client.request(request).await?;
        let status = response.status();
        let body = response.into_body().collect().await?.to_bytes();
        if !status.is_success() {
            return Ok(cache_clear_error_response(status, &body));
        }
        cleared_slots.push(slot_id);
    }
    Ok(json_response(
        StatusCode::OK,
        json!({
            "ok": true,
            "message": "KV cache cleared",
            "cache_policy": "cleared_each_run",
            "cleared_slots": cleared_slots,
        }),
    ))
}

fn cache_clear_error_response(status: StatusCode, body: &[u8]) -> Response<Body> {
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
    json_response(StatusCode::CONFLICT, json!({"error": {"message": message}}))
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

fn backend_protocol_not_supported(target: &RuntimeProxyTarget, endpoint: &str) -> Response<Body> {
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({
            "error": {
                "code": "backend_protocol_not_supported",
                "message": format!(
                    "{endpoint} is not supported by backend protocol {}",
                    target.protocol.as_str(),
                ),
                "backend": target.backend_id,
                "external_server_protocol": target.protocol.as_str(),
                "client_endpoint": target.client_endpoint,
            }
        }),
    )
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
