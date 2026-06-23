use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

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
use omniinfer_core::gateway_auth::{GatewayAccessPolicy, RequestAuthContext, authorize_request};
use omniinfer_core::request_normalization::normalize_chat_request;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

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
    access_policy: GatewayAccessPolicy,
    client: Client<HttpConnector, Full<HyperBytes>>,
    shutdown: Arc<tokio::sync::Mutex<Option<oneshot::Sender<()>>>>,
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
        access_policy: config.access_policy,
        client: Client::builder(TokioExecutor::new()).build_http(),
        shutdown: Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx))),
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

fn normalize_chat_body(body: HyperBytes, default_thinking: bool) -> Result<HyperBytes> {
    let payload: serde_json::Value = serde_json::from_slice(&body)?;
    let normalized = normalize_chat_request(payload, default_thinking)?;
    Ok(HyperBytes::from(serde_json::to_vec(&normalized.payload)?))
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
}
