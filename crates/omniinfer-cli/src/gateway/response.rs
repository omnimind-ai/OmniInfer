use axum::body::Body;
use axum::http::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    CONNECTION, CONTENT_LENGTH, HeaderMap, HeaderName, HeaderValue, TRANSFER_ENCODING,
};
use axum::http::{Response, StatusCode};

pub(super) fn should_forward_response_header(name: &HeaderName) -> bool {
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

pub(super) fn json_response(status: StatusCode, payload: serde_json::Value) -> Response<Body> {
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    let mut response = Response::builder()
        .status(status)
        .header("content-type", "application/json; charset=utf-8")
        .body(Body::from(body))
        .expect("response should build");
    add_cors_headers(response.headers_mut());
    response
}

pub(super) fn cors_response(status: StatusCode) -> Response<Body> {
    let mut response = Response::builder()
        .status(status)
        .body(Body::empty())
        .expect("response should build");
    add_cors_headers(response.headers_mut());
    response
}

pub(super) fn add_cors_headers(headers: &mut HeaderMap) {
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
