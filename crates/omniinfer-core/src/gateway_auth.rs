use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayAccessPolicy {
    pub api_key: String,
    pub admin_api_key: String,
    pub allow_insecure_lan: bool,
    pub allow_remote_management: bool,
    pub trust_proxy_headers: bool,
}

impl Default for GatewayAccessPolicy {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            admin_api_key: String::new(),
            allow_insecure_lan: false,
            allow_remote_management: false,
            trust_proxy_headers: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestAuthContext {
    pub method: String,
    pub path: String,
    pub client_ip: String,
    pub authorization: Option<String>,
    pub x_api_key: Option<String>,
    pub cf_connecting_ip: Option<String>,
    pub x_forwarded_for: Option<String>,
    pub x_real_ip: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GatewayAuthError {
    #[error(
        "remote clients may only access inference endpoints; management endpoints are local-only"
    )]
    RemoteManagementForbidden,
    #[error("missing or invalid API key")]
    MissingOrInvalidApiKey,
    #[error("remote access requires an API key")]
    RemoteAccessRequiresApiKey,
}

impl GatewayAuthError {
    pub fn status_code(&self) -> u16 {
        match self {
            GatewayAuthError::RemoteManagementForbidden => 403,
            GatewayAuthError::MissingOrInvalidApiKey => 401,
            GatewayAuthError::RemoteAccessRequiresApiKey => 403,
        }
    }
}

pub fn authorize_request(
    policy: &GatewayAccessPolicy,
    request: &RequestAuthContext,
) -> Result<(), GatewayAuthError> {
    if !is_remote_client(policy, request) {
        return Ok(());
    }
    let management_endpoint = !is_public_endpoint(&request.method, &request.path);
    if management_endpoint && !policy.allow_remote_management {
        return Err(GatewayAuthError::RemoteManagementForbidden);
    }
    if remote_request_authenticated(policy, request, management_endpoint) {
        return Ok(());
    }
    if required_key(policy, management_endpoint).is_empty() {
        Err(GatewayAuthError::RemoteAccessRequiresApiKey)
    } else {
        Err(GatewayAuthError::MissingOrInvalidApiKey)
    }
}

pub fn should_require_remote_api_key(host: &str) -> bool {
    !is_loopback_bind_host(host)
}

pub fn is_remote_request(policy: &GatewayAccessPolicy, request: &RequestAuthContext) -> bool {
    is_remote_client(policy, request)
}

fn is_remote_client(policy: &GatewayAccessPolicy, request: &RequestAuthContext) -> bool {
    if policy.trust_proxy_headers && proxy_header_remote_address(request).is_some() {
        return true;
    }
    !is_loopback_address(&request.client_ip)
}

fn remote_request_authenticated(
    policy: &GatewayAccessPolicy,
    request: &RequestAuthContext,
    management_endpoint: bool,
) -> bool {
    let api_key = required_key(policy, management_endpoint);
    if api_key.is_empty() {
        return policy.allow_insecure_lan && !management_endpoint;
    }
    let token = bearer_token(request)
        .or_else(|| request.x_api_key.as_deref().map(str::trim))
        .unwrap_or("");
    !token.is_empty() && constant_time_eq(token.as_bytes(), api_key.as_bytes())
}

fn required_key(policy: &GatewayAccessPolicy, management_endpoint: bool) -> &str {
    if management_endpoint {
        let admin_api_key = policy.admin_api_key.trim();
        if !admin_api_key.is_empty() {
            return admin_api_key;
        }
    }
    policy.api_key.trim()
}

fn bearer_token(request: &RequestAuthContext) -> Option<&str> {
    let authorization = request.authorization.as_deref()?.trim();
    authorization
        .get(..7)
        .filter(|prefix| prefix.eq_ignore_ascii_case("bearer "))
        .map(|_| authorization[7..].trim())
}

fn is_public_endpoint(method: &str, path: &str) -> bool {
    match method.to_ascii_uppercase().as_str() {
        "GET" => matches!(path, "/health" | "/v1/models"),
        "POST" => matches!(
            path,
            "/v1/chat/completions" | "/v1/messages" | "/tokenize" | "/detokenize"
        ),
        _ => false,
    }
}

fn proxy_header_remote_address(request: &RequestAuthContext) -> Option<&str> {
    [
        request.cf_connecting_ip.as_deref(),
        request.x_forwarded_for.as_deref(),
        request.x_real_ip.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(first_header_ip)
}

fn first_header_ip(value: &str) -> Option<&str> {
    value
        .split(',')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn is_loopback_bind_host(host: &str) -> bool {
    let value = host.trim().trim_start_matches('[').trim_end_matches(']');
    matches!(value, "127.0.0.1" | "localhost" | "::1")
}

fn is_loopback_address(host: &str) -> bool {
    let value = host.trim().trim_start_matches('[').trim_end_matches(']');
    value == "localhost" || value == "::1" || value.starts_with("127.")
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(method: &str, path: &str) -> RequestAuthContext {
        RequestAuthContext {
            method: method.to_string(),
            path: path.to_string(),
            client_ip: "203.0.113.10".to_string(),
            authorization: None,
            x_api_key: None,
            cf_connecting_ip: None,
            x_forwarded_for: None,
            x_real_ip: None,
        }
    }

    #[test]
    fn local_requests_do_not_need_api_key() {
        let mut request = request("GET", "/omni/state");
        request.client_ip = "127.0.0.1".to_string();
        assert_eq!(
            authorize_request(&GatewayAccessPolicy::default(), &request),
            Ok(())
        );
    }

    #[test]
    fn remote_public_endpoint_requires_api_key() {
        let policy = GatewayAccessPolicy {
            api_key: "secret".to_string(),
            ..GatewayAccessPolicy::default()
        };
        assert_eq!(
            authorize_request(&policy, &request("GET", "/health")).unwrap_err(),
            GatewayAuthError::MissingOrInvalidApiKey
        );
    }

    #[test]
    fn remote_public_endpoint_accepts_bearer_key() {
        let policy = GatewayAccessPolicy {
            api_key: "secret".to_string(),
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("GET", "/health");
        request.authorization = Some("Bearer secret".to_string());
        assert_eq!(authorize_request(&policy, &request), Ok(()));
    }

    #[test]
    fn remote_public_endpoint_accepts_x_api_key() {
        let policy = GatewayAccessPolicy {
            api_key: "secret".to_string(),
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("POST", "/v1/chat/completions");
        request.x_api_key = Some("secret".to_string());
        assert_eq!(authorize_request(&policy, &request), Ok(()));
    }

    #[test]
    fn remote_management_endpoint_is_local_only() {
        let policy = GatewayAccessPolicy {
            api_key: "secret".to_string(),
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("POST", "/omni/shutdown");
        request.authorization = Some("Bearer secret".to_string());
        let error = authorize_request(&policy, &request).unwrap_err();
        assert_eq!(error, GatewayAuthError::RemoteManagementForbidden);
        assert_eq!(error.status_code(), 403);
    }

    #[test]
    fn remote_management_endpoint_can_be_explicitly_exposed() {
        let policy = GatewayAccessPolicy {
            api_key: "secret".to_string(),
            allow_remote_management: true,
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("POST", "/omni/shutdown");
        request.authorization = Some("Bearer secret".to_string());
        assert_eq!(authorize_request(&policy, &request), Ok(()));
    }

    #[test]
    fn remote_management_endpoint_prefers_admin_api_key() {
        let policy = GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("POST", "/omni/model/select");
        request.authorization = Some("Bearer inference".to_string());
        assert_eq!(
            authorize_request(&policy, &request).unwrap_err(),
            GatewayAuthError::MissingOrInvalidApiKey
        );
        request.authorization = Some("Bearer admin".to_string());
        assert_eq!(authorize_request(&policy, &request), Ok(()));
    }

    #[test]
    fn remote_public_endpoint_still_uses_inference_api_key() {
        let policy = GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("POST", "/v1/chat/completions");
        request.authorization = Some("Bearer admin".to_string());
        assert_eq!(
            authorize_request(&policy, &request).unwrap_err(),
            GatewayAuthError::MissingOrInvalidApiKey
        );
        request.authorization = Some("Bearer inference".to_string());
        assert_eq!(authorize_request(&policy, &request), Ok(()));
    }

    #[test]
    fn proxy_headers_mark_loopback_peer_as_remote_when_trusted() {
        let policy = GatewayAccessPolicy {
            api_key: "secret".to_string(),
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        };
        let mut request = request("GET", "/health");
        request.client_ip = "127.0.0.1".to_string();
        request.cf_connecting_ip = Some("203.0.113.10".to_string());
        assert_eq!(
            authorize_request(&policy, &request).unwrap_err(),
            GatewayAuthError::MissingOrInvalidApiKey
        );
    }

    #[test]
    fn allow_insecure_lan_only_skips_key_for_public_endpoints() {
        let policy = GatewayAccessPolicy {
            allow_insecure_lan: true,
            ..GatewayAccessPolicy::default()
        };
        assert_eq!(
            authorize_request(&policy, &request("GET", "/health")),
            Ok(())
        );
        assert_eq!(
            authorize_request(&policy, &request("POST", "/omni/shutdown")).unwrap_err(),
            GatewayAuthError::RemoteManagementForbidden
        );
    }

    #[test]
    fn remote_bind_hosts_require_api_key() {
        assert!(should_require_remote_api_key("0.0.0.0"));
        assert!(should_require_remote_api_key("192.168.1.10"));
        assert!(!should_require_remote_api_key("127.0.0.1"));
        assert!(!should_require_remote_api_key("::1"));
    }
}
