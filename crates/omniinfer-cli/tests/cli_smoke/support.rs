pub(super) use assert_cmd::Command;
pub(super) use omniinfer_core::http_client;
pub(super) use predicates::prelude::*;
pub(super) use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
pub(super) use std::process::{Command as StdCommand, Stdio};
use std::sync::mpsc;
pub(super) use std::thread;
pub(super) use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) struct TestGateway {
    pub(super) port: u16,
    request_rx: mpsc::Receiver<String>,
    handle: thread::JoinHandle<()>,
}

impl TestGateway {
    pub(super) fn start(responses: Vec<Response>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test gateway");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking test gateway");
        let port = listener.local_addr().expect("local addr").port();
        let (request_tx, request_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            for response_body in responses {
                let mut stream = accept_test_request(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("set request read timeout");
                let request = read_http_request(&mut stream);
                request_tx.send(request).expect("send request");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.content_type,
                    response_body.content_len()
                );
                stream.write_all(response.as_bytes()).expect("write header");
                response_body.write_body(&mut stream);
                stream.flush().expect("flush response");
            }
        });
        Self {
            port,
            request_rx,
            handle,
        }
    }

    pub(super) fn request(&self) -> String {
        self.request_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("receive request")
    }

    pub(super) fn join(self) {
        self.handle.join().expect("server thread");
    }
}

fn accept_test_request(listener: &TcpListener) -> std::net::TcpStream {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return stream,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    panic!("timed out waiting for test gateway request");
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("accept request: {error}"),
        }
    }
}

pub(super) struct Response {
    body: ResponseBody,
    content_type: String,
}

enum ResponseBody {
    Text(String),
    Chunks(Vec<String>),
}

impl Response {
    pub(super) fn new(body: &str) -> Self {
        Self::with_content_type(body, "application/json")
    }

    pub(super) fn with_content_type(body: &str, content_type: &str) -> Self {
        Self {
            body: ResponseBody::Text(body.to_string()),
            content_type: content_type.to_string(),
        }
    }

    pub(super) fn chunks(chunks: &[&str], content_type: &str) -> Self {
        Self {
            body: ResponseBody::Chunks(chunks.iter().map(|chunk| (*chunk).to_string()).collect()),
            content_type: content_type.to_string(),
        }
    }

    fn content_len(&self) -> usize {
        match &self.body {
            ResponseBody::Text(body) => body.len(),
            ResponseBody::Chunks(chunks) => chunks.iter().map(String::len).sum(),
        }
    }

    fn write_body(&self, stream: &mut impl Write) {
        match &self.body {
            ResponseBody::Text(body) => {
                stream.write_all(body.as_bytes()).expect("write body");
            }
            ResponseBody::Chunks(chunks) => {
                for chunk in chunks {
                    stream.write_all(chunk.as_bytes()).expect("write chunk");
                    stream.flush().expect("flush chunk");
                    thread::sleep(Duration::from_millis(15));
                }
            }
        }
    }
}

pub(super) fn temp_repo_root(test_name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-{test_name}-{nanos}"))
}

pub(super) fn install_fake_backend(root: &std::path::Path, backend_id: &str) {
    let binary_name = if cfg!(windows) {
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
        .join(binary_name);
    fs::create_dir_all(launcher.parent().unwrap()).expect("create fake backend dir");
    fs::write(&launcher, "#!/usr/bin/env bash\nexit 0\n").expect("write fake backend");
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(&launcher)
            .expect("fake backend metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).expect("chmod fake backend");
    }
}

pub(super) fn install_fake_runtime_server_in_root(
    runtime_root: &std::path::Path,
    backend_id: &str,
) {
    let launcher_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let launcher = runtime_root
        .join(backend_id)
        .join("bin")
        .join(launcher_name);
    fs::create_dir_all(launcher.parent().unwrap()).expect("create isolated fake runtime dir");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("fake_llama_server.rs");
    let mut command = StdCommand::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
    command
        .arg("--edition=2021")
        .arg(&fixture)
        .arg("-o")
        .arg(&launcher);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let status = command.status().expect("compile isolated fake runtime");
    assert!(
        status.success(),
        "failed to compile fake runtime fixture {}",
        fixture.display()
    );
}

pub(super) fn install_failing_runtime_in_root(runtime_root: &std::path::Path, backend_id: &str) {
    let launcher_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let launcher = runtime_root
        .join(backend_id)
        .join("bin")
        .join(launcher_name);
    fs::create_dir_all(launcher.parent().unwrap()).expect("create failing runtime dir");
    fs::copy(assert_cmd::cargo::cargo_bin("omniinfer"), &launcher)
        .expect("copy failing runtime fixture");
}

#[cfg(windows)]
pub(super) fn compile_fake_wsl(root: &std::path::Path) -> std::path::PathBuf {
    let launcher = root.join("fake-wsl.exe");
    fs::create_dir_all(root).expect("create fake WSL root");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("fake_wsl.rs");
    let mut command = StdCommand::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
    command
        .arg("--edition=2021")
        .arg(&fixture)
        .arg("-o")
        .arg(&launcher);
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
    let status = command.status().expect("compile fake WSL fixture");
    assert!(
        status.success(),
        "failed to compile fake WSL fixture {}",
        fixture.display()
    );
    launcher
}

#[cfg(windows)]
pub(super) fn write_wsl_python_runtime_fixture(root: &std::path::Path) -> std::path::PathBuf {
    use sha2::Digest;

    let fixture = root.join("wsl-python-fixture");
    fs::create_dir_all(&fixture).expect("create WSL Python fixture");
    let archive = fixture.join("uv.tar.gz");
    let file = fs::File::create(&archive).expect("create uv archive");
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut tar = tar::Builder::new(encoder);
    let contents = b"fake uv";
    let mut header = tar::Header::new_gnu();
    header.set_path("uv-x86_64-unknown-linux-gnu/uv").unwrap();
    header.set_size(contents.len() as u64);
    header.set_mode(0o755);
    header.set_cksum();
    tar.append(&header, contents.as_slice())
        .expect("append fake uv");
    tar.finish().expect("finish fake uv archive");
    drop(tar);
    let sha256 = format!(
        "{:x}",
        sha2::Sha256::digest(fs::read(&archive).expect("read fake uv archive"))
    );
    let catalog = fixture.join("catalog.json");
    fs::write(
        &catalog,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 2,
            "mirrors": [],
            "sources": {},
            "python_runtimes": {
                "windows": {
                    "vllm-wsl2-cuda": {
                        "source": "vllm-project/vllm",
                        "tag": "v0.24.0",
                        "package": "vllm",
                        "python": "3.12",
                        "launcher": "vllm",
                        "uv": {
                            "x86_64": {
                                "version": "0.11.16",
                                "url": format!("file://{}", archive.display()),
                                "sha256": sha256
                            }
                        },
                        "variants": {
                            "x86_64": {
                                "version": "0.24.0+cu129",
                                "accelerator": "cuda",
                                "runtime_version": "12.9",
                                "torch_backend": "cu129",
                                "minimum_driver": "576.02",
                                "url": "https://github.com/vllm-project/vllm/releases/download/v0.24.0/fake.whl",
                                "sha256": "1".repeat(64)
                            }
                        }
                    }
                }
            },
            "platforms": {}
        }))
        .expect("serialize WSL Python catalog"),
    )
    .expect("write WSL Python catalog");
    catalog
}

#[cfg(windows)]
pub(super) fn write_wsl_rocm_runtime_fixture(root: &std::path::Path) -> std::path::PathBuf {
    use sha2::Digest;

    let catalog = write_wsl_python_runtime_fixture(root);
    let fixture = root.join("wsl-python-fixture");
    let archive = fixture.join("uv.tar.gz");
    let uv_sha256 = format!(
        "{:x}",
        sha2::Sha256::digest(fs::read(&archive).expect("read fake uv archive"))
    );
    let key = fixture.join("rocm.gpg.key");
    let rocdxg = fixture.join("rocdxg.deb");
    fs::write(&key, b"fake verified ROCm key").expect("write fake ROCm key");
    fs::write(&rocdxg, b"fake verified ROCDXG package").expect("write fake ROCDXG");
    let key_sha256 = format!(
        "{:x}",
        sha2::Sha256::digest(fs::read(&key).expect("read fake ROCm key"))
    );
    let rocdxg_sha256 = format!(
        "{:x}",
        sha2::Sha256::digest(fs::read(&rocdxg).expect("read fake ROCDXG"))
    );
    let package_versions = [
        ("comgr", "3.0.0.70203-90~24.04"),
        ("hipblas", "3.2.0.70203-90~24.04"),
        ("hipblaslt", "1.2.2.70203-90~24.04"),
        ("hipfft", "1.0.22.70203-90~24.04"),
        ("hiprand", "3.1.0.70203-90~24.04"),
        ("hip-runtime-amd", "7.2.53211.70203-90~24.04"),
        ("hipsolver", "3.2.0.70203-90~24.04"),
        ("hipsparse", "4.2.0.70203-90~24.04"),
        ("hipsparselt", "0.2.6.70203-90~24.04"),
        ("hsa-rocr", "1.18.0.70203-90~24.04"),
        ("libpython3.12-dev", "3.12.3-1ubuntu0.15"),
        ("miopen-hip", "3.5.1.70203-90~24.04"),
        ("openmp-extras-runtime", "20.70.0.70203-90~24.04"),
        ("python3.12-dev", "3.12.3-1ubuntu0.15"),
        ("rccl", "2.27.7.70203-90~24.04"),
        ("rocblas", "5.2.0.70203-90~24.04"),
        ("rocfft", "1.0.36.70203-90~24.04"),
        ("rocm-hip-runtime", "7.2.3.70203-90~24.04"),
        ("rocm-core", "7.2.3.70203-90~24.04"),
        ("rocm-device-libs", "1.0.0.70203-90~24.04"),
        ("rocm-language-runtime", "7.2.3.70203-90~24.04"),
        ("rocm-llvm", "22.0.0.26084.70203-90~24.04"),
        ("rocm-smi-lib", "7.8.0.70203-90~24.04"),
        ("rocminfo", "1.0.0.70203-90~24.04"),
        ("rocprofiler-register", "0.6.0.70203-90~24.04"),
        ("rocrand", "4.2.0.70203-90~24.04"),
        ("rocsolver", "3.32.0.70203-90~24.04"),
        ("rocsparse", "4.2.0.70203-90~24.04"),
        ("roctracer", "4.1.70203.70203-90~24.04"),
    ];
    let mut packages = serde_json::Map::new();
    let mut package_assets = serde_json::Map::new();
    packages.insert(
        "libopenmpi3t64".to_string(),
        serde_json::Value::String("4.1.6-7ubuntu2".to_string()),
    );
    for (name, version) in package_versions {
        let contents = format!("fake verified ROCm package {name}={version}");
        let package = fixture.join(format!("{name}.deb"));
        fs::write(&package, contents.as_bytes()).expect("write fake ROCm package");
        let sha256 = format!("{:x}", sha2::Sha256::digest(contents.as_bytes()));
        let filename = format!("{name}_{version}_amd64.deb");
        packages.insert(
            name.to_string(),
            serde_json::Value::String(version.to_string()),
        );
        package_assets.insert(
            name.to_string(),
            serde_json::json!({
                "version": version,
                "url": format!("file://{}", package.display()),
                "filename": filename,
                "size": contents.len(),
                "sha256": sha256,
            }),
        );
    }
    fs::write(
        &catalog,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 2,
            "mirrors": [],
            "sources": {},
            "python_runtimes": {
                "windows": {
                    "vllm-wsl2-rocm": {
                        "source": "vllm-project/vllm",
                        "tag": "v0.26.0",
                        "package": "vllm",
                        "python": "3.12",
                        "launcher": "vllm",
                        "uv": {
                            "x86_64": {
                                "version": "0.11.16",
                                "url": format!("file://{}", archive.display()),
                                "sha256": uv_sha256
                            }
                        },
                        "variants": {
                            "x86_64": {
                                "version": "0.26.0+rocm723",
                                "reported_version": "0.26.0",
                                "accelerator": "rocm",
                                "runtime_version": "7.2.3",
                                "reported_runtime_version": "7.2.53211",
                                "torch_backend": "rocm723",
                                "build_commit": "f2654939e69b4069b13977e9aef3e31d4dcaf051",
                                "index_url": "https://wheels.vllm.ai/rocm/f2654939e69b4069b13977e9aef3e31d4dcaf051/rocm723",
                                "url": "https://wheels.vllm.ai/rocm/f2654939e69b4069b13977e9aef3e31d4dcaf051/fake.whl",
                                "sha256": "2".repeat(64),
                                "rocm_system": {
                                    "apt_repository": "https://repo.radeon.com/rocm/apt/7.2.3 noble main",
                                    "repository_key": {
                                        "version": "7.2.3",
                                        "url": format!("file://{}", key.display()),
                                        "sha256": key_sha256
                                    },
                                    "packages": packages,
                                    "package_assets": package_assets,
                                    "rocdxg": {
                                        "version": "1.2.0",
                                        "url": format!("file://{}", rocdxg.display()),
                                        "sha256": rocdxg_sha256
                                    },
                                    "required_gfx": ["gfx1151", "gfx1150"],
                                    "minimum_windows_release": "26.2.2"
                                }
                            }
                        }
                    }
                }
            },
            "platforms": {}
        }))
        .expect("serialize WSL ROCm catalog"),
    )
    .expect("write WSL ROCm catalog");
    catalog
}

pub(super) fn test_external_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-cpu"
    } else {
        "llama.cpp-linux"
    }
}

pub(super) fn test_secondary_backend_id() -> &'static str {
    if cfg!(target_os = "macos") {
        "llama.cpp-mac-intel"
    } else if cfg!(target_os = "windows") {
        "llama.cpp-windows-arm64"
    } else {
        "llama.cpp-linux"
    }
}

pub(super) fn test_runtime_platform_dir() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

#[cfg(unix)]
pub(super) fn install_fake_runtime_server(root: &std::path::Path, backend_id: &str) {
    let launcher = root
        .join(".local")
        .join("runtime")
        .join(test_runtime_platform_dir())
        .join(backend_id)
        .join("bin")
        .join("llama-server");
    fs::create_dir_all(launcher.parent().unwrap()).expect("create fake runtime dir");
    fs::write(
        &launcher,
        r#"#!/usr/bin/env bash
port=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
placement_mode="$(cat "$(dirname "$0")/placement-mode" 2>/dev/null || printf partial)"
if [ "$placement_mode" = "full" ]; then
  printf '%s\n' \
    'load_tensors: offloaded 42/42 layers to GPU' \
    'load_tensors: CPU_Mapped model buffer size = 32.00 MiB' \
    'load_tensors: CUDA0 model buffer size = 2048.00 MiB' \
    'llama_kv_cache: CUDA0 KV buffer size = 80.00 MiB' \
    'sched_reserve: CUDA0 compute buffer size = 72.00 MiB'
else
  printf '%s\n' \
    'load_tensors: offloaded 2/4 layers to GPU' \
    'load_tensors: CPU_Mapped model buffer size = 8.00 MiB' \
    'load_tensors: CUDA0 model buffer size = 16.00 MiB'
fi
exec python3 - "$port" <<'PY'
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
            self._json({"status": "ok"})
        else:
            self._json({"ok": True})
    def do_POST(self):
        self._json({
            "choices": [{"message": {"content": "fake backend"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#,
    )
    .expect("write fake runtime");
    let mut permissions = fs::metadata(&launcher)
        .expect("fake runtime metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod fake runtime");
}

pub(super) fn fake_cloudflared_launcher(root: &std::path::Path) -> std::path::PathBuf {
    fake_cloudflared_launcher_with_url(root, "https://example-test.trycloudflare.com")
}

pub(super) fn fake_cloudflared_launcher_with_url(
    root: &std::path::Path,
    url: &str,
) -> std::path::PathBuf {
    #[cfg(unix)]
    {
        fake_cloudflared_launcher_unix(root, url)
    }
    #[cfg(windows)]
    {
        fake_cloudflared_launcher_windows(root, url)
    }
}

#[cfg(unix)]
pub(super) fn fake_cloudflared_launcher_unix(
    root: &std::path::Path,
    url: &str,
) -> std::path::PathBuf {
    let launcher = root.join("fake-cloudflared.sh");
    let output = root.join("cloudflared.args");
    fs::write(
        &launcher,
        format!(
            "#!/usr/bin/env bash\nprintf '%s ' \"$@\" > '{}'\necho 'Your quick Tunnel has been created! Visit it at {}'\nwhile true; do echo 'cloudflared heartbeat'; sleep 0.1; done\n",
            output.display(),
            url
        ),
    )
    .expect("write fake cloudflared");
    let mut permissions = fs::metadata(&launcher)
        .expect("cloudflared metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&launcher, permissions).expect("chmod cloudflared");
    launcher
}

#[cfg(windows)]
pub(super) fn fake_cloudflared_launcher_windows(
    root: &std::path::Path,
    url: &str,
) -> std::path::PathBuf {
    let launcher = root.join("fake-cloudflared.cmd");
    let output = root.join("cloudflared.args");
    fs::write(
        &launcher,
        format!(
            "@echo off\r\necho %* > \"{}\"\r\necho Your quick Tunnel has been created! Visit it at {}\r\npowershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command \"Start-Sleep -Seconds 30\" > nul\r\n",
            output.display(),
            url
        ),
    )
    .expect("write fake cloudflared");
    launcher
}

fn read_http_request(stream: &mut impl Read) -> String {
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut expected_len = None;
    loop {
        let bytes = stream.read(&mut buffer).expect("read request");
        if bytes == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..bytes]);
        if expected_len.is_none()
            && let Some(header_end) = find_header_end(&raw)
        {
            let header = String::from_utf8_lossy(&raw[..header_end]);
            let content_length = header
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            expected_len = Some(header_end + 4 + content_length);
        }
        if let Some(length) = expected_len
            && raw.len() >= length
        {
            break;
        }
    }
    String::from_utf8_lossy(&raw).to_string()
}

fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|window| window == b"\r\n\r\n")
}

pub(super) fn request_body_json(request: &str) -> serde_json::Value {
    let body = request
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("request body separator");
    serde_json::from_str(body).expect("request body json")
}

pub(super) fn wait_for_file(path: std::path::PathBuf) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(text) = fs::read_to_string(&path) {
            return text;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {}", path.display());
}

#[cfg(unix)]
pub(super) fn install_fake_nvidia_smi(root: &std::path::Path, free_mib: u64) -> std::path::PathBuf {
    let executable = root.join("nvidia-smi");
    fs::create_dir_all(root).expect("create fake nvidia-smi directory");
    fs::write(
        &executable,
        format!(
            r#"#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index,uuid,memory.free"*) printf '0, GPU-TEST, {free_mib}\n' ;;
  *"--query-gpu=index,uuid,memory.used"*) printf '0, GPU-TEST, 0\n' ;;
  *"--query-gpu=index"*) printf '0\n' ;;
  *"--query-compute-apps="*) exit 0 ;;
  *) exit 0 ;;
esac
"#
        ),
    )
    .expect("write fake nvidia-smi");
    let mut permissions = fs::metadata(&executable)
        .expect("fake nvidia-smi metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&executable, permissions).expect("chmod fake nvidia-smi");
    executable
}

pub(super) fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    listener.local_addr().expect("local addr").port()
}

pub(super) fn wait_for_http_json(port: u16, path: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    let url = format!("http://127.0.0.1:{port}{path}");
    while Instant::now() < deadline {
        if let Ok(response) = http_client::get_json(&url, Duration::from_secs(1)) {
            if response.status < 400 {
                return response.body;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for {url}");
}

pub(super) fn stop_rust_serve(
    source_root: &std::path::Path,
    state_root: &std::path::Path,
    port: u16,
) {
    let mut stop = Command::cargo_bin("omniinfer").expect("binary exists");
    stop.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", source_root)
        .env("OMNIINFER_RUST_STATE_ROOT", state_root)
        .args(["serve", "stop", "--port"])
        .arg(port.to_string())
        .assert()
        .success();
    assert!(wait_for_port_closed(port));
}

pub(super) fn wait_for_port_closed(port: u16) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}

pub(super) fn wait_for_process_exit(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50));
            }
            Ok(None) | Err(_) => return None,
        }
    }
}

#[cfg(unix)]
pub(super) fn send_sigint(child: &std::process::Child) {
    let status = StdCommand::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("send SIGINT");
    assert!(status.success(), "SIGINT failed for pid {}", child.id());
}
