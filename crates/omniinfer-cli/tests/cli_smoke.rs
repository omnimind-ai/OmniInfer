use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn help_lists_core_commands() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("advisor"))
        .stdout(predicate::str::contains("serve"));
}

#[test]
fn completion_generates_bash_script() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.args(["completion", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_omniinfer-rs"));
}

#[test]
fn strict_mode_reports_unported_commands_without_fallback() {
    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .args(["advisor", "system"])
        .assert()
        .success()
        .stdout(predicate::str::contains("implementation pending"));
}

#[test]
fn backend_stop_posts_to_local_gateway() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test gateway");
    let port = listener.local_addr().expect("local addr").port();
    let (request_tx, request_rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept backend stop");
        let request = read_http_request(&mut stream);
        request_tx.send(request).expect("send request");
        let body = br#"{"stopped":true}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(response.as_bytes()).expect("write header");
        stream.write_all(body).expect("write body");
        stream.flush().expect("flush response");
    });

    let root = temp_repo_root("backend-stop");
    fs::create_dir_all(root.join("config")).expect("create config dir");
    fs::write(
        root.join("config").join("omniinfer.json"),
        format!(r#"{{"host":"127.0.0.1","port":{port}}}"#),
    )
    .expect("write config");

    let mut cmd = Command::cargo_bin("omniinfer-rs").expect("binary exists");
    cmd.env("OMNIINFER_RUST_STRICT", "1")
        .env("OMNIINFER_RUST_REPO_ROOT", &root)
        .args(["backend", "stop"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Current backend process stopped"));

    let request = request_rx.recv().expect("receive request");
    assert!(request.starts_with("POST /omni/backend/stop HTTP/1.1"));
    handle.join().expect("server thread");
    fs::remove_dir_all(root).ok();
}

fn temp_repo_root(test_name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("omniinfer-rs-{test_name}-{nanos}"))
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
