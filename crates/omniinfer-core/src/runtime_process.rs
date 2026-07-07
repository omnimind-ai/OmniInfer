use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::runtime_plan::ExternalRuntimePlan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProcessOptions {
    pub log_path: PathBuf,
    pub env: Vec<(String, String)>,
    pub startup_timeout: Duration,
    pub health_host: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProcessInfo {
    pub pid: u32,
    pub port: u16,
    pub command: Vec<String>,
    pub log_path: PathBuf,
}

#[derive(Debug)]
pub struct RuntimeProcess {
    child: Child,
    log_handle: File,
    info: RuntimeProcessInfo,
}

#[derive(Debug, Error)]
pub enum RuntimeProcessError {
    #[error("runtime command is empty")]
    EmptyCommand,
    #[error("failed to create runtime log directory {path}: {source}")]
    CreateLogDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open runtime log {path}: {source}")]
    OpenLog {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to duplicate runtime log handle {path}: {source}")]
    CloneLog {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to spawn runtime process: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("runtime exited before becoming ready")]
    EarlyExit,
    #[error("runtime did not become ready in time")]
    ReadyTimeout,
}

impl RuntimeProcess {
    pub fn start(
        plan: &ExternalRuntimePlan,
        options: RuntimeProcessOptions,
    ) -> Result<Self, RuntimeProcessError> {
        let executable = plan
            .command
            .first()
            .ok_or(RuntimeProcessError::EmptyCommand)?;
        if let Some(parent) = options.log_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                RuntimeProcessError::CreateLogDir {
                    path: parent.display().to_string(),
                    source,
                }
            })?;
        }
        let log_handle = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&options.log_path)
            .map_err(|source| RuntimeProcessError::OpenLog {
                path: options.log_path.display().to_string(),
                source,
            })?;
        let stdout = log_handle
            .try_clone()
            .map_err(|source| RuntimeProcessError::CloneLog {
                path: options.log_path.display().to_string(),
                source,
            })?;
        let stderr = log_handle
            .try_clone()
            .map_err(|source| RuntimeProcessError::CloneLog {
                path: options.log_path.display().to_string(),
                source,
            })?;
        let mut command = Command::new(executable);
        command
            .args(plan.command.iter().skip(1))
            .current_dir(&plan.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        for (key, value) in &options.env {
            command.env(key, value);
        }
        hide_child_window(&mut command);
        let mut child = command.spawn()?;
        if !wait_http_ready(
            &options.health_host,
            plan.port,
            options.startup_timeout,
            &mut child,
        )? {
            let _ = terminate_child(&mut child, Duration::from_secs(2));
            return Err(RuntimeProcessError::ReadyTimeout);
        }
        let info = RuntimeProcessInfo {
            pid: child.id(),
            port: plan.port,
            command: plan.command.clone(),
            log_path: options.log_path,
        };
        Ok(Self {
            child,
            log_handle,
            info,
        })
    }

    pub fn info(&self) -> &RuntimeProcessInfo {
        &self.info
    }

    pub fn stop(&mut self, grace: Duration) -> Result<(), RuntimeProcessError> {
        terminate_child(&mut self.child, grace)?;
        self.log_handle.sync_all().ok();
        Ok(())
    }
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        let _ = self.stop(Duration::from_secs(1));
    }
}

fn wait_http_ready(
    host: &str,
    port: u16,
    timeout: Duration,
    child: &mut Child,
) -> Result<bool, RuntimeProcessError> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if child.try_wait()?.is_some() {
            return Err(RuntimeProcessError::EarlyExit);
        }
        if health_endpoint_ready(host, port, Duration::from_millis(500)) {
            return Ok(true);
        }
        thread::sleep(Duration::from_millis(100));
    }
    if child.try_wait()?.is_some() {
        return Err(RuntimeProcessError::EarlyExit);
    }
    Ok(false)
}

fn health_endpoint_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let Ok(mut stream) = TcpStream::connect((host, port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let request =
        format!("GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    if reader.read_line(&mut status_line).is_err() {
        return false;
    }
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| (200..300).contains(&status))
}

fn terminate_child(child: &mut Child, grace: Duration) -> Result<(), RuntimeProcessError> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    terminate_process(child.id());
    let deadline = Instant::now() + grace;
    while Instant::now() < deadline {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    child.kill()?;
    let _ = child.wait();
    Ok(())
}

#[cfg(unix)]
fn terminate_process(pid: u32) {
    let _ = Command::new("kill").arg(pid.to_string()).status();
}

#[cfg(windows)]
fn terminate_process(pid: u32) {
    let mut command = Command::new("taskkill");
    hide_child_window(&mut command);
    let _ = command
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status();
}

fn hide_child_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

#[allow(dead_code)]
fn _path_exists(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::net::TcpListener;

    use super::*;

    #[test]
    fn accepts_empty_success_health_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                if line == "\r\n" || line == "\n" {
                    break;
                }
                line.clear();
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
        });
        assert!(health_endpoint_ready(
            "127.0.0.1",
            port,
            Duration::from_secs(1)
        ));
        handle.join().unwrap();
    }

    #[test]
    fn starts_ready_process_and_stops_on_drop() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let root = temp_root("runtime-process-ready");
        let script = write_test_server(&root, port);
        let plan = ExternalRuntimePlan {
            command: test_script_command(&script),
            cwd: root.clone(),
            port,
            ctx_size: None,
            log_file_name: "runtime.log".to_string(),
            proxy_model_ref: None,
        };
        let process = RuntimeProcess::start(
            &plan,
            RuntimeProcessOptions {
                log_path: root.join("runtime.log"),
                env: Vec::new(),
                startup_timeout: Duration::from_secs(5),
                health_host: "127.0.0.1".to_string(),
            },
        )
        .unwrap();
        assert!(process.info().pid > 0);
        let pid = process.info().pid;
        drop(process);
        assert!(process_exited(pid, Duration::from_secs(3)));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn returns_early_exit_for_failed_process() {
        let root = temp_root("runtime-process-fail");
        let script = write_failed_process(&root);
        let plan = ExternalRuntimePlan {
            command: test_script_command(&script),
            cwd: root.clone(),
            port: 9,
            ctx_size: None,
            log_file_name: "runtime.log".to_string(),
            proxy_model_ref: None,
        };
        let error = RuntimeProcess::start(
            &plan,
            RuntimeProcessOptions {
                log_path: root.join("runtime.log"),
                env: Vec::new(),
                startup_timeout: Duration::from_secs(1),
                health_host: "127.0.0.1".to_string(),
            },
        )
        .unwrap_err();
        assert!(
            matches!(error, RuntimeProcessError::EarlyExit),
            "unexpected error: {error:?}"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn drop_kills_unready_process() {
        let root = temp_root("runtime-process-unready");
        let script = write_sleep_process(&root);
        let plan = ExternalRuntimePlan {
            command: test_script_command(&script),
            cwd: root.clone(),
            port: 9,
            ctx_size: None,
            log_file_name: "runtime.log".to_string(),
            proxy_model_ref: None,
        };
        let error = RuntimeProcess::start(
            &plan,
            RuntimeProcessOptions {
                log_path: root.join("runtime.log"),
                env: Vec::new(),
                startup_timeout: Duration::from_millis(250),
                health_host: "127.0.0.1".to_string(),
            },
        )
        .unwrap_err();
        assert!(matches!(error, RuntimeProcessError::ReadyTimeout));
        fs::remove_dir_all(root).ok();
    }

    fn write_test_server(root: &Path, port: u16) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        #[cfg(windows)]
        {
            let executable = root.join("server.exe");
            compile_test_exe(
                root,
                "server.rs",
                &executable,
                &format!(
                    r##"
use std::io::{{BufRead, BufReader, Write}};
use std::net::{{TcpListener, TcpStream}};

fn main() {{
    let listener = TcpListener::bind("127.0.0.1:{port}").unwrap();
    for stream in listener.incoming().flatten() {{
        handle(stream);
    }}
}}

fn handle(mut stream: TcpStream) {{
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {{
        return;
    }}
    loop {{
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {{
            return;
        }}
        if line == "\r\n" || line == "\n" || line.is_empty() {{
            break;
        }}
    }}
    let body = r#"{{"status":"ok"}}"#;
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {{}}\r\nConnection: close\r\n\r\n",
        body.as_bytes().len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(body.as_bytes());
}}
"##
                ),
            );
            return executable;
        }
        #[cfg(not(windows))]
        {
            let script = root.join("server.sh");
            fs::write(
                &script,
                format!(
                    r#"#!/usr/bin/env bash
python3 - <<'PY'
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        raw = json.dumps({{"status": "ok"}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

HTTPServer(("127.0.0.1", {port}), Handler).serve_forever()
PY
"#
                ),
            )
            .unwrap();
            make_executable(&script);
            script
        }
    }

    fn write_failed_process(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        #[cfg(windows)]
        {
            let executable = root.join("fail.exe");
            compile_test_exe(
                root,
                "fail.rs",
                &executable,
                "fn main() { std::process::exit(7); }\n",
            );
            executable
        }
        #[cfg(not(windows))]
        {
            let script = root.join("fail.sh");
            fs::write(&script, "#!/usr/bin/env bash\nexit 7\n").unwrap();
            make_executable(&script);
            script
        }
    }

    fn write_sleep_process(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        #[cfg(windows)]
        {
            let executable = root.join("sleep.exe");
            compile_test_exe(
                root,
                "sleep.rs",
                &executable,
                r#"
fn main() {
    std::thread::sleep(std::time::Duration::from_secs(30));
}
"#,
            );
            executable
        }
        #[cfg(not(windows))]
        {
            let script = root.join("sleep.sh");
            fs::write(&script, "#!/usr/bin/env bash\nsleep 30\n").unwrap();
            make_executable(&script);
            script
        }
    }

    #[cfg(windows)]
    fn compile_test_exe(root: &Path, source_name: &str, executable: &Path, code: &str) {
        let source = root.join(source_name);
        fs::write(&source, code).unwrap();
        let status = Command::new("rustc")
            .arg("--edition=2021")
            .arg(&source)
            .arg("-o")
            .arg(executable)
            .status()
            .expect("compile Windows test process");
        assert!(status.success(), "failed to compile Windows test process");
    }

    fn temp_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("omniinfer-{name}-{nanos}"))
    }

    #[cfg(unix)]
    fn make_executable(path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    #[cfg(unix)]
    fn test_script_command(path: &Path) -> Vec<String> {
        vec!["bash".to_string(), path.display().to_string()]
    }

    #[cfg(windows)]
    fn test_script_command(path: &Path) -> Vec<String> {
        vec![path.display().to_string()]
    }

    fn process_exited(pid: u32, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if !process_exists(pid) {
                return true;
            }
            thread::sleep(Duration::from_millis(50));
        }
        false
    }

    #[cfg(unix)]
    fn process_exists(pid: u32) -> bool {
        Path::new("/proc").join(pid.to_string()).exists()
    }

    #[cfg(windows)]
    fn process_exists(pid: u32) -> bool {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}
