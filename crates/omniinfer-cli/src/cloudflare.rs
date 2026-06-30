use std::env;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use omniinfer_core::paths;

use crate::{detach_child_process, hide_child_window};

pub(crate) fn resolve_cloudflared(explicit_path: Option<&str>) -> Result<PathBuf> {
    if let Some(path) = explicit_path.filter(|value| !value.trim().is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            anyhow::bail!("cloudflared was not found at {}", path.display());
        }
        return Ok(path);
    }

    let managed = managed_cloudflared_path();
    if managed.is_file() {
        return Ok(managed);
    }

    if let Some(path) = find_executable_in_path(cloudflared_executable_name()) {
        return Ok(path);
    }

    anyhow::bail!("cloudflared was not found. Install it in PATH or pass --cloudflared-path.")
}

fn managed_cloudflared_path() -> PathBuf {
    paths::local_dir()
        .join("tools")
        .join("cloudflared")
        .join(cloudflared_executable_name())
}

fn cloudflared_executable_name() -> &'static str {
    if cfg!(windows) {
        "cloudflared.exe"
    } else {
        "cloudflared"
    }
}

fn find_executable_in_path(name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 && candidate.is_file() {
        return Some(candidate.to_path_buf());
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

pub(crate) fn start_cloudflare_quick_tunnel(
    cloudflared: &Path,
    local_url: &str,
    log_path: &Path,
    detach: bool,
) -> Result<(std::process::Child, String)> {
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let mut command = ProcessCommand::new(cloudflared);
    command
        .args(["tunnel", "--url", local_url])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_child_window(&mut command);
    if detach {
        detach_child_process(&mut command);
    }
    let mut child = command.spawn()?;

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture cloudflared stdout"))?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture cloudflared stderr"))?;
    let (line_tx, line_rx) = mpsc::channel();
    spawn_cloudflared_reader(stdout_pipe, stdout, line_tx.clone());
    spawn_cloudflared_reader(stderr_pipe, stderr, line_tx);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut tail = Vec::new();
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!(
                "cloudflared exited before creating a Quick Tunnel with status {status}.{}",
                format_log_tail(&tail)
            );
        }
        match line_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                if tail.len() == 10 {
                    tail.remove(0);
                }
                tail.push(line.clone());
                if let Some(url) = parse_trycloudflare_url(&line) {
                    return Ok((child, url));
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    anyhow::bail!(
        "Timed out waiting for Cloudflare Quick Tunnel URL.{}",
        format_log_tail(&tail)
    )
}

fn spawn_cloudflared_reader<R: std::io::Read + Send + 'static>(
    stream: R,
    mut log: std::fs::File,
    line_tx: mpsc::Sender<String>,
) {
    thread::spawn(move || {
        let reader = BufReader::new(stream);
        for line in reader.lines().map_while(Result::ok) {
            use std::io::Write;
            let _ = writeln!(log, "{line}");
            let _ = log.flush();
            let _ = line_tx.send(line);
        }
    });
}

fn parse_trycloudflare_url(line: &str) -> Option<String> {
    line.split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | '(' | ')' | '[' | ']'))
        .find(|part| part.starts_with("https://") && part.contains(".trycloudflare.com"))
        .map(|part| {
            part.trim_end_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '/')
                .trim_end_matches('/')
                .to_string()
        })
}

fn format_log_tail(lines: &[String]) -> String {
    if lines.is_empty() {
        String::new()
    } else {
        format!("\ncloudflared log tail:\n{}", lines.join("\n"))
    }
}
