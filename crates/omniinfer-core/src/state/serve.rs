use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::paths;

#[derive(Debug, Error)]
pub enum ServeStateError {
    #[error("failed to read serve state file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse serve state file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to create serve state directory {path}: {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write serve state file {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode serve state file {path}: {source}")]
    Encode {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("another serve operation already owns port {port}")]
    Locked { port: u16 },
    #[error("failed to lock serve state for port {port}: {source}")]
    Lock {
        port: u16,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_time: u64,
    pub executable: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessIdentityStatus {
    Running,
    Exited,
    Mismatched,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ServePidInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gateway_process: Option<ProcessIdentity>,
    pub cloudflared_pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloudflared_process: Option<ProcessIdentity>,
    pub port: Option<u16>,
    pub log: Option<String>,
    pub public_url: Option<String>,
    pub openai_base_url: Option<String>,
    pub backend: Option<String>,
    pub model: Option<String>,
    pub mmproj: Option<String>,
    pub ctx_size: Option<u32>,
    pub backend_ready: Option<bool>,
    pub backend_pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_process: Option<ProcessIdentity>,
    pub backend_port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_process_owned: Option<bool>,
}

pub struct ServePortLock {
    file: File,
    #[allow(dead_code)]
    path: PathBuf,
}

impl Drop for ServePortLock {
    fn drop(&mut self) {
        let _ = File::unlock(&self.file);
    }
}

pub fn try_lock_serve_port(port: u16) -> Result<ServePortLock, ServeStateError> {
    let path = paths::local_run_dir().join(format!("serve-{port}.lock"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| ServeStateError::CreateDir {
            path: parent.display().to_string(),
            source,
        })?;
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|source| ServeStateError::Lock { port, source })?;
    match file.try_lock() {
        Ok(()) => Ok(ServePortLock { file, path }),
        Err(fs::TryLockError::WouldBlock) => Err(ServeStateError::Locked { port }),
        Err(fs::TryLockError::Error(source)) => Err(ServeStateError::Lock { port, source }),
    }
}

pub fn capture_process_identity(pid: u32) -> Option<ProcessIdentity> {
    process_snapshot(pid).map(|(identity, _)| identity)
}

fn process_snapshot(pid: u32) -> Option<(ProcessIdentity, sysinfo::ProcessStatus)> {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

    let pid_value = sysinfo::Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid_value]),
        true,
        ProcessRefreshKind::nothing()
            .with_cmd(UpdateKind::Always)
            .with_exe(UpdateKind::Always)
            .without_tasks(),
    );
    let process = system.process(pid_value)?;
    Some((
        ProcessIdentity {
            pid,
            start_time: process.start_time(),
            executable: process
                .exe()
                .map(|value| value.to_string_lossy().into_owned()),
            name: process.name().to_string_lossy().into_owned(),
        },
        process.status(),
    ))
}

pub fn process_identity_status(identity: &ProcessIdentity) -> ProcessIdentityStatus {
    let Some((current, status)) = process_snapshot(identity.pid) else {
        return ProcessIdentityStatus::Exited;
    };
    if matches!(
        status,
        sysinfo::ProcessStatus::Zombie | sysinfo::ProcessStatus::Dead
    ) {
        return ProcessIdentityStatus::Exited;
    }
    let matches = if identity.start_time > 0 && current.start_time > 0 {
        current.start_time == identity.start_time
    } else if let Some(expected) = identity.executable.as_ref() {
        current.executable.as_ref() == Some(expected)
    } else {
        !identity.name.is_empty() && current.name == identity.name
    };
    if !matches {
        ProcessIdentityStatus::Mismatched
    } else {
        ProcessIdentityStatus::Running
    }
}

pub fn load_serve_pid_info(port: u16) -> Result<Option<ServePidInfo>, ServeStateError> {
    let path = paths::serve_pid_file(port);
    if !path.is_file() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).map_err(|source| ServeStateError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let info = serde_json::from_str(&raw).map_err(|source| ServeStateError::Parse {
        path: path.display().to_string(),
        source,
    })?;
    Ok(Some(info))
}

pub fn save_serve_pid_info(info: &ServePidInfo) -> Result<(), ServeStateError> {
    let port = info.port.unwrap_or(9000);
    let path = paths::serve_pid_file(port);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| ServeStateError::CreateDir {
            path: parent.display().to_string(),
            source,
        })?;
    }
    let raw = serde_json::to_string_pretty(info).map_err(|source| ServeStateError::Encode {
        path: path.display().to_string(),
        source,
    })?;
    atomic_write(&path, format!("{raw}\n").as_bytes()).map_err(|source| ServeStateError::Write {
        path: path.display().to_string(),
        source,
    })
}

pub fn remove_serve_pid_info_if_run_id(port: u16, run_id: &str) -> Result<bool, ServeStateError> {
    let Some(info) = load_serve_pid_info(port)? else {
        return Ok(false);
    };
    if info.run_id.as_deref() != Some(run_id) {
        return Ok(false);
    }
    remove_serve_pid_info(port).map_err(|source| ServeStateError::Write {
        path: paths::serve_pid_file(port).display().to_string(),
        source,
    })?;
    Ok(true)
}

fn atomic_write(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "state path has no parent")
    })?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "state path has no file name",
            )
        })?;
    let temp_path = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        rand::random::<u64>()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.write_all(contents)?;
        file.sync_all()?;
        replace_file(&temp_path, path)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::rename(from, to)
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let from = from
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let to = to
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let success = unsafe {
        MoveFileExW(
            from.as_ptr(),
            to.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if success == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn remove_serve_pid_info(port: u16) -> Result<(), std::io::Error> {
    let path = paths::serve_pid_file(port);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn list_serve_pid_infos() -> Result<Vec<ServePidInfo>, ServeStateError> {
    let dir = paths::local_run_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(ServeStateError::Read {
                path: dir.display().to_string(),
                source,
            });
        }
    };
    let mut infos = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("serve-") || !name.ends_with(".json") {
            continue;
        }
        let raw = fs::read_to_string(&path).map_err(|source| ServeStateError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let info: ServePidInfo =
            serde_json::from_str(&raw).map_err(|source| ServeStateError::Parse {
                path: path.display().to_string(),
                source,
            })?;
        infos.push(info);
    }
    infos.sort_by_key(|info| info.port.unwrap_or(0));
    Ok(infos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::process::Command;
    #[cfg(unix)]
    use std::time::{Duration, Instant};

    #[test]
    fn serializes_python_compatible_pid_info() {
        let info = ServePidInfo {
            run_id: Some("run-test".to_string()),
            phase: Some("ready".to_string()),
            pid: Some(123),
            gateway_process: None,
            cloudflared_pid: Some(124),
            cloudflared_process: None,
            port: Some(9000),
            log: Some("/tmp/serve.log".to_string()),
            public_url: Some("https://example.trycloudflare.com".to_string()),
            openai_base_url: Some("https://example.trycloudflare.com/v1".to_string()),
            backend: Some("llama.cpp-linux-cuda".to_string()),
            model: Some("/models/model.gguf".to_string()),
            mmproj: None,
            ctx_size: Some(8192),
            backend_ready: Some(true),
            backend_pid: Some(456),
            backend_process: None,
            backend_port: Some(12345),
            backend_process_owned: Some(true),
        };
        let value = serde_json::to_value(&info).unwrap();
        assert_eq!(value["pid"], 123);
        assert_eq!(value["cloudflared_pid"], 124);
        assert_eq!(value["port"], 9000);
        assert_eq!(value["public_url"], "https://example.trycloudflare.com");
        assert_eq!(
            value["openai_base_url"],
            "https://example.trycloudflare.com/v1"
        );
        assert_eq!(value["backend_ready"], true);
        assert_eq!(value["backend_process_owned"], true);
    }

    #[test]
    fn process_identity_rejects_pid_reuse_metadata() {
        let mut identity = capture_process_identity(std::process::id()).expect("current process");
        assert_eq!(
            process_identity_status(&identity),
            ProcessIdentityStatus::Running
        );
        identity.start_time = identity.start_time.saturating_add(1);
        assert_eq!(
            process_identity_status(&identity),
            ProcessIdentityStatus::Mismatched
        );
    }

    #[cfg(unix)]
    #[test]
    fn process_identity_treats_unreaped_zombie_as_exited() {
        let mut child = Command::new("sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .expect("spawn child");
        let identity = capture_process_identity(child.id()).expect("capture child identity");
        child.kill().expect("terminate child without reaping it");

        let deadline = Instant::now() + Duration::from_secs(2);
        while process_identity_status(&identity) == ProcessIdentityStatus::Running
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            process_identity_status(&identity),
            ProcessIdentityStatus::Exited
        );
        let _ = child.wait();
    }

    #[test]
    fn serve_port_lock_is_exclusive() {
        let port = 29_000 + (std::process::id() % 10_000) as u16;
        let first = try_lock_serve_port(port).expect("first lock");
        assert!(matches!(
            try_lock_serve_port(port),
            Err(ServeStateError::Locked { .. })
        ));
        drop(first);
        try_lock_serve_port(port).expect("lock after release");
    }

    #[test]
    fn atomic_write_replaces_complete_json() {
        let root = std::env::temp_dir().join(format!(
            "omniinfer-serve-state-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        fs::create_dir_all(&root).expect("create temporary state directory");
        let path = root.join("serve.json");
        atomic_write(&path, br#"{"phase":"starting"}"#).expect("write starting state");
        atomic_write(&path, br#"{"phase":"ready"}"#).expect("replace ready state");
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("read state")).expect("valid JSON");
        assert_eq!(value["phase"], "ready");
        assert_eq!(
            fs::read_dir(&root).expect("read state directory").count(),
            1
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn deserializes_legacy_state_without_identity() {
        let info: ServePidInfo = serde_json::from_str(
            r#"{"pid":123,"cloudflared_pid":124,"port":9000,"backend_pid":456}"#,
        )
        .expect("legacy serve state");
        assert_eq!(info.pid, Some(123));
        assert_eq!(info.cloudflared_pid, Some(124));
        assert!(info.run_id.is_none());
        assert!(info.gateway_process.is_none());
        assert!(info.backend_process_owned.is_none());
    }
}
