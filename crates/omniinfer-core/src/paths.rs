use std::path::{Path, PathBuf};

const ROOT_OVERRIDE_ENV: &str = "OMNIINFER_RUST_REPO_ROOT";

pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os(ROOT_OVERRIDE_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(root);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate path should be under crates/<name>")
        .to_path_buf()
}

pub fn local_dir() -> PathBuf {
    repo_root().join(".local")
}

pub fn local_config_dir() -> PathBuf {
    local_dir().join("config")
}

pub fn local_logs_dir() -> PathBuf {
    local_dir().join("logs")
}

pub fn local_run_dir() -> PathBuf {
    local_dir().join("run")
}

pub fn state_file() -> PathBuf {
    local_config_dir().join("state.json")
}

pub fn legacy_state_file() -> PathBuf {
    local_config_dir().join("cli_state.json")
}

pub fn serve_pid_file(port: u16) -> PathBuf {
    local_run_dir().join(format!("serve-{port}.json"))
}
