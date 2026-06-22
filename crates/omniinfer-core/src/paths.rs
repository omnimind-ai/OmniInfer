use std::path::{Path, PathBuf};

pub fn repo_root() -> PathBuf {
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

pub fn state_file() -> PathBuf {
    local_config_dir().join("state.json")
}

pub fn legacy_state_file() -> PathBuf {
    local_config_dir().join("cli_state.json")
}
