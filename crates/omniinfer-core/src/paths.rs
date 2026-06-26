use std::path::{Path, PathBuf};

const ROOT_OVERRIDE_ENV: &str = "OMNIINFER_RUST_REPO_ROOT";
const STATE_ROOT_OVERRIDE_ENV: &str = "OMNIINFER_RUST_STATE_ROOT";

pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os(ROOT_OVERRIDE_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(root);
    }
    if let Some(root) = executable_root().filter(|root| looks_like_omniinfer_root(root)) {
        return root;
    }
    manifest_repo_root()
}

fn executable_root() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

fn looks_like_omniinfer_root(root: &Path) -> bool {
    root.join("omniinfer.py").is_file()
        || root.join("service_core").is_dir()
        || root.join("runtime").is_dir()
}

fn manifest_repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate path should be under crates/<name>")
        .to_path_buf()
}

pub fn local_dir() -> PathBuf {
    state_root().join(".local")
}

pub fn config_dir() -> PathBuf {
    state_root().join("config")
}

fn state_root() -> PathBuf {
    if let Some(root) = std::env::var_os(STATE_ROOT_OVERRIDE_ENV).filter(|value| !value.is_empty())
    {
        return PathBuf::from(root);
    }
    repo_root()
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

pub fn backend_profile_dir() -> PathBuf {
    local_config_dir().join("backend_profiles")
}

pub fn backend_profile_file(backend_id: &str) -> PathBuf {
    backend_profile_dir().join(format!("{backend_id}.json"))
}

pub fn serve_pid_file(port: u16) -> PathBuf {
    local_run_dir().join(format!("serve-{port}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_repo_root_is_workspace_root() {
        let root = manifest_repo_root();
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("crates").is_dir());
    }

    #[test]
    fn portable_markers_identify_root() {
        let root = tempfile_root("portable-markers");
        std::fs::create_dir_all(root.join("runtime")).expect("create runtime marker");
        assert!(looks_like_omniinfer_root(&root));
    }

    #[test]
    fn empty_directory_is_not_portable_root() {
        let root = tempfile_root("empty-root");
        assert!(!looks_like_omniinfer_root(&root));
    }

    fn tempfile_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("omniinfer-paths-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp root");
        root
    }
}
