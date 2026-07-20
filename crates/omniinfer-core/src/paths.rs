use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const ROOT_OVERRIDE_ENV: &str = "OMNIINFER_RUST_REPO_ROOT";
const STATE_ROOT_OVERRIDE_ENV: &str = "OMNIINFER_STATE_ROOT";
const LEGACY_STATE_ROOT_OVERRIDE_ENV: &str = "OMNIINFER_RUST_STATE_ROOT";
const RUNTIME_ROOT_OVERRIDE_ENV: &str = "OMNIINFER_RUNTIME_ROOT";

static STATE_ROOT_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();
static RUNTIME_ROOT_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

pub fn configure_cli_roots(
    state_root: Option<PathBuf>,
    runtime_root: Option<PathBuf>,
) -> Result<(), String> {
    if let Some(root) = state_root {
        set_once(&STATE_ROOT_OVERRIDE, absolute_cli_path(root), "state root")?;
    }
    if let Some(root) = runtime_root {
        set_once(
            &RUNTIME_ROOT_OVERRIDE,
            absolute_cli_path(root),
            "runtime root",
        )?;
    }
    Ok(())
}

fn set_once(cell: &OnceLock<PathBuf>, value: PathBuf, label: &str) -> Result<(), String> {
    if let Some(existing) = cell.get() {
        if existing == &value {
            return Ok(());
        }
        return Err(format!(
            "{label} is already configured as {}",
            existing.display()
        ));
    }
    cell.set(value)
        .map_err(|_| format!("failed to configure {label}"))
}

fn absolute_cli_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&path))
            .unwrap_or(path)
    }
}

pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os(ROOT_OVERRIDE_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(root);
    }
    let manifest_root = manifest_repo_root();
    if let Some(executable_root) = executable_root() {
        return resolve_repo_root(executable_root, manifest_root);
    }
    manifest_root
}

fn executable_root() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

fn looks_like_omniinfer_root(root: &Path) -> bool {
    root.join("Cargo.toml").is_file()
        || root.join("runtime").is_dir()
        || looks_like_cli_only_package_root(root)
}

fn looks_like_cli_only_package_root(root: &Path) -> bool {
    root.join("VERSION").is_file()
        && (root.join("omniinfer").is_file() || root.join("omniinfer.exe").is_file())
}

fn manifest_repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate path should be under crates/<name>")
        .to_path_buf()
}

fn resolve_repo_root(executable_root: PathBuf, manifest_root: PathBuf) -> PathBuf {
    if looks_like_omniinfer_root(&executable_root) {
        return executable_root;
    }
    if executable_root.starts_with(&manifest_root) {
        return manifest_root;
    }
    executable_root
}

pub fn local_dir() -> PathBuf {
    state_root().join(".local")
}

pub fn config_dir() -> PathBuf {
    state_root().join("config")
}

pub fn state_root() -> PathBuf {
    if let Some(root) = STATE_ROOT_OVERRIDE.get() {
        return root.clone();
    }
    if let Some(root) = std::env::var_os(STATE_ROOT_OVERRIDE_ENV)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var_os(LEGACY_STATE_ROOT_OVERRIDE_ENV).filter(|value| !value.is_empty())
        })
    {
        return PathBuf::from(root);
    }
    repo_root()
}

pub fn runtime_root_override() -> Option<PathBuf> {
    RUNTIME_ROOT_OVERRIDE.get().cloned().or_else(|| {
        std::env::var_os(RUNTIME_ROOT_OVERRIDE_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
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

pub fn admin_keys_file() -> PathBuf {
    local_config_dir().join("admin_keys.json")
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
    fn cli_only_package_markers_identify_root() {
        let root = tempfile_root("cli-only-markers");
        std::fs::write(root.join("VERSION"), "0.3.2").expect("write version marker");
        std::fs::write(root.join("omniinfer"), "").expect("write launcher marker");
        assert!(looks_like_omniinfer_root(&root));
    }

    #[test]
    fn empty_directory_is_not_portable_root() {
        let root = tempfile_root("empty-root");
        assert!(!looks_like_omniinfer_root(&root));
    }

    #[test]
    fn source_checkout_binary_uses_manifest_root() {
        let manifest_root = tempfile_root("source-manifest");
        let executable_root = manifest_root.join("target").join("release");
        std::fs::create_dir_all(&executable_root).expect("create executable root");
        assert_eq!(
            resolve_repo_root(executable_root, manifest_root.clone()),
            manifest_root
        );
    }

    #[test]
    fn installed_single_binary_uses_executable_root() {
        let manifest_root = tempfile_root("installed-manifest");
        let executable_root = tempfile_root("installed-bin");
        assert_eq!(
            resolve_repo_root(executable_root.clone(), manifest_root),
            executable_root
        );
    }

    fn tempfile_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("omniinfer-paths-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp root");
        root
    }
}
