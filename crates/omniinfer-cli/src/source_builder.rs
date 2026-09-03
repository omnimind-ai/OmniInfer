use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use omniinfer_core::{backend_registry, paths};

pub(crate) fn build_backend(backend: &str, build_args: &[String]) -> Result<()> {
    let repo_root = paths::repo_root();
    let scripts_root = repo_root.join("scripts").join("platforms");
    if !scripts_root.is_dir() {
        anyhow::bail!(
            "Source backend builds are only available from a source checkout, not packaged releases."
        );
    }

    let registry = backend_registry::BackendRegistry::load_current();
    if registry.get(backend).is_none() {
        anyhow::bail!("Unsupported backend: {backend}");
    }

    let script = source_build_script(&scripts_root, backend);
    if !script.is_file() {
        anyhow::bail!(
            "No source build script is available for {backend} on {}.",
            std::env::consts::OS
        );
    }

    println!("Building backend from source: {backend}");
    println!("Build script: {}", script.display());
    let mut command = source_build_command(&script, build_args);
    command.current_dir(&repo_root);
    paths::propagate_cli_roots(&mut command);
    crate::hide_child_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("start source build script {}", script.display()))?;
    if !status.success() {
        anyhow::bail!(
            "Source build failed for {backend} with status {}.",
            status.code().map_or_else(
                || "terminated by signal".to_string(),
                |code| code.to_string()
            )
        );
    }
    Ok(())
}

fn source_build_script(scripts_root: &Path, backend: &str) -> PathBuf {
    #[cfg(windows)]
    {
        scripts_root.join("windows").join(backend).join("build.ps1")
    }
    #[cfg(target_os = "macos")]
    {
        scripts_root.join("macos").join(backend).join("build.sh")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        scripts_root.join("linux").join(backend).join("build.sh")
    }
}

fn source_build_command(script: &Path, build_args: &[String]) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("powershell.exe");
        command.args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ]);
        command.arg(script);
        command.args(build_args);
        command
    }
    #[cfg(unix)]
    {
        let mut command = Command::new("bash");
        command.arg(script).arg("--from-source");
        command.args(build_args);
        command
    }
}
