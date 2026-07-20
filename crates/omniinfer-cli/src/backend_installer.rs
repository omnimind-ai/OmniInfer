use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use omniinfer_core::{backend_registry, paths};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const DEFAULT_CATALOG: &str = include_str!("../../../scripts/prebuilt_backends.json");

#[derive(Debug, Clone)]
pub(crate) struct InstallOptions {
    pub(crate) backend: String,
    pub(crate) dry_run: bool,
    pub(crate) from_source: bool,
}

#[derive(Debug, Deserialize)]
struct PrebuiltCatalog {
    #[serde(default)]
    mirrors: Vec<String>,
    platforms: BTreeMap<String, BTreeMap<String, PrebuiltEntry>>,
}

#[derive(Debug, Clone, Deserialize)]
struct PrebuiltEntry {
    source: Option<String>,
    tag: Option<String>,
    url: String,
    archive: String,
    launcher: String,
    sha256: Option<String>,
    #[serde(default)]
    companion_assets: Vec<CompanionAsset>,
    #[serde(default)]
    required_files: Vec<String>,
    submodule_path: Option<String>,
    submodule_commit: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompanionAsset {
    url: String,
    archive: String,
    sha256: Option<String>,
    files: Vec<String>,
}

#[derive(Debug)]
struct DownloadedArchive {
    url: String,
    bytes: Vec<u8>,
    sha256: String,
    catalog_sha256: Option<String>,
    archive: String,
    role: String,
}

pub(crate) fn install_backend(options: InstallOptions) -> Result<()> {
    if options.from_source {
        anyhow::bail!(
            "Source builds require a source checkout. Use `scripts/install-from-source.sh` or run the backend build script from a cloned repository."
        );
    }

    let platform = current_platform_name();
    let registry = backend_registry::BackendRegistry::load_current();
    let spec = registry
        .get(&options.backend)
        .ok_or_else(|| anyhow::anyhow!("Unsupported backend: {}", options.backend))?;
    let catalog = load_catalog()?;
    let runtime_dir = PathBuf::from(&spec.runtime_dir);
    let entry_result = catalog_entry(&catalog, platform, &options.backend);
    if spec.binary_exists() {
        match entry_result.as_ref() {
            Ok(entry) => {
                let missing = missing_required_runtime_files(&runtime_dir, entry)?;
                if missing.is_empty() {
                    println!("Backend already installed: {}", options.backend);
                    if let Some(path) = &spec.launcher_path {
                        println!("Launcher: {path}");
                    }
                    return Ok(());
                }
                println!(
                    "Existing backend is incomplete; reinstalling {} (missing: {})",
                    options.backend,
                    missing.join(", ")
                );
            }
            Err(_) => {
                println!("Backend already installed: {}", options.backend);
                if let Some(path) = &spec.launcher_path {
                    println!("Launcher: {path}");
                }
                return Ok(());
            }
        }
    }
    let entry = entry_result?;
    let models_dir = spec.models_dir.as_ref().map(PathBuf::from);

    println!("Prebuilt backend: {}/{}", platform, options.backend);
    println!("  source: {}", entry.source.as_deref().unwrap_or("-"));
    println!("  tag: {}", entry.tag.as_deref().unwrap_or("-"));
    println!("  runtime: {}", runtime_dir.display());
    println!("  launcher: {}", entry.launcher);
    if let Some(note) = source_checkout_version_note(&entry) {
        println!("  version note: {note}");
    }
    print_asset_plan(
        "runtime",
        &catalog,
        &entry.url,
        entry.sha256.as_deref(),
        options.dry_run,
    );
    for (index, asset) in entry.companion_assets.iter().enumerate() {
        print_asset_plan(
            &format!("companion {}", index + 1),
            &catalog,
            &asset.url,
            asset.sha256.as_deref(),
            options.dry_run,
        );
    }
    if options.dry_run {
        return Ok(());
    }

    let archives = download_entry_archives(&catalog, entry)?;
    fs::create_dir_all(&runtime_dir)
        .with_context(|| format!("create runtime dir {}", runtime_dir.display()))?;
    if let Some(models_dir) = models_dir {
        fs::create_dir_all(&models_dir)
            .with_context(|| format!("create models dir {}", models_dir.display()))?;
    }

    let extracted_dir = temp_install_dir(&options.backend)?;
    fs::create_dir_all(&extracted_dir)
        .with_context(|| format!("create temp dir {}", extracted_dir.display()))?;
    let result = prepare_and_install_runtime(&extracted_dir, &runtime_dir, entry, &archives)
        .and_then(|launcher| {
            write_install_manifest(&runtime_dir, platform, &options.backend, entry, &archives)?;
            Ok(launcher)
        });
    let cleanup = fs::remove_dir_all(&extracted_dir);
    let launcher = result?;
    if let Err(error) = cleanup {
        eprintln!(
            "warning: failed to remove temp dir {}: {error}",
            extracted_dir.display()
        );
    }

    println!("Prebuilt backend installed: {}", launcher.display());
    Ok(())
}

fn load_catalog() -> Result<PrebuiltCatalog> {
    if let Some(path) =
        std::env::var_os("OMNIINFER_PREBUILT_CATALOG").filter(|value| !value.is_empty())
    {
        let path = PathBuf::from(path);
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("read prebuilt catalog {}", path.display()))?;
        return Ok(serde_json::from_str(&raw)
            .with_context(|| format!("parse prebuilt catalog {}", path.display()))?);
    }
    Ok(serde_json::from_str(DEFAULT_CATALOG).context("parse built-in prebuilt catalog")?)
}

fn catalog_entry<'a>(
    catalog: &'a PrebuiltCatalog,
    platform: &str,
    backend: &str,
) -> Result<&'a PrebuiltEntry> {
    catalog
        .platforms
        .get(platform)
        .ok_or_else(|| anyhow::anyhow!("no prebuilt catalog entries for platform: {platform}"))?
        .get(backend)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no prebuilt archive is configured for {platform}/{backend}. Use `omniinfer build {backend} --from-source` from a source checkout."
            )
        })
}

fn current_platform_name() -> &'static str {
    match std::env::consts::OS {
        "windows" => "windows",
        "macos" => "macos",
        _ => "linux",
    }
}

fn mirror_urls(catalog: &PrebuiltCatalog, url: &str) -> Vec<String> {
    let mut urls = Vec::new();
    if let Ok(prefixes) = std::env::var("OMNIINFER_PREBUILT_MIRROR_PREFIXES") {
        for prefix in prefixes
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            urls.push(format!("{prefix}{url}"));
        }
    }
    for prefix in &catalog.mirrors {
        if !prefix.trim().is_empty() {
            urls.push(format!("{}{}", prefix.trim(), url));
        }
    }
    urls.push(url.to_string());
    urls
}

fn print_asset_plan(
    role: &str,
    catalog: &PrebuiltCatalog,
    url: &str,
    expected_sha256: Option<&str>,
    dry_run: bool,
) {
    if let Some(expected) = expected_sha256 {
        println!("  {role} sha256: {expected}");
    } else {
        println!("  {role} checksum: not provided by catalog; recording downloaded archive digest");
    }
    if dry_run {
        for candidate in mirror_urls(catalog, url) {
            println!("  {role} would try: {candidate}");
        }
    }
}

fn download_entry_archives(
    catalog: &PrebuiltCatalog,
    entry: &PrebuiltEntry,
) -> Result<Vec<DownloadedArchive>> {
    let mut archives = Vec::with_capacity(1 + entry.companion_assets.len());
    archives.push(download_archive(
        &mirror_urls(catalog, &entry.url),
        entry.sha256.as_deref(),
        "runtime",
        &entry.archive,
    )?);
    for (index, asset) in entry.companion_assets.iter().enumerate() {
        archives.push(download_archive(
            &mirror_urls(catalog, &asset.url),
            asset.sha256.as_deref(),
            &format!("companion {}", index + 1),
            &asset.archive,
        )?);
    }
    Ok(archives)
}

fn download_archive(
    urls: &[String],
    expected_sha256: Option<&str>,
    role: &str,
    archive_type: &str,
) -> Result<DownloadedArchive> {
    let mut last_error = String::new();
    for url in urls {
        match read_url_bytes(url) {
            Ok(bytes) => {
                let sha256 = sha256_hex(&bytes);
                if let Some(expected) = expected_sha256
                    && !expected.eq_ignore_ascii_case(&sha256)
                {
                    last_error =
                        format!("checksum mismatch for {url}: expected {expected}, got {sha256}");
                    continue;
                }
                return Ok(DownloadedArchive {
                    url: url.clone(),
                    bytes,
                    sha256,
                    catalog_sha256: expected_sha256.map(str::to_string),
                    archive: archive_type.to_string(),
                    role: role.to_string(),
                });
            }
            Err(error) => {
                last_error = error.to_string();
            }
        }
    }
    anyhow::bail!("failed to download prebuilt archive; last error: {last_error}")
}

fn read_url_bytes(url: &str) -> Result<Vec<u8>> {
    println!("Downloading prebuilt archive: {url}");
    if let Some(path) = url.strip_prefix("file://") {
        return Ok(fs::read(path).with_context(|| format!("read local archive {path}"))?);
    }
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .new_agent();
    let mut response = agent
        .get(url)
        .header("User-Agent", "OmniInfer-prebuilt-installer")
        .call()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(response
        .body_mut()
        .with_config()
        .limit(512 * 1024 * 1024)
        .read_to_vec()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn temp_install_dir(backend: &str) -> Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let name = format!(
        "omni-prebuilt-{}-{}-{timestamp}",
        sanitize_name(backend),
        std::process::id()
    );
    Ok(std::env::temp_dir().join(name))
}

fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn extract_archive(bytes: &[u8], archive_type: &str, destination: &Path) -> Result<()> {
    match archive_type.to_ascii_lowercase().as_str() {
        "tar.gz" | "tgz" => {
            let decoder = GzDecoder::new(Cursor::new(bytes));
            let mut archive = tar::Archive::new(decoder);
            for entry in archive.entries()? {
                let mut entry = entry?;
                let path = entry.path()?.to_path_buf();
                validate_archive_path(&path)?;
                let entry_type = entry.header().entry_type();
                if !(entry_type.is_file() || entry_type.is_dir()) {
                    anyhow::bail!("unsupported tar entry type for {}", path.display());
                }
                entry.unpack_in(destination)?;
            }
            Ok(())
        }
        "zip" => {
            let reader = Cursor::new(bytes);
            let mut archive = zip::ZipArchive::new(reader)?;
            for index in 0..archive.len() {
                let mut file = archive.by_index(index)?;
                let enclosed = file
                    .enclosed_name()
                    .ok_or_else(|| anyhow::anyhow!("unsafe zip path: {}", file.name()))?
                    .to_path_buf();
                validate_archive_path(&enclosed)?;
                let target = destination.join(&enclosed);
                if file.is_dir() {
                    fs::create_dir_all(&target)?;
                } else {
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let mut out = File::create(&target)?;
                    std::io::copy(&mut file, &mut out)?;
                    #[cfg(unix)]
                    if let Some(mode) = file.unix_mode() {
                        use std::os::unix::fs::PermissionsExt;
                        fs::set_permissions(&target, fs::Permissions::from_mode(mode))?;
                    }
                }
            }
            Ok(())
        }
        other => anyhow::bail!("unsupported prebuilt archive type: {other}"),
    }
}

fn validate_archive_path(path: &Path) -> Result<()> {
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => anyhow::bail!("unsafe archive path: {}", path.display()),
        }
    }
    Ok(())
}

fn prepare_and_install_runtime(
    work_dir: &Path,
    runtime_dir: &Path,
    entry: &PrebuiltEntry,
    archives: &[DownloadedArchive],
) -> Result<PathBuf> {
    let primary = archives
        .first()
        .ok_or_else(|| anyhow::anyhow!("prebuilt catalog produced no runtime archive"))?;
    let primary_dir = work_dir.join("asset-0");
    fs::create_dir_all(&primary_dir)?;
    extract_archive(&primary.bytes, &primary.archive, &primary_dir)?;
    let launcher = find_launcher(&primary_dir, &entry.launcher)?;
    let source_dir = launcher
        .parent()
        .ok_or_else(|| anyhow::anyhow!("launcher has no parent directory"))?;
    let staged_bin = work_dir.join("staged-bin");
    fs::create_dir_all(&staged_bin)?;
    copy_directory_contents(source_dir, &staged_bin)?;

    if archives.len() != 1 + entry.companion_assets.len() {
        anyhow::bail!("downloaded prebuilt asset count does not match the catalog");
    }
    for (index, asset) in entry.companion_assets.iter().enumerate() {
        let archive = &archives[index + 1];
        let asset_dir = work_dir.join(format!("asset-{}", index + 1));
        fs::create_dir_all(&asset_dir)?;
        extract_archive(&archive.bytes, &archive.archive, &asset_dir)?;
        if asset.files.is_empty() {
            anyhow::bail!("companion asset {} does not declare any files", index + 1);
        }
        for file in &asset.files {
            copy_named_asset_file(&asset_dir, &staged_bin, file)?;
        }
    }
    validate_required_runtime_files(&staged_bin, entry)?;

    install_staged_runtime(&staged_bin, runtime_dir, &entry.launcher)
}

fn install_staged_runtime(
    staged_bin: &Path,
    runtime_dir: &Path,
    launcher_name: &str,
) -> Result<PathBuf> {
    let bin_dir = runtime_dir.join("bin");
    let logs_dir = runtime_dir.join("logs");
    fs::create_dir_all(&logs_dir)?;
    let suffix = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let next_dir = runtime_dir.join(format!("bin.installing-{suffix}"));
    let backup_dir = runtime_dir.join(format!("bin.backup-{suffix}"));
    copy_dir_recursive(staged_bin, &next_dir)?;
    let next_launcher = next_dir.join(launcher_name);
    if !next_launcher.is_file() {
        let _ = fs::remove_dir_all(&next_dir);
        anyhow::bail!(
            "prebuilt install failed: {} was not staged",
            next_launcher.display()
        );
    }
    make_executable(&next_launcher)?;

    if bin_dir.exists() {
        fs::rename(&bin_dir, &backup_dir).with_context(|| {
            format!(
                "move existing runtime {} to {}",
                bin_dir.display(),
                backup_dir.display()
            )
        })?;
    }
    if let Err(error) = fs::rename(&next_dir, &bin_dir) {
        if backup_dir.exists() {
            let _ = fs::rename(&backup_dir, &bin_dir);
        }
        let _ = fs::remove_dir_all(&next_dir);
        return Err(error).with_context(|| {
            format!(
                "activate staged runtime {} as {}",
                next_dir.display(),
                bin_dir.display()
            )
        });
    }
    if backup_dir.exists()
        && let Err(error) = fs::remove_dir_all(&backup_dir)
    {
        eprintln!(
            "warning: failed to remove old runtime backup {}: {error}",
            backup_dir.display()
        );
    }

    let installed_launcher = bin_dir.join(launcher_name);
    if !installed_launcher.is_file() {
        anyhow::bail!(
            "prebuilt install failed: {} was not created",
            installed_launcher.display()
        );
    }
    make_executable(&installed_launcher)?;
    Ok(installed_launcher)
}

fn copy_named_asset_file(source_root: &Path, target_root: &Path, file: &str) -> Result<()> {
    let relative = Path::new(file);
    validate_archive_path(relative)?;
    if relative.components().count() != 1 {
        anyhow::bail!("companion asset file must be a plain file name: {file}");
    }
    let source = find_launcher(source_root, file)
        .with_context(|| format!("required companion file {file} was not found"))?;
    let target = target_root.join(file);
    fs::copy(&source, &target).with_context(|| {
        format!(
            "copy companion file {} to {}",
            source.display(),
            target.display()
        )
    })?;
    Ok(())
}

fn missing_required_runtime_files(
    runtime_dir: &Path,
    entry: &PrebuiltEntry,
) -> Result<Vec<String>> {
    let bin_dir = runtime_dir.join("bin");
    let mut required = Vec::with_capacity(1 + entry.required_files.len());
    required.push(entry.launcher.as_str());
    required.extend(entry.required_files.iter().map(String::as_str));
    let mut missing = Vec::new();
    for file in required {
        let relative = Path::new(file);
        validate_archive_path(relative)?;
        if !bin_dir.join(relative).is_file() {
            missing.push(file.to_string());
        }
    }
    Ok(missing)
}

fn validate_required_runtime_files(bin_dir: &Path, entry: &PrebuiltEntry) -> Result<()> {
    let mut required = Vec::with_capacity(1 + entry.required_files.len());
    required.push(entry.launcher.as_str());
    required.extend(entry.required_files.iter().map(String::as_str));
    let mut missing = Vec::new();
    for file in required {
        let relative = Path::new(file);
        validate_archive_path(relative)?;
        if !bin_dir.join(relative).is_file() {
            missing.push(file);
        }
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "prebuilt install is incomplete; missing required files: {}",
            missing.join(", ")
        );
    }
    Ok(())
}

fn find_launcher(root: &Path, launcher: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    collect_launcher_matches(root, launcher, &mut matches)?;
    matches.sort_by(|left, right| {
        left.components()
            .count()
            .cmp(&right.components().count())
            .then_with(|| left.cmp(right))
    });
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("launcher {launcher:?} was not found in extracted archive"))
}

fn collect_launcher_matches(root: &Path, launcher: &str, matches: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_launcher_matches(&path, launcher, matches)?;
        } else if file_type.is_file()
            && path.file_name().and_then(|value| value.to_str()) == Some(launcher)
        {
            matches.push(path);
        }
    }
    Ok(())
}

fn copy_directory_contents(source_dir: &Path, target_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let source = entry.path();
        let target = target_dir.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_dir_recursive(&source, &target)?;
        } else if file_type.is_file() {
            fs::copy(&source, &target)?;
            make_executable_if_source_is_executable(&source, &target)?;
        }
    }
    Ok(())
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    copy_directory_contents(source, target)
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(permissions.mode() | 0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

fn make_executable_if_source_is_executable(source: &Path, target: &Path) -> Result<()> {
    #[cfg(not(unix))]
    let _ = (source, target);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let source_mode = fs::metadata(source)?.permissions().mode();
        if source_mode & 0o111 != 0 {
            let mut permissions = fs::metadata(target)?.permissions();
            permissions.set_mode(permissions.mode() | 0o755);
            fs::set_permissions(target, permissions)?;
        }
    }
    Ok(())
}

fn write_install_manifest(
    runtime_dir: &Path,
    platform: &str,
    backend: &str,
    entry: &PrebuiltEntry,
    archives: &[DownloadedArchive],
) -> Result<()> {
    let installed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let primary = archives
        .first()
        .ok_or_else(|| anyhow::anyhow!("prebuilt manifest requires a runtime archive"))?;
    let asset_records = archives
        .iter()
        .map(|archive| {
            json!({
                "role": archive.role,
                "url": archive.url,
                "archive": archive.archive,
                "archive_sha256": archive.sha256,
                "catalog_sha256": archive.catalog_sha256,
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema_version": 3,
        "installed_at": installed_at,
        "platform": platform,
        "backend": backend,
        "source": entry.source,
        "tag": entry.tag,
        "url": primary.url,
        "archive_sha256": primary.sha256,
        "catalog_sha256": entry.sha256,
        "archive": entry.archive,
        "launcher": entry.launcher,
        "required_files": entry.required_files,
        "assets": asset_records,
        "submodule_path": entry.submodule_path,
        "submodule_commit": entry.submodule_commit,
    });
    fs::write(
        runtime_dir.join("prebuilt.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

fn source_checkout_version_note(entry: &PrebuiltEntry) -> Option<String> {
    let submodule_path = entry.submodule_path.as_deref()?;
    if !paths::repo_root().join(submodule_path).exists() {
        return None;
    }
    let expected = entry.submodule_commit.as_deref()?;
    let actual = git_rev_parse(submodule_path)?;
    if actual == expected {
        Some(format!("{submodule_path} matches {expected}"))
    } else {
        Some(format!(
            "{submodule_path} is {actual}, catalog expects {expected}"
        ))
    }
}

fn git_rev_parse(path: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["-C", path, "rev-parse", "HEAD"])
        .current_dir(paths::repo_root())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}
