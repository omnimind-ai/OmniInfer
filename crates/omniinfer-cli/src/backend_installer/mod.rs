use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use omniinfer_core::{backend_registry, paths};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::prebuilt_catalog::{
    PrebuiltCatalog, PrebuiltEntry, current_platform_name, load_catalog,
};

#[derive(Debug, Clone)]
pub(crate) struct InstallOptions {
    pub(crate) backend: String,
    pub(crate) dry_run: bool,
    pub(crate) from_source: bool,
    pub(crate) json: bool,
    pub(crate) wsl_distro: Option<String>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TarLinkKind {
    Symbolic,
    Hard,
}

#[derive(Clone, Debug)]
struct TarLink {
    path: PathBuf,
    target: PathBuf,
    resolved_target: PathBuf,
    kind: TarLinkKind,
}

pub(super) struct InstallReporter {
    backend: String,
    json: bool,
    sequence: u64,
}

impl InstallReporter {
    fn new(backend: &str, json: bool) -> Self {
        Self {
            backend: backend.to_string(),
            json,
            sequence: 0,
        }
    }

    pub(super) fn human(&self, message: impl AsRef<str>) {
        if !self.json {
            println!("{}", message.as_ref());
        }
    }

    pub(super) fn event(&mut self, event: &str, fields: Value) {
        if !self.json {
            return;
        }
        self.sequence += 1;
        let mut payload = json!({
            "schema_version": 1,
            "sequence": self.sequence,
            "event": event,
            "backend": self.backend,
        });
        if let (Some(target), Some(source)) = (payload.as_object_mut(), fields.as_object()) {
            for (key, value) in source {
                target.insert(key.clone(), value.clone());
            }
        }
        println!(
            "{}",
            serde_json::to_string(&payload).expect("serialize install event")
        );
        let _ = std::io::stdout().flush();
    }
}

pub(crate) fn install_backend(options: InstallOptions) -> Result<()> {
    let mut reporter = InstallReporter::new(&options.backend, options.json);
    let result = install_backend_inner(&options, &mut reporter);
    if let Err(error) = &result {
        reporter.event(
            "error",
            json!({
                "message": error.to_string(),
            }),
        );
    }
    result
}

fn install_backend_inner(options: &InstallOptions, reporter: &mut InstallReporter) -> Result<()> {
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
    reporter.event(
        "install_started",
        json!({
            "platform": platform,
            "state_root": paths::state_root(),
            "runtime_root": paths::runtime_root_override(),
            "runtime_dir": runtime_dir,
            "dry_run": options.dry_run,
        }),
    );
    if let Some(entry) = catalog.python_runtime(platform, &options.backend) {
        return crate::wsl_runtime_installer::install_wsl_python_runtime(
            &options.backend,
            &runtime_dir,
            entry,
            options.wsl_distro.as_deref(),
            options.dry_run,
            reporter,
            &catalog,
        );
    }
    if options.wsl_distro.is_some() {
        anyhow::bail!("--wsl-distro is only valid for a managed WSL2 backend");
    }
    let entry_result = catalog_entry(&catalog, platform, &options.backend);
    if spec.binary_exists() {
        match entry_result.as_ref() {
            Ok(entry) => {
                let missing = missing_required_runtime_files(&runtime_dir, entry)?;
                if missing.is_empty() {
                    if let Some(reason) =
                        existing_prebuilt_verification_failure(&runtime_dir, entry)
                    {
                        reporter.human(format!(
                            "Existing prebuilt backend is not verified by the current catalog; reinstalling {} ({reason})",
                            options.backend
                        ));
                        reporter.event(
                            "repair_started",
                            json!({ "reason": reason, "missing_files": [] }),
                        );
                    } else {
                        reporter.human(format!("Backend already installed: {}", options.backend));
                        if let Some(path) = &spec.launcher_path {
                            reporter.human(format!("Launcher: {path}"));
                        }
                        reporter.event(
                            "already_installed",
                            json!({
                                "runtime_dir": runtime_dir,
                                "launcher": spec.launcher_path,
                            }),
                        );
                        return Ok(());
                    }
                } else {
                    reporter.human(format!(
                        "Existing backend is incomplete; reinstalling {} (missing: {})",
                        options.backend,
                        missing.join(", ")
                    ));
                    reporter.event("repair_started", json!({ "missing_files": missing }));
                }
            }
            Err(_) => {
                reporter.human(format!("Backend already installed: {}", options.backend));
                if let Some(path) = &spec.launcher_path {
                    reporter.human(format!("Launcher: {path}"));
                }
                reporter.event(
                    "already_installed",
                    json!({
                        "runtime_dir": runtime_dir,
                        "launcher": spec.launcher_path,
                    }),
                );
                return Ok(());
            }
        }
    }
    let entry = entry_result?;
    let models_dir = spec.models_dir.as_ref().map(PathBuf::from);

    reporter.human(format!(
        "Prebuilt backend: {}/{}",
        platform, options.backend
    ));
    reporter.human(format!(
        "  source: {}",
        entry.source.as_deref().unwrap_or("-")
    ));
    reporter.human(format!(
        "  tag: {}",
        catalog.resolved_tag(entry).unwrap_or("-")
    ));
    reporter.human(format!("  runtime: {}", runtime_dir.display()));
    reporter.human(format!("  launcher: {}", entry.launcher));
    if let Some(note) = source_checkout_version_note(&catalog, entry) {
        reporter.human(format!("  version note: {note}"));
    }
    print_asset_plan(
        reporter,
        "runtime",
        1,
        1 + entry.companion_assets.len(),
        &catalog,
        &entry.url,
        entry.sha256.as_deref(),
        options.dry_run,
    );
    for (index, asset) in entry.companion_assets.iter().enumerate() {
        print_asset_plan(
            reporter,
            &format!("companion {}", index + 1),
            index + 2,
            1 + entry.companion_assets.len(),
            &catalog,
            &asset.url,
            asset.sha256.as_deref(),
            options.dry_run,
        );
    }
    if options.dry_run {
        reporter.event("dry_run_completed", json!({ "runtime_dir": runtime_dir }));
        return Ok(());
    }

    let archives = download_entry_archives(&catalog, entry, reporter)?;
    fs::create_dir_all(&runtime_dir)
        .with_context(|| format!("create runtime dir {}", runtime_dir.display()))?;
    if let Some(models_dir) = models_dir {
        fs::create_dir_all(&models_dir)
            .with_context(|| format!("create models dir {}", models_dir.display()))?;
    }

    let extracted_dir = temp_install_dir(&options.backend)?;
    fs::create_dir_all(&extracted_dir)
        .with_context(|| format!("create temp dir {}", extracted_dir.display()))?;
    reporter.event(
        "staging_started",
        json!({
            "asset_count": archives.len(),
            "runtime_dir": runtime_dir,
        }),
    );
    let result = prepare_and_install_runtime(&extracted_dir, &runtime_dir, entry, &archives)
        .and_then(|launcher| {
            write_install_manifest(
                &runtime_dir,
                platform,
                &options.backend,
                &catalog,
                entry,
                &archives,
            )?;
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

    reporter.human(format!(
        "Prebuilt backend installed: {}",
        launcher.display()
    ));
    reporter.event(
        "completed",
        json!({
            "runtime_dir": runtime_dir,
            "launcher": launcher,
            "manifest": runtime_dir.join("prebuilt.json"),
        }),
    );
    Ok(())
}

fn catalog_entry<'a>(
    catalog: &'a PrebuiltCatalog,
    platform: &str,
    backend: &str,
) -> Result<&'a PrebuiltEntry> {
    if let Some(excluded) =
        catalog.excluded_release_asset(platform, backend, std::env::consts::ARCH)
    {
        let tag = catalog
            .sources
            .get(&excluded.source)
            .and_then(|source| source.tag.as_deref())
            .unwrap_or("unknown release");
        anyhow::bail!(
            "official prebuilt archive for {platform}/{backend} ({tag}, {}) is intentionally unavailable: {}. Use `omniinfer build {backend} --from-source` from a source checkout.",
            excluded.architecture,
            excluded.reason
        );
    }
    catalog.entry(platform, backend).ok_or_else(|| {
            anyhow::anyhow!(
                "no prebuilt archive is configured for {platform}/{backend}. Use `omniinfer build {backend} --from-source` from a source checkout."
            )
        })
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
    reporter: &mut InstallReporter,
    role: &str,
    asset_index: usize,
    asset_count: usize,
    catalog: &PrebuiltCatalog,
    url: &str,
    expected_sha256: Option<&str>,
    dry_run: bool,
) {
    if let Some(expected) = expected_sha256 {
        reporter.human(format!("  {role} sha256: {expected}"));
    } else {
        reporter.human(format!(
            "  {role} checksum: not provided by catalog; recording downloaded archive digest"
        ));
    }
    let candidates = mirror_urls(catalog, url);
    reporter.event(
        "asset_planned",
        json!({
            "role": role,
            "asset_index": asset_index,
            "asset_count": asset_count,
            "url": url,
            "candidate_urls": candidates,
            "expected_sha256": expected_sha256,
        }),
    );
    if dry_run {
        for candidate in candidates {
            reporter.human(format!("  {role} would try: {candidate}"));
        }
    }
}

fn download_entry_archives(
    catalog: &PrebuiltCatalog,
    entry: &PrebuiltEntry,
    reporter: &mut InstallReporter,
) -> Result<Vec<DownloadedArchive>> {
    let asset_count = 1 + entry.companion_assets.len();
    let mut archives = Vec::with_capacity(asset_count);
    archives.push(download_archive(
        &mirror_urls(catalog, &entry.url),
        entry.sha256.as_deref(),
        "runtime",
        &entry.archive,
        1,
        asset_count,
        reporter,
    )?);
    for (index, asset) in entry.companion_assets.iter().enumerate() {
        archives.push(download_archive(
            &mirror_urls(catalog, &asset.url),
            asset.sha256.as_deref(),
            &format!("companion {}", index + 1),
            &asset.archive,
            index + 2,
            asset_count,
            reporter,
        )?);
    }
    Ok(archives)
}

fn download_archive(
    urls: &[String],
    expected_sha256: Option<&str>,
    role: &str,
    archive_type: &str,
    asset_index: usize,
    asset_count: usize,
    reporter: &mut InstallReporter,
) -> Result<DownloadedArchive> {
    let mut last_error = String::new();
    for url in urls {
        match read_url_bytes(url, role, asset_index, asset_count, reporter) {
            Ok(bytes) => {
                let sha256 = sha256_hex(&bytes);
                if let Some(expected) = expected_sha256
                    && !expected.eq_ignore_ascii_case(&sha256)
                {
                    last_error =
                        format!("checksum mismatch for {url}: expected {expected}, got {sha256}");
                    reporter.event(
                        "checksum_failed",
                        json!({
                            "role": role,
                            "asset_index": asset_index,
                            "asset_count": asset_count,
                            "url": url,
                            "expected_sha256": expected,
                            "actual_sha256": sha256,
                        }),
                    );
                    continue;
                }
                reporter.event(
                    "checksum_verified",
                    json!({
                        "role": role,
                        "asset_index": asset_index,
                        "asset_count": asset_count,
                        "url": url,
                        "bytes": bytes.len(),
                        "sha256": sha256,
                        "expected_sha256": expected_sha256,
                    }),
                );
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
                reporter.event(
                    "download_failed",
                    json!({
                        "role": role,
                        "asset_index": asset_index,
                        "asset_count": asset_count,
                        "url": url,
                        "message": last_error,
                    }),
                );
            }
        }
    }
    anyhow::bail!("failed to download prebuilt archive; last error: {last_error}")
}

pub(super) fn download_verified_asset(
    catalog: &PrebuiltCatalog,
    url: &str,
    expected_sha256: &str,
    role: &str,
    reporter: &mut InstallReporter,
) -> Result<Vec<u8>> {
    download_archive(
        &mirror_urls(catalog, url),
        Some(expected_sha256),
        role,
        "asset",
        1,
        1,
        reporter,
    )
    .map(|archive| archive.bytes)
}

fn read_url_bytes(
    url: &str,
    role: &str,
    asset_index: usize,
    asset_count: usize,
    reporter: &mut InstallReporter,
) -> Result<Vec<u8>> {
    reporter.human(format!("Downloading prebuilt archive: {url}"));
    reporter.event(
        "download_started",
        json!({
            "role": role,
            "asset_index": asset_index,
            "asset_count": asset_count,
            "url": url,
        }),
    );
    if let Some(path) = url.strip_prefix("file://") {
        let file = File::open(path).with_context(|| format!("read local archive {path}"))?;
        let total = file.metadata().ok().map(|metadata| metadata.len());
        return read_with_progress(file, total, url, role, asset_index, asset_count, reporter);
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
    let total = response.body().content_length();
    read_with_progress(
        response.body_mut().as_reader(),
        total,
        url,
        role,
        asset_index,
        asset_count,
        reporter,
    )
}

fn read_with_progress(
    mut reader: impl Read,
    total: Option<u64>,
    url: &str,
    role: &str,
    asset_index: usize,
    asset_count: usize,
    reporter: &mut InstallReporter,
) -> Result<Vec<u8>> {
    const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
    const REPORT_INTERVAL_BYTES: u64 = 1024 * 1024;
    if total.is_some_and(|value| value > MAX_ARCHIVE_BYTES) {
        anyhow::bail!("prebuilt archive exceeds the 512 MiB limit");
    }
    let mut bytes = Vec::with_capacity(total.unwrap_or_default().min(16 * 1024 * 1024) as usize);
    let mut buffer = [0_u8; 64 * 1024];
    let mut downloaded = 0_u64;
    let mut next_report = REPORT_INTERVAL_BYTES;
    let mut last_reported = None;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if count == 0 {
            break;
        }
        downloaded += count as u64;
        if downloaded > MAX_ARCHIVE_BYTES {
            anyhow::bail!("prebuilt archive exceeds the 512 MiB limit");
        }
        bytes.extend_from_slice(&buffer[..count]);
        if downloaded >= next_report || total.is_some_and(|value| downloaded >= value) {
            reporter.event(
                "download_progress",
                json!({
                    "role": role,
                    "asset_index": asset_index,
                    "asset_count": asset_count,
                    "url": url,
                    "bytes_downloaded": downloaded,
                    "bytes_total": total,
                }),
            );
            last_reported = Some(downloaded);
            next_report = downloaded.saturating_add(REPORT_INTERVAL_BYTES);
        }
    }
    if last_reported != Some(downloaded) {
        reporter.event(
            "download_progress",
            json!({
                "role": role,
                "asset_index": asset_index,
                "asset_count": asset_count,
                "url": url,
                "bytes_downloaded": downloaded,
                "bytes_total": total,
            }),
        );
    }
    Ok(bytes)
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

mod archive;

use archive::*;
mod staging;

use staging::*;
fn source_checkout_version_note(
    catalog: &PrebuiltCatalog,
    entry: &PrebuiltEntry,
) -> Option<String> {
    let submodule_path = catalog.resolved_submodule_path(entry)?;
    if !paths::repo_root().join(submodule_path).exists() {
        return None;
    }
    let expected = catalog.resolved_submodule_commit(entry)?;
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

#[cfg(all(test, unix))]
mod tests;
