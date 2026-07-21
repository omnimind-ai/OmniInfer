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

struct InstallReporter {
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

    fn human(&self, message: impl AsRef<str>) {
        if !self.json {
            println!("{}", message.as_ref());
        }
    }

    fn event(&mut self, event: &str, fields: Value) {
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
    let entry_result = catalog_entry(&catalog, platform, &options.backend);
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

fn extract_archive(bytes: &[u8], archive_type: &str, destination: &Path) -> Result<()> {
    match archive_type.to_ascii_lowercase().as_str() {
        "tar.gz" | "tgz" => extract_tar_archive(bytes, destination),
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

fn extract_tar_archive(bytes: &[u8], destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    let destination_root = fs::canonicalize(destination)
        .with_context(|| format!("resolve tar destination {}", destination.display()))?;
    let links = inspect_tar_entries(bytes)?;

    let decoder = GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let entry_type = entry.header().entry_type();
        if (entry_type.is_file() || entry_type.is_dir()) && !entry.unpack_in(&destination_root)? {
            anyhow::bail!("unsafe tar path: {}", path.display());
        }
    }

    create_tar_links(&destination_root, &links)
}

fn inspect_tar_entries(bytes: &[u8]) -> Result<Vec<TarLink>> {
    let decoder = GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    let mut links = Vec::new();
    for entry in archive.entries()? {
        let entry = entry?;
        let path = entry.path()?.to_path_buf();
        validate_archive_path(&path)?;
        let entry_type = entry.header().entry_type();
        if entry_type.is_file() || entry_type.is_dir() {
            continue;
        }
        let kind = if entry_type.is_symlink() {
            TarLinkKind::Symbolic
        } else if entry_type.is_hard_link() {
            TarLinkKind::Hard
        } else {
            anyhow::bail!("unsupported tar entry type for {}", path.display());
        };
        let target = entry
            .link_name()?
            .ok_or_else(|| anyhow::anyhow!("tar link {} has no target", path.display()))?
            .into_owned();
        let resolved_target = resolve_tar_link_target(&path, &target, kind)?;
        links.push(TarLink {
            path,
            target,
            resolved_target,
            kind,
        });
    }
    validate_tar_link_graph(&links)?;
    Ok(links)
}

fn resolve_tar_link_target(path: &Path, target: &Path, kind: TarLinkKind) -> Result<PathBuf> {
    let mut resolved = PathBuf::new();
    if kind == TarLinkKind::Symbolic
        && let Some(parent) = path.parent()
    {
        resolved.push(parent);
    }
    for component in target.components() {
        match component {
            Component::Normal(value) => resolved.push(value),
            Component::CurDir => {}
            Component::ParentDir => {
                if !resolved.pop() {
                    anyhow::bail!(
                        "tar link target escapes staging root: {} -> {}",
                        path.display(),
                        target.display()
                    );
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!(
                    "tar link target must be relative: {} -> {}",
                    path.display(),
                    target.display()
                );
            }
        }
    }
    if resolved.as_os_str().is_empty() {
        anyhow::bail!(
            "tar link target is empty: {} -> {}",
            path.display(),
            target.display()
        );
    }
    Ok(resolved)
}

fn validate_tar_link_graph(links: &[TarLink]) -> Result<()> {
    let symbolic_links = links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
        .map(|link| (link.path.clone(), link.resolved_target.clone()))
        .collect::<HashMap<_, _>>();
    if symbolic_links.len()
        != links
            .iter()
            .filter(|link| link.kind == TarLinkKind::Symbolic)
            .count()
    {
        anyhow::bail!("tar archive contains duplicate symbolic link paths");
    }

    for link in links {
        let mut ancestor = link.path.parent();
        while let Some(path) = ancestor {
            if symbolic_links.contains_key(path) {
                anyhow::bail!(
                    "tar link path traverses another symbolic link: {}",
                    link.path.display()
                );
            }
            ancestor = path.parent();
        }

        let mut current = link.resolved_target.as_path();
        let mut visited = HashSet::new();
        while let Some(next) = symbolic_links.get(current) {
            if !visited.insert(current.to_path_buf()) {
                anyhow::bail!("tar symbolic link cycle at {}", link.path.display());
            }
            current = next;
        }
    }
    Ok(())
}

fn create_tar_links(destination_root: &Path, links: &[TarLink]) -> Result<()> {
    let symbolic_targets = links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
        .map(|link| (link.path.clone(), link.resolved_target.clone()))
        .collect::<HashMap<_, _>>();

    for link in links.iter().filter(|link| link.kind == TarLinkKind::Hard) {
        let final_target = resolve_final_tar_target(&link.resolved_target, &symbolic_targets)?;
        let source = canonical_target_in_root(destination_root, &final_target, link)?;
        if !source.is_file() {
            anyhow::bail!("tar hard link target is not a file: {}", source.display());
        }
        let target = checked_link_destination(destination_root, &link.path)?;
        fs::hard_link(&source, &target).with_context(|| {
            format!(
                "create tar hard link {} -> {}",
                target.display(),
                source.display()
            )
        })?;
    }

    for link in links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
    {
        let final_target = resolve_final_tar_target(&link.resolved_target, &symbolic_targets)?;
        let source = canonical_target_in_root(destination_root, &final_target, link)?;
        let target = checked_link_destination(destination_root, &link.path)?;
        create_symbolic_link(&link.target, &target, &source)?;
    }

    for link in links
        .iter()
        .filter(|link| link.kind == TarLinkKind::Symbolic)
    {
        let target = destination_root.join(&link.path);
        let resolved = fs::canonicalize(&target)
            .with_context(|| format!("resolve extracted tar link {}", target.display()))?;
        if !resolved.starts_with(destination_root) {
            anyhow::bail!(
                "extracted tar link escapes staging root: {}",
                target.display()
            );
        }
    }
    Ok(())
}

fn resolve_final_tar_target(
    target: &Path,
    symbolic_targets: &HashMap<PathBuf, PathBuf>,
) -> Result<PathBuf> {
    let mut current = target.to_path_buf();
    let mut visited = HashSet::new();
    while let Some(next) = symbolic_targets.get(&current) {
        if !visited.insert(current.clone()) {
            anyhow::bail!("tar symbolic link cycle at {}", current.display());
        }
        current = next.clone();
    }
    Ok(current)
}

fn canonical_target_in_root(
    destination_root: &Path,
    relative_target: &Path,
    link: &TarLink,
) -> Result<PathBuf> {
    let target = destination_root.join(relative_target);
    let canonical = fs::canonicalize(&target).with_context(|| {
        format!(
            "tar link target does not exist: {} -> {}",
            link.path.display(),
            link.target.display()
        )
    })?;
    if !canonical.starts_with(destination_root) {
        anyhow::bail!(
            "tar link target escapes staging root: {} -> {}",
            link.path.display(),
            link.target.display()
        );
    }
    Ok(canonical)
}

fn checked_link_destination(destination_root: &Path, relative: &Path) -> Result<PathBuf> {
    let target = destination_root.join(relative);
    let parent = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("tar link has no parent: {}", relative.display()))?;
    fs::create_dir_all(parent)?;
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("resolve tar link parent {}", parent.display()))?;
    if !canonical_parent.starts_with(destination_root) {
        anyhow::bail!("tar link path escapes staging root: {}", relative.display());
    }
    if fs::symlink_metadata(&target).is_ok() {
        anyhow::bail!("tar link path already exists: {}", relative.display());
    }
    Ok(target)
}

fn create_symbolic_link(target: &Path, link: &Path, resolved_target: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let _ = resolved_target;
        std::os::unix::fs::symlink(target, link)?;
    }
    #[cfg(windows)]
    {
        if resolved_target.is_dir() {
            std::os::windows::fs::symlink_dir(target, link)?;
        } else {
            std::os::windows::fs::symlink_file(target, link)?;
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link, resolved_target);
        anyhow::bail!("symbolic links are not supported on this platform");
    }
    Ok(())
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

fn existing_prebuilt_verification_failure(
    runtime_dir: &Path,
    entry: &PrebuiltEntry,
) -> Option<String> {
    let manifest_path = runtime_dir.join("prebuilt.json");
    if !manifest_path.is_file() {
        return None;
    }
    let raw = match fs::read_to_string(&manifest_path) {
        Ok(raw) => raw,
        Err(error) => return Some(format!("cannot read prebuilt manifest: {error}")),
    };
    let manifest: Value = match serde_json::from_str(&raw) {
        Ok(manifest) => manifest,
        Err(error) => return Some(format!("cannot parse prebuilt manifest: {error}")),
    };
    if let Some(expected) = entry.sha256.as_deref() {
        let actual = manifest
            .get("archive_sha256")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !expected.eq_ignore_ascii_case(actual) {
            return Some(format!(
                "runtime archive digest is {actual:?}, expected {expected}"
            ));
        }
    }
    let assets = manifest.get("assets").and_then(Value::as_array);
    for (index, companion) in entry.companion_assets.iter().enumerate() {
        let Some(expected) = companion.sha256.as_deref() else {
            continue;
        };
        let actual = assets
            .and_then(|items| items.get(index + 1))
            .and_then(|asset| asset.get("archive_sha256"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !expected.eq_ignore_ascii_case(actual) {
            return Some(format!(
                "companion {} archive digest is {actual:?}, expected {expected}",
                index + 1
            ));
        }
    }
    None
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
    let source_root = fs::canonicalize(source_dir)
        .with_context(|| format!("resolve copy source {}", source_dir.display()))?;
    copy_directory_contents_in_root(source_dir, target_dir, &source_root)
}

fn copy_directory_contents_in_root(
    source_dir: &Path,
    target_dir: &Path,
    source_root: &Path,
) -> Result<()> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let source = entry.path();
        let target = target_dir.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            copy_safe_symbolic_link(&source, &target, source_root)?;
        } else if file_type.is_dir() {
            fs::create_dir_all(&target)?;
            copy_directory_contents_in_root(&source, &target, source_root)?;
        } else if file_type.is_file() {
            fs::copy(&source, &target)?;
            make_executable_if_source_is_executable(&source, &target)?;
        } else {
            anyhow::bail!("unsupported runtime file type: {}", source.display());
        }
    }
    Ok(())
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    copy_directory_contents(source, target)
}

fn copy_safe_symbolic_link(source: &Path, target: &Path, source_root: &Path) -> Result<()> {
    let link_target = fs::read_link(source)
        .with_context(|| format!("read runtime symbolic link {}", source.display()))?;
    if link_target.is_absolute() {
        anyhow::bail!(
            "runtime symbolic link target must be relative: {}",
            source.display()
        );
    }
    let resolved = fs::canonicalize(
        source
            .parent()
            .ok_or_else(|| anyhow::anyhow!("runtime symbolic link has no parent"))?
            .join(&link_target),
    )
    .with_context(|| format!("resolve runtime symbolic link {}", source.display()))?;
    if !resolved.starts_with(source_root) {
        anyhow::bail!(
            "runtime symbolic link escapes source root: {}",
            source.display()
        );
    }
    create_symbolic_link(&link_target, target, &resolved)
        .with_context(|| format!("copy runtime symbolic link {}", source.display()))
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
    catalog: &PrebuiltCatalog,
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
        "tag": catalog.resolved_tag(entry),
        "url": primary.url,
        "archive_sha256": primary.sha256,
        "catalog_sha256": entry.sha256,
        "archive": entry.archive,
        "launcher": entry.launcher,
        "required_files": entry.required_files,
        "assets": asset_records,
        "submodule_path": catalog.resolved_submodule_path(entry),
        "submodule_commit": catalog.resolved_submodule_commit(entry),
    });
    fs::write(
        runtime_dir.join("prebuilt.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

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
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io;

    enum TestTarEntry<'a> {
        File(&'a str, &'a [u8]),
        Symlink(&'a str, &'a str),
        HardLink(&'a str, &'a str),
        Special(&'a str),
        RawFilePath(&'a str),
    }

    #[test]
    fn tar_links_extract_and_survive_runtime_staging() {
        let archive = build_test_tar(&[
            TestTarEntry::File("runtime/llama-server", b"launcher"),
            TestTarEntry::File("runtime/libreal.dylib", b"library"),
            TestTarEntry::Symlink("runtime/libalias.dylib", "libreal.dylib"),
            TestTarEntry::HardLink("runtime/libhard.dylib", "runtime/libreal.dylib"),
        ]);
        let extracted = test_dir("safe-links-extracted");
        extract_archive(&archive, "tar.gz", &extracted).expect("extract safe links");

        let symlink = extracted.join("runtime/libalias.dylib");
        assert!(
            fs::symlink_metadata(&symlink)
                .expect("symlink metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_link(&symlink).unwrap(), Path::new("libreal.dylib"));
        assert_eq!(
            fs::read(extracted.join("runtime/libhard.dylib")).unwrap(),
            b"library"
        );

        let staged = test_dir("safe-links-staged");
        copy_dir_recursive(&extracted.join("runtime"), &staged).expect("stage runtime links");
        let staged_symlink = staged.join("libalias.dylib");
        assert!(
            fs::symlink_metadata(&staged_symlink)
                .expect("staged symlink metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::canonicalize(&staged_symlink).unwrap(),
            fs::canonicalize(staged.join("libreal.dylib")).unwrap()
        );

        fs::remove_dir_all(extracted).ok();
        fs::remove_dir_all(staged).ok();
    }

    #[test]
    fn tar_extractor_rejects_unsafe_paths_links_and_entry_types() {
        let cases = [
            (
                "absolute-path",
                build_test_tar(&[TestTarEntry::RawFilePath("/tmp/escape")]),
                "unsafe archive path",
            ),
            (
                "parent-path",
                build_test_tar(&[TestTarEntry::RawFilePath("../escape")]),
                "unsafe archive path",
            ),
            (
                "absolute-link",
                build_test_tar(&[
                    TestTarEntry::File("runtime/real", b"safe"),
                    TestTarEntry::Symlink("runtime/link", "/tmp/escape"),
                ]),
                "tar link target must be relative",
            ),
            (
                "escaping-link",
                build_test_tar(&[
                    TestTarEntry::File("runtime/real", b"safe"),
                    TestTarEntry::Symlink("runtime/link", "../../escape"),
                ]),
                "tar link target escapes staging root",
            ),
            (
                "dangling-link",
                build_test_tar(&[TestTarEntry::Symlink("runtime/link", "missing")]),
                "tar link target does not exist",
            ),
            (
                "link-cycle",
                build_test_tar(&[
                    TestTarEntry::Symlink("runtime/one", "two"),
                    TestTarEntry::Symlink("runtime/two", "one"),
                ]),
                "tar symbolic link cycle",
            ),
            (
                "escaping-hard-link",
                build_test_tar(&[TestTarEntry::HardLink("runtime/link", "../escape")]),
                "tar link target escapes staging root",
            ),
            (
                "special-entry",
                build_test_tar(&[TestTarEntry::Special("runtime/device")]),
                "unsupported tar entry type",
            ),
        ];

        for (name, archive, expected) in cases {
            let destination = test_dir(name);
            let error = extract_archive(&archive, "tar.gz", &destination)
                .expect_err("unsafe tar must fail");
            assert!(
                error.to_string().contains(expected),
                "{name}: expected {expected:?}, got {error:#}"
            );
            fs::remove_dir_all(destination).ok();
        }
    }

    #[test]
    fn tar_extractor_rejects_canonical_target_outside_staging() {
        let destination = test_dir("canonical-target-destination");
        let outside = test_dir("canonical-target-outside");
        fs::write(outside.join("library.dylib"), "outside").unwrap();
        fs::create_dir_all(destination.join("runtime")).unwrap();
        std::os::unix::fs::symlink(
            outside.join("library.dylib"),
            destination.join("runtime/external"),
        )
        .unwrap();
        let archive = build_test_tar(&[TestTarEntry::Symlink("runtime/link.dylib", "external")]);

        let error = extract_archive(&archive, "tar.gz", &destination)
            .expect_err("canonical target outside staging must fail");
        assert!(
            error
                .to_string()
                .contains("tar link target escapes staging root")
        );

        fs::remove_dir_all(destination).ok();
        fs::remove_dir_all(outside).ok();
    }

    #[test]
    fn runtime_staging_rejects_symbolic_links_outside_source_root() {
        let source = test_dir("copy-external-source");
        let destination = test_dir("copy-external-destination");
        let outside = test_dir("copy-external-outside");
        fs::write(outside.join("library.dylib"), "outside").unwrap();
        let outside_name = outside.file_name().expect("outside directory name");
        std::os::unix::fs::symlink(
            Path::new("..").join(outside_name).join("library.dylib"),
            source.join("external.dylib"),
        )
        .unwrap();

        let error = copy_dir_recursive(&source, &destination)
            .expect_err("runtime staging must reject external symbolic link");
        assert!(
            format!("{error:#}").contains("runtime symbolic link escapes source root"),
            "{error:#}"
        );

        fs::remove_dir_all(source).ok();
        fs::remove_dir_all(destination).ok();
        fs::remove_dir_all(outside).ok();
    }

    fn test_dir(name: &str) -> PathBuf {
        let path = temp_install_dir(name).expect("test temp path");
        fs::remove_dir_all(&path).ok();
        fs::create_dir_all(&path).expect("create test directory");
        path
    }

    fn build_test_tar(entries: &[TestTarEntry<'_>]) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for entry in entries {
            match entry {
                TestTarEntry::File(path, contents) => {
                    let mut header = tar::Header::new_gnu();
                    header.set_path(path).unwrap();
                    header.set_size(contents.len() as u64);
                    header.set_mode(0o755);
                    header.set_cksum();
                    builder.append(&header, *contents).unwrap();
                }
                TestTarEntry::Symlink(path, target) => {
                    append_test_link(&mut builder, path, target, tar::EntryType::Symlink);
                }
                TestTarEntry::HardLink(path, target) => {
                    append_test_link(&mut builder, path, target, tar::EntryType::Link);
                }
                TestTarEntry::Special(path) => {
                    let mut header = tar::Header::new_gnu();
                    header.set_path(path).unwrap();
                    header.set_size(0);
                    header.set_mode(0o600);
                    header.set_entry_type(tar::EntryType::Char);
                    header.set_cksum();
                    builder.append(&header, io::empty()).unwrap();
                }
                TestTarEntry::RawFilePath(path) => {
                    let mut header = tar::Header::new_gnu();
                    header.set_size(0);
                    header.set_mode(0o600);
                    set_raw_header_value(&mut header, 0, 100, path.as_bytes());
                    header.set_cksum();
                    builder.append(&header, io::empty()).unwrap();
                }
            }
        }
        let encoder = builder.into_inner().expect("finish tar");
        encoder.finish().expect("finish gzip")
    }

    fn append_test_link(
        builder: &mut tar::Builder<GzEncoder<Vec<u8>>>,
        path: &str,
        target: &str,
        entry_type: tar::EntryType,
    ) {
        let mut header = tar::Header::new_gnu();
        header.set_path(path).unwrap();
        header.set_link_name(target).unwrap();
        header.set_size(0);
        header.set_mode(0o777);
        header.set_entry_type(entry_type);
        header.set_cksum();
        builder.append(&header, io::empty()).unwrap();
    }

    fn set_raw_header_value(header: &mut tar::Header, offset: usize, size: usize, value: &[u8]) {
        assert!(value.len() < size);
        let bytes = header.as_mut_bytes();
        bytes[offset..offset + size].fill(0);
        bytes[offset..offset + value.len()].copy_from_slice(value);
    }
}
