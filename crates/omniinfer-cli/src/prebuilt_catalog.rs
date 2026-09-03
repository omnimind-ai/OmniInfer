use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_CATALOG: &str = include_str!("../../../scripts/prebuilt_backends.json");
pub(crate) const REQUIRED_ROCM_SYSTEM_PACKAGES: &[&str] = &[
    "comgr",
    "hipblas",
    "hipblaslt",
    "hipfft",
    "hiprand",
    "hip-runtime-amd",
    "hipsolver",
    "hipsparse",
    "hipsparselt",
    "hsa-rocr",
    "libopenmpi3t64",
    "libpython3.12-dev",
    "miopen-hip",
    "openmp-extras-runtime",
    "python3.12-dev",
    "rccl",
    "rocblas",
    "rocfft",
    "rocm-hip-runtime",
    "rocm-core",
    "rocm-device-libs",
    "rocm-language-runtime",
    "rocm-llvm",
    "rocm-smi-lib",
    "rocminfo",
    "rocprofiler-register",
    "rocrand",
    "rocsolver",
    "rocsparse",
    "roctracer",
];
pub(crate) const REQUIRED_ROCM_PACKAGE_ASSETS: &[&str] = &[
    "comgr",
    "hipblas",
    "hipblaslt",
    "hipfft",
    "hiprand",
    "hip-runtime-amd",
    "hipsolver",
    "hipsparse",
    "hipsparselt",
    "hsa-rocr",
    "libpython3.12-dev",
    "miopen-hip",
    "openmp-extras-runtime",
    "python3.12-dev",
    "rccl",
    "rocblas",
    "rocfft",
    "rocm-hip-runtime",
    "rocm-core",
    "rocm-device-libs",
    "rocm-language-runtime",
    "rocm-llvm",
    "rocm-smi-lib",
    "rocminfo",
    "rocprofiler-register",
    "rocrand",
    "rocsolver",
    "rocsparse",
    "roctracer",
];

#[derive(Debug, Deserialize)]
pub(crate) struct PrebuiltCatalog {
    #[serde(default)]
    pub(crate) schema_version: u32,
    #[serde(default)]
    pub(crate) mirrors: Vec<String>,
    #[serde(default)]
    pub(crate) sources: BTreeMap<String, SourceMetadata>,
    #[serde(default)]
    pub(crate) python_runtimes: BTreeMap<String, BTreeMap<String, PythonRuntimeEntry>>,
    #[serde(default)]
    pub(crate) excluded_release_assets: Vec<ExcludedReleaseAsset>,
    pub(crate) platforms: BTreeMap<String, BTreeMap<String, PrebuiltEntry>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExcludedReleaseAsset {
    pub(crate) source: String,
    pub(crate) platform: String,
    pub(crate) backend: String,
    pub(crate) architecture: String,
    pub(crate) url: String,
    pub(crate) sha256: String,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PythonRuntimeEntry {
    pub(crate) source: String,
    pub(crate) tag: String,
    pub(crate) package: String,
    pub(crate) python: String,
    pub(crate) launcher: String,
    pub(crate) uv: BTreeMap<String, ToolAsset>,
    pub(crate) variants: BTreeMap<String, PythonRuntimeVariant>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ToolAsset {
    pub(crate) version: String,
    pub(crate) url: String,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct PythonRuntimeVariant {
    pub(crate) version: String,
    pub(crate) reported_version: Option<String>,
    pub(crate) accelerator: String,
    pub(crate) runtime_version: String,
    pub(crate) reported_runtime_version: Option<String>,
    pub(crate) torch_backend: Option<String>,
    pub(crate) minimum_driver: Option<String>,
    pub(crate) build_commit: Option<String>,
    pub(crate) index_url: Option<String>,
    pub(crate) rocm_system: Option<RocmSystemRuntime>,
    pub(crate) url: String,
    pub(crate) sha256: String,
}

impl PythonRuntimeVariant {
    pub(crate) fn reported_version(&self) -> &str {
        self.reported_version.as_deref().unwrap_or(&self.version)
    }

    pub(crate) fn reported_runtime_version(&self) -> &str {
        self.reported_runtime_version
            .as_deref()
            .unwrap_or(&self.runtime_version)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RocmSystemRuntime {
    pub(crate) apt_repository: String,
    pub(crate) repository_key: ToolAsset,
    pub(crate) packages: BTreeMap<String, String>,
    pub(crate) package_assets: BTreeMap<String, RocmPackageAsset>,
    pub(crate) rocdxg: ToolAsset,
    pub(crate) required_gfx: Vec<String>,
    pub(crate) minimum_windows_release: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RocmPackageAsset {
    pub(crate) version: String,
    pub(crate) url: String,
    pub(crate) filename: String,
    pub(crate) size: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SourceMetadata {
    pub(crate) tag: Option<String>,
    pub(crate) submodule_tag: Option<String>,
    pub(crate) submodule_path: Option<String>,
    pub(crate) submodule_commit: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PrebuiltEntry {
    pub(crate) source: Option<String>,
    pub(crate) tag: Option<String>,
    pub(crate) url: String,
    pub(crate) archive: String,
    pub(crate) launcher: String,
    pub(crate) sha256: Option<String>,
    #[serde(default)]
    pub(crate) companion_assets: Vec<CompanionAsset>,
    #[serde(default)]
    pub(crate) required_files: Vec<String>,
    pub(crate) submodule_path: Option<String>,
    pub(crate) submodule_commit: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CompanionAsset {
    pub(crate) url: String,
    pub(crate) archive: String,
    pub(crate) sha256: Option<String>,
    pub(crate) files: Vec<String>,
}

impl PrebuiltCatalog {
    pub(crate) fn entry(&self, platform: &str, backend: &str) -> Option<&PrebuiltEntry> {
        self.platforms.get(platform)?.get(backend)
    }

    pub(crate) fn python_runtime(
        &self,
        platform: &str,
        backend: &str,
    ) -> Option<&PythonRuntimeEntry> {
        self.python_runtimes.get(platform)?.get(backend)
    }

    pub(crate) fn source_metadata(&self, entry: &PrebuiltEntry) -> Option<&SourceMetadata> {
        self.sources.get(entry.source.as_deref()?)
    }

    pub(crate) fn excluded_release_asset(
        &self,
        platform: &str,
        backend: &str,
        architecture: &str,
    ) -> Option<&ExcludedReleaseAsset> {
        self.excluded_release_assets.iter().find(|entry| {
            entry.platform == platform
                && entry.backend == backend
                && entry.architecture == architecture
        })
    }

    pub(crate) fn resolved_tag<'a>(&'a self, entry: &'a PrebuiltEntry) -> Option<&'a str> {
        entry.tag.as_deref().or_else(|| {
            self.source_metadata(entry)
                .and_then(|source| source.tag.as_deref())
        })
    }

    pub(crate) fn resolved_submodule_path<'a>(
        &'a self,
        entry: &'a PrebuiltEntry,
    ) -> Option<&'a str> {
        entry.submodule_path.as_deref().or_else(|| {
            self.source_metadata(entry)
                .and_then(|source| source.submodule_path.as_deref())
        })
    }

    pub(crate) fn resolved_submodule_commit<'a>(
        &'a self,
        entry: &'a PrebuiltEntry,
    ) -> Option<&'a str> {
        entry.submodule_commit.as_deref().or_else(|| {
            self.source_metadata(entry)
                .and_then(|source| source.submodule_commit.as_deref())
        })
    }
}

pub(crate) fn load_catalog() -> Result<PrebuiltCatalog> {
    let catalog = if let Some(path) =
        std::env::var_os("OMNIINFER_PREBUILT_CATALOG").filter(|value| !value.is_empty())
    {
        let path = PathBuf::from(path);
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("read prebuilt catalog {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("parse prebuilt catalog {}", path.display()))?
    } else {
        serde_json::from_str(DEFAULT_CATALOG).context("parse built-in prebuilt catalog")?
    };
    validate_catalog(&catalog)?;
    Ok(catalog)
}

pub(crate) fn current_platform_name() -> &'static str {
    match std::env::consts::OS {
        "windows" => "windows",
        "macos" => "macos",
        _ => "linux",
    }
}

pub(crate) fn installable_backend_ids() -> BTreeSet<String> {
    load_catalog()
        .ok()
        .map(|catalog| {
            let platform = current_platform_name();
            let mut ids = catalog
                .platforms
                .get(platform)
                .into_iter()
                .flat_map(|entries| entries.keys().cloned())
                .collect::<BTreeSet<_>>();
            ids.extend(
                catalog
                    .python_runtimes
                    .get(platform)
                    .into_iter()
                    .flat_map(|entries| entries.keys().cloned()),
            );
            ids
        })
        .unwrap_or_default()
}

fn validate_catalog(catalog: &PrebuiltCatalog) -> Result<()> {
    if catalog.schema_version < 3 {
        return Ok(());
    }
    if catalog.sources.is_empty() {
        anyhow::bail!("prebuilt catalog schema 3 or newer requires source metadata");
    }
    for (source_name, source) in &catalog.sources {
        let tag = source
            .tag
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("catalog source {source_name} has no tag"))?;
        let submodule_tag = source
            .submodule_tag
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("catalog source {source_name} has no submodule tag"))?;
        if source
            .submodule_path
            .as_deref()
            .is_none_or(|value| value.is_empty())
        {
            anyhow::bail!("catalog source {source_name} has no submodule path");
        }
        let commit = source
            .submodule_commit
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("catalog source {source_name} has no commit"))?;
        if commit.len() != 40
            || !commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            anyhow::bail!("catalog source {source_name} has an invalid commit");
        }
        if tag.contains('/') || submodule_tag.contains('/') {
            anyhow::bail!("catalog source {source_name} has an invalid runtime or submodule tag");
        }
    }
    for (platform, entries) in &catalog.platforms {
        for (backend, entry) in entries {
            validate_sha256(entry.sha256.as_deref(), platform, backend, "runtime")?;
            let source_name = entry
                .source
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("{platform}/{backend} has no source"))?;
            if !catalog.sources.contains_key(source_name) {
                anyhow::bail!("{platform}/{backend} references unknown source {source_name}");
            }
            let tag = catalog.resolved_tag(entry).ok_or_else(|| {
                anyhow::anyhow!("{platform}/{backend} has no resolved source tag")
            })?;
            validate_asset_url(&entry.url, platform, backend, "runtime", tag)?;
            for (index, asset) in entry.companion_assets.iter().enumerate() {
                let role = format!("companion {}", index + 1);
                validate_sha256(asset.sha256.as_deref(), platform, backend, &role)?;
                validate_asset_url(&asset.url, platform, backend, &role, tag)?;
            }
        }
    }
    for entry in &catalog.excluded_release_assets {
        let source = catalog.sources.get(&entry.source).ok_or_else(|| {
            anyhow::anyhow!(
                "excluded {}/{} references unknown source {}",
                entry.platform,
                entry.backend,
                entry.source
            )
        })?;
        let tag = source.tag.as_deref().ok_or_else(|| {
            anyhow::anyhow!("excluded source {} has no release tag", entry.source)
        })?;
        if entry.platform.trim().is_empty()
            || entry.backend.trim().is_empty()
            || entry.architecture.trim().is_empty()
            || entry.reason.trim().is_empty()
        {
            anyhow::bail!("excluded release asset metadata is incomplete");
        }
        validate_sha256(
            Some(&entry.sha256),
            &entry.platform,
            &entry.backend,
            "excluded runtime",
        )?;
        validate_asset_url(
            &entry.url,
            &entry.platform,
            &entry.backend,
            "excluded runtime",
            tag,
        )?;
        if catalog.entry(&entry.platform, &entry.backend).is_some() {
            anyhow::bail!(
                "{}/{} cannot be both installable and excluded",
                entry.platform,
                entry.backend
            );
        }
    }
    for (platform, entries) in &catalog.python_runtimes {
        for (backend, entry) in entries {
            if entry.source.trim().is_empty()
                || entry.package.trim().is_empty()
                || entry.python.trim().is_empty()
                || entry.launcher.trim().is_empty()
            {
                anyhow::bail!("{platform}/{backend} Python runtime metadata is incomplete");
            }
            if entry.tag.trim().is_empty() || entry.tag.contains('/') {
                anyhow::bail!("{platform}/{backend} Python runtime has an invalid tag");
            }
            catalog.sources.get(&entry.source).ok_or_else(|| {
                anyhow::anyhow!(
                    "{platform}/{backend} Python runtime references unknown source {}",
                    entry.source
                )
            })?;
            if entry.variants.is_empty() || entry.uv.is_empty() {
                anyhow::bail!(
                    "{platform}/{backend} Python runtime has no architecture variants or managed uv assets"
                );
            }
            for (architecture, variant) in &entry.variants {
                let role = format!("Python wheel ({architecture})");
                validate_sha256(Some(&variant.sha256), platform, backend, &role)?;
                validate_python_asset_url(
                    &variant.url,
                    variant.build_commit.as_deref(),
                    platform,
                    backend,
                    &role,
                    &entry.tag,
                )?;
                if variant.version.trim().is_empty()
                    || !variant
                        .version
                        .starts_with(entry.tag.trim_start_matches('v'))
                    || variant.reported_version().trim().is_empty()
                    || !variant
                        .reported_version()
                        .starts_with(entry.tag.trim_start_matches('v'))
                    || variant.runtime_version.trim().is_empty()
                    || variant.reported_runtime_version().trim().is_empty()
                {
                    anyhow::bail!(
                        "{platform}/{backend} {architecture} has invalid accelerator compatibility metadata"
                    );
                }
                match variant.accelerator.as_str() {
                    "cuda" => {
                        if variant
                            .torch_backend
                            .as_deref()
                            .is_none_or(|value| !value.starts_with("cu"))
                            || variant
                                .minimum_driver
                                .as_deref()
                                .and_then(parse_version_triplet)
                                .is_none()
                            || variant.rocm_system.is_some()
                        {
                            anyhow::bail!(
                                "{platform}/{backend} {architecture} has invalid CUDA compatibility metadata"
                            );
                        }
                    }
                    "rocm" => {
                        let system = variant.rocm_system.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "{platform}/{backend} {architecture} has no ROCm system metadata"
                            )
                        })?;
                        if variant
                            .torch_backend
                            .as_deref()
                            .is_none_or(|value| !value.starts_with("rocm"))
                            || variant.index_url.as_deref().is_none_or(|value| {
                                !value.starts_with("https://wheels.vllm.ai/rocm/")
                            })
                            || system.apt_repository.trim().is_empty()
                            || system.packages.len() != REQUIRED_ROCM_SYSTEM_PACKAGES.len()
                            || REQUIRED_ROCM_SYSTEM_PACKAGES
                                .iter()
                                .any(|name| !system.packages.contains_key(*name))
                            || system.package_assets.len() != REQUIRED_ROCM_PACKAGE_ASSETS.len()
                            || REQUIRED_ROCM_PACKAGE_ASSETS
                                .iter()
                                .any(|name| !system.package_assets.contains_key(*name))
                            || system.required_gfx.is_empty()
                            || system.minimum_windows_release.trim().is_empty()
                        {
                            anyhow::bail!(
                                "{platform}/{backend} {architecture} has invalid ROCm compatibility metadata"
                            );
                        }
                        validate_sha256(
                            Some(&system.repository_key.sha256),
                            platform,
                            backend,
                            "ROCm repository key",
                        )?;
                        validate_sha256(
                            Some(&system.rocdxg.sha256),
                            platform,
                            backend,
                            "ROCDXG runtime",
                        )?;
                        let repository_url = system
                            .apt_repository
                            .split_whitespace()
                            .next()
                            .unwrap_or_default()
                            .trim_end_matches('/');
                        let ubuntu_python_pool =
                            "https://security.ubuntu.com/ubuntu/pool/main/p/python3.12/";
                        for (name, asset) in &system.package_assets {
                            validate_sha256(
                                Some(&asset.sha256),
                                platform,
                                backend,
                                &format!("ROCm package {name}"),
                            )?;
                            let expected_version = system
                                .packages
                                .get(name)
                                .map(String::as_str)
                                .unwrap_or_default();
                            if asset.version != expected_version
                                || asset.size == 0
                                || asset.filename.contains(['/', '\\'])
                                || !asset.filename.ends_with(".deb")
                                || !(asset.url.starts_with(&format!("{repository_url}/"))
                                    || (matches!(
                                        name.as_str(),
                                        "python3.12-dev" | "libpython3.12-dev"
                                    ) && asset.url.starts_with(ubuntu_python_pool)))
                                || !asset.url.ends_with(&asset.filename)
                            {
                                anyhow::bail!(
                                    "{platform}/{backend} {architecture} has invalid ROCm package asset {name}"
                                );
                            }
                        }
                        if !system
                            .repository_key
                            .url
                            .starts_with("https://repo.radeon.com/")
                            || !system
                                .rocdxg
                                .url
                                .starts_with("https://github.com/ROCm/librocdxg/")
                            || !system
                                .apt_repository
                                .starts_with("https://repo.radeon.com/rocm/apt/")
                            || !system.apt_repository.ends_with(" noble main")
                            || system.repository_key.version != variant.runtime_version
                            || system.rocdxg.version.trim().is_empty()
                            || variant.index_url.as_deref().is_none_or(|url| {
                                variant
                                    .build_commit
                                    .as_deref()
                                    .is_none_or(|commit| !url.contains(commit))
                            })
                        {
                            anyhow::bail!(
                                "{platform}/{backend} {architecture} has non-canonical ROCm system assets"
                            );
                        }
                    }
                    other => anyhow::bail!(
                        "{platform}/{backend} {architecture} has unsupported accelerator {other}"
                    ),
                }
                let uv = entry.uv.get(architecture).ok_or_else(|| {
                    anyhow::anyhow!("{platform}/{backend} {architecture} has no managed uv asset")
                })?;
                validate_sha256(
                    Some(&uv.sha256),
                    platform,
                    backend,
                    &format!("uv ({architecture})"),
                )?;
                if uv.version.trim().is_empty()
                    || !uv.url.starts_with("https://")
                    || !uv.url.contains(&format!("/download/{}/", uv.version))
                {
                    anyhow::bail!("{platform}/{backend} {architecture} has invalid uv metadata");
                }
            }
        }
    }
    Ok(())
}

fn parse_version_triplet(value: &str) -> Option<(u32, u32, u32)> {
    let mut parts = value.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    parts.next().is_none().then_some((major, minor, patch))
}

fn validate_asset_url(
    url: &str,
    platform: &str,
    backend: &str,
    role: &str,
    tag: &str,
) -> Result<()> {
    if !url.starts_with("https://") {
        anyhow::bail!("{platform}/{backend} {role} asset requires a canonical HTTPS URL");
    }
    if !url.contains(&format!("/download/{tag}/")) {
        anyhow::bail!("{platform}/{backend} {role} URL does not match source tag {tag}");
    }
    Ok(())
}

fn validate_python_asset_url(
    url: &str,
    build_commit: Option<&str>,
    platform: &str,
    backend: &str,
    role: &str,
    tag: &str,
) -> Result<()> {
    if url.starts_with("https://github.com/") {
        return validate_asset_url(url, platform, backend, role, tag);
    }
    let Some(commit) = build_commit else {
        anyhow::bail!("{platform}/{backend} {role} non-release asset has no build commit");
    };
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || !url.starts_with("https://wheels.vllm.ai/rocm/")
        || !url.contains(commit)
    {
        anyhow::bail!("{platform}/{backend} {role} has invalid wheel provenance");
    }
    Ok(())
}

fn validate_sha256(value: Option<&str>, platform: &str, backend: &str, role: &str) -> Result<()> {
    let Some(value) = value else {
        anyhow::bail!("{platform}/{backend} {role} asset has no pinned SHA256");
    };
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("{platform}/{backend} {role} asset has an invalid SHA256");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_catalog_is_complete() {
        let catalog: PrebuiltCatalog =
            serde_json::from_str(DEFAULT_CATALOG).expect("parse built-in catalog");
        validate_catalog(&catalog).expect("validate built-in catalog");
        assert!(
            catalog
                .python_runtime("windows", "vllm-wsl2-cuda")
                .is_some()
        );
        assert!(
            catalog
                .python_runtime("windows", "vllm-wsl2-rocm")
                .is_some()
        );
        assert!(catalog.entry("linux", "vla.cpp-linux-cuda").is_none());
        assert!(
            catalog
                .excluded_release_asset("linux", "vla.cpp-linux-cuda", "x86_64")
                .is_some()
        );
    }

    #[test]
    fn rejects_companion_url_with_mismatched_source_tag() {
        let mut value: serde_json::Value =
            serde_json::from_str(DEFAULT_CATALOG).expect("parse built-in catalog");
        let source_tag = value["sources"]["ggml-org/llama.cpp"]["tag"]
            .as_str()
            .expect("llama.cpp source tag")
            .to_string();
        value["platforms"]["windows"]["llama.cpp-cuda"]["companion_assets"][0]["url"] =
            serde_json::Value::String(
                "https://github.com/ggml-org/llama.cpp/releases/download/b9999/runtime.zip"
                    .to_string(),
            );
        let catalog: PrebuiltCatalog = serde_json::from_value(value).expect("parse test catalog");
        let error = validate_catalog(&catalog).expect_err("reject mismatched companion tag");
        assert!(
            error
                .to_string()
                .contains(&format!("does not match source tag {source_tag}"))
        );
    }
}
