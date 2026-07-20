use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

const DEFAULT_CATALOG: &str = include_str!("../../../scripts/prebuilt_backends.json");

#[derive(Debug, Deserialize)]
pub(crate) struct PrebuiltCatalog {
    #[serde(default)]
    pub(crate) schema_version: u32,
    #[serde(default)]
    pub(crate) mirrors: Vec<String>,
    #[serde(default)]
    pub(crate) sources: BTreeMap<String, SourceMetadata>,
    pub(crate) platforms: BTreeMap<String, BTreeMap<String, PrebuiltEntry>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SourceMetadata {
    pub(crate) tag: Option<String>,
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

    pub(crate) fn source_metadata(&self, entry: &PrebuiltEntry) -> Option<&SourceMetadata> {
        self.sources.get(entry.source.as_deref()?)
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
        .and_then(|catalog| {
            catalog
                .platforms
                .get(current_platform_name())
                .map(|entries| entries.keys().cloned().collect())
        })
        .unwrap_or_default()
}

fn validate_catalog(catalog: &PrebuiltCatalog) -> Result<()> {
    if catalog.schema_version < 3 {
        return Ok(());
    }
    if catalog.sources.is_empty() {
        anyhow::bail!("prebuilt catalog schema 3 requires source metadata");
    }
    for (source_name, source) in &catalog.sources {
        let tag = source
            .tag
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("catalog source {source_name} has no tag"))?;
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
        if tag.contains('/') {
            anyhow::bail!("catalog source {source_name} has an invalid tag");
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
    Ok(())
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
    }

    #[test]
    fn schema_three_rejects_unpinned_companion_url() {
        let mut value: serde_json::Value =
            serde_json::from_str(DEFAULT_CATALOG).expect("parse built-in catalog");
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
                .contains("does not match source tag b9500")
        );
    }
}
