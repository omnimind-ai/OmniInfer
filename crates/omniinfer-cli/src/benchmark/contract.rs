use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::BENCHMARK_SCHEMA_VERSION;

const BENCHMARK_CONTRACT_VERSION: &str = "1.0.0";
const MANIFEST_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/contract/manifest.json"
));
const SCHEMA_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/contract/schema.json"
));
const CATALOG_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/contract/catalog-index.json"
));
const SNAPSHOT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/contract/snapshot.json"
));

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DigestEntry {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SourceProvenance {
    catalog_url: String,
    schema_url: String,
    source_commit: String,
    source_timestamp: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    artifact_base_url: String,
    catalog_schema_version: u64,
    contract_version: String,
    manifest: DigestEntry,
    schema_version: String,
    source: SourceProvenance,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFiles {
    #[serde(rename = "schema.json")]
    schema: DigestEntry,
    #[serde(rename = "catalog-index.json")]
    catalog: DigestEntry,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    artifact_base_url: String,
    catalog_schema_version: u64,
    contract_version: String,
    files: ManifestFiles,
    policy_version: String,
    schema_version: String,
    source: SourceProvenance,
}

#[derive(Debug)]
pub(super) struct BenchmarkContract {
    catalog: Value,
    schema: Value,
}

impl BenchmarkContract {
    pub(super) fn load_embedded() -> Result<Self> {
        Self::from_bytes(MANIFEST_BYTES, SCHEMA_BYTES, CATALOG_BYTES, SNAPSHOT_BYTES)
            .context("vendored OmniStudio benchmark contract is invalid")
    }

    fn from_bytes(
        manifest_raw: &[u8],
        schema_raw: &[u8],
        catalog_raw: &[u8],
        snapshot_raw: &[u8],
    ) -> Result<Self> {
        let snapshot: Snapshot = serde_json::from_slice(snapshot_raw)
            .context("failed to parse benchmark contract snapshot.json")?;
        verify_digest("manifest.json", manifest_raw, &snapshot.manifest)?;
        let manifest: Manifest = serde_json::from_slice(manifest_raw)
            .context("failed to parse benchmark contract manifest.json")?;
        verify_digest("schema.json", schema_raw, &manifest.files.schema)?;
        verify_digest("catalog-index.json", catalog_raw, &manifest.files.catalog)?;

        if manifest.contract_version != BENCHMARK_CONTRACT_VERSION
            || snapshot.contract_version != BENCHMARK_CONTRACT_VERSION
        {
            anyhow::bail!(
                "unsupported benchmark contract version: manifest={}, snapshot={}, supported={BENCHMARK_CONTRACT_VERSION}",
                manifest.contract_version,
                snapshot.contract_version,
            );
        }
        if manifest.schema_version != BENCHMARK_SCHEMA_VERSION
            || snapshot.schema_version != BENCHMARK_SCHEMA_VERSION
        {
            anyhow::bail!(
                "incompatible benchmark schema version: manifest={}, snapshot={}, generator={BENCHMARK_SCHEMA_VERSION}",
                manifest.schema_version,
                snapshot.schema_version,
            );
        }
        if manifest.artifact_base_url != snapshot.artifact_base_url
            || !manifest.artifact_base_url.starts_with("https://")
        {
            anyhow::bail!("benchmark contract source URL is inconsistent or not HTTPS");
        }
        if manifest.catalog_schema_version != snapshot.catalog_schema_version
            || manifest.source != snapshot.source
        {
            anyhow::bail!("benchmark contract provenance is inconsistent");
        }
        if manifest.policy_version.trim().is_empty() {
            anyhow::bail!("benchmark contract policy version is empty");
        }

        let schema: Value = serde_json::from_slice(schema_raw)
            .context("failed to parse benchmark contract schema.json")?;
        if schema.get("$schema").and_then(Value::as_str)
            != Some("http://json-schema.org/draft-07/schema#")
            || schema
                .pointer("/properties/schema_version/const")
                .and_then(Value::as_str)
                != Some(BENCHMARK_SCHEMA_VERSION)
        {
            anyhow::bail!("benchmark schema declaration does not match the manifest");
        }
        jsonschema::draft7::meta::validate(&schema).map_err(|error| {
            anyhow::anyhow!("vendored benchmark schema is not valid Draft-07: {error}")
        })?;

        let catalog: Value = serde_json::from_slice(catalog_raw)
            .context("failed to parse benchmark contract catalog-index.json")?;
        if catalog.get("data_kind").and_then(Value::as_str) != Some("benchmark-catalog-index")
            || catalog.get("contract_version").and_then(Value::as_str)
                != Some(BENCHMARK_CONTRACT_VERSION)
            || catalog.get("schema_version").and_then(Value::as_str)
                != Some(BENCHMARK_SCHEMA_VERSION)
            || catalog
                .get("catalog_schema_version")
                .and_then(Value::as_u64)
                != Some(manifest.catalog_schema_version)
            || catalog.get("source_commit").and_then(Value::as_str)
                != Some(manifest.source.source_commit.as_str())
            || catalog.get("source_timestamp").and_then(Value::as_str)
                != Some(manifest.source.source_timestamp.as_str())
        {
            anyhow::bail!("benchmark catalog metadata does not match the manifest");
        }
        for field in ["backends", "devices", "models", "platforms"] {
            if catalog.get(field).and_then(Value::as_array).is_none() {
                anyhow::bail!("benchmark catalog field {field:?} is missing or not an array");
            }
        }
        if catalog.get("benchmarks").is_some() {
            anyhow::bail!("benchmark contract must not contain measurement records");
        }
        Ok(Self { catalog, schema })
    }

    pub(super) fn validate_references(
        &self,
        model_id: &str,
        format: &str,
        quantization: &str,
        backend_id: &str,
        device_id_or_name: &str,
        platform_id: &str,
    ) -> Result<()> {
        let platform_exists = entries(&self.catalog, "platforms")?
            .iter()
            .any(|entry| string(entry, "id") == Some(platform_id));
        if !platform_exists {
            anyhow::bail!("unknown benchmark catalog platform: {platform_id}");
        }

        let backend = entries(&self.catalog, "backends")?
            .iter()
            .find(|entry| string(entry, "id") == Some(backend_id))
            .ok_or_else(|| anyhow::anyhow!("unknown benchmark catalog backend: {backend_id}"))?;
        if !string_array(backend, "platforms")?.contains(&platform_id) {
            anyhow::bail!(
                "benchmark backend {backend_id:?} is not available on platform {platform_id:?}"
            );
        }

        let device = entries(&self.catalog, "devices")?
            .iter()
            .find(|entry| {
                string(entry, "id") == Some(device_id_or_name)
                    || string(entry, "name") == Some(device_id_or_name)
            })
            .ok_or_else(|| {
                anyhow::anyhow!("unknown benchmark catalog device ID or name: {device_id_or_name}")
            })?;
        if !string_array(device, "platforms")?.contains(&platform_id) {
            anyhow::bail!(
                "benchmark device {device_id_or_name:?} is not available on platform {platform_id:?}"
            );
        }
        if !string_array(device, "backend_ids")?.contains(&backend_id) {
            anyhow::bail!(
                "benchmark device {device_id_or_name:?} does not support backend {backend_id:?}"
            );
        }

        let model = entries(&self.catalog, "models")?
            .iter()
            .find(|entry| string(entry, "id") == Some(model_id))
            .ok_or_else(|| anyhow::anyhow!("unknown benchmark catalog model: {model_id}"))?;
        let supported = model
            .get("support")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("benchmark model {model_id:?} has no support entries"))?
            .iter()
            .any(|entry| {
                string(entry, "platform_id") == Some(platform_id)
                    && string(entry, "backend_id") == Some(backend_id)
                    && string(entry, "format") == Some(format)
                    && string_array(entry, "quantizations")
                        .is_ok_and(|values| values.contains(&quantization))
            });
        if !supported {
            anyhow::bail!(
                "benchmark combination is not in the catalog: model={model_id}, platform={platform_id}, backend={backend_id}, format={format}, quantization={quantization}"
            );
        }
        Ok(())
    }

    pub(super) fn validate_submission(&self, submission: &Value) -> Result<()> {
        let validator = jsonschema::draft7::options()
            .should_validate_formats(true)
            .build(&self.schema)
            .context("failed to compile vendored benchmark schema")?;
        let errors = validator
            .iter_errors(submission)
            .take(5)
            .map(|error| format!("{}: {error}", error.instance_path()))
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            anyhow::bail!(
                "generated benchmark JSON does not satisfy the vendored Schema {BENCHMARK_SCHEMA_VERSION}: {}",
                errors.join("; ")
            );
        }
        Ok(())
    }
}

pub(super) fn benchmark_platform() -> &'static str {
    std::env::consts::OS
}

fn verify_digest(label: &str, raw: &[u8], expected: &DigestEntry) -> Result<()> {
    let actual_bytes = u64::try_from(raw.len()).context("contract artifact is too large")?;
    let actual_hash = format!("{:x}", Sha256::digest(raw));
    if actual_bytes != expected.bytes || actual_hash != expected.sha256 {
        anyhow::bail!(
            "{label} integrity mismatch: expected {} bytes / {}, got {actual_bytes} bytes / {actual_hash}",
            expected.bytes,
            expected.sha256,
        );
    }
    Ok(())
}

fn entries<'a>(catalog: &'a Value, field: &str) -> Result<&'a [Value]> {
    catalog
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| anyhow::anyhow!("benchmark catalog field {field:?} is invalid"))
}

fn string<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn string_array<'a>(value: &'a Value, field: &str) -> Result<Vec<&'a str>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("benchmark catalog field {field:?} is invalid"))?
        .iter()
        .map(|item| {
            item.as_str()
                .ok_or_else(|| anyhow::anyhow!("benchmark catalog field {field:?} is invalid"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded() -> BenchmarkContract {
        BenchmarkContract::load_embedded().expect("embedded contract is valid")
    }

    #[test]
    fn accepts_known_complete_combination() {
        let contract = embedded();
        contract
            .validate_references(
                "qwen3-5-2b",
                "GGUF",
                "Q4_K_M",
                "llama.cpp-linux-cuda",
                "rtx-5090",
                "linux",
            )
            .unwrap();
        contract
            .validate_references(
                "qwen3-5-2b",
                "GGUF",
                "Q4_K_M",
                "llama.cpp-linux-cuda",
                "NVIDIA GeForce RTX 5090",
                "linux",
            )
            .unwrap();
    }

    #[test]
    fn rejects_unknown_catalog_references() {
        let contract = embedded();
        let error = contract
            .validate_references(
                "not-a-model",
                "GGUF",
                "Q4_K_M",
                "llama.cpp-linux-cuda",
                "rtx-5090",
                "linux",
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unknown benchmark catalog model")
        );

        let error = contract
            .validate_references(
                "qwen3-5-2b",
                "GGUF",
                "Q4_K_M",
                "llama.cpp-linux-cuda",
                "not-a-device",
                "linux",
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unknown benchmark catalog device")
        );
    }

    #[test]
    fn rejects_individually_valid_but_unsupported_combination() {
        let error = embedded()
            .validate_references(
                "qwen3-5-2b",
                "GGUF",
                "Q4_0",
                "llama.cpp-linux-cuda",
                "rtx-5090",
                "linux",
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("combination is not in the catalog")
        );
    }

    #[test]
    fn rejects_device_backend_platform_mismatch() {
        let error = embedded()
            .validate_references(
                "qwen3-5-2b",
                "GGUF",
                "Q4_K_M",
                "llama.cpp-linux-cuda",
                "radeon-8060s",
                "linux",
            )
            .unwrap_err();
        assert!(error.to_string().contains("not available on platform"));
    }

    #[test]
    fn rejects_tampered_artifact() {
        let mut schema = SCHEMA_BYTES.to_vec();
        schema.push(b'\n');
        let error =
            BenchmarkContract::from_bytes(MANIFEST_BYTES, &schema, CATALOG_BYTES, SNAPSHOT_BYTES)
                .unwrap_err();
        assert!(error.to_string().contains("schema.json integrity mismatch"));
    }

    #[test]
    fn rejects_incompatible_contract_version() {
        let mut manifest: Value = serde_json::from_slice(MANIFEST_BYTES).unwrap();
        manifest["contract_version"] = Value::String("2.0.0".to_string());
        let manifest_raw = format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap());
        let mut snapshot: Value = serde_json::from_slice(SNAPSHOT_BYTES).unwrap();
        snapshot["contract_version"] = Value::String("2.0.0".to_string());
        snapshot["manifest"]["bytes"] = Value::from(manifest_raw.len() as u64);
        snapshot["manifest"]["sha256"] =
            Value::String(format!("{:x}", Sha256::digest(manifest_raw.as_bytes())));
        let snapshot_raw = format!("{}\n", serde_json::to_string_pretty(&snapshot).unwrap());
        let error = BenchmarkContract::from_bytes(
            manifest_raw.as_bytes(),
            SCHEMA_BYTES,
            CATALOG_BYTES,
            snapshot_raw.as_bytes(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported benchmark contract version")
        );
    }

    #[test]
    fn schema_rejects_incomplete_submission() {
        let error = embedded()
            .validate_submission(&serde_json::json!({}))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not satisfy the vendored Schema")
        );
    }
}
