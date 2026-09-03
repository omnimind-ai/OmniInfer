use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use omniinfer_core::{config, paths};
use rand::random;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use url::Url;

use crate::{
    BenchRunArgs, BenchmarkAccelerator, BenchmarkPrivilegeLevel, get_local_json, json_bool,
    json_str, json_u64, post_local_json_for_config,
};

const BENCHMARK_SCHEMA_VERSION: &str = "1.4.0";
const MAX_MEASUREMENT_CV: f64 = 0.05;
const IGNORE_EOS_PROTOCOL_NOTE: &str =
    "fixed_length_generation=true; ignore_eos=true; completion_tokens=max_tokens";
const DEFAULT_PROMPT: &str = "Write a detailed but concise explanation of why local language-model inference speed varies across hardware and runtimes.";
const MODEL_FORMATS: &[&str] = &[
    "GGUF",
    "MLX",
    "Safetensors",
    "MNN",
    "TFLite",
    "LiteRT",
    "LITERTLM",
    "ONNX",
    "ExecuTorch",
    "Other",
];

#[derive(Debug, Clone)]
struct Measurement {
    prompt_tokens: u64,
    completion_tokens: u64,
    prefill_tps: f64,
    decode_tps: f64,
    prefill_duration_ms: f64,
    decode_duration_ms: f64,
    ttft_ms: Option<f64>,
    wall_time_ms: f64,
}

pub(crate) fn run(args: &BenchRunArgs) -> Result<()> {
    validate_metadata(args)?;
    let contract = BenchmarkContract::load_embedded()?;
    let config = config::load_app_config().unwrap_or_default();
    let state = get_local_json("/omni/state", Duration::from_secs(10))?;
    if !json_bool(&state, "backend_ready").unwrap_or(false) {
        anyhow::bail!(
            "No benchmarkable runtime is ready. Load a model first with `omniinfer load -m <model>`."
        );
    }
    json_str(&state, "model")
        .ok_or_else(|| anyhow::anyhow!("OmniInfer state does not identify the loaded model."))?;
    let loaded_backend = json_str(&state, "backend")
        .ok_or_else(|| anyhow::anyhow!("OmniInfer state does not identify the loaded backend."))?;
    if !valid_catalog_id(loaded_backend) {
        anyhow::bail!(
            "Loaded backend ID {loaded_backend:?} cannot be represented by the benchmark schema."
        );
    }
    if let Some(expected) = args.backend_id.as_deref()
        && expected != loaded_backend
    {
        anyhow::bail!("Loaded backend is {loaded_backend}, but --backend-id requested {expected}.");
    }

    let launch_args = command_array(&state, "launch_command")?;
    validate_cache_isolation(loaded_backend, &launch_args, &state)?;
    let (device_name, soc) = resolve_device(args, loaded_backend, &state)?;
    contract.validate_references(
        &args.catalog_model_id,
        &args.model_format,
        &args.quantization,
        loaded_backend,
        &soc,
        benchmark_platform(),
    )?;
    let (backend_version, build_command) =
        resolve_runtime_provenance(args, loaded_backend, &state)?;
    let detected = detect_optimizations(loaded_backend, &launch_args);
    let (optimization_mode, optimizations) = resolve_optimization_declaration(args, &detected)?;
    let run_command = match args.run_command.as_deref() {
        Some(command) => validated_command("--run-command", command)?.to_string(),
        None if launch_args.is_empty() => anyhow::bail!(
            "The loaded runtime does not expose a launch command. Pass --run-command with the effective runtime command."
        ),
        None => validated_command(
            "captured runtime command",
            &command_text(&redact_command_args(&launch_args)),
        )?
        .to_string(),
    };
    let execution = resolve_execution(args, loaded_backend, &run_command)?;

    let context_size = args
        .context_size
        .or_else(|| json_u64(&state, "ctx_size").and_then(|value| u32::try_from(value).ok()))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Context size is unavailable from the loaded runtime. Pass --context-size explicitly."
            )
        })?;
    let batch_size = args
        .batch_size
        .or_else(|| infer_positive_flag(&launch_args, &["-b", "--batch-size", "--batch_size"]))
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Batch size is unavailable from the launch command. Pass --batch-size explicitly."
            )
        })?;
    let (prompt, prompt_source) = read_prompt(args)?;
    let prompt_sha256 = format!("{:x}", Sha256::digest(prompt.as_bytes()));
    let mut request = json!({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": false,
        "think": false,
        "cache_prompt": false,
    });
    if args.ignore_eos {
        request
            .as_object_mut()
            .expect("benchmark request is a JSON object")
            .insert("ignore_eos".to_string(), Value::Bool(true));
    }
    let timeout = Duration::from_secs(u64::from(args.timeout_seconds));

    for index in 0..args.warmup_runs {
        post_local_json_for_config("/v1/chat/completions", &request, timeout, &config)
            .with_context(|| format!("warmup run {} failed", index + 1))?;
    }

    let started_at = OffsetDateTime::now_utc();
    let mut measurements = Vec::with_capacity(usize::from(args.runs));
    for index in 0..args.runs {
        let reset =
            post_local_json_for_config("/omni/cache/clear", &json!({}), timeout, &config)
                .with_context(|| format!("cache reset before measured run {} failed", index + 1))?;
        if !json_bool(&reset, "ok").unwrap_or(false) {
            anyhow::bail!(
                "cache reset before measured run {} did not provide a successful acknowledgement",
                index + 1
            );
        }
        let started = Instant::now();
        let response =
            post_local_json_for_config("/v1/chat/completions", &request, timeout, &config)
                .with_context(|| format!("measured run {} failed", index + 1))?;
        let measurement = extract_measurement(&response, started.elapsed())
            .with_context(|| format!("measured run {} returned incomplete metrics", index + 1))?;
        if args.ignore_eos && measurement.completion_tokens != u64::from(args.max_tokens) {
            anyhow::bail!(
                "measured run {} with --ignore-eos returned completion_tokens={}; expected max_tokens={} (fixed-length generation requires completion_tokens == --max-tokens)",
                index + 1,
                measurement.completion_tokens,
                args.max_tokens,
            );
        }
        let progress = format!(
            "Run {}/{}: pp={} tg={} prefill={:.3} tok/s decode={:.3} tok/s",
            index + 1,
            args.runs,
            measurement.prompt_tokens,
            measurement.completion_tokens,
            measurement.prefill_tps,
            measurement.decode_tps,
        );
        if args.json {
            eprintln!("{progress}");
        } else {
            println!("{progress}");
        }
        measurements.push(measurement);
    }

    let pp = consistent_token_count(&measurements, true)?;
    let tg = consistent_token_count(&measurements, false)?;
    validate_measurement_stability(&measurements, MAX_MEASUREMENT_CV)?;
    if pp > 1_048_576 || tg > 1_048_576 {
        anyhow::bail!("Measured PP/TG exceeds the submission contract limit of 1,048,576 tokens.");
    }
    if context_size > 4_194_304 {
        anyhow::bail!("Context size exceeds the submission contract limit of 4,194,304.");
    }
    if batch_size > 1_048_576 {
        anyhow::bail!("Batch size exceeds the submission contract limit of 1,048,576.");
    }
    if u64::from(context_size) < pp.saturating_add(tg) {
        anyhow::bail!(
            "Measured pp + tg is {}, which exceeds context size {context_size}.",
            pp.saturating_add(tg)
        );
    }
    let benchmark_id = args.benchmark_id.clone().unwrap_or_else(|| {
        generated_benchmark_id(&args.catalog_model_id, loaded_backend, started_at)
    });
    validate_benchmark_id(&benchmark_id)?;
    let payload = build_submission(BuildSubmission {
        args,
        benchmark_id: &benchmark_id,
        loaded_backend,
        backend_version: &backend_version,
        build_command: &build_command,
        run_command: &run_command,
        execution: &execution,
        optimization_mode,
        optimizations: &optimizations,
        context_size,
        batch_size,
        device_name: &device_name,
        soc: &soc,
        prompt_source: &prompt_source,
        prompt_sha256: &prompt_sha256,
        started_at,
        measurements: &measurements,
        pp,
        tg,
    })?;
    contract.validate_submission(&payload)?;
    let destination = result_path(args.output.as_deref(), &benchmark_id)?;
    write_result_atomic(&destination, &payload)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        eprintln!("Benchmark saved: {}", destination.display());
        eprintln!("Schema: {BENCHMARK_SCHEMA_VERSION}");
    } else {
        println!("Benchmark saved: {}", destination.display());
        println!("Schema: {BENCHMARK_SCHEMA_VERSION}");
    }
    Ok(())
}

pub(crate) fn list(json_output: bool) -> Result<()> {
    let directory = paths::benchmark_results_dir();
    let mut rows = Vec::new();
    if directory.is_dir() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json")
                || path.is_symlink()
            {
                continue;
            }
            let Ok(raw) = fs::read(&path) else {
                continue;
            };
            let Ok(payload) = serde_json::from_slice::<Value>(&raw) else {
                continue;
            };
            if json_str(&payload, "schema_version") != Some(BENCHMARK_SCHEMA_VERSION) {
                continue;
            }
            rows.push(json!({
                "benchmark_id": json_str(&payload, "benchmark_id"),
                "model_id": payload.pointer("/model/catalog_model_id").and_then(Value::as_str),
                "backend_id": payload.pointer("/backend/catalog_backend_id").and_then(Value::as_str),
                "started_at": payload.pointer("/protocol/started_at").and_then(Value::as_str),
                "path": path.display().to_string(),
            }));
        }
    }
    rows.sort_by(|left, right| {
        left["started_at"]
            .as_str()
            .cmp(&right["started_at"].as_str())
            .reverse()
    });
    if json_output {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("No archived benchmark results in {}", directory.display());
        return Ok(());
    }
    println!("Archived benchmark results:");
    for row in rows {
        println!(
            "  {}  model={}  backend={}  {}",
            json_str(&row, "benchmark_id").unwrap_or("-"),
            json_str(&row, "model_id").unwrap_or("-"),
            json_str(&row, "backend_id").unwrap_or("-"),
            json_str(&row, "path").unwrap_or("-"),
        );
    }
    Ok(())
}

mod validation;

use validation::*;
mod cache;
use cache::*;
mod contract;
use contract::*;
mod environment;
use environment::*;
mod result;

use result::*;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_sensitive_runtime_arguments() {
        let args = vec![
            "runtime".to_string(),
            "--api-key".to_string(),
            "secret-value".to_string(),
            "TOKEN=another-secret".to_string(),
            "--port".to_string(),
            "9000".to_string(),
        ];
        assert_eq!(
            redact_command_args(&args),
            vec![
                "runtime",
                "--api-key",
                "<redacted>",
                "TOKEN=<redacted>",
                "--port",
                "9000",
            ]
        );
    }

    #[test]
    fn detects_known_optimization_markers() {
        let detected = detect_optimizations(
            "turboquant-mac",
            &["runtime".to_string(), "--dflash-mode".to_string()],
        );
        assert_eq!(
            detected.into_iter().collect::<Vec<_>>(),
            vec!["dflash", "turboquant"]
        );
    }

    #[test]
    fn infers_known_single_accelerator_backends() {
        assert_eq!(
            infer_accelerator("llama.cpp-linux-cuda"),
            Some(BenchmarkAccelerator::Gpu)
        );
        assert_eq!(
            infer_accelerator("llama.cpp-android-htp"),
            Some(BenchmarkAccelerator::Htp)
        );
        assert_eq!(
            infer_accelerator("llama.cpp-linux"),
            Some(BenchmarkAccelerator::Cpu)
        );
        assert_eq!(infer_accelerator("third-party-runtime"), None);
    }

    #[test]
    fn extracts_backend_and_observed_measurements() {
        let backend = json!({
            "usage": {"prompt_tokens": 64, "completion_tokens": 16},
            "timings": {
                "prompt_ms": 400.0,
                "predicted_ms": 800.0
            }
        });
        let value = extract_measurement(&backend, Duration::from_millis(1300)).unwrap();
        assert_eq!(value.prefill_tps, 160.0);
        assert_eq!(value.decode_tps, 20.0);

        let observed = json!({
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "omniinfer_metrics": {
                "ttft_ms": 100,
                "decode_ms": 500,
                "observed_prefill_tps": 200.0,
                "observed_decode_tps": 20.0
            }
        });
        let value = extract_measurement(&observed, Duration::from_millis(650)).unwrap();
        assert_eq!(value.prefill_tps, 200.0);
        assert_eq!(value.decode_tps, 20.0);
        assert_eq!(value.ttft_ms, Some(100.0));

        let upstream_rates = json!({
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "timings": {
                "prompt_per_second": 400.0,
                "predicted_per_second": 40.0
            },
            "omniinfer_metrics": {
                "ttft_ms": 100,
                "decode_ms": 500,
                "observed_prefill_tps": 200.0,
                "observed_decode_tps": 20.0
            }
        });
        let value = extract_measurement(&upstream_rates, Duration::from_millis(650)).unwrap();
        assert_eq!(value.prefill_tps, 400.0);
        assert_eq!(value.decode_tps, 40.0);
    }

    #[test]
    fn infers_common_batch_flags() {
        assert_eq!(
            infer_positive_flag(&["runtime".into(), "-b".into(), "256".into()], &["-b"]),
            Some(256)
        );
        assert_eq!(
            infer_positive_flag(
                &["runtime".into(), "--batch-size=64".into()],
                &["--batch-size"]
            ),
            Some(64)
        );
    }

    #[test]
    fn accepts_only_effective_cache_isolation_flags() {
        let state = json!({"mmproj": null});
        let isolated = [
            "llama-server",
            "--cache-ram",
            "8192",
            "--cache-ram=0",
            "--cache-idle-slots",
            "--no-cache-idle-slots",
            "--cache-prompt",
            "--no-cache-prompt",
            "--slot-prompt-similarity",
            "0",
            "--slot-save-path",
            "/tmp/benchmark-slots",
        ]
        .map(str::to_string);
        assert!(validate_cache_isolation("llama.cpp-linux-cuda", &isolated, &state).is_ok());

        let mut unsafe_args = isolated.to_vec();
        unsafe_args.push("--cache-ram=8192".to_string());
        assert!(validate_cache_isolation("llama.cpp-linux-cuda", &unsafe_args, &state).is_err());
        assert!(validate_cache_isolation("vllm-linux-cuda", &isolated, &state).is_err());
    }

    #[test]
    fn rejects_unstable_measurements() {
        let measurement = |prefill_tps, decode_tps| Measurement {
            prompt_tokens: 64,
            completion_tokens: 16,
            prefill_tps,
            decode_tps,
            prefill_duration_ms: 64_000.0 / prefill_tps,
            decode_duration_ms: 16_000.0 / decode_tps,
            ttft_ms: None,
            wall_time_ms: 1_000.0,
        };
        let stable = [
            measurement(100.0, 20.0),
            measurement(101.0, 20.1),
            measurement(99.0, 19.9),
        ];
        assert!(validate_measurement_stability(&stable, 0.05).is_ok());
        let unstable = [
            measurement(100.0, 20.0),
            measurement(140.0, 20.0),
            measurement(80.0, 20.0),
        ];
        assert!(validate_measurement_stability(&unstable, 0.05).is_err());
    }

    #[test]
    fn requires_immutable_hugging_face_model_urls() {
        assert!(
            validate_https_url(
                "model",
                "https://huggingface.co/owner/model/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/model.gguf",
            )
            .is_ok()
        );
        assert!(
            validate_https_url(
                "model",
                "https://huggingface.co/owner/model/resolve/main/model.gguf",
            )
            .is_err()
        );
    }
}
