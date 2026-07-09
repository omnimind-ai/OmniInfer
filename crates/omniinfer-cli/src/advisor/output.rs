use super::*;
use std::io::IsTerminal;

use crossterm::style::{Color, Stylize, style};

pub fn print_system(payload: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(payload)?);
        return Ok(());
    }
    let host = payload.get("host").and_then(Value::as_object);
    let cuda = payload.get("cuda").and_then(Value::as_object);
    let summary = payload.get("summary").and_then(Value::as_object);
    let color = color_output_enabled();
    println!("{}", heading("OmniInfer Advisor System", color));
    println!("{}", section("Host", color));
    println!(
        "  System: {} ({})",
        host.and_then(|value| value.get("system"))
            .and_then(Value::as_str)
            .unwrap_or("-"),
        host.and_then(|value| value.get("machine"))
            .and_then(Value::as_str)
            .unwrap_or("-")
    );
    println!(
        "  CPU: {} cores",
        host.and_then(|value| value.get("cpu_cores"))
            .and_then(Value::as_u64)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  RAM: {} GiB available / {} GiB total",
        host.and_then(|value| value.get("available_ram_gib"))
            .and_then(Value::as_f64)
            .map(format_gib)
            .unwrap_or_else(|| "-".to_string()),
        host.and_then(|value| value.get("total_ram_gib"))
            .and_then(Value::as_f64)
            .map(format_gib)
            .unwrap_or_else(|| "-".to_string())
    );

    println!("{}", section("GPU", color));
    let devices = cuda
        .and_then(|value| value.get("visible_devices"))
        .and_then(Value::as_array)
        .or_else(|| {
            cuda.and_then(|value| value.get("devices"))
                .and_then(Value::as_array)
        });
    match devices {
        Some(devices) if !devices.is_empty() => {
            for device in devices {
                println!(
                    "  CUDA {}: {} free={} GiB total={} GiB util={}",
                    json_str(device, "index").unwrap_or("-"),
                    json_str(device, "name").unwrap_or("-"),
                    json_number_string(device, "free_gib"),
                    json_number_string(device, "total_gib"),
                    json_u64(device, "utilization_pct")
                        .map(|value| format!("{value}%"))
                        .unwrap_or_else(|| "-".to_string())
                );
            }
        }
        _ => println!("  CUDA: none detected"),
    }

    println!("{}", section("Backend readiness", color));
    let installed_count = summary
        .and_then(|value| value.get("installed_backends"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let recommended_installed = summary
        .and_then(|value| value.get("recommended_installed_backend"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            if installed_count == 0 {
                "none (no runtime installed)".to_string()
            } else {
                "none".to_string()
            }
        });
    println!(
        "  Recommended installed backend: {}",
        state_text(
            &recommended_installed,
            !recommended_installed.starts_with("none"),
            color,
        )
    );
    if installed_count == 0
        && let Some(candidate) = summary
            .and_then(|value| value.get("recommended_backend_to_install"))
            .and_then(Value::as_str)
    {
        println!("  Recommended backend to install: {candidate}");
        println!("  Install command: omniinfer backend install {candidate}");
    }
    print_usable_backends(payload);
    Ok(())
}

pub fn print_inspect(payload: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(payload)?);
        return Ok(());
    }
    println!("OmniInfer Advisor Inspect");
    println!("Model: {}", json_str(payload, "model").unwrap_or("-"));
    println!("Format: {}", json_str(payload, "format").unwrap_or("-"));
    println!(
        "Artifact: {}",
        json_str(payload, "artifact_kind").unwrap_or("-")
    );
    println!(
        "Exists: {}",
        if json_bool(payload, "exists").unwrap_or(false) {
            "yes"
        } else {
            "no"
        }
    );
    println!("Size: {} GiB", json_number_string(payload, "size_gib"));
    println!(
        "Quantization: {}",
        json_str(payload, "quantization").unwrap_or("-")
    );
    println!("Params: {}B", json_number_string(payload, "params_b"));
    print_warnings(payload);
    Ok(())
}

pub fn print_fit(payload: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(payload)?);
        return Ok(());
    }
    println!("OmniInfer Advisor Fit");
    let model = payload.get("model").unwrap_or(&Value::Null);
    println!("Model: {}", json_str(model, "model").unwrap_or("-"));
    println!(
        "Context size: {}",
        json_u64(payload, "context_size")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    let recommended = payload.get("recommended").unwrap_or(&Value::Null);
    if recommended.is_object() {
        println!(
            "Recommended backend: {}",
            json_str(recommended, "backend").unwrap_or("-")
        );
        println!("Fit: {}", json_str(recommended, "fit").unwrap_or("-"));
        println!(
            "Confidence: {}",
            json_str(recommended, "recommendation_confidence").unwrap_or("-")
        );
        if let Some(evidence) = recommended.get("evidence") {
            println!("Evidence: {}", json_str(evidence, "level").unwrap_or("-"));
        }
        println!(
            "Installed: {}",
            if json_bool(recommended, "installed").unwrap_or(false) {
                "yes"
            } else {
                "no"
            }
        );
        println!(
            "Memory: {} GiB required / {} GiB available",
            json_number_string(recommended, "memory_required_gib"),
            json_number_string(recommended, "memory_available_gib")
        );
        print_memory_breakdown(recommended.get("memory_breakdown").unwrap_or(&Value::Null));
        print_string_array(recommended, "why_recommended", "Why", 5);
    } else {
        println!("Recommended backend: -");
    }
    if let Some(command) = json_str(payload, "next_command") {
        println!("Next command: {command}");
    }
    if let Some(alternatives) = payload.get("alternatives").and_then(Value::as_array)
        && !alternatives.is_empty()
    {
        println!("Alternatives:");
        for candidate in alternatives.iter().take(5) {
            println!(
                "  {}: fit={}, installed={}",
                json_str(candidate, "backend").unwrap_or("-"),
                json_str(candidate, "fit").unwrap_or("-"),
                if json_bool(candidate, "installed").unwrap_or(false) {
                    "yes"
                } else {
                    "no"
                }
            );
            if let Some(reason) = candidate
                .get("why_not")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .next()
            {
                println!("    why_not={reason}");
            }
        }
    }
    print_warnings(payload);
    Ok(())
}

pub fn print_plan(payload: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(payload)?);
        return Ok(());
    }
    println!("OmniInfer Advisor Plan");
    let model = payload.get("model").unwrap_or(&Value::Null);
    let planning = payload.get("planning_hardware").unwrap_or(&Value::Null);
    println!("Model: {}", json_str(model, "model").unwrap_or("-"));
    println!(
        "Context size: {}",
        json_u64(payload, "context_size")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "Planning hardware: free_vram={} GiB, available_ram={} GiB, cpu_cores={}",
        json_number_string(planning, "gpu_vram_free_gib"),
        json_number_string(planning, "available_ram_gib"),
        json_u64(planning, "cpu_cores")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    if let Some(estimate) = model.get("estimate") {
        print_memory_breakdown(estimate.get("breakdown").unwrap_or(&Value::Null));
    }
    if let Some(recommended) = payload
        .get("recommended_path")
        .filter(|value| value.is_object())
    {
        println!(
            "Recommended path: {} ({})",
            json_str(recommended, "path").unwrap_or("-"),
            json_str(recommended, "fit").unwrap_or("-")
        );
    }
    println!("Run paths:");
    if let Some(paths) = payload.get("run_paths").and_then(Value::as_array) {
        for path in paths {
            println!(
                "  {}: feasible={}, fit={}, speed={}",
                json_str(path, "path").unwrap_or("-"),
                if json_bool(path, "feasible_now").unwrap_or(false) {
                    "yes"
                } else {
                    "no"
                },
                json_str(path, "fit").unwrap_or("-"),
                json_str(path, "estimated_relative_speed").unwrap_or("-")
            );
        }
    }
    print_string_array(payload, "next_commands", "Next command", usize::MAX);
    print_warnings(payload);
    Ok(())
}

pub fn print_recommend(payload: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(payload)?);
        return Ok(());
    }
    println!("OmniInfer Advisor Recommend");
    println!("Task: {}", json_str(payload, "task").unwrap_or("any"));
    println!(
        "Models scanned: {}",
        json_u64(payload, "models_scanned").unwrap_or(0)
    );
    let rows = payload
        .get("recommendations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if rows.is_empty() {
        println!("No local model recommendations found.");
        return Ok(());
    }
    for (index, row) in rows.iter().enumerate() {
        let model = row.get("model").unwrap_or(&Value::Null);
        let recommended = row.get("recommended").unwrap_or(&Value::Null);
        let evidence = row.get("evidence").unwrap_or(&Value::Null);
        println!("{}. {}", index + 1, json_str(model, "model").unwrap_or("-"));
        println!(
            "   backend={} fit={} score={} confidence={} evidence={}",
            json_str(recommended, "backend").unwrap_or("-"),
            json_str(recommended, "fit").unwrap_or("-"),
            json_number_string(row, "score"),
            json_str(row, "recommendation_confidence").unwrap_or("-"),
            json_str(evidence, "level").unwrap_or("-")
        );
        if let Some(command) = json_str(row, "next_command") {
            println!("   command={command}");
        }
    }
    Ok(())
}

fn print_usable_backends(payload: &Value) {
    println!("Usable backends:");
    let backends = payload
        .get("backends")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|backend| json_bool(backend, "installed").unwrap_or(false))
        .filter(|backend| json_bool(backend, "hardware_compatible").unwrap_or(false))
        .collect::<Vec<_>>();
    if backends.is_empty() {
        println!("  none installed and compatible");
        return;
    }
    let rows = backends
        .iter()
        .map(|backend| {
            let capabilities = backend
                .get("capabilities")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .take(4)
                .collect::<Vec<_>>()
                .join(", ");
            vec![
                json_str(backend, "id").unwrap_or("-").to_string(),
                json_str(backend, "family").unwrap_or("-").to_string(),
                if capabilities.is_empty() {
                    "-".to_string()
                } else {
                    capabilities
                },
            ]
        })
        .collect::<Vec<_>>();
    print_table(&["Backend", "Family", "Capabilities"], &rows, "  ");
    let all_count = payload
        .get("backends")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let hidden_count = all_count.saturating_sub(backends.len());
    if hidden_count > 0 {
        println!(
            "Hidden backends: {hidden_count} unavailable or incompatible; use --json for the full probe."
        );
    }
}

fn print_table(headers: &[&str], rows: &[Vec<String>], indent: &str) {
    let widths = headers
        .iter()
        .enumerate()
        .map(|(index, header)| {
            rows.iter()
                .filter_map(|row| row.get(index))
                .map(String::len)
                .chain(std::iter::once(header.len()))
                .max()
                .unwrap_or(header.len())
        })
        .collect::<Vec<_>>();
    print!("{indent}");
    for (index, header) in headers.iter().enumerate() {
        if index > 0 {
            print!("  ");
        }
        print!("{header:<width$}", width = widths[index]);
    }
    println!();
    print!("{indent}");
    for (index, width) in widths.iter().enumerate() {
        if index > 0 {
            print!("  ");
        }
        print!("{:-<width$}", "");
    }
    println!();
    for row in rows {
        print!("{indent}");
        for (index, width) in widths.iter().enumerate() {
            if index > 0 {
                print!("  ");
            }
            let value = row.get(index).map(String::as_str).unwrap_or("");
            print!("{value:<width$}");
        }
        println!();
    }
}

fn color_output_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

fn heading(value: &str, color: bool) -> String {
    if color {
        format!("{}", style(value).with(Color::Cyan))
    } else {
        value.to_string()
    }
}

fn section(value: &str, color: bool) -> String {
    if color {
        format!("{}", style(value).with(Color::Blue))
    } else {
        value.to_string()
    }
}

fn state_text(value: &str, healthy: bool, color: bool) -> String {
    if !color {
        return value.to_string();
    }
    let color = if healthy {
        Color::Green
    } else {
        Color::DarkGrey
    };
    format!("{}", style(value).with(color))
}

fn print_memory_breakdown(breakdown: &Value) {
    if !breakdown.is_object() {
        return;
    }
    let fields = [
        ("weights", "weights_gib"),
        ("mmproj", "mmproj_gib"),
        ("kv", "kv_cache_gib"),
        ("activation", "activation_gib"),
        ("framework", "framework_overhead_gib"),
        ("slack", "allocator_slack_gib"),
    ];
    let rendered = fields
        .iter()
        .filter_map(|(label, key)| {
            breakdown
                .get(*key)
                .and_then(Value::as_f64)
                .map(|value| format!("{label}={} GiB", format_gib(value)))
        })
        .collect::<Vec<_>>();
    if !rendered.is_empty() {
        println!("Memory breakdown: {}", rendered.join(", "));
    }
}

fn print_string_array(payload: &Value, key: &str, label: &str, limit: usize) {
    if let Some(values) = payload.get(key).and_then(Value::as_array) {
        for value in values.iter().filter_map(Value::as_str).take(limit) {
            println!("{label}: {value}");
        }
    }
}

fn print_warnings(payload: &Value) {
    print_string_array(payload, "warnings", "Warning", usize::MAX);
}

fn json_number_string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(format_gib)
        .unwrap_or_else(|| "-".to_string())
}
