use std::process::Command;

use anyhow::Result;
use serde_json::{Value, json};

use crate::{current_system_name, json_bool, json_str, json_u64};

pub fn system_payload(backends: Value) -> Value {
    let backends = normalize_backends(backends);
    let cuda_devices = query_cuda_devices();
    let visible_filter = std::env::var("OMNIINFER_CUDA_VISIBLE_DEVICES")
        .ok()
        .or_else(|| std::env::var("CUDA_VISIBLE_DEVICES").ok())
        .filter(|value| !value.trim().is_empty());
    let visible_devices = filter_visible_cuda_devices(&cuda_devices, visible_filter.as_deref());
    let best_free_device = best_cuda_device(if visible_devices.is_empty() {
        &cuda_devices
    } else {
        &visible_devices
    });
    let installed_backends = backends
        .iter()
        .filter(|backend| json_bool(backend, "installed").unwrap_or(false))
        .filter_map(|backend| json_str(backend, "id").map(str::to_string))
        .collect::<Vec<_>>();
    let compatible_backends = backends
        .iter()
        .filter(|backend| json_bool(backend, "hardware_compatible").unwrap_or(false))
        .filter_map(|backend| json_str(backend, "id").map(str::to_string))
        .collect::<Vec<_>>();
    let recommended_installed_backend = recommended_installed_backend(&backends);

    json!({
        "object": "advisor.system",
        "host": host_payload(),
        "cuda": {
            "devices": cuda_devices,
            "visible_filter": visible_filter,
            "visible_devices": visible_devices,
            "best_free_device": best_free_device,
        },
        "backends": backends,
        "summary": {
            "installed_backends": installed_backends,
            "compatible_backends": compatible_backends,
            "recommended_installed_backend": recommended_installed_backend,
        },
    })
}

pub fn print_system(payload: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(payload)?);
        return Ok(());
    }
    let host = payload.get("host").and_then(Value::as_object);
    let cuda = payload.get("cuda").and_then(Value::as_object);
    let summary = payload.get("summary").and_then(Value::as_object);
    println!("OmniInfer Advisor System");
    println!(
        "System: {} ({})",
        host.and_then(|value| value.get("system"))
            .and_then(Value::as_str)
            .unwrap_or("-"),
        host.and_then(|value| value.get("machine"))
            .and_then(Value::as_str)
            .unwrap_or("-")
    );
    println!(
        "CPU cores: {}",
        host.and_then(|value| value.get("cpu_cores"))
            .and_then(Value::as_u64)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "RAM: {} GiB available / {} GiB total",
        host.and_then(|value| value.get("available_ram_gib"))
            .and_then(Value::as_f64)
            .map(format_gib)
            .unwrap_or_else(|| "-".to_string()),
        host.and_then(|value| value.get("total_ram_gib"))
            .and_then(Value::as_f64)
            .map(format_gib)
            .unwrap_or_else(|| "-".to_string())
    );

    let devices = cuda
        .and_then(|value| value.get("visible_devices"))
        .and_then(Value::as_array)
        .or_else(|| {
            cuda.and_then(|value| value.get("devices"))
                .and_then(Value::as_array)
        });
    match devices {
        Some(devices) if !devices.is_empty() => {
            println!("CUDA devices:");
            for device in devices {
                println!(
                    "  GPU {}: {} free={} GiB total={} GiB util={}%",
                    json_str(device, "index").unwrap_or("-"),
                    json_str(device, "name").unwrap_or("-"),
                    json_number_string(device, "free_gib"),
                    json_number_string(device, "total_gib"),
                    json_u64(device, "utilization_pct")
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_string())
                );
            }
        }
        _ => println!("CUDA devices: none detected"),
    }

    println!(
        "Recommended installed backend: {}",
        summary
            .and_then(|value| value.get("recommended_installed_backend"))
            .and_then(Value::as_str)
            .unwrap_or("-")
    );
    print_usable_backends(payload);
    Ok(())
}

fn host_payload() -> Value {
    let (available_ram_gib, total_ram_gib) = linux_meminfo_gib().unwrap_or((None, None));
    json!({
        "system": current_system_name(),
        "platform": platform_string(),
        "machine": std::env::consts::ARCH,
        "processor": std::env::consts::ARCH,
        "cpu_cores": std::thread::available_parallelism().map(usize::from).unwrap_or(0),
        "available_ram_gib": available_ram_gib,
        "total_ram_gib": total_ram_gib,
    })
}

fn platform_string() -> String {
    Command::new("uname")
        .args(["-srmo"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH))
}

fn linux_meminfo_gib() -> Option<(Option<f64>, Option<f64>)> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available = None;
    let mut total = None;
    for line in raw.lines() {
        if let Some(value) = parse_meminfo_kib(line, "MemAvailable:") {
            available = Some(round_gib(value as f64 / 1024.0 / 1024.0));
        } else if let Some(value) = parse_meminfo_kib(line, "MemTotal:") {
            total = Some(round_gib(value as f64 / 1024.0 / 1024.0));
        }
    }
    Some((available, total))
}

fn parse_meminfo_kib(line: &str, key: &str) -> Option<u64> {
    let value = line.strip_prefix(key)?.split_whitespace().next()?;
    value.parse().ok()
}

fn query_cuda_devices() -> Vec<Value> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_nvidia_smi_line)
        .collect()
}

fn parse_nvidia_smi_line(line: &str) -> Option<Value> {
    let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() < 6 {
        return None;
    }
    Some(json!({
        "index": parts[0],
        "name": parts[1],
        "total_gib": mib_to_gib(parts[2].parse().ok()?),
        "free_gib": mib_to_gib(parts[3].parse().ok()?),
        "used_gib": mib_to_gib(parts[4].parse().ok()?),
        "utilization_pct": parts[5].parse::<u64>().ok()?,
    }))
}

fn filter_visible_cuda_devices(devices: &[Value], visible_filter: Option<&str>) -> Vec<Value> {
    let Some(visible_filter) = visible_filter else {
        return devices.to_vec();
    };
    let values = visible_filter
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Vec::new();
    }
    devices
        .iter()
        .filter(|device| {
            let index = json_str(device, "index").unwrap_or("");
            values.iter().any(|value| *value == index)
        })
        .cloned()
        .collect()
}

fn best_cuda_device(devices: &[Value]) -> Option<Value> {
    devices
        .iter()
        .max_by(|left, right| {
            let left = left
                .get("free_gib")
                .and_then(Value::as_f64)
                .unwrap_or_default();
            let right = right
                .get("free_gib")
                .and_then(Value::as_f64)
                .unwrap_or_default();
            left.partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

fn normalize_backends(payload: Value) -> Vec<Value> {
    let recommended = json_str(&payload, "recommended").map(str::to_string);
    payload
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|backend| {
            let mut backend = backend.clone();
            let map = backend.as_object_mut()?;
            let installed = map
                .get("binary_exists")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let compatibility = map
                .get("compatibility")
                .and_then(Value::as_str)
                .unwrap_or("");
            let hardware_compatible = matches!(compatibility, "installed" | "compatible");
            map.insert("installed".to_string(), Value::Bool(installed));
            map.insert(
                "hardware_compatible".to_string(),
                Value::Bool(hardware_compatible),
            );
            if let Some(id) = map.get("id").and_then(Value::as_str) {
                map.insert(
                    "priority".to_string(),
                    Value::Number(priority_for_backend(id, recommended.as_deref()).into()),
                );
            }
            Some(backend)
        })
        .collect()
}

fn recommended_installed_backend(backends: &[Value]) -> Option<String> {
    backends
        .iter()
        .filter(|backend| json_bool(backend, "installed").unwrap_or(false))
        .filter(|backend| json_bool(backend, "hardware_compatible").unwrap_or(false))
        .min_by_key(|backend| json_u64(backend, "priority").unwrap_or(999))
        .and_then(|backend| json_str(backend, "id"))
        .map(str::to_string)
}

fn priority_for_backend(id: &str, recommended: Option<&str>) -> u64 {
    if recommended == Some(id) {
        return 0;
    }
    match id {
        value if value.contains("cuda") => 1,
        value if value.contains("vllm") => 2,
        value if value.contains("cpu") || value.ends_with("-linux") => 3,
        _ => 99,
    }
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

fn json_number_string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(format_gib)
        .unwrap_or_else(|| "-".to_string())
}

fn mib_to_gib(value: f64) -> f64 {
    round_gib(value / 1024.0)
}

fn round_gib(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn format_gib(value: f64) -> String {
    let rounded = round_gib(value);
    if (rounded.fract()).abs() < f64::EPSILON {
        format!("{rounded:.1}")
    } else {
        format!("{rounded:.2}")
    }
}
