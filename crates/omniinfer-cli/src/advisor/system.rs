use super::*;

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
    let recommended_backend_to_install = recommended_backend_to_install(&backends);

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
            "recommended_backend_to_install": recommended_backend_to_install,
        },
    })
}

fn host_payload() -> Value {
    let (available_ram_gib, total_ram_gib) = system_memory_gib()
        .or_else(linux_meminfo_gib)
        .unwrap_or((None, None));
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

fn system_memory_gib() -> Option<(Option<f64>, Option<f64>)> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available = bytes_to_gib(system.available_memory())?;
    let total = bytes_to_gib(system.total_memory())?;
    Some((Some(available), Some(total)))
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
    let installable = prebuilt_catalog::installable_backend_ids();
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
            if let Some(id) = map.get("id").and_then(Value::as_str).map(str::to_string) {
                let prebuilt_installable = installable.contains(&id);
                map.insert(
                    "prebuilt_installable".to_string(),
                    Value::Bool(prebuilt_installable),
                );
                map.insert(
                    "install_command".to_string(),
                    prebuilt_installable
                        .then(|| Value::String(format!("omniinfer backend install {id}")))
                        .unwrap_or(Value::Null),
                );
                map.insert(
                    "priority".to_string(),
                    Value::Number(backend_priority(&id).into()),
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

fn recommended_backend_to_install(backends: &[Value]) -> Option<String> {
    backends
        .iter()
        .filter(|backend| !json_bool(backend, "installed").unwrap_or(false))
        .filter(|backend| json_bool(backend, "hardware_compatible").unwrap_or(false))
        .filter(|backend| json_bool(backend, "prebuilt_installable").unwrap_or(false))
        .min_by_key(|backend| json_u64(backend, "priority").unwrap_or(999))
        .and_then(|backend| json_str(backend, "id"))
        .map(str::to_string)
}

fn mib_to_gib(value: f64) -> f64 {
    round_gib(value / 1024.0)
}

fn bytes_to_gib(value: u64) -> Option<f64> {
    if value == 0 {
        None
    } else {
        Some(round_gib(value as f64 / 1024.0 / 1024.0 / 1024.0))
    }
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use super::*;

    #[test]
    fn install_recommendation_requires_catalog_entry() {
        let backends = normalize_backends(json!({
            "recommended": "ik_llama.cpp-cuda",
            "data": [
                {
                    "id": "ik_llama.cpp-cuda",
                    "binary_exists": false,
                    "compatibility": "compatible"
                },
                {
                    "id": "llama.cpp-cuda",
                    "binary_exists": false,
                    "compatibility": "compatible"
                }
            ]
        }));
        assert_eq!(
            recommended_backend_to_install(&backends).as_deref(),
            Some("llama.cpp-cuda")
        );
        let ik = backends
            .iter()
            .find(|backend| json_str(backend, "id") == Some("ik_llama.cpp-cuda"))
            .unwrap();
        let llama = backends
            .iter()
            .find(|backend| json_str(backend, "id") == Some("llama.cpp-cuda"))
            .unwrap();
        assert_eq!(ik["prebuilt_installable"], false);
        assert!(ik["install_command"].is_null());
        assert_eq!(llama["prebuilt_installable"], true);
        assert_eq!(
            llama["install_command"],
            "omniinfer backend install llama.cpp-cuda"
        );
    }
}
