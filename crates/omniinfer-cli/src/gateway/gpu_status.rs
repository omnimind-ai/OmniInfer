use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use anyhow::Result;
use omniinfer_core::backend_registry;
use serde_json::{Value, json};

use super::LoadedRuntimeSummary;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeCudaSelection {
    pub(super) visible_devices: String,
    pub(super) warning: Option<String>,
}

pub(super) fn runtime_env_for_backend(
    backend: &backend_registry::BackendSpec,
    launch_args: &[String],
) -> (Vec<(String, String)>, Option<RuntimeCudaSelection>) {
    let mut env = Vec::new();
    if let Some(launcher) = backend.launcher_path.as_deref()
        && let Some(parent) = PathBuf::from(launcher).parent()
        && std::env::consts::OS != "windows"
    {
        let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let value = if existing.is_empty() {
            parent.display().to_string()
        } else {
            format!("{}:{existing}", parent.display())
        };
        env.push(("LD_LIBRARY_PATH".to_string(), value));
    }
    let mut cuda_selection = None;
    if backend.capabilities.iter().any(|cap| cap == "cuda") {
        cuda_selection = select_cuda_visible_devices(launch_args);
    }
    if let Some(selection) = cuda_selection.as_ref() {
        env.push((
            "CUDA_VISIBLE_DEVICES".to_string(),
            selection.visible_devices.clone(),
        ));
    }
    (env, cuda_selection)
}

fn select_cuda_visible_devices(launch_args: &[String]) -> Option<RuntimeCudaSelection> {
    if let Ok(devices) = std::env::var("OMNIINFER_CUDA_VISIBLE_DEVICES")
        && !devices.trim().is_empty()
    {
        return Some(RuntimeCudaSelection {
            visible_devices: devices,
            warning: None,
        });
    }
    if let Ok(devices) = std::env::var("CUDA_VISIBLE_DEVICES")
        && !devices.trim().is_empty()
    {
        return Some(RuntimeCudaSelection {
            visible_devices: devices,
            warning: None,
        });
    }
    if uses_explicit_cuda_device_args(launch_args) {
        return None;
    }
    select_idle_cuda_device_from_nvidia_smi().map(|device| RuntimeCudaSelection {
        visible_devices: device.index,
        warning: device.warning,
    })
}

pub(super) fn uses_explicit_cuda_device_args(args: &[String]) -> bool {
    args.iter().any(|token| {
        let flag = token.split_once('=').map(|(flag, _)| flag).unwrap_or(token);
        matches!(
            flag,
            "--tensor-split" | "--split-mode" | "--main-gpu" | "--device" | "-mg"
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CudaDeviceChoice {
    pub(super) index: String,
    pub(super) warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CudaDeviceUsage {
    index: String,
    uuid: String,
    used_memory_mib: u64,
    compute_processes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GpuStatusDevice {
    pub(super) index: String,
    pub(super) uuid: String,
    pub(super) name: String,
    pub(super) memory_total_mib: u64,
    pub(super) memory_used_mib: u64,
    pub(super) memory_free_mib: u64,
    pub(super) utilization_gpu_percent: Option<u32>,
    pub(super) processes: Vec<GpuStatusProcess>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GpuStatusProcess {
    gpu_uuid: String,
    pid: u32,
    process_name: String,
    pub(super) display_name: String,
    used_memory_mib: Option<u64>,
    pub(super) owner_model: Option<String>,
    pub(super) owner_admin_id: Option<String>,
    pub(super) owner_type: String,
    pub(super) owner_name: Option<String>,
}

pub(super) fn gpu_status_payload(devices: &[GpuStatusDevice]) -> Value {
    json!({
        "object": "list",
        "data": devices.iter().map(gpu_status_device_payload).collect::<Vec<_>>(),
    })
}

pub(super) fn gpu_status_device_payload(device: &GpuStatusDevice) -> Value {
    json!({
        "index": parse_cuda_index(&device.index),
        "uuid": device.uuid,
        "name": device.name,
        "memory_total_mb": device.memory_total_mib,
        "memory_used_mb": device.memory_used_mib,
        "memory_free_mb": device.memory_free_mib,
        "utilization_gpu": device.utilization_gpu_percent,
        "processes": device.processes.iter().map(gpu_status_process_payload).collect::<Vec<_>>(),
    })
}

fn gpu_status_process_payload(process: &GpuStatusProcess) -> Value {
    json!({
        "pid": process.pid,
        "name": process.process_name,
        "display_name": process.display_name,
        "used_memory_mb": process.used_memory_mib,
        "owner_model": process.owner_model,
        "owner_admin_id": process.owner_admin_id,
        "owner_type": process.owner_type,
        "owner_name": process.owner_name,
    })
}

pub(super) fn query_nvidia_smi_gpu_status(
    loaded: &[LoadedRuntimeSummary],
) -> Result<Vec<GpuStatusDevice>> {
    let gpu_output = ProcessCommand::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !gpu_output.status.success() {
        anyhow::bail!("nvidia-smi GPU query failed");
    }
    let mut devices = parse_gpu_status_rows(&String::from_utf8_lossy(&gpu_output.stdout));
    if let Ok(process_output) = ProcessCommand::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        && process_output.status.success()
    {
        let processes =
            parse_gpu_process_rows(&String::from_utf8_lossy(&process_output.stdout), loaded);
        apply_gpu_process_rows(&mut devices, processes);
    }
    Ok(devices)
}

pub(super) fn parse_gpu_status_rows(text: &str) -> Vec<GpuStatusDevice> {
    text.lines()
        .filter_map(|line| {
            let parts = split_csv_row(line);
            if parts.len() < 7 {
                return None;
            }
            Some(GpuStatusDevice {
                index: parts[0].clone(),
                uuid: parts[1].clone(),
                name: parts[2].clone(),
                memory_total_mib: parse_mib(&parts[3])?,
                memory_used_mib: parse_mib(&parts[4])?,
                memory_free_mib: parse_mib(&parts[5])?,
                utilization_gpu_percent: parse_optional_u32(&parts[6]),
                processes: Vec::new(),
            })
        })
        .collect()
}

pub(super) fn parse_gpu_process_rows(
    text: &str,
    loaded: &[LoadedRuntimeSummary],
) -> Vec<GpuStatusProcess> {
    text.lines()
        .filter_map(|line| {
            let parts = split_csv_row(line);
            if parts.len() < 4 {
                return None;
            }
            let pid = parts[1].parse::<u32>().ok()?;
            let owner = loaded.iter().find(|runtime| runtime.backend_pid == pid);
            let process_name = parts[2].clone();
            let (owner_type, owner_name) = if let Some(owner) = owner {
                (
                    "admin".to_string(),
                    owner
                        .owner_admin_id
                        .clone()
                        .or_else(|| Some("admin".to_string())),
                )
            } else {
                ("user".to_string(), process_username(pid))
            };
            Some(GpuStatusProcess {
                gpu_uuid: parts[0].clone(),
                pid,
                display_name: short_process_name(&process_name),
                process_name,
                used_memory_mib: parse_mib(&parts[3]),
                owner_model: owner.map(|runtime| runtime.id.clone()),
                owner_admin_id: owner.and_then(|runtime| runtime.owner_admin_id.clone()),
                owner_type,
                owner_name,
            })
        })
        .collect()
}

pub(super) fn apply_gpu_process_rows(
    devices: &mut [GpuStatusDevice],
    processes: Vec<GpuStatusProcess>,
) {
    for process in processes {
        if let Some(device) = devices
            .iter_mut()
            .find(|device| device.uuid == process.gpu_uuid)
        {
            device.processes.push(process);
        }
    }
}

fn split_csv_row(line: &str) -> Vec<String> {
    line.split(',')
        .map(|part| part.trim().to_string())
        .collect()
}

fn parse_mib(value: &str) -> Option<u64> {
    value
        .trim()
        .strip_suffix("MiB")
        .unwrap_or(value.trim())
        .trim()
        .parse()
        .ok()
}

fn parse_optional_u32(value: &str) -> Option<u32> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("[not supported]") {
        return None;
    }
    value.parse().ok()
}

fn short_process_name(value: &str) -> String {
    let value = value.trim();
    PathBuf::from(value)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(value)
        .to_string()
}

fn process_username(pid: u32) -> Option<String> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let uid = status.lines().find_map(|line| {
        line.strip_prefix("Uid:")
            .and_then(|rest| rest.split_whitespace().next())
    })?;
    username_for_uid(uid)
}

fn username_for_uid(uid: &str) -> Option<String> {
    let passwd = std::fs::read_to_string("/etc/passwd").ok()?;
    passwd.lines().find_map(|line| {
        let mut parts = line.split(':');
        let name = parts.next()?;
        let _password = parts.next()?;
        let user_id = parts.next()?;
        (user_id == uid).then(|| name.to_string())
    })
}

fn select_idle_cuda_device_from_nvidia_smi() -> Option<CudaDeviceChoice> {
    let output = ProcessCommand::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let gpu_rows = String::from_utf8_lossy(&output.stdout);
    let mut devices = parse_cuda_gpu_rows(&gpu_rows);
    if devices.is_empty() {
        return None;
    }
    if let Ok(process_output) = ProcessCommand::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ])
        .output()
        && process_output.status.success()
    {
        let process_rows = String::from_utf8_lossy(&process_output.stdout);
        apply_cuda_process_rows(&mut devices, &process_rows);
    }
    select_cuda_device_from_usage(&devices)
}

pub(super) fn parse_cuda_gpu_rows(text: &str) -> Vec<CudaDeviceUsage> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let index = parts.next()?.to_string();
            let uuid = parts.next()?.to_string();
            let used_memory_mib = parts.next()?.parse().ok()?;
            Some(CudaDeviceUsage {
                index,
                uuid,
                used_memory_mib,
                compute_processes: 0,
            })
        })
        .collect()
}

pub(super) fn apply_cuda_process_rows(devices: &mut [CudaDeviceUsage], text: &str) {
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Some(uuid) = line.split(',').next().map(str::trim) else {
            continue;
        };
        if let Some(device) = devices.iter_mut().find(|device| device.uuid == uuid) {
            device.compute_processes = device.compute_processes.saturating_add(1);
        }
    }
}

pub(super) fn select_cuda_device_from_usage(
    devices: &[CudaDeviceUsage],
) -> Option<CudaDeviceChoice> {
    let idle = devices
        .iter()
        .filter(|device| device.compute_processes == 0)
        .min_by_key(|device| parse_cuda_index(&device.index));
    if let Some(device) = idle {
        return Some(CudaDeviceChoice {
            index: device.index.clone(),
            warning: None,
        });
    }
    let selected = devices.iter().min_by_key(|device| {
        (
            device.compute_processes,
            device.used_memory_mib,
            parse_cuda_index(&device.index),
        )
    })?;
    Some(CudaDeviceChoice {
        index: selected.index.clone(),
        warning: Some(cuda_all_busy_warning()),
    })
}

fn parse_cuda_index(index: &str) -> u32 {
    index.parse().unwrap_or(u32::MAX)
}

fn cuda_all_busy_warning() -> String {
    "Warning: all CUDA GPUs appear to be in use; OmniInfer selected the least-used GPU and set CUDA_VISIBLE_DEVICES for the backend process."
        .to_string()
}
