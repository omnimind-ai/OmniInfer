use std::path::{Path, PathBuf};

use super::registry::{BackendSpec, HostInfo, HostSystem};

pub(super) fn gpu_backend_ids(host: HostInfo) -> &'static [&'static str] {
    match host.system {
        HostSystem::Linux => &[
            "llama.cpp-linux-cuda",
            "llama.cpp-linux-rocm",
            "llama.cpp-linux-vulkan",
            "stable-diffusion.cpp-linux-vulkan",
            "omniinfer-native-linux",
            "ik_llama.cpp-linux-cuda",
            "vllm-linux-cuda",
            "freetoken-linux-cuda",
            "vla.cpp-linux-cuda",
        ],
        HostSystem::Windows => &[
            "llama.cpp-cuda",
            "llama.cpp-vulkan",
            "stable-diffusion.cpp-vulkan",
            "llama.cpp-sycl",
            "llama.cpp-hip",
            "ik_llama.cpp-cuda",
            "vllm-wsl2-cuda",
            "vllm-wsl2-rocm",
        ],
        _ => &[],
    }
}

pub(super) fn is_hardware_compatible(host: HostInfo, spec: &BackendSpec) -> bool {
    let caps = spec
        .capabilities
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if caps.contains(&"arm64") && !is_arm64(host.machine) {
        return false;
    }
    if caps.contains(&"x64") && !is_x86_64(host.machine) {
        return false;
    }
    if caps.contains(&"s390x") && host.machine != "s390x" {
        return false;
    }
    if caps.contains(&"openvino") || caps.contains(&"eagle3") {
        return spec.binary_exists();
    }
    if !gpu_backend_ids(host).contains(&spec.id.as_str()) {
        return true;
    }
    if caps.contains(&"cuda") {
        return if caps.contains(&"cuda13") {
            cuda13_driver_detected()
        } else {
            cuda_detected()
        };
    }
    if caps.contains(&"rocm") || caps.contains(&"hip") {
        return rocm_detected(host);
    }
    if caps.contains(&"metal") {
        return host.system == HostSystem::Mac || host.system == HostSystem::Ios;
    }
    if caps.contains(&"vulkan") {
        return vulkan_detected();
    }
    spec.binary_exists()
}

fn is_arm64(machine: &str) -> bool {
    machine.eq_ignore_ascii_case("aarch64") || machine.eq_ignore_ascii_case("arm64")
}

fn is_x86_64(machine: &str) -> bool {
    machine.eq_ignore_ascii_case("x86_64") || machine.eq_ignore_ascii_case("amd64")
}

fn cuda_detected() -> bool {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false)
}

fn cuda13_driver_detected() -> bool {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .filter_map(parse_nvidia_driver_branch)
                    .any(|branch| branch >= 580)
        })
        .unwrap_or(false)
}

pub(super) fn parse_nvidia_driver_branch(value: &str) -> Option<u32> {
    value.trim().split('.').next()?.parse().ok()
}

fn rocm_detected(host: HostInfo) -> bool {
    if host.system == HostSystem::Windows {
        return windows_amd_gpu_detected();
    }
    std::process::Command::new("rocm-smi")
        .arg("--showmeminfo")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn windows_amd_gpu_detected() -> bool {
    let mut powershell = std::process::Command::new("powershell.exe");
    powershell.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_VideoController | ForEach-Object Name",
    ]);
    hide_child_window(&mut powershell);
    if powershell
        .output()
        .is_ok_and(|output| output.status.success() && output_mentions_amd_gpu(&output.stdout))
    {
        return true;
    }

    let mut registry = std::process::Command::new("reg.exe");
    registry.args([
        "query",
        r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
        "/s",
        "/v",
        "DriverDesc",
    ]);
    hide_child_window(&mut registry);
    registry
        .output()
        .is_ok_and(|output| output.status.success() && output_mentions_amd_gpu(&output.stdout))
}

pub(super) fn output_mentions_amd_gpu(output: &[u8]) -> bool {
    let names = String::from_utf8_lossy(output).to_ascii_lowercase();
    names.contains("radeon")
        || names
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|token| token == "amd")
}

fn hide_child_window(command: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

fn vulkan_detected() -> bool {
    std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

pub(super) fn embedded_module_exists(runtime_dir: &Path, module_name: &str) -> bool {
    embedded_site_roots(runtime_dir)
        .iter()
        .any(|root| module_path_exists(root, module_name))
}

fn embedded_site_roots(runtime_dir: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let bases = [runtime_dir.to_path_buf(), runtime_dir.join("venv")];
    for base in bases {
        for candidate in [
            base.join("Lib").join("site-packages"),
            base.join("lib").join("site-packages"),
        ] {
            if candidate.is_dir() && !roots.contains(&candidate) {
                roots.push(candidate);
            }
        }
        let pattern_root = base.join("lib");
        if let Ok(entries) = std::fs::read_dir(pattern_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !name.starts_with("python") {
                    continue;
                }
                for site_name in ["site-packages", "dist-packages"] {
                    let candidate = path.join(site_name);
                    if candidate.is_dir() && !roots.contains(&candidate) {
                        roots.push(candidate);
                    }
                }
            }
        }
    }
    roots
}

fn module_path_exists(site_root: &Path, module_name: &str) -> bool {
    let module_path = module_name
        .split('.')
        .fold(site_root.to_path_buf(), |path, item| path.join(item));
    if module_path.is_dir() || module_path.with_extension("py").is_file() {
        return true;
    }
    let Some(parent) = module_path.parent() else {
        return false;
    };
    let Some(name) = module_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    std::fs::read_dir(parent)
        .map(|entries| {
            entries.flatten().any(|entry| {
                let path = entry.path();
                path.file_stem().and_then(|stem| stem.to_str()) == Some(name)
                    && matches!(
                        path.extension().and_then(|ext| ext.to_str()),
                        Some("so" | "pyd" | "dll" | "dylib")
                    )
            })
        })
        .unwrap_or(false)
}
