use std::{io::IsTerminal, time::Duration};

use anyhow::Result;
use crossterm::style::{Color, Stylize, style};
use omniinfer_core::{backend_profiles, backend_registry, config, local_state};

use crate::{BackendScope, json_bool, json_str, post_local_json_for_config_with_autostart};

pub(crate) fn print_backend_list(scope: BackendScope) -> Result<()> {
    let payload = rust_backend_payload(scope.clone());
    let rows = payload
        .get("data")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();

    let title = match scope {
        BackendScope::Compatible => "Compatible backends",
        BackendScope::Installed => "Installed backends",
        BackendScope::All => "Available backends",
    };
    println!("{title}");
    let color = color_output_enabled();
    let width = rows
        .iter()
        .filter_map(|item| json_str(item, "id"))
        .map(str::len)
        .chain(std::iter::once("Backend".len()))
        .max()
        .unwrap_or("Backend".len());
    println!("{:<width$}  Selected  Runtime", "Backend");
    println!("{:<width$}  --------  ---------", "-".repeat(width));
    if rows.is_empty() {
        println!("(none)");
        return Ok(());
    }
    let mut missing_runtime_count = 0_usize;
    for item in &rows {
        let backend = json_str(&item, "id").unwrap_or("");
        let selected = if json_bool(&item, "selected").unwrap_or(false) {
            "yes"
        } else {
            ""
        };
        let runtime_exists = json_bool(&item, "binary_exists").unwrap_or(false);
        if !runtime_exists {
            missing_runtime_count += 1;
        }
        let runtime = if runtime_exists {
            "installed"
        } else {
            "missing"
        };
        println!(
            "{}  {}  {}",
            styled_backend_cell(backend, width, color),
            styled_state_cell(selected, 8, StateStyle::Selected, color),
            styled_state_cell(runtime, 9, runtime_style(runtime_exists), color)
        );
    }
    if matches!(scope, BackendScope::Compatible) && missing_runtime_count == rows.len() {
        println!("Install a runtime: omniinfer backend install <backend>");
    }
    Ok(())
}

fn color_output_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

fn styled_backend_cell(value: &str, width: usize, color: bool) -> String {
    let padded = format!("{value:<width$}");
    if color {
        format!("{}", style(padded).with(Color::Blue))
    } else {
        padded
    }
}

#[derive(Clone, Copy)]
enum StateStyle {
    Selected,
    Installed,
    Missing,
}

fn runtime_style(runtime_exists: bool) -> StateStyle {
    if runtime_exists {
        StateStyle::Installed
    } else {
        StateStyle::Missing
    }
}

fn styled_state_cell(value: &str, width: usize, style_kind: StateStyle, color: bool) -> String {
    let padded = format!("{value:<width$}");
    if !color || value.is_empty() {
        return padded;
    }
    match style_kind {
        StateStyle::Selected | StateStyle::Installed => {
            format!("{}", style(padded).with(Color::Green))
        }
        StateStyle::Missing => format!("{}", style(padded).with(Color::DarkGrey)),
    }
}

pub(crate) fn rust_backend_payload(scope: BackendScope) -> serde_json::Value {
    backend_registry::BackendRegistry::load_current().api_payload(match scope {
        BackendScope::Installed => backend_registry::BackendScope::Installed,
        BackendScope::Compatible => backend_registry::BackendScope::Compatible,
        BackendScope::All => backend_registry::BackendScope::All,
    })
}

pub(crate) fn select_backend(backend: &str) -> Result<()> {
    let config = config::load_app_config().unwrap_or_default();
    select_backend_for_config(backend, &config)
}

pub(crate) fn select_backend_for_config(backend: &str, config: &config::AppConfig) -> Result<()> {
    select_backend_for_config_with_autostart(backend, config, true)
}

pub(crate) fn select_backend_for_config_with_autostart(
    backend: &str,
    config: &config::AppConfig,
    autostart: bool,
) -> Result<()> {
    let backends = rust_backend_payload(BackendScope::All);
    let rows = backends
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Unable to read backend list."))?;
    let backend_payload = rows
        .iter()
        .find(|item| json_str(item, "id") == Some(backend))
        .ok_or_else(|| {
            let available = rows
                .iter()
                .filter_map(|item| json_str(item, "id"))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::anyhow!("Unsupported backend: {backend}\nAvailable backends: {available}")
        })?;

    let _payload = post_local_json_for_config_with_autostart(
        "/omni/backend/select",
        &serde_json::json!({ "backend": backend }),
        Duration::from_secs(30),
        config,
        autostart,
    )?;
    local_state::save_selected_backend(backend)?;
    let profile = backend_profiles::ensure_backend_profile_template(backend_payload)?;
    println!("Selected backend: {backend}");
    if let Some(models_dir) = json_str(backend_payload, "models_dir") {
        println!("Models directory: {models_dir}");
    }
    println!(
        "Backend config: {} ({})",
        profile.path.display(),
        if profile.created {
            "created"
        } else {
            "already exists"
        }
    );
    Ok(())
}
