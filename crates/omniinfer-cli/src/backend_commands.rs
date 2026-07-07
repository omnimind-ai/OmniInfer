use std::time::Duration;

use anyhow::Result;
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
    let width = rows
        .iter()
        .filter_map(|item| json_str(item, "id"))
        .map(str::len)
        .chain(std::iter::once("Backend".len()))
        .max()
        .unwrap_or("Backend".len());
    println!("{:<width$}  Selected  Installed", "Backend");
    println!("{:<width$}  --------  ---------", "-".repeat(width));
    if rows.is_empty() {
        println!("(none)");
        return Ok(());
    }
    for item in rows {
        let backend = json_str(&item, "id").unwrap_or("");
        let selected = if json_bool(&item, "selected").unwrap_or(false) {
            "yes"
        } else {
            ""
        };
        let installed = if json_bool(&item, "binary_exists").unwrap_or(false) {
            "yes"
        } else {
            ""
        };
        println!("{backend:<width$}  {selected:<8}  {installed:<9}");
    }
    Ok(())
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
