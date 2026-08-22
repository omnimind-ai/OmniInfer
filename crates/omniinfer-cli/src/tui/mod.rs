use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::time::Duration;

use anyhow::Result;
use omniinfer_core::{
    chat_stream, config, http_client, local_state, model_load, paths, serve_state,
};
use serde_json::Value;

mod models;
mod render;

use crate::serve::{ForegroundCtrlCHandler, install_foreground_ctrl_c_handler, stop_process};
use crate::{
    BackendScope, ServeArgs, advisor, backend_installer, get_local_json_for_config, json_bool,
    json_str, json_u64, load_model_with_request_for_config, post_local_json_for_config,
    rust_backend_payload, select_backend_for_config, serve_orchestrated, stop_serve,
    wait_for_gateway_ready,
};

use models::{
    advisor_model_summary, advisor_recommendation_map, discover_local_models, model_context_label,
    model_provider_label, model_quant_label, model_size_label, prompt_model_path, same_path,
};
use render::{
    MessageKind, ModelMenuContext, ModelMenuItem, NoticeKind, clear_screen, is_interactive, notice,
    print_chat_header, print_header, print_health_kv, print_help, print_kv, print_message_header,
    print_section, print_tui_performance, prompt_default, select_menu, select_model_menu,
};

#[derive(Debug, Clone)]
struct MenuItem {
    label: String,
    details: Vec<String>,
    selected: bool,
}

#[derive(Debug)]
struct ChatSession {
    backend: String,
    reasoning_visible: bool,
    messages: Vec<Value>,
    last_usage: Option<Value>,
}

pub fn run() -> Result<()> {
    if !is_interactive() {
        anyhow::bail!("OmniInfer TUI requires an interactive terminal.");
    }
    clear_screen();
    print_header("OmniInfer", "Local inference console");
    let config = config::load_app_config().unwrap_or_default();
    let _gateway = TuiGatewayGuard::ensure(&config)?;
    let state = local_state::load_state().unwrap_or_default();
    let backend = match state.selected_model.clone() {
        Some(model) if Path::new(&model.model).exists() => {
            match load_remembered_model(&config, &model) {
                Ok(backend) => backend,
                Err(error) => {
                    notice(
                        &format!("Could not load previous model: {error}"),
                        NoticeKind::Warning,
                    );
                    setup_model_flow(&config)?
                }
            }
        }
        _ => setup_model_flow(&config)?,
    };
    chat_loop(&config, backend)?;
    Ok(())
}

struct TuiGatewayGuard {
    port: u16,
    owned: bool,
    child: Option<Child>,
    _interrupt: Option<ForegroundCtrlCHandler>,
}

impl TuiGatewayGuard {
    fn ensure(config: &config::AppConfig) -> Result<Self> {
        if get_running_state(config).is_some() {
            return Ok(Self {
                port: config.port,
                owned: false,
                child: None,
                _interrupt: None,
            });
        }
        print_section("Service", "Starting local OmniInfer gateway");
        print_kv("Port", &config.port.to_string());
        let interrupt = install_foreground_ctrl_c_handler(config.port, true)?;
        let mut command = ProcessCommand::new(std::env::current_exe()?);
        paths::propagate_cli_roots(&mut command);
        command
            .arg("gateway")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(config.port.to_string())
            .current_dir(paths::repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let guard = Self {
            port: config.port,
            owned: true,
            child: Some(command.spawn()?),
            _interrupt: Some(interrupt),
        };
        let child_pid = guard
            .child
            .as_ref()
            .expect("owned gateway has a child")
            .id();
        let gateway_process =
            serve_state::capture_process_identity(child_pid).ok_or_else(|| {
                anyhow::anyhow!("gateway exited before its process identity could be recorded")
            })?;
        if let Err(error) = serve_state::save_serve_pid_info(&serve_state::ServePidInfo {
            phase: Some("starting".to_string()),
            pid: Some(child_pid),
            gateway_process: Some(gateway_process),
            port: Some(config.port),
            ..Default::default()
        }) {
            return Err(error.into());
        }
        guard
            ._interrupt
            .as_ref()
            .expect("owned gateway has an interrupt handler")
            .arm();
        if guard
            ._interrupt
            .as_ref()
            .is_some_and(ForegroundCtrlCHandler::interrupted)
        {
            return Err(anyhow::anyhow!("startup interrupted"));
        }
        wait_for_gateway_ready(config)?;
        notice("Local gateway ready", NoticeKind::Success);
        println!();
        Ok(guard)
    }
}

impl Drop for TuiGatewayGuard {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _ = stop_serve(self.port);
        if let Some(child) = self.child.as_mut()
            && child.try_wait().ok().flatten().is_none()
        {
            stop_process(child.id());
            let _ = child.wait();
        }
    }
}

pub fn run_server(args: &ServeArgs) -> Result<()> {
    if !is_interactive() {
        return serve_orchestrated(args);
    }
    clear_screen();
    print_header("OmniInfer Server", "Interactive gateway launcher");
    let config = config::load_app_config().unwrap_or_default();
    let backend = choose_backend()?.ok_or_else(|| anyhow::anyhow!("No backend selected."))?;
    let model = choose_model(&config, true, Some(&backend))?
        .ok_or_else(|| anyhow::anyhow!("No model selected."))?;
    let mut args = args.clone();
    args.backend = Some(backend);
    args.model = Some(model.display().to_string());
    serve_orchestrated(&args)
}

mod model_flow;

use model_flow::*;
mod chat_session;

use chat_session::chat_loop;
#[allow(dead_code)]
fn _loaded_services() -> Vec<serve_state::ServePidInfo> {
    serve_state::list_serve_pid_infos().unwrap_or_default()
}

#[cfg(test)]
mod backend_model_tests {
    use super::*;

    #[test]
    fn vla_backend_accepts_only_supported_checkpoint_files() {
        let backend = SelectedBackendInfo {
            family: "vla.cpp".to_string(),
            model_artifact: "vla-artifact".to_string(),
        };
        assert!(model_supported_by_backend(
            Path::new("smolvla.gguf"),
            Some(&backend)
        ));
        assert!(model_supported_by_backend(
            Path::new("smolvla.safetensors"),
            Some(&backend)
        ));
        assert!(!model_supported_by_backend(
            Path::new("weights.bin"),
            Some(&backend)
        ));
        assert!(!model_supported_by_backend(
            Path::new("config.json"),
            Some(&backend)
        ));
    }

    #[test]
    fn chat_backends_do_not_claim_vla_safetensors() {
        let backend = SelectedBackendInfo {
            family: "llama.cpp".to_string(),
            model_artifact: "file".to_string(),
        };
        assert!(model_supported_by_backend(
            Path::new("chat.gguf"),
            Some(&backend)
        ));
        assert!(!model_supported_by_backend(
            Path::new("weights.safetensors"),
            Some(&backend)
        ));
    }

    #[test]
    fn remembered_model_reuse_requires_matching_request_defaults() {
        let model = local_state::SelectedModel {
            model: "/models/model.gguf".to_string(),
            mmproj: None,
            no_mmproj: false,
            ctx_size: Some(4096),
            request_defaults: serde_json::from_value(serde_json::json!({"max_tokens": 64}))
                .unwrap(),
        };
        assert!(state_matches_remembered_model(
            &serde_json::json!({
                "model_path": "/models/model.gguf",
                "request_defaults": {"max_tokens": 64}
            }),
            &model,
        ));
        assert!(!state_matches_remembered_model(
            &serde_json::json!({
                "model_path": "/models/model.gguf",
                "request_defaults": {"max_tokens": 128}
            }),
            &model,
        ));
    }

    #[test]
    fn model_picker_uses_explicit_backend_over_persisted_selection() {
        let backends = serde_json::json!({
            "data": [
                {
                    "id": "persisted-backend",
                    "family": "llama.cpp",
                    "model_artifact": "file",
                    "selected": true
                },
                {
                    "id": "chosen-backend",
                    "family": "vla.cpp",
                    "model_artifact": "vla-artifact",
                    "selected": false
                }
            ]
        });
        let selected = selected_backend_info(&backends, Some("chosen-backend"))
            .expect("chosen backend should be present");
        assert_eq!(selected.family, "vla.cpp");
        assert!(model_supported_by_backend(
            Path::new("model.safetensors"),
            Some(&selected)
        ));
        assert!(!model_supported_by_backend(
            Path::new("model.bin"),
            Some(&selected)
        ));
        assert_eq!(
            selected_backend_line(&backends, Some("chosen-backend")),
            "Backend: chosen-backend (not installed)"
        );
    }
}
