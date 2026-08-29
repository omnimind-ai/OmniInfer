use super::*;

pub(super) fn start_runtime_with_cold_start_policy(
    backend_id: &str,
    plan: &omniinfer_core::runtime_plan::ExternalRuntimePlan,
    options: RuntimeProcessOptions,
    startup_cancelled: &AtomicBool,
) -> Result<RuntimeProcess, RuntimeProcessError> {
    let Some(initial_timeout) =
        wsl_rocm_cold_start_retry_timeout(backend_id, options.startup_timeout)
    else {
        return RuntimeProcess::start_cancellable(plan, options, startup_cancelled);
    };

    let total_timeout = options.startup_timeout;
    retry_after_ready_timeout(
        total_timeout,
        initial_timeout,
        WSL_ROCM_COLD_START_RETRY_COOLDOWN,
        startup_cancelled,
        |attempt_timeout| {
            let mut attempt_options = options.clone();
            attempt_options.startup_timeout = attempt_timeout;
            RuntimeProcess::start_cancellable(plan, attempt_options, startup_cancelled)
        },
    )
}

pub(super) fn wsl_rocm_cold_start_retry_timeout(
    backend_id: &str,
    total_timeout: Duration,
) -> Option<Duration> {
    (backend_id == "vllm-wsl2-rocm" && total_timeout >= WSL_ROCM_COLD_START_RETRY_MINIMUM_BUDGET)
        .then_some(WSL_ROCM_COLD_START_INITIAL_ATTEMPT)
}

pub(super) fn retry_after_ready_timeout<T>(
    total_timeout: Duration,
    initial_timeout: Duration,
    cooldown: Duration,
    startup_cancelled: &AtomicBool,
    mut attempt: impl FnMut(Duration) -> Result<T, RuntimeProcessError>,
) -> Result<T, RuntimeProcessError> {
    let started = Instant::now();
    match attempt(initial_timeout) {
        Err(RuntimeProcessError::ReadyTimeout) => {
            let remaining_before_cooldown = total_timeout.saturating_sub(started.elapsed());
            if remaining_before_cooldown <= cooldown {
                return Err(RuntimeProcessError::ReadyTimeout);
            }
            eprintln!(
                "OmniInfer: WSL2 ROCm cold start did not become ready after {} seconds; cooling down for {} seconds before retry",
                initial_timeout.as_secs(),
                cooldown.as_secs()
            );
            let cooldown_deadline = Instant::now() + cooldown;
            while Instant::now() < cooldown_deadline {
                if startup_cancelled.load(Ordering::SeqCst) {
                    return Err(RuntimeProcessError::Interrupted);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let remaining = total_timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(RuntimeProcessError::ReadyTimeout);
            }
            eprintln!(
                "OmniInfer: retrying WSL2 ROCm cold start once with the remaining {} seconds",
                remaining.as_secs()
            );
            attempt(remaining)
        }
        result => result,
    }
}

pub(super) fn annotate_restore_state(
    payload: &mut Value,
    persistent_state: &local_state::LocalState,
    loaded_runtimes: &BTreeMap<String, LoadedRustRuntime>,
) {
    let Some(selected) = persistent_state.selected_model.as_ref() else {
        payload["restore_selection"] = Value::Null;
        payload["restore_status"] = json!("not_configured");
        payload["restore_completed"] = json!(false);
        return;
    };
    let completed = loaded_runtimes.values().any(|loaded| {
        loaded.route_state == RuntimeRouteState::Ready
            && persistent_state
                .selected_backend
                .as_deref()
                .is_none_or(|backend| loaded.backend_id == backend)
            && loaded.model == selected.model
            && loaded.mmproj == selected.mmproj
            && loaded.ctx_size == selected.ctx_size
            && loaded.request_defaults == selected.request_defaults
    });
    payload["restore_selection"] = json!({
        "backend": persistent_state.selected_backend,
        "model": selected.model,
        "mmproj": selected.mmproj,
        "no_mmproj": selected.no_mmproj,
        "ctx_size": selected.ctx_size,
        "request_defaults": selected.request_defaults,
    });
    payload["restore_status"] = json!(if completed { "loaded" } else { "pending" });
    payload["restore_completed"] = json!(completed);
}

pub(super) fn same_load_configuration(
    loaded: &LoadedRustRuntime,
    backend_id: &str,
    model_path: &str,
    mmproj: Option<&str>,
    ctx_size: Option<u32>,
    launch_args: &[String],
) -> bool {
    loaded.backend_id == backend_id
        && loaded.model == model_path
        && loaded.mmproj.as_deref() == mmproj
        && loaded.ctx_size == ctx_size
        && loaded.launch_args == launch_args
}

pub(super) fn model_load_response(loaded: &LoadedRustRuntime, already_loaded: bool) -> Value {
    let info = loaded.process.info();
    let mut response = json!({
        "ok": true,
        "already_loaded": already_loaded,
        "requires_reload": false,
        "model": loaded.model_key,
        "owner_admin_id": loaded.owner_admin_id,
        "selected_backend": loaded.backend_id,
        "selected_model": loaded.model,
        "selected_public_model_id": loaded.public_model_id,
        "selected_mmproj": loaded.mmproj,
        "selected_ctx_size": loaded.ctx_size,
        "request_defaults": loaded.request_defaults,
        "backend_pid": info.pid,
        "backend_port": info.port,
        "generation": loaded.generation,
        "route_state": loaded.route_state.as_str(),
        "allocation_id": loaded.allocation_id.get(),
        "resource_budget": resource_budget_payload(&loaded.resource_budget),
        "runtime_placement": runtime_placement_payload(loaded.runtime_placement.as_ref()),
        "speculative_admission": loaded.speculative_admission.as_ref().map(|admission| json!({
            "speculative": true,
            "device": admission.device,
            "estimated_cuda_bytes": admission.estimated,
            "exclusive_reservation_bytes": admission.exclusive,
            "shortfall_bytes": admission.shortfall,
            "waived_allocator_slack_bytes": admission.waived_allocator_slack,
        })),
        "launch_command": info.command,
        "log_path": info.log_path.display().to_string(),
        "external_server_protocol": loaded.external_server_protocol.as_str(),
        "client_endpoint": loaded.client_endpoint,
        "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
    });
    if let Some(visible_devices) = loaded.cuda_visible_devices.as_deref() {
        response["cuda_visible_devices"] = json!(visible_devices);
    }
    if let Some(warning) = loaded.cuda_warning.as_deref() {
        response["warning"] = json!(warning);
    }
    response
}

pub(super) struct RequestedRuntimeConfig<'a> {
    pub(super) backend_id: &'a str,
    pub(super) model_key: &'a str,
    pub(super) model_path: &'a str,
    pub(super) public_model_id: Option<&'a str>,
    pub(super) mmproj: Option<&'a str>,
    pub(super) ctx_size: Option<u32>,
    pub(super) request_defaults: &'a Map<String, Value>,
    pub(super) launch_args: &'a [String],
}

pub(super) fn reload_required_response(
    loaded: &LoadedRustRuntime,
    requested: &RequestedRuntimeConfig<'_>,
) -> Value {
    json!({
        "ok": false,
        "already_loaded": true,
        "requires_reload": true,
        "error": {
            "code": "model_reload_required",
            "message": format!(
                "model '{}' is already loaded with different runtime settings; unload it before selecting the new configuration",
                requested.model_key,
            ),
        },
        "current": {
            "backend": loaded.backend_id,
            "model": loaded.model_key,
            "model_path": loaded.model,
            "public_model_id": loaded.public_model_id,
            "mmproj": loaded.mmproj,
            "ctx_size": loaded.ctx_size,
            "request_defaults": loaded.request_defaults,
            "launch_args": loaded.launch_args,
        },
        "requested": {
            "backend": requested.backend_id,
            "model": requested.model_key,
            "model_path": requested.model_path,
            "public_model_id": requested.public_model_id,
            "mmproj": requested.mmproj,
            "ctx_size": requested.ctx_size,
            "request_defaults": requested.request_defaults,
            "launch_args": requested.launch_args,
        },
    })
}

pub(super) fn loaded_runtime_payload(loaded: &LoadedRustRuntime) -> Value {
    let info = loaded.process.info();
    json!({
        "id": loaded.model_key,
        "owner_admin_id": loaded.owner_admin_id,
        "backend": loaded.backend_id,
        "model": loaded.model_key,
        "model_path": loaded.model,
        "public_model_id": loaded.public_model_id,
        "mmproj": loaded.mmproj,
        "ctx_size": loaded.ctx_size,
        "request_defaults": loaded.request_defaults,
        "runtime_mode": "external_server",
        "backend_pid": info.pid,
        "backend_port": info.port,
        "generation": loaded.generation,
        "route_state": loaded.route_state.as_str(),
        "allocation_id": loaded.allocation_id.get(),
        "resource_budget": resource_budget_payload(&loaded.resource_budget),
        "runtime_placement": runtime_placement_payload(loaded.runtime_placement.as_ref()),
        "speculative_admission": loaded.speculative_admission.as_ref().map(|admission| json!({
            "speculative": true,
            "device": admission.device,
            "estimated_cuda_bytes": admission.estimated,
            "exclusive_reservation_bytes": admission.exclusive,
            "shortfall_bytes": admission.shortfall,
            "waived_allocator_slack_bytes": admission.waived_allocator_slack,
        })),
        "launch_args": loaded.launch_args,
        "cuda_visible_devices": loaded.cuda_visible_devices,
        "warning": loaded.cuda_warning,
        "launch_command": info.command,
        "proxy_model": loaded.proxy_model_ref,
        "external_server_protocol": loaded.external_server_protocol.as_str(),
        "client_endpoint": loaded.client_endpoint,
        "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
        "backend_log": info.log_path.display().to_string(),
    })
}

pub(super) fn runtime_placement_payload(placement: Option<&RuntimePlacement>) -> Value {
    placement.map_or(Value::Null, |placement| {
        json!({
            "source": "llama.cpp_startup_log",
            "policy": placement.policy.as_str(),
            "requested_gpu_layers": placement.policy.requested_gpu_layers(),
            "mode": placement.mode,
            "offloaded_layers": placement.offloaded_layers,
            "total_layers": placement.total_layers,
            "reported_buffer_bytes": domain_bytes_payload(&placement.reported_bytes),
            "reconciled_budget": resource_budget_payload(&placement.reconciled_budget),
        })
    })
}
