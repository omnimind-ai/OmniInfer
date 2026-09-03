use std::collections::BTreeMap;
use std::fs;
use std::net::{IpAddr, TcpListener};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use omniinfer_core::backend_args::parse_backend_load_extra_args;
use omniinfer_core::backend_registry::{self, BackendRegistry, BackendScope};
use omniinfer_core::local_state;
use omniinfer_core::model_artifacts::{discover_llama_cpp_model_artifacts, maybe_auto_mmproj};
use omniinfer_core::model_load::DEFAULT_LOAD_CONTEXT_SIZE;
use omniinfer_core::resource_ledger::{
    AllocationId, BudgetComponent, MemoryDomain, ReservationId, ResourceBudget, ResourceCapacity,
    ResourceLedger, ResourceLedgerError,
};
use omniinfer_core::runtime_plan::{
    ExternalRuntimeRequest, ExternalServerProtocol, build_external_runtime_plan,
};
use omniinfer_core::runtime_process::{RuntimeProcess, RuntimeProcessError, RuntimeProcessOptions};
use serde_json::{Map, Value, json};
use url::Url;

use super::gpu_status::runtime_env_for_backend;

const WSL_ROCM_COLD_START_RETRY_MINIMUM_BUDGET: Duration = Duration::from_secs(360);
const WSL_ROCM_COLD_START_INITIAL_ATTEMPT: Duration = Duration::from_secs(120);
const WSL_ROCM_COLD_START_RETRY_COOLDOWN: Duration = Duration::from_secs(90);
const SPECULATIVE_ALLOCATOR_SLACK_LIMIT: u64 = 1024 * 1024 * 1024;
const ATTACHED_RUNTIME_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const ATTACHED_RUNTIME_PROBE_INTERVAL: Duration = Duration::from_secs(1);

pub(super) struct RustRuntimeManager {
    selected_backend: Option<String>,
    loaded: BTreeMap<String, LoadedRustRuntime>,
    attached: BTreeMap<String, AttachedExternalRuntime>,
    speculative_domains: BTreeMap<MemoryDomain, AllocationId>,
    default_model_key: Option<String>,
    resource_ledger: Option<ResourceLedger>,
    next_capacity_snapshot: u64,
    next_generation: u64,
}

impl Default for RustRuntimeManager {
    fn default() -> Self {
        Self {
            selected_backend: None,
            loaded: BTreeMap::new(),
            attached: BTreeMap::new(),
            speculative_domains: BTreeMap::new(),
            default_model_key: None,
            resource_ledger: None,
            next_capacity_snapshot: 1,
            next_generation: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeProxyTarget {
    pub(super) base_url: Option<String>,
    pub(super) client_endpoint: String,
    pub(super) protocol: ExternalServerProtocol,
    pub(super) backend_id: String,
    pub(super) model: Option<String>,
    pub(super) request_defaults: Map<String, Value>,
    pub(super) generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeRouteState {
    Ready,
    Draining,
    Failed,
}

impl RuntimeRouteState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Failed => "failed",
        }
    }
}

struct LoadedRustRuntime {
    model_key: String,
    owner_admin_id: Option<String>,
    backend_id: String,
    model: String,
    public_model_id: Option<String>,
    mmproj: Option<String>,
    ctx_size: Option<u32>,
    request_defaults: Map<String, Value>,
    launch_args: Vec<String>,
    cuda_visible_devices: Option<String>,
    cuda_warning: Option<String>,
    speculative_admission: Option<SpeculativeAdmission>,
    runtime_placement: Option<RuntimePlacement>,
    external_server_protocol: ExternalServerProtocol,
    client_endpoint: String,
    process: RuntimeProcess,
    proxy_model_ref: Option<String>,
    generation: u64,
    route_state: RuntimeRouteState,
    allocation_id: AllocationId,
    resource_budget: ResourceBudget,
}

struct AttachedExternalRuntime {
    model_key: String,
    owner_admin_id: Option<String>,
    backend_id: String,
    request_defaults: Map<String, Value>,
    external_server_protocol: ExternalServerProtocol,
    client_endpoint: String,
    backend_port: u16,
    upstream_models: Vec<String>,
    generation: u64,
    route_state: RuntimeRouteState,
    last_error: Option<String>,
    monitor: AttachedRuntimeMonitor,
}

struct AttachedRuntimeMonitor {
    failure: Arc<Mutex<Option<String>>>,
    stop_tx: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl AttachedRuntimeMonitor {
    fn start(endpoint: String, expected_models: Vec<String>) -> std::io::Result<Self> {
        let failure = Arc::new(Mutex::new(None));
        let monitor_failure = Arc::clone(&failure);
        let (stop_tx, stop_rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("omniinfer-external-runtime-monitor".to_string())
            .spawn(move || {
                loop {
                    match stop_rx.recv_timeout(ATTACHED_RUNTIME_PROBE_INTERVAL) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    let error = match probe_attached_runtime(&endpoint) {
                        Ok(models) if models == expected_models => continue,
                        Ok(models) => format!(
                            "external runtime identity changed at {endpoint}: expected models [{}], observed [{}]; route withdrawn, attach again after verifying the server",
                            expected_models.join(", "),
                            models.join(", "),
                        ),
                        Err(error) => format!(
                            "external runtime became unavailable at {endpoint}: {error}; route withdrawn, restart the server and attach it again"
                        ),
                    };
                    *monitor_failure
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                    break;
                }
            })?;
        Ok(Self {
            failure,
            stop_tx: Some(stop_tx),
            thread: Some(thread),
        })
    }

    fn failure(&self) -> Option<String> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl Drop for AttachedRuntimeMonitor {
    fn drop(&mut self) {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttachRuntimeErrorKind {
    BadRequest,
    Conflict,
    Unavailable,
}

#[derive(Debug)]
pub(super) struct AttachRuntimeError {
    pub(super) kind: AttachRuntimeErrorKind,
    pub(super) message: String,
}

impl AttachRuntimeError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            kind: AttachRuntimeErrorKind::BadRequest,
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            kind: AttachRuntimeErrorKind::Conflict,
            message: message.into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            kind: AttachRuntimeErrorKind::Unavailable,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpeculativeAdmission {
    device: String,
    estimated: u64,
    exclusive: u64,
    shortfall: u64,
    waived_allocator_slack: u64,
}

fn speculative_admission_payload(admission: Option<&SpeculativeAdmission>) -> Value {
    admission.map_or(Value::Null, |admission| {
        json!({
            "speculative": true,
            "device": admission.device,
            "estimated_cuda_bytes": admission.estimated,
            "exclusive_reservation_bytes": admission.exclusive,
            "shortfall_bytes": admission.shortfall,
            "waived_allocator_slack_bytes": admission.waived_allocator_slack,
        })
    })
}

#[derive(Debug, Clone)]
pub(super) struct LoadedRuntimeSummary {
    pub(super) id: String,
    pub(super) owner_admin_id: Option<String>,
    pub(super) backend_pid: u32,
}

pub(super) enum LoadModelOutcome {
    Success(Value),
    ReloadRequired(Value),
}

impl RustRuntimeManager {
    pub(super) fn select_backend(&mut self, backend_id: &str) -> Result<Value> {
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(backend_id)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {backend_id}"))?;
        if self.selected_backend.as_deref() != Some(backend_id) {
            self.stop_runtime()?;
        }
        self.selected_backend = Some(backend_id.to_string());
        local_state::save_selected_backend(backend_id)?;
        Ok(json!({
            "ok": true,
            "selected_backend": backend_id,
            "binary_exists": backend.binary_exists(),
            "models_dir": backend.models_dir,
        }))
    }

    pub(super) fn stop_runtime(&mut self) -> Result<Value> {
        let keys = self.loaded.keys().cloned().collect::<Vec<_>>();
        let mut failures = Vec::new();
        for key in keys {
            let stop_result = {
                let loaded = self
                    .loaded
                    .get_mut(&key)
                    .expect("runtime key came from the loaded map");
                loaded.route_state = RuntimeRouteState::Draining;
                loaded.process.stop(Duration::from_secs(8))
            };
            match stop_result {
                Ok(()) => self.remove_runtime_and_release(&key),
                Err(error) => {
                    if let Some(loaded) = self.loaded.get_mut(&key) {
                        loaded.route_state = RuntimeRouteState::Failed;
                    }
                    failures.push(format!("{key}: {error}"));
                }
            }
        }
        let detached_external_runtimes = self.attached.len();
        self.attached.clear();
        self.select_fallback_default();
        if !failures.is_empty() {
            anyhow::bail!("failed to stop runtimes: {}", failures.join("; "));
        }
        let selected_model_preserved = local_state::load_state()
            .ok()
            .is_some_and(|state| state.selected_model.is_some());
        Ok(json!({
            "ok": true,
            "stopped": true,
            "detached_external_runtimes": detached_external_runtimes,
            "selected_backend": self.selected_backend,
            "selected_model_preserved": selected_model_preserved,
            "restore_status": if selected_model_preserved { "pending" } else { "not_configured" },
        }))
    }

    pub(super) fn has_loaded_runtime(&mut self) -> bool {
        self.reap_exited_runtimes();
        self.loaded
            .values()
            .any(|loaded| loaded.route_state == RuntimeRouteState::Ready)
            || self
                .attached
                .values()
                .any(|attached| attached.route_state == RuntimeRouteState::Ready)
    }

    pub(super) fn load_model(
        &mut self,
        payload: Value,
        backend_host: String,
        startup_timeout: Duration,
        owner_admin_id: Option<String>,
        startup_cancelled: &AtomicBool,
    ) -> Result<LoadModelOutcome> {
        if startup_cancelled.load(Ordering::SeqCst) {
            anyhow::bail!("gateway is shutting down")
        }
        self.reap_exited_runtimes();
        let model = json_required_str(&payload, "model")?.to_string();
        let requested_request_defaults = request_defaults_from_payload(&payload)?;
        let public_model_id = payload
            .get("public_model_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);
        let requested_model_key = public_model_id.clone().unwrap_or_else(|| model.clone());
        if self.attached.contains_key(&requested_model_key) {
            anyhow::bail!(
                "model '{requested_model_key}' is registered as an external attachment; detach it before starting a managed runtime"
            );
        }
        let requested_backend = self.resolve_requested_backend(&payload)?;
        let no_mmproj = no_mmproj_from_payload(&payload)?;
        let registry = BackendRegistry::load_current();
        let backend = registry
            .get(&requested_backend)
            .ok_or_else(|| anyhow::anyhow!("unsupported backend: {requested_backend}"))?;
        if backend.runtime_mode != "external_server" {
            anyhow::bail!(
                "{} is an embedded backend. Python control-plane fallback has been removed; use an external-server backend or a backend adapter service.",
                backend.id
            );
        }
        if !backend.binary_exists() {
            anyhow::bail!(
                "backend launcher not found: {}",
                backend.launcher_path.as_deref().unwrap_or("(unset)")
            );
        }
        let resolved_model = resolve_model_for_backend(&model, backend)?;
        if no_mmproj && payload.get("mmproj").is_some_and(|value| !value.is_null()) {
            anyhow::bail!("no_mmproj cannot be combined with mmproj");
        }
        let explicit_mmproj = payload
            .get("mmproj")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(|value| resolve_path_for_backend(value, backend, "mmproj file"))
            .transpose()?;
        let mmproj_path = select_mmproj_path(
            no_mmproj,
            explicit_mmproj,
            resolved_model.mmproj_path,
            maybe_auto_mmproj(backend.models_dir.as_deref(), &resolved_model.model_path),
        );
        if mmproj_path.is_some() && !backend.supports_mmproj {
            anyhow::bail!("{} does not support mmproj inputs", backend.id);
        }
        let requested_ctx_size = payload
            .get("ctx_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        let launch_args = payload
            .get("launch_args")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            });
        let effective_launch_args = merged_launch_args(
            &backend.id,
            &backend.family,
            &backend.default_args,
            launch_args.as_deref(),
        );
        let placement_policy = llama_cpp_cuda_placement_policy(backend, &effective_launch_args)?;
        let effective_launch_args =
            managed_placement_evidence_args(&effective_launch_args, placement_policy)?;
        let launch_args_have_ctx =
            launch_args_have_ctx_size(&backend.family, &effective_launch_args);
        let launch_args_ctx_size =
            parse_backend_load_extra_args(&backend.id, &backend.family, &effective_launch_args)
                .ok()
                .and_then(|parsed| parsed.ctx_size);
        let ctx_size = requested_ctx_size.or(launch_args_ctx_size).or_else(|| {
            (backend.supports_ctx_size && !launch_args_have_ctx)
                .then_some(DEFAULT_LOAD_CONTEXT_SIZE)
        });
        if let Some(loaded_key) = self.matching_loaded_model_key(
            &requested_model_key,
            &resolved_model.model_path,
            public_model_id.as_deref(),
        ) {
            let loaded = self
                .loaded
                .get(&loaded_key)
                .expect("matched runtime should remain registered");
            if same_load_configuration(
                loaded,
                &backend.id,
                &resolved_model.model_path,
                mmproj_path.as_deref(),
                ctx_size,
                &effective_launch_args,
            ) {
                local_state::save_selected_backend(&backend.id)?;
                local_state::save_selected_model_with_no_mmproj(
                    &resolved_model.model_path,
                    mmproj_path.as_deref(),
                    no_mmproj,
                    ctx_size,
                    &requested_request_defaults,
                )?;
                let loaded_key = self.promote_loaded_model_key(
                    &loaded_key,
                    &requested_model_key,
                    public_model_id.as_deref(),
                );
                let loaded = self
                    .loaded
                    .get_mut(&loaded_key)
                    .expect("promoted runtime should remain registered");
                loaded.request_defaults = requested_request_defaults.clone();
                let response = model_load_response(loaded, true);
                self.default_model_key = Some(loaded_key);
                return Ok(LoadModelOutcome::Success(response));
            }
            let requested = RequestedRuntimeConfig {
                backend_id: &backend.id,
                model_key: &requested_model_key,
                model_path: &resolved_model.model_path,
                public_model_id: public_model_id.as_deref(),
                mmproj: mmproj_path.as_deref(),
                ctx_size,
                request_defaults: &requested_request_defaults,
                launch_args: &effective_launch_args,
            };
            return Ok(LoadModelOutcome::ReloadRequired(reload_required_response(
                loaded, &requested,
            )));
        }
        let requested_port = payload
            .get("backend_port")
            .and_then(Value::as_u64)
            .filter(|value| (1..=u64::from(u16::MAX)).contains(value))
            .and_then(|value| u16::try_from(value).ok());
        let port = requested_port
            .map(Ok)
            .unwrap_or_else(|| pick_runtime_port(&backend_host))?;
        if requested_port.is_some() {
            ensure_runtime_port_available(&backend_host, port)?;
        }
        let backend_payload = serde_json::to_value(backend)?;
        let plan = build_external_runtime_plan(&ExternalRuntimeRequest {
            backend: backend_payload,
            model_path: resolved_model.model_path.clone(),
            mmproj_path: mmproj_path.clone(),
            host: backend_host.clone(),
            port,
            ctx_size,
            launch_args: Some(effective_launch_args.clone()),
        })?;
        let log_path = PathBuf::from(&backend.runtime_dir)
            .join("logs")
            .join(model_log_file_name(
                &plan.log_file_name,
                &requested_model_key,
            ));
        let (runtime_env, cuda_selection) =
            runtime_env_for_backend(backend, &effective_launch_args);
        let budget_cuda_devices = if backend.capabilities.iter().any(|value| value == "cuda") {
            match cuda_selection.as_ref() {
                Some(selection) => Some(selection.visible_devices.clone()),
                None => Some(detect_cuda_device_ids()?.join(",")),
            }
        } else {
            None
        };
        let replicate_across_domains =
            cuda_selection.is_none() && budget_cuda_devices.is_some() && placement_policy.is_none();
        let resource_budget = build_runtime_resource_budget(
            &payload,
            backend,
            &resolved_model.model_path,
            mmproj_path.as_deref(),
            plan.ctx_size.unwrap_or(DEFAULT_LOAD_CONTEXT_SIZE),
            &effective_launch_args,
            budget_cuda_devices.as_deref(),
            replicate_across_domains,
        )?;
        let budget_vulkan_devices = resource_budget
            .domains()
            .keys()
            .filter_map(|domain| match domain {
                MemoryDomain::Vulkan(device) => Some(device.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let reconcile_policy = placement_policy;
        let selected_cuda_devices = resource_budget
            .domains()
            .keys()
            .filter(|domain| matches!(domain, MemoryDomain::Cuda(_)))
            .count();
        let use_provisional_reservation = reconcile_policy
            .is_some_and(|policy| policy.permits_partial_offload() || selected_cuda_devices > 1);
        let initial_reservation = if use_provisional_reservation {
            self.reserve_llama_cpp_placement_resources(
                &requested_model_key,
                &resource_budget,
                budget_cuda_devices.as_deref(),
                &budget_vulkan_devices,
            )
        } else {
            self.reserve_runtime_resources(
                &requested_model_key,
                &resource_budget,
                budget_cuda_devices.as_deref(),
                &budget_vulkan_devices,
            )
        };
        let (reservation_id, speculative) = match initial_reservation {
            Ok(reservation_id) => (reservation_id, None),
            Err(error) if !use_provisional_reservation && is_cuda_capacity_exhaustion(&error) => {
                let decision = speculative_reservation(
                    backend,
                    &payload,
                    &resource_budget,
                    replicate_across_domains,
                    self.resource_ledger
                        .as_ref()
                        .map(|ledger| ledger.snapshot()),
                )?;
                let Some(decision) = decision else {
                    return Err(error);
                };
                let reservation_id = self
                    .resource_ledger
                    .as_mut()
                    .expect("speculative admission requires a resource ledger")
                    .reserve(&requested_model_key, decision.budget.clone())?;
                (reservation_id, Some(decision))
            }
            Err(error) => return Err(error),
        };
        if let Some(speculative) = speculative.as_ref() {
            eprintln!(
                "warning: speculative llama.cpp CUDA admission backend={} device={} estimated={} available={} shortfall={} waived_allocator_slack={} exclusive_reservation={}",
                backend.id,
                speculative.device,
                speculative.estimated,
                speculative.available,
                speculative.shortfall,
                speculative.waived_slack,
                speculative.available,
            );
        }
        let log_start_offset = fs::metadata(&log_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let transaction = self.with_reservation(reservation_id, |manager| {
            if requested_port.is_some() {
                ensure_runtime_port_available(&backend_host, port)?;
            }
            let mut process = start_runtime_with_cold_start_policy(
                &backend.id,
                &plan,
                RuntimeProcessOptions {
                    log_path: log_path.clone(),
                    env: runtime_env,
                    startup_timeout,
                    health_host: backend_host.clone(),
                },
                startup_cancelled,
            )?;
            let runtime_placement = if let Some(policy) = reconcile_policy {
                let placement = parse_llama_cpp_runtime_placement(
                    &log_path,
                    log_start_offset,
                    budget_cuda_devices
                        .as_deref()
                        .expect("CUDA reconciliation requires selected devices"),
                    policy,
                );
                let placement = match placement {
                    Ok(placement) => placement,
                    Err(error) => {
                        let cleanup = process.stop(Duration::from_secs(8));
                        return Err(match cleanup {
                            Ok(()) => error.context(format!(
                                "failed to reconcile llama.cpp placement (log: {})",
                                log_path.display()
                            )),
                            Err(cleanup) => anyhow::anyhow!(
                                "failed to reconcile llama.cpp placement: {error}; runtime cleanup failed: {cleanup}; log: {}",
                                log_path.display()
                            ),
                        });
                    }
                };
                if let Err(error) = manager
                    .resource_ledger
                    .as_mut()
                    .expect("reservation requires a resource ledger")
                    .reconcile_reservation(
                        reservation_id,
                        placement.reconciled_budget.clone(),
                    )
                {
                    let cleanup = process.stop(Duration::from_secs(8));
                    return Err(match cleanup {
                        Ok(()) => anyhow::Error::new(error).context(format!(
                            "llama.cpp placement exceeds safe reconciled capacity (log: {})",
                            log_path.display()
                        )),
                        Err(cleanup) => anyhow::anyhow!(
                            "llama.cpp placement exceeds safe reconciled capacity: {error}; runtime cleanup failed: {cleanup}; log: {}",
                            log_path.display()
                        ),
                    });
                }
                Some(placement)
            } else {
                None
            };
            local_state::save_selected_backend(&backend.id)?;
            local_state::save_selected_model_with_no_mmproj(
                &resolved_model.model_path,
                mmproj_path.as_deref(),
                no_mmproj,
                plan.ctx_size,
                &requested_request_defaults,
            )?;
            let generation = manager.take_generation()?;
            let allocation_id = manager
                .resource_ledger
                .as_mut()
                .expect("reservation requires a resource ledger")
                .commit(reservation_id)?;
            Ok((process, generation, allocation_id, runtime_placement))
        })?;
        let (process, generation, allocation_id, runtime_placement) = transaction;
        let committed_budget = runtime_placement
            .as_ref()
            .map(|placement| placement.reconciled_budget.clone())
            .unwrap_or_else(|| resource_budget.clone());
        if let Some(speculative) = speculative.as_ref() {
            self.speculative_domains.insert(
                MemoryDomain::Cuda(speculative.device.clone()),
                allocation_id,
            );
        }
        self.selected_backend = Some(backend.id.clone());
        self.loaded.insert(
            requested_model_key.clone(),
            LoadedRustRuntime {
                model_key: requested_model_key.clone(),
                owner_admin_id: owner_admin_id.clone(),
                backend_id: backend.id.clone(),
                model: resolved_model.model_path.clone(),
                public_model_id: public_model_id.clone(),
                mmproj: mmproj_path.clone(),
                ctx_size: plan.ctx_size,
                request_defaults: requested_request_defaults,
                launch_args: effective_launch_args,
                cuda_visible_devices: cuda_selection
                    .as_ref()
                    .map(|selection| selection.visible_devices.clone()),
                cuda_warning: cuda_selection
                    .as_ref()
                    .and_then(|selection| selection.warning.clone()),
                speculative_admission: speculative.map(|value| SpeculativeAdmission {
                    device: value.device,
                    estimated: value.estimated,
                    exclusive: value.available,
                    shortfall: value.shortfall,
                    waived_allocator_slack: value.waived_slack,
                }),
                runtime_placement,
                external_server_protocol: plan.protocol,
                client_endpoint: plan.client_endpoint.clone(),
                proxy_model_ref: plan.proxy_model_ref.clone(),
                process,
                generation,
                route_state: RuntimeRouteState::Ready,
                allocation_id,
                resource_budget: committed_budget,
            },
        );
        self.default_model_key = Some(requested_model_key.clone());
        let loaded = self
            .loaded
            .get(&requested_model_key)
            .expect("newly loaded runtime should be registered");
        Ok(LoadModelOutcome::Success(model_load_response(
            loaded, false,
        )))
    }

    pub(super) fn attach_runtime(
        &mut self,
        payload: Value,
        owner_admin_id: Option<String>,
        gateway_port: u16,
    ) -> std::result::Result<Value, AttachRuntimeError> {
        self.reap_exited_runtimes();
        let model = required_attach_string(&payload, "model")?;
        let backend_id = required_attach_string(&payload, "backend")?;
        let protocol_text = required_attach_string(&payload, "external_server_protocol")?;
        let protocol = ExternalServerProtocol::parse(&protocol_text).ok_or_else(|| {
            AttachRuntimeError::bad_request(format!(
                "unsupported external_server_protocol: {protocol_text}"
            ))
        })?;
        if protocol != ExternalServerProtocol::LlamaCppServer {
            return Err(AttachRuntimeError::bad_request(
                "external attachment currently supports only llama.cpp-server",
            ));
        }
        let registry = BackendRegistry::load_current();
        let backend = registry.get(&backend_id).ok_or_else(|| {
            AttachRuntimeError::bad_request(format!("unsupported backend: {backend_id}"))
        })?;
        if backend.runtime_mode != "external_server" {
            return Err(AttachRuntimeError::bad_request(format!(
                "{} is not an external-server backend",
                backend.id
            )));
        }
        if backend.external_server_protocol.as_deref() != Some(protocol.as_str()) {
            return Err(AttachRuntimeError::bad_request(format!(
                "backend {} uses protocol {}, not {}",
                backend.id,
                backend
                    .external_server_protocol
                    .as_deref()
                    .unwrap_or("unknown"),
                protocol.as_str(),
            )));
        }
        let request_defaults = match payload.get("request_defaults") {
            None => Map::new(),
            Some(Value::Object(defaults)) => defaults.clone(),
            Some(_) => {
                return Err(AttachRuntimeError::bad_request(
                    "request_defaults must be an object",
                ));
            }
        };
        let endpoint_text = required_attach_string(&payload, "client_endpoint")?;
        let (client_endpoint, backend_port) =
            normalize_attach_endpoint(&endpoint_text, gateway_port)?;

        if let Some(loaded) = self.loaded.get(&model) {
            return Err(AttachRuntimeError::conflict(format!(
                "model '{}' is already managed by OmniInfer at {}",
                model, loaded.client_endpoint
            )));
        }
        if let Some((key, _)) = self
            .loaded
            .iter()
            .find(|(_, loaded)| loaded.client_endpoint == client_endpoint)
        {
            return Err(AttachRuntimeError::conflict(format!(
                "client_endpoint {client_endpoint} is already owned by managed model '{key}'"
            )));
        }
        if let Some((key, _)) = self
            .attached
            .iter()
            .find(|(key, attached)| **key != model && attached.client_endpoint == client_endpoint)
        {
            return Err(AttachRuntimeError::conflict(format!(
                "client_endpoint {client_endpoint} is already attached as model '{key}'"
            )));
        }

        let upstream_models = probe_attached_runtime(&client_endpoint).map_err(|message| {
            AttachRuntimeError::unavailable(format!(
                "external runtime at {client_endpoint} is unavailable: {message}"
            ))
        })?;
        if !upstream_models.iter().any(|candidate| candidate == &model) {
            return Err(AttachRuntimeError::conflict(format!(
                "external runtime model mismatch: requested '{model}', upstream exposes [{}]",
                upstream_models.join(", ")
            )));
        }

        if let Some(attached) = self.attached.get_mut(&model) {
            if attached.owner_admin_id.as_deref() != owner_admin_id.as_deref()
                && attached.owner_admin_id.is_some()
            {
                return Err(AttachRuntimeError::conflict(format!(
                    "model '{}' is owned by another administrator",
                    model
                )));
            }
            if attached.route_state == RuntimeRouteState::Ready
                && attached.backend_id == backend_id
                && attached.external_server_protocol == protocol
                && attached.client_endpoint == client_endpoint
                && attached.upstream_models == upstream_models
            {
                attached.request_defaults = request_defaults;
                attached.owner_admin_id = owner_admin_id;
                attached.route_state = RuntimeRouteState::Ready;
                attached.last_error = None;
                self.default_model_key = Some(model.clone());
                return Ok(attached_runtime_response(attached, true));
            }
            if attached.route_state == RuntimeRouteState::Ready {
                return Err(AttachRuntimeError::conflict(format!(
                    "model '{}' is already attached with different settings; detach it first",
                    model
                )));
            }
        }

        self.attached.remove(&model);
        let monitor =
            AttachedRuntimeMonitor::start(client_endpoint.clone(), upstream_models.clone())
                .map_err(|error| {
                    AttachRuntimeError::unavailable(format!(
                        "failed to start external runtime health monitor: {error}"
                    ))
                })?;
        let generation = self
            .take_generation()
            .map_err(|error| AttachRuntimeError::bad_request(error.to_string()))?;
        self.attached.insert(
            model.clone(),
            AttachedExternalRuntime {
                model_key: model.clone(),
                owner_admin_id,
                backend_id,
                request_defaults,
                external_server_protocol: protocol,
                client_endpoint,
                backend_port,
                upstream_models,
                generation,
                route_state: RuntimeRouteState::Ready,
                last_error: None,
                monitor,
            },
        );
        self.default_model_key = Some(model.clone());
        Ok(attached_runtime_response(
            self.attached
                .get(&model)
                .expect("new external attachment must be registered"),
            false,
        ))
    }

    pub(super) fn detach_runtime(
        &mut self,
        requested_model: Option<&str>,
        admin_id: Option<&str>,
    ) -> std::result::Result<Value, AttachRuntimeError> {
        self.reap_exited_runtimes();
        let key = match requested_model
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(model) => self.resolve_attached_model_key(model).ok_or_else(|| {
                AttachRuntimeError::bad_request(format!(
                    "external attachment is not registered: {model}"
                ))
            })?,
            None => self
                .default_model_key
                .as_ref()
                .filter(|key| self.attached.contains_key(*key))
                .cloned()
                .or_else(|| {
                    (self.attached.len() == 1)
                        .then(|| self.attached.keys().next().cloned())
                        .flatten()
                })
                .ok_or_else(|| {
                    AttachRuntimeError::bad_request(
                        "field 'model' is required when no external attachment is the default",
                    )
                })?,
        };
        let attached = self
            .attached
            .get(&key)
            .expect("resolved external attachment must exist");
        if let Some(owner) = attached.owner_admin_id.as_deref()
            && let Some(admin_id) = admin_id
            && owner != admin_id
        {
            return Err(AttachRuntimeError::conflict(format!(
                "model '{key}' is owned by admin '{owner}' and cannot be detached by admin '{admin_id}'"
            )));
        }
        let attached = self
            .attached
            .remove(&key)
            .expect("resolved external attachment must be removable");
        self.select_fallback_default();
        Ok(json!({
            "ok": true,
            "detached": true,
            "model": attached.model_key,
            "client_endpoint": attached.client_endpoint,
            "runtime_ownership": "external",
            "process_left_running": true,
            "invalidated_generation": attached.generation,
        }))
    }

    pub(super) fn unload_model(&mut self, model: &str, admin_id: Option<&str>) -> Result<Value> {
        if self.resolve_attached_model_key(model).is_some() {
            return self
                .detach_runtime(Some(model), admin_id)
                .map_err(|error| anyhow::anyhow!(error.message));
        }
        let model_key = self
            .resolve_loaded_model_key(model)
            .ok_or_else(|| anyhow::anyhow!("model is not loaded: {model}"))?;
        let owner = self
            .loaded
            .get(&model_key)
            .and_then(|runtime| runtime.owner_admin_id.as_deref())
            .map(str::to_string);
        if let Some(owner) = owner.as_deref()
            && let Some(admin_id) = admin_id
            && owner != admin_id
        {
            anyhow::bail!(
                "model '{model_key}' is owned by admin '{owner}' and cannot be unloaded by admin '{admin_id}'"
            );
        }
        let (generation, stop_result) = {
            let loaded = self
                .loaded
                .get_mut(&model_key)
                .expect("resolved runtime key must exist");
            loaded.route_state = RuntimeRouteState::Draining;
            (
                loaded.generation,
                loaded.process.stop(Duration::from_secs(8)),
            )
        };
        if let Err(error) = stop_result {
            if let Some(loaded) = self.loaded.get_mut(&model_key) {
                loaded.route_state = RuntimeRouteState::Failed;
            }
            return Err(error.into());
        }
        self.remove_runtime_and_release(&model_key);
        self.select_fallback_default();
        Ok(json!({
            "ok": true,
            "unloaded": true,
            "model": model_key,
            "owner_admin_id": owner,
            "invalidated_generation": generation,
            "resources_released": true,
        }))
    }

    pub(super) fn resolve_requested_backend(&self, payload: &Value) -> Result<String> {
        payload
            .get("backend")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .or_else(|| self.selected_backend.clone())
            .or_else(|| {
                BackendRegistry::load_current()
                    .api_payload(BackendScope::Installed)
                    .get("recommended")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| anyhow::anyhow!("no installed backend available"))
    }

    pub(super) fn proxy_base_for_model(&mut self, requested_model: Option<&str>) -> Option<String> {
        self.proxy_target_for_model(requested_model)
            .and_then(|target| target.base_url)
    }

    pub(super) fn proxy_target_for_model(
        &mut self,
        requested_model: Option<&str>,
    ) -> Option<RuntimeProxyTarget> {
        self.reap_exited_runtimes();
        let key = self.resolve_proxy_model_key(requested_model)?;
        if let Some(loaded) = self.loaded.get(&key) {
            if loaded.route_state != RuntimeRouteState::Ready {
                return None;
            }
            return Some(RuntimeProxyTarget {
                base_url: loaded
                    .external_server_protocol
                    .is_http_transport()
                    .then(|| loaded.client_endpoint.clone()),
                client_endpoint: loaded.client_endpoint.clone(),
                protocol: loaded.external_server_protocol,
                backend_id: loaded.backend_id.clone(),
                model: loaded.proxy_model_ref.clone(),
                request_defaults: loaded.request_defaults.clone(),
                generation: loaded.generation,
            });
        }
        let attached = self.attached.get(&key)?;
        (attached.route_state == RuntimeRouteState::Ready).then(|| RuntimeProxyTarget {
            base_url: attached
                .external_server_protocol
                .is_http_transport()
                .then(|| attached.client_endpoint.clone()),
            client_endpoint: attached.client_endpoint.clone(),
            protocol: attached.external_server_protocol,
            backend_id: attached.backend_id.clone(),
            model: Some(attached.model_key.clone()),
            request_defaults: attached.request_defaults.clone(),
            generation: attached.generation,
        })
    }

    pub(super) fn unavailable_route_message(
        &self,
        requested_model: Option<&str>,
    ) -> Option<String> {
        let attached = match requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some("omniinfer" | "local") | None => self
                .attached
                .values()
                .filter(|attached| attached.route_state == RuntimeRouteState::Failed)
                .next_back(),
            Some(model) => self
                .resolve_attached_model_key(model)
                .and_then(|key| self.attached.get(&key))
                .filter(|attached| attached.route_state == RuntimeRouteState::Failed),
        };
        attached.and_then(|attached| attached.last_error.clone())
    }

    fn resolve_proxy_model_key(&self, requested_model: Option<&str>) -> Option<String> {
        match requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some("omniinfer" | "local") => self.default_model_key.clone(),
            Some(model) => self.resolve_loaded_model_key(model),
            None => self.default_model_key.clone(),
        }
    }

    fn resolve_loaded_model_key(&self, requested: &str) -> Option<String> {
        let requested = requested.trim();
        if requested.is_empty() {
            return None;
        }
        if self.loaded.contains_key(requested) {
            return Some(requested.to_string());
        }
        if self.attached.contains_key(requested) {
            return Some(requested.to_string());
        }
        self.loaded.iter().find_map(|(key, loaded)| {
            (loaded.public_model_id.as_deref() == Some(requested)
                || loaded.model == requested
                || loaded.proxy_model_ref.as_deref() == Some(requested))
            .then(|| key.clone())
        })
    }

    fn resolve_attached_model_key(&self, requested: &str) -> Option<String> {
        let requested = requested.trim();
        self.attached
            .contains_key(requested)
            .then(|| requested.to_string())
    }

    fn matching_loaded_model_key(
        &self,
        requested_key: &str,
        model_path: &str,
        public_model_id: Option<&str>,
    ) -> Option<String> {
        if self.loaded.contains_key(requested_key) {
            return Some(requested_key.to_string());
        }
        self.loaded.iter().find_map(|(key, loaded)| {
            let compatible_public_id = loaded.public_model_id.is_none()
                || public_model_id.is_none()
                || loaded.public_model_id.as_deref() == public_model_id;
            (loaded.model == model_path && compatible_public_id).then(|| key.clone())
        })
    }

    fn promote_loaded_model_key(
        &mut self,
        loaded_key: &str,
        requested_key: &str,
        public_model_id: Option<&str>,
    ) -> String {
        if loaded_key == requested_key || public_model_id.is_none() {
            return loaded_key.to_string();
        }
        let Some(mut loaded) = self.loaded.remove(loaded_key) else {
            return loaded_key.to_string();
        };
        if loaded.public_model_id.is_some() {
            self.loaded.insert(loaded_key.to_string(), loaded);
            return loaded_key.to_string();
        }
        loaded.model_key = requested_key.to_string();
        loaded.public_model_id = public_model_id.map(str::to_string);
        self.loaded.insert(requested_key.to_string(), loaded);
        requested_key.to_string()
    }

    pub(super) fn loaded_models_payload(&mut self) -> Value {
        self.reap_exited_runtimes();
        let mut data = self
            .loaded
            .values()
            .map(loaded_runtime_payload)
            .collect::<Vec<_>>();
        data.extend(self.attached.values().map(attached_runtime_payload));
        json!({
            "object": "list",
            "data": data,
        })
    }

    pub(super) fn loaded_runtime_summaries(&mut self) -> Vec<LoadedRuntimeSummary> {
        self.reap_exited_runtimes();
        self.loaded
            .values()
            .filter(|loaded| loaded.route_state == RuntimeRouteState::Ready)
            .map(|loaded| LoadedRuntimeSummary {
                id: loaded.model_key.clone(),
                owner_admin_id: loaded.owner_admin_id.clone(),
                backend_pid: loaded.process.info().pid,
            })
            .collect()
    }

    pub(super) fn snapshot(&mut self) -> Value {
        self.reap_exited_runtimes();
        let persistent_state = local_state::load_state().unwrap_or_default();
        let selected_backend = self
            .selected_backend
            .clone()
            .or_else(|| persistent_state.selected_backend.clone());
        let mut loaded_models = self
            .loaded
            .values()
            .map(loaded_runtime_payload)
            .collect::<Vec<_>>();
        loaded_models.extend(self.attached.values().map(attached_runtime_payload));
        let mut payload = self
            .default_model_key
            .as_ref()
            .and_then(|default_key| {
                self.loaded
                    .get(default_key)
                    .map(|loaded| managed_runtime_snapshot_payload(loaded, &loaded_models))
                    .or_else(|| {
                        self.attached.get(default_key).map(|attached| {
                            attached_runtime_snapshot_payload(attached, &loaded_models)
                        })
                    })
            })
            .unwrap_or_else(|| {
                let failure = self
                    .attached
                    .values()
                    .filter(|attached| attached.route_state == RuntimeRouteState::Failed)
                    .next_back();
                empty_runtime_snapshot_payload(selected_backend.as_deref(), &loaded_models, failure)
            });
        annotate_restore_state(&mut payload, &persistent_state, &self.loaded);
        payload["resource_ledger"] = self.resource_ledger_payload();
        payload
    }

    fn with_reservation<T>(
        &mut self,
        reservation_id: ReservationId,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let result = operation(self);
        if result.is_err() {
            self.resource_ledger
                .as_mut()
                .expect("reservation requires a resource ledger")
                .rollback(reservation_id);
        }
        result
    }

    fn take_generation(&mut self) -> Result<u64> {
        let generation = self.next_generation;
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("runtime generation overflow"))?;
        Ok(generation)
    }

    fn reserve_runtime_resources(
        &mut self,
        request_id: &str,
        budget: &ResourceBudget,
        cuda_visible_devices: Option<&str>,
        vulkan_devices: &[String],
    ) -> Result<ReservationId> {
        self.reject_exclusive_domains(budget)?;
        self.refresh_resource_capacity(cuda_visible_devices, vulkan_devices)?;
        Ok(self
            .resource_ledger
            .as_mut()
            .expect("resource ledger was initialized")
            .reserve(request_id, budget.clone())?)
    }

    fn reserve_llama_cpp_placement_resources(
        &mut self,
        request_id: &str,
        estimated: &ResourceBudget,
        cuda_visible_devices: Option<&str>,
        vulkan_devices: &[String],
    ) -> Result<ReservationId> {
        self.reject_exclusive_domains(estimated)?;
        self.refresh_resource_capacity(cuda_visible_devices, vulkan_devices)?;
        let provisional = provisional_llama_cpp_placement_budget(
            estimated,
            &self
                .resource_ledger
                .as_ref()
                .expect("resource ledger was initialized")
                .snapshot(),
        )?;
        Ok(self
            .resource_ledger
            .as_mut()
            .expect("resource ledger was initialized")
            .reserve(request_id, provisional)?)
    }

    fn reject_exclusive_domains(&self, budget: &ResourceBudget) -> Result<()> {
        for domain in budget.domains().keys() {
            if self.speculative_domains.contains_key(domain) {
                anyhow::bail!(
                    "CUDA device {} is exclusively held by a speculative runtime",
                    domain.key(),
                );
            }
        }
        Ok(())
    }

    fn refresh_resource_capacity(
        &mut self,
        cuda_visible_devices: Option<&str>,
        vulkan_devices: &[String],
    ) -> Result<()> {
        let observed = detect_available_resources(cuda_visible_devices, vulkan_devices)?;
        let current_usage = self
            .resource_ledger
            .as_ref()
            .map(|ledger| ledger.snapshot())
            .map(|snapshot| {
                merge_domain_totals(&snapshot.reserved, &snapshot.committed)
                    .map(|usage| (snapshot, usage))
            })
            .transpose()?;
        let mut capacities = current_usage
            .as_ref()
            .map(|(snapshot, _)| snapshot.capacities.clone())
            .unwrap_or_default();
        for (domain, available) in observed {
            let used = current_usage
                .as_ref()
                .and_then(|(_, usage)| usage.get(&domain))
                .copied()
                .unwrap_or(0);
            capacities.insert(
                domain,
                available
                    .checked_add(used)
                    .ok_or_else(|| anyhow::anyhow!("resource capacity overflow"))?,
            );
        }
        let snapshot_id = self.next_capacity_snapshot;
        self.next_capacity_snapshot = self
            .next_capacity_snapshot
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("resource capacity snapshot overflow"))?;
        let capacity = ResourceCapacity::new(snapshot_id, capacities)?;
        match self.resource_ledger.as_mut() {
            Some(ledger) => ledger.update_capacity(capacity)?,
            None => self.resource_ledger = Some(ResourceLedger::new(capacity)),
        }
        Ok(())
    }

    fn reap_exited_runtimes(&mut self) {
        let exited = self
            .loaded
            .iter_mut()
            .filter_map(|(key, loaded)| match loaded.process.has_exited() {
                Ok(true) => Some(key.clone()),
                Ok(false) => None,
                Err(_) => {
                    loaded.route_state = RuntimeRouteState::Failed;
                    None
                }
            })
            .collect::<Vec<_>>();
        for key in exited {
            self.remove_runtime_and_release(&key);
        }
        self.refresh_attached_runtimes();
        self.select_fallback_default();
    }

    fn refresh_attached_runtimes(&mut self) {
        for attached in self
            .attached
            .values_mut()
            .filter(|attached| attached.route_state == RuntimeRouteState::Ready)
        {
            if let Some(error) = attached.monitor.failure() {
                attached.route_state = RuntimeRouteState::Failed;
                attached.last_error = Some(error);
            }
        }
    }

    fn remove_runtime_and_release(&mut self, key: &str) {
        if let Some(loaded) = self.loaded.remove(key) {
            self.clear_speculative_owner(loaded.allocation_id);
            if let Some(ledger) = self.resource_ledger.as_mut() {
                ledger.release(loaded.allocation_id);
            }
        }
    }

    fn clear_speculative_owner(&mut self, allocation_id: AllocationId) {
        self.speculative_domains
            .retain(|_, current_owner| *current_owner != allocation_id);
    }

    fn select_fallback_default(&mut self) {
        if self.default_model_key.as_ref().is_some_and(|key| {
            self.loaded
                .get(key)
                .is_some_and(|loaded| loaded.route_state == RuntimeRouteState::Ready)
                || self
                    .attached
                    .get(key)
                    .is_some_and(|attached| attached.route_state == RuntimeRouteState::Ready)
        }) {
            return;
        }
        self.default_model_key = self
            .attached
            .iter()
            .rev()
            .find_map(|(key, attached)| {
                (attached.route_state == RuntimeRouteState::Ready).then(|| key.clone())
            })
            .or_else(|| {
                self.loaded.iter().rev().find_map(|(key, loaded)| {
                    (loaded.route_state == RuntimeRouteState::Ready).then(|| key.clone())
                })
            });
    }

    fn resource_ledger_payload(&self) -> Value {
        let Some(ledger) = self.resource_ledger.as_ref() else {
            return Value::Null;
        };
        let snapshot = ledger.snapshot();
        let available = snapshot.available().unwrap_or_default();
        json!({
            "capacity_snapshot_id": snapshot.capacity_snapshot_id,
            "capacity_bytes": domain_bytes_payload(&snapshot.capacities),
            "reserved_bytes": domain_bytes_payload(&snapshot.reserved),
            "committed_bytes": domain_bytes_payload(&snapshot.committed),
            "available_bytes": domain_bytes_payload(&available),
        })
    }
}

fn required_attach_string(
    payload: &Value,
    field: &'static str,
) -> std::result::Result<String, AttachRuntimeError> {
    let value = payload
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AttachRuntimeError::bad_request(format!("field '{field}' is required")))?;
    if value.len() > 2048 {
        return Err(AttachRuntimeError::bad_request(format!(
            "field '{field}' exceeds 2048 bytes"
        )));
    }
    Ok(value.to_string())
}

fn normalize_attach_endpoint(
    value: &str,
    gateway_port: u16,
) -> std::result::Result<(String, u16), AttachRuntimeError> {
    let endpoint = Url::parse(value).map_err(|_| {
        AttachRuntimeError::bad_request("client_endpoint must be an absolute HTTP URL")
    })?;
    if endpoint.scheme() != "http"
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
        || endpoint.path() != "/"
    {
        return Err(AttachRuntimeError::bad_request(
            "client_endpoint must be an HTTP origin without credentials, path, query, or fragment",
        ));
    }
    let host = endpoint
        .host_str()
        .ok_or_else(|| AttachRuntimeError::bad_request("client_endpoint host is required"))?;
    let parsed_host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    let loopback = parsed_host.eq_ignore_ascii_case("localhost")
        || parsed_host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    if !loopback {
        return Err(AttachRuntimeError::bad_request(format!(
            "client_endpoint must resolve explicitly to loopback, got: {host}"
        )));
    }
    let port = endpoint
        .port_or_known_default()
        .ok_or_else(|| AttachRuntimeError::bad_request("client_endpoint port is required"))?;
    if port == gateway_port {
        return Err(AttachRuntimeError::conflict(
            "client_endpoint cannot point to the current OmniInfer gateway",
        ));
    }
    let display_host = parsed_host
        .parse::<IpAddr>()
        .ok()
        .filter(IpAddr::is_ipv6)
        .map(|_| format!("[{parsed_host}]"))
        .unwrap_or_else(|| parsed_host.to_ascii_lowercase());
    Ok((format!("http://{display_host}:{port}"), port))
}

fn probe_attached_runtime(endpoint: &str) -> std::result::Result<Vec<String>, String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(ATTACHED_RUNTIME_PROBE_TIMEOUT))
        .proxy(None)
        .build()
        .new_agent();
    agent
        .get(&format!("{endpoint}/health"))
        .call()
        .map_err(|error| format!("GET /health failed: {error}"))?;
    let mut response = agent
        .get(&format!("{endpoint}/v1/models"))
        .call()
        .map_err(|error| format!("GET /v1/models failed: {error}"))?;
    let payload: Value = response
        .body_mut()
        .read_json()
        .map_err(|error| format!("GET /v1/models returned invalid JSON: {error}"))?;
    let data = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "GET /v1/models response is missing array field 'data'".to_string())?;
    let mut models = data
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str))
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    if models.is_empty() {
        return Err("GET /v1/models did not expose any model IDs".to_string());
    }
    Ok(models)
}

fn ensure_runtime_port_available(host: &str, port: u16) -> Result<()> {
    TcpListener::bind((host, port)).map_err(|error| {
        anyhow::anyhow!(
            "backend_port {port} on {host} is already occupied or unavailable; managed runtime startup was not attempted: {error}"
        )
    })?;
    Ok(())
}

fn attached_runtime_response(attached: &AttachedExternalRuntime, already_attached: bool) -> Value {
    json!({
        "ok": true,
        "already_attached": already_attached,
        "model": attached.model_key,
        "selected_backend": attached.backend_id,
        "request_defaults": attached.request_defaults,
        "backend_pid": null,
        "backend_port": attached.backend_port,
        "generation": attached.generation,
        "route_state": attached.route_state.as_str(),
        "runtime_ownership": "external",
        "process_owned": false,
        "external_server_protocol": attached.external_server_protocol.as_str(),
        "client_endpoint": attached.client_endpoint,
        "upstream_models": attached.upstream_models,
        "persisted_for_restore": false,
    })
}

fn attached_runtime_payload(attached: &AttachedExternalRuntime) -> Value {
    json!({
        "id": attached.model_key,
        "owner_admin_id": attached.owner_admin_id,
        "backend": attached.backend_id,
        "model": attached.model_key,
        "model_path": null,
        "public_model_id": null,
        "mmproj": null,
        "ctx_size": null,
        "request_defaults": attached.request_defaults,
        "runtime_mode": "external_server",
        "runtime_ownership": "external",
        "process_owned": false,
        "backend_pid": null,
        "backend_port": attached.backend_port,
        "generation": attached.generation,
        "route_state": attached.route_state.as_str(),
        "runtime_error": attached.last_error,
        "allocation_id": null,
        "resource_budget": null,
        "runtime_placement": null,
        "speculative_admission": null,
        "launch_args": [],
        "cuda_visible_devices": null,
        "warning": null,
        "launch_command": [],
        "proxy_model": attached.model_key,
        "external_server_protocol": attached.external_server_protocol.as_str(),
        "client_endpoint": attached.client_endpoint,
        "openai_compatible": attached.external_server_protocol.is_openai_compatible(),
        "backend_log": null,
        "upstream_models": attached.upstream_models,
        "persisted_for_restore": false,
    })
}

fn managed_runtime_snapshot_payload(loaded: &LoadedRustRuntime, loaded_models: &[Value]) -> Value {
    let info = loaded.process.info();
    let mut payload = loaded_runtime_payload(loaded);
    payload["backend_ready"] = json!(true);
    payload["effective_parameters"] = json!({});
    payload["runtime"] = json!({
        "mode": "external_server",
        "ownership": "managed",
        "host": "127.0.0.1",
        "port": info.port,
        "pid": info.pid,
        "cuda_visible_devices": loaded.cuda_visible_devices,
        "launch_command": info.command,
        "log_path": info.log_path.display().to_string(),
        "proxy_model_ref": loaded.proxy_model_ref,
        "external_server_protocol": loaded.external_server_protocol.as_str(),
        "client_endpoint": loaded.client_endpoint,
        "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
    });
    payload["log_path"] = json!(info.log_path.display().to_string());
    payload["loaded_models"] = json!(loaded_models);
    payload["default_model"] = json!(loaded.model_key);
    payload
}

fn attached_runtime_snapshot_payload(
    attached: &AttachedExternalRuntime,
    loaded_models: &[Value],
) -> Value {
    let mut payload = attached_runtime_payload(attached);
    payload["backend_ready"] = json!(true);
    payload["effective_parameters"] = json!({});
    payload["runtime"] = json!({
        "mode": "external_server",
        "ownership": "external",
        "host": "loopback",
        "port": attached.backend_port,
        "pid": null,
        "launch_command": [],
        "log_path": null,
        "proxy_model_ref": attached.model_key,
        "external_server_protocol": attached.external_server_protocol.as_str(),
        "client_endpoint": attached.client_endpoint,
        "openai_compatible": attached.external_server_protocol.is_openai_compatible(),
    });
    payload["log_path"] = Value::Null;
    payload["loaded_models"] = json!(loaded_models);
    payload["default_model"] = json!(attached.model_key);
    payload
}

fn empty_runtime_snapshot_payload(
    selected_backend: Option<&str>,
    loaded_models: &[Value],
    failure: Option<&AttachedExternalRuntime>,
) -> Value {
    json!({
        "backend": selected_backend,
        "backend_ready": false,
        "model": null,
        "public_model_id": null,
        "mmproj": null,
        "ctx_size": null,
        "request_defaults": {},
        "runtime_mode": null,
        "runtime_ownership": null,
        "process_owned": null,
        "runtime_error": failure.and_then(|attached| attached.last_error.as_deref()),
        "backend_pid": null,
        "backend_port": null,
        "launch_args": [],
        "cuda_visible_devices": null,
        "warning": null,
        "launch_command": [],
        "proxy_model": null,
        "external_server_protocol": null,
        "client_endpoint": null,
        "openai_compatible": false,
        "backend_log": null,
        "runtime_placement": null,
        "effective_parameters": {},
        "runtime": null,
        "loaded_models": loaded_models,
        "default_model": null,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpeculativeReservation {
    budget: ResourceBudget,
    device: String,
    estimated: u64,
    available: u64,
    shortfall: u64,
    waived_slack: u64,
}

fn is_cuda_capacity_exhaustion(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<ResourceLedgerError>()
        .is_some_and(|error| {
            matches!(
                error,
                ResourceLedgerError::InsufficientCapacity { domain, .. }
                    if domain.starts_with("cuda:")
            )
        })
}

fn speculative_reservation(
    backend: &backend_registry::BackendSpec,
    payload: &Value,
    budget: &ResourceBudget,
    replicate_across_domains: bool,
    snapshot: Option<omniinfer_core::resource_ledger::ResourceLedgerSnapshot>,
) -> Result<Option<SpeculativeReservation>> {
    if backend.family != "llama.cpp"
        || !backend.id.starts_with("llama.cpp-")
        || !backend.capabilities.iter().any(|cap| cap == "cuda")
        || replicate_across_domains
        || payload
            .get("resource_budget_bytes")
            .and_then(Value::as_u64)
            .is_some_and(|bytes| bytes > 0)
    {
        return Ok(None);
    }
    let cuda = budget
        .domains()
        .iter()
        .filter(|(domain, _)| matches!(domain, MemoryDomain::Cuda(_)))
        .collect::<Vec<_>>();
    if cuda.len() != 1 {
        return Ok(None);
    }
    let (domain, estimated) = cuda[0];
    let Some(snapshot) = snapshot else {
        return Ok(None);
    };
    let reserved = snapshot.reserved.get(domain).copied().unwrap_or(0);
    let committed = snapshot.committed.get(domain).copied().unwrap_or(0);
    if reserved != 0 || committed != 0 {
        return Ok(None);
    }
    let available = snapshot.available()?.get(domain).copied().unwrap_or(0);
    if available >= *estimated {
        return Ok(None);
    }
    let slack = budget
        .components()
        .iter()
        .find(|component| component.domain == *domain && component.name == "allocator_slack")
        .map(|component| component.bytes)
        .unwrap_or(0);
    let waiver_limit = slack.min(SPECULATIVE_ALLOCATOR_SLACK_LIMIT);
    let shortfall = estimated.saturating_sub(available);
    if estimated.saturating_sub(slack) > available || shortfall > waiver_limit {
        return Ok(None);
    }
    let device = match domain {
        MemoryDomain::Cuda(device) => device.clone(),
        _ => unreachable!(),
    };
    let budget = ResourceBudget::from_components(vec![BudgetComponent {
        name: "llama_cpp_speculative_cuda_exclusive".to_string(),
        domain: domain.clone(),
        bytes: available,
    }])?;
    Ok(Some(SpeculativeReservation {
        budget,
        device,
        estimated: *estimated,
        available,
        shortfall,
        waived_slack: shortfall,
    }))
}

mod lifecycle;

use lifecycle::*;
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

mod resources;

use resources::*;
mod model_config;

use model_config::*;
mod placement;

use placement::*;
pub(super) fn pick_runtime_port(host: &str) -> Result<u16> {
    let listener = std::net::TcpListener::bind((host, 0))?;
    Ok(listener.local_addr()?.port())
}

fn no_mmproj_from_payload(payload: &Value) -> Result<bool> {
    match payload.get("no_mmproj") {
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => anyhow::bail!("no_mmproj must be a boolean"),
        None => Ok(false),
    }
}

fn select_mmproj_path(
    no_mmproj: bool,
    explicit: Option<String>,
    discovered: Option<String>,
    automatic: Option<String>,
) -> Option<String> {
    if no_mmproj {
        None
    } else {
        explicit.or(discovered).or(automatic)
    }
}

#[cfg(test)]
mod tests;
