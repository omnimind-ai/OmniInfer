use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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

use super::gpu_status::runtime_env_for_backend;

const WSL_ROCM_COLD_START_RETRY_MINIMUM_BUDGET: Duration = Duration::from_secs(360);
const WSL_ROCM_COLD_START_INITIAL_ATTEMPT: Duration = Duration::from_secs(120);
const WSL_ROCM_COLD_START_RETRY_COOLDOWN: Duration = Duration::from_secs(90);
const SPECULATIVE_ALLOCATOR_SLACK_LIMIT: u64 = 1024 * 1024 * 1024;

pub(super) struct RustRuntimeManager {
    selected_backend: Option<String>,
    loaded: BTreeMap<String, LoadedRustRuntime>,
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
        let port = payload
            .get("backend_port")
            .and_then(Value::as_u64)
            .filter(|value| (1..=u64::from(u16::MAX)).contains(value))
            .and_then(|value| u16::try_from(value).ok())
            .map(Ok)
            .unwrap_or_else(|| pick_runtime_port(&backend_host))?;
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
        let resource_budget = build_runtime_resource_budget(
            &payload,
            backend,
            &resolved_model.model_path,
            mmproj_path.as_deref(),
            plan.ctx_size.unwrap_or(DEFAULT_LOAD_CONTEXT_SIZE),
            &effective_launch_args,
            budget_cuda_devices.as_deref(),
            cuda_selection.is_none() && budget_cuda_devices.is_some(),
        )?;
        let budget_vulkan_devices = resource_budget
            .domains()
            .keys()
            .filter_map(|domain| match domain {
                MemoryDomain::Vulkan(device) => Some(device.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let reconcile_policy = placement_policy.filter(|policy| {
            policy.permits_partial_offload()
                && resource_budget
                    .domains()
                    .keys()
                    .filter(|domain| matches!(domain, MemoryDomain::Cuda(_)))
                    .count()
                    == 1
        });
        let initial_reservation = if reconcile_policy.is_some() {
            self.reserve_partial_offload_resources(
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
            Err(error) if reconcile_policy.is_none() && is_cuda_capacity_exhaustion(&error) => {
                let decision = speculative_reservation(
                    backend,
                    &payload,
                    &resource_budget,
                    cuda_selection.is_none() && budget_cuda_devices.is_some(),
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

    pub(super) fn unload_model(&mut self, model: &str, admin_id: Option<&str>) -> Result<Value> {
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
        let loaded = self.loaded.get(&key)?;
        if loaded.route_state != RuntimeRouteState::Ready {
            return None;
        }
        Some(RuntimeProxyTarget {
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
        })
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
        self.loaded.iter().find_map(|(key, loaded)| {
            (loaded.public_model_id.as_deref() == Some(requested)
                || loaded.model == requested
                || loaded.proxy_model_ref.as_deref() == Some(requested))
            .then(|| key.clone())
        })
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
        json!({
            "object": "list",
            "data": self.loaded.values().map(loaded_runtime_payload).collect::<Vec<_>>(),
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
        let loaded_models = self
            .loaded
            .values()
            .map(loaded_runtime_payload)
            .collect::<Vec<_>>();
        let mut payload = match self
            .default_model_key
            .as_ref()
            .and_then(|default_key| self.loaded.get(default_key))
        {
            Some(loaded) => {
                let info = loaded.process.info();
                json!({
                    "backend": loaded.backend_id,
                    "backend_ready": true,
                    "model": loaded.model_key,
                    "model_path": loaded.model,
                    "public_model_id": loaded.public_model_id,
                    "owner_admin_id": loaded.owner_admin_id,
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
                    "speculative_admission": speculative_admission_payload(
                        loaded.speculative_admission.as_ref(),
                    ),
                    "launch_args": loaded.launch_args,
                    "cuda_visible_devices": loaded.cuda_visible_devices,
                    "warning": loaded.cuda_warning,
                    "launch_command": info.command,
                    "proxy_model": loaded.proxy_model_ref,
                    "external_server_protocol": loaded.external_server_protocol.as_str(),
                    "client_endpoint": loaded.client_endpoint,
                    "openai_compatible": loaded.external_server_protocol.is_openai_compatible(),
                    "backend_log": info.log_path.display().to_string(),
                    "effective_parameters": {},
                    "runtime": {
                        "mode": "external_server",
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
                    },
                    "log_path": info.log_path.display().to_string(),
                    "loaded_models": loaded_models,
                    "default_model": loaded.model_key,
                })
            }
            None => json!({
                "backend": selected_backend,
                "backend_ready": false,
                "model": null,
                "public_model_id": null,
                "mmproj": null,
                "ctx_size": null,
                "request_defaults": {},
                "runtime_mode": null,
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
            }),
        };
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

    fn reserve_partial_offload_resources(
        &mut self,
        request_id: &str,
        estimated: &ResourceBudget,
        cuda_visible_devices: Option<&str>,
        vulkan_devices: &[String],
    ) -> Result<ReservationId> {
        self.reject_exclusive_domains(estimated)?;
        self.refresh_resource_capacity(cuda_visible_devices, vulkan_devices)?;
        let provisional = provisional_partial_offload_budget(
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
        self.select_fallback_default();
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
        }) {
            return;
        }
        self.default_model_key = self.loaded.iter().rev().find_map(|(key, loaded)| {
            (loaded.route_state == RuntimeRouteState::Ready).then(|| key.clone())
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
