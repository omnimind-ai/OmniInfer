from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from service_core.backends import BACKEND_PRIORITY, BackendSpec
from service_core.platforms import HostPlatform, discover_llama_cpp_model_artifacts, maybe_auto_mmproj
from service_core.platforms.common import (
    bytes_to_gib,
    get_available_memory_bytes,
    get_total_memory_bytes,
    get_available_rocm_memory_bytes,
    hidden_subprocess_kwargs,
)


DEFAULT_CONTEXT_SIZE = 8192
GPU_MEMORY_MARGIN_GIB = 0.5
CPU_MEMORY_MARGIN_GIB = 1.0
BYTES_PER_GIB = float(1024 ** 3)
QUANT_PATTERNS = (
    "UD-Q8_K_XL",
    "UD-Q8_K_L",
    "UD-Q8_K_M",
    "UD-Q8_K_S",
    "UD-Q6_K_XL",
    "UD-Q6_K_L",
    "UD-Q6_K_M",
    "UD-Q6_K_S",
    "UD-Q5_K_XL",
    "UD-Q5_K_L",
    "UD-Q5_K_M",
    "UD-Q5_K_S",
    "UD-Q4_K_XL",
    "UD-Q4_K_L",
    "UD-Q4_K_M",
    "UD-Q4_K_S",
    "UD-Q3_K_XL",
    "UD-Q3_K_L",
    "UD-Q3_K_M",
    "UD-Q3_K_S",
    "UD-Q2_K_XL",
    "UD-Q2_K_L",
    "UD-Q2_K_M",
    "UD-Q2_K_S",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "Q4_0",
    "Q3_K_M",
    "Q2_K",
    "BF16",
    "F16",
    "F32",
)


def system_snapshot(platform_obj: HostPlatform, backends: dict[str, BackendSpec]) -> dict[str, Any]:
    cuda_devices = _query_cuda_devices()
    visible_filter = os.environ.get("OMNIINFER_CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_cuda_devices = _filter_visible_cuda_devices(cuda_devices, visible_filter)
    installed_set = {backend.id for backend in backends.values() if backend.binary_exists}
    compatible_set = {
        backend.id
        for backend in backends.values()
        if _system_hardware_compatible(backend, platform_obj, installed_set=installed_set, cuda_devices=cuda_devices)
    }
    return {
        "object": "advisor.system",
        "host": {
            "system": platform_obj.system_name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_cores": os.cpu_count(),
            "available_ram_gib": _available_ram_gib(),
            "total_ram_gib": _total_ram_gib(),
        },
        "cuda": {
            "devices": cuda_devices,
            "visible_filter": visible_filter,
            "visible_devices": visible_cuda_devices,
            "best_free_device": _best_cuda_device(visible_cuda_devices or cuda_devices),
        },
        "backends": [
            _backend_payload(backend, installed=installed_set, compatible=compatible_set)
            for backend in sorted(backends.values(), key=lambda item: BACKEND_PRIORITY.get(item.id, 999))
        ],
        "summary": {
            "installed_backends": sorted(installed_set),
            "compatible_backends": sorted(compatible_set),
            "recommended_installed_backend": _recommended_backend_id(
                [backend for backend in backends.values() if backend.id in installed_set and backend.id in compatible_set]
            ),
        },
    }


def inspect_model(model: str, *, mmproj: str | None = None) -> dict[str, Any]:
    resolved_model, resolved_mmproj, artifact_kind, warnings = _resolve_model_artifact(model, mmproj)
    name_source = resolved_model if isinstance(resolved_model, str) else str(resolved_model)
    model_path = resolved_model if isinstance(resolved_model, Path) else None
    mmproj_path = resolved_mmproj if isinstance(resolved_mmproj, Path) else None
    size_gib = _path_size_gib(model_path)
    mmproj_size_gib = _path_size_gib(mmproj_path)
    quant = _infer_quantization(name_source)
    params_b = _infer_params_b(name_source)
    capabilities = _infer_model_capabilities(name_source, mmproj_path)
    model_format = _model_format(artifact_kind, model_path, name_source)
    return {
        "object": "advisor.model",
        "input": model,
        "model": str(resolved_model),
        "model_path": str(model_path) if model_path else None,
        "mmproj": str(mmproj_path) if mmproj_path else None,
        "format": model_format,
        "artifact_kind": artifact_kind,
        "exists": model_path.exists() if model_path else False,
        "size_gib": size_gib,
        "mmproj_size_gib": mmproj_size_gib,
        "quantization": quant,
        "params_b": params_b,
        "capabilities": capabilities,
        "estimate": _memory_estimate(size_gib, mmproj_size_gib, params_b, DEFAULT_CONTEXT_SIZE),
        "warnings": warnings,
    }


def fit_model(
    model: str,
    *,
    platform_obj: HostPlatform,
    backends: dict[str, BackendSpec],
    mmproj: str | None = None,
    ctx_size: int | None = None,
    backend_filter: str | None = None,
) -> dict[str, Any]:
    model_info = inspect_model(model, mmproj=mmproj)
    context = ctx_size or DEFAULT_CONTEXT_SIZE
    estimate = _memory_estimate(
        model_info.get("size_gib"),
        model_info.get("mmproj_size_gib"),
        model_info.get("params_b"),
        context,
    )
    model_info["estimate"] = estimate
    candidates = [
        _backend_fit_payload(backend, platform_obj, model_info, estimate, context)
        for backend in sorted(backends.values(), key=lambda item: BACKEND_PRIORITY.get(item.id, 999))
        if backend_filter is None or backend.id == backend_filter
    ]
    compatible = [candidate for candidate in candidates if candidate["compatible"]]
    recommended = _recommended_candidate(compatible)
    if recommended is not None:
        recommended["why_recommended"] = _why_recommended(recommended, model_info)
    command = _next_load_command(model_info, recommended, context)
    warnings = list(model_info.get("warnings", []))
    if recommended and not recommended.get("installed"):
        warnings.append(f"recommended backend {recommended['backend']} is compatible but not installed")
    if not compatible:
        warnings.append("no compatible backend was found for this model reference")
    return {
        "object": "advisor.fit",
        "model": model_info,
        "context_size": context,
        "recommended": recommended,
        "alternatives": [
            _with_why_not(candidate, recommended, model_info)
            for candidate in compatible
            if recommended is None or candidate["backend"] != recommended["backend"]
        ],
        "all_backends": candidates,
        "next_command": command,
        "warnings": warnings,
    }


def recommend_models(
    *,
    platform_obj: HostPlatform,
    backends: dict[str, BackendSpec],
    task: str | None = None,
    limit: int = 5,
    ctx_size: int | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for models_dir in _candidate_model_dirs(backends):
        for model_path in _iter_local_models(models_dir):
            key = str(model_path)
            if key in seen:
                continue
            seen.add(key)
            try:
                fit = fit_model(
                    str(model_path),
                    platform_obj=platform_obj,
                    backends=backends,
                    ctx_size=ctx_size,
                )
            except (OSError, ValueError):
                continue
            recommended = fit.get("recommended")
            if not isinstance(recommended, dict):
                continue
            if task and not _task_matches_model(task, fit["model"]):
                continue
            score = _recommendation_score(recommended, fit["model"])
            evidence = recommended.get("evidence") if isinstance(recommended.get("evidence"), dict) else {}
            candidates.append(
                {
                    "score": score,
                    "model": fit["model"],
                    "recommended": recommended,
                    "evidence": evidence,
                    "recommendation_confidence": recommended.get("recommendation_confidence"),
                    "why_recommended": recommended.get("why_recommended", []),
                    "next_command": fit.get("next_command"),
                    "warnings": fit.get("warnings", []),
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), item["model"]["model"]))
    return {
        "object": "advisor.recommend",
        "task": task or "any",
        "context_size": ctx_size or DEFAULT_CONTEXT_SIZE,
        "models_scanned": len(seen),
        "returned": min(limit, len(candidates)),
        "recommendations": candidates[:limit],
    }


def plan_model(
    model: str,
    *,
    platform_obj: HostPlatform,
    backends: dict[str, BackendSpec],
    mmproj: str | None = None,
    ctx_size: int | None = None,
    gpu_vram_gib: float | None = None,
    ram_gib: float | None = None,
    cpu_cores: int | None = None,
) -> dict[str, Any]:
    context = ctx_size or DEFAULT_CONTEXT_SIZE
    model_info = inspect_model(model, mmproj=mmproj)
    estimate = _memory_estimate(
        model_info.get("size_gib"),
        model_info.get("mmproj_size_gib"),
        model_info.get("params_b"),
        context,
    )
    model_info["estimate"] = estimate
    system = system_snapshot(platform_obj, backends)
    current = _current_hardware(system)
    simulated = _apply_hardware_overrides(current, gpu_vram_gib=gpu_vram_gib, ram_gib=ram_gib, cpu_cores=cpu_cores)
    paths = [
        _plan_run_path("gpu", estimate, simulated),
        _plan_run_path("cpu_offload", estimate, simulated),
        _plan_run_path("cpu_only", estimate, simulated),
    ]
    recommended = _recommended_plan_path(paths)
    return {
        "object": "advisor.plan",
        "model": model_info,
        "context_size": context,
        "current_hardware": current,
        "planning_hardware": simulated,
        "run_paths": paths,
        "recommended_path": recommended,
        "upgrade_deltas": _upgrade_deltas(paths, simulated),
        "estimate_notice": "Hardware planning uses local advisor heuristics; backend logs and benchmark runs remain authoritative.",
        "next_commands": _plan_next_commands(model_info, context, recommended),
        "warnings": list(model_info.get("warnings", [])),
    }


def _backend_payload(backend: BackendSpec, *, installed: set[str], compatible: set[str]) -> dict[str, Any]:
    return {
        "id": backend.id,
        "label": backend.label,
        "family": backend.family,
        "installed": backend.id in installed,
        "hardware_compatible": backend.id in compatible,
        "runtime_mode": backend.runtime_mode,
        "model_artifact": backend.model_artifact,
        "supports_mmproj": backend.supports_mmproj,
        "supports_ctx_size": backend.supports_ctx_size,
        "capabilities": backend.capabilities,
        "launcher_path": backend.launcher_path,
        "runtime_dir": backend.runtime_dir,
        "models_dir": backend.models_dir,
        "priority": BACKEND_PRIORITY.get(backend.id, 999),
    }


def _system_hardware_compatible(
    backend: BackendSpec,
    platform_obj: HostPlatform,
    *,
    installed_set: set[str],
    cuda_devices: list[dict[str, Any]],
) -> bool:
    caps = set(backend.capabilities)
    machine = platform.machine().lower()
    if "arm64" in caps and machine not in {"arm64", "aarch64"}:
        return False
    if "s390x" in caps and machine != "s390x":
        return False
    if "openvino" in caps or "eagle3" in caps:
        return backend.id in installed_set
    if backend.id not in platform_obj.gpu_backend_ids:
        return True
    if "cuda" in caps:
        return bool(cuda_devices)
    if "metal" in caps:
        return True
    return platform_obj.is_hardware_compatible(backend)


def _current_hardware(system_payload: dict[str, Any]) -> dict[str, Any]:
    host = system_payload.get("host") if isinstance(system_payload.get("host"), dict) else {}
    cuda = system_payload.get("cuda") if isinstance(system_payload.get("cuda"), dict) else {}
    devices = cuda.get("visible_devices") or cuda.get("devices") or []
    best = cuda.get("best_free_device") if isinstance(cuda.get("best_free_device"), dict) else None
    return {
        "available_ram_gib": host.get("available_ram_gib"),
        "total_ram_gib": host.get("total_ram_gib"),
        "cpu_cores": host.get("cpu_cores"),
        "gpu_vram_free_gib": best.get("free_gib") if best else None,
        "gpu_vram_total_gib": best.get("total_gib") if best else None,
        "gpu_name": best.get("name") if best else None,
        "gpu_count": len(devices) if isinstance(devices, list) else 0,
    }


def _apply_hardware_overrides(
    current: dict[str, Any],
    *,
    gpu_vram_gib: float | None,
    ram_gib: float | None,
    cpu_cores: int | None,
) -> dict[str, Any]:
    result = dict(current)
    if gpu_vram_gib is not None:
        result["gpu_vram_free_gib"] = float(gpu_vram_gib)
        result["gpu_vram_total_gib"] = float(gpu_vram_gib)
        result["simulated_gpu_vram_gib"] = float(gpu_vram_gib)
    if ram_gib is not None:
        result["available_ram_gib"] = float(ram_gib)
        result["total_ram_gib"] = float(ram_gib)
        result["simulated_ram_gib"] = float(ram_gib)
    if cpu_cores is not None:
        result["cpu_cores"] = int(cpu_cores)
        result["simulated_cpu_cores"] = int(cpu_cores)
    return result


def _plan_run_path(path: str, estimate: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    required = float(estimate.get("estimated_gpu_memory_gib") or estimate.get("estimated_ram_gib") or 0.0)
    cpu_cores = int(hardware.get("cpu_cores") or os.cpu_count() or 1)
    if path == "gpu":
        available = _float_or_none(hardware.get("gpu_vram_free_gib"))
        minimum = {
            "vram_gib": round(required, 2),
            "ram_gib": round(max(4.0, required * 0.25), 2),
            "cpu_cores": max(2, min(cpu_cores, 4)),
        }
        recommended = {
            "vram_gib": round(required * 1.2 + GPU_MEMORY_MARGIN_GIB, 2),
            "ram_gib": round(max(8.0, required * 0.35), 2),
            "cpu_cores": max(4, min(cpu_cores, 8)),
        }
        fit = _fit_level(required, available, GPU_MEMORY_MARGIN_GIB)
        feasible = available is not None and fit in {"good", "marginal"}
        notes = ["fastest path when the selected backend can fully or mostly use GPU memory"]
    elif path == "cpu_offload":
        available = _float_or_none(hardware.get("available_ram_gib"))
        minimum = {
            "vram_gib": 2.0,
            "ram_gib": round(required, 2),
            "cpu_cores": max(4, min(cpu_cores, 8)),
        }
        recommended = {
            "vram_gib": 4.0,
            "ram_gib": round(required * 1.25 + CPU_MEMORY_MARGIN_GIB, 2),
            "cpu_cores": max(8, min(cpu_cores, 16)),
        }
        fit = _fit_level(required, available, CPU_MEMORY_MARGIN_GIB)
        feasible = available is not None and fit in {"good", "marginal"}
        notes = ["uses system RAM as the primary pool and GPU for partial acceleration when backend supports it"]
    else:
        available = _float_or_none(hardware.get("available_ram_gib"))
        minimum = {
            "vram_gib": None,
            "ram_gib": round(required, 2),
            "cpu_cores": max(4, min(cpu_cores, 8)),
        }
        recommended = {
            "vram_gib": None,
            "ram_gib": round(required * 1.35 + CPU_MEMORY_MARGIN_GIB, 2),
            "cpu_cores": max(8, min(cpu_cores, 32)),
        }
        fit = _fit_level(required, available, CPU_MEMORY_MARGIN_GIB)
        feasible = available is not None and fit in {"good", "marginal"}
        notes = ["lowest GPU requirement, usually slowest for chat generation"]
    return {
        "path": path,
        "feasible_now": feasible,
        "fit": fit,
        "memory_required_gib": round(required, 2) if required else None,
        "memory_available_gib": available,
        "minimum": minimum,
        "recommended": recommended,
        "estimated_relative_speed": _relative_speed(path, cpu_cores),
        "notes": notes,
    }


def _recommended_plan_path(paths: list[dict[str, Any]]) -> dict[str, Any] | None:
    rank_path = {"gpu": 0, "cpu_offload": 1, "cpu_only": 2}
    rank_fit = {"good": 0, "marginal": 1, "too_tight": 2, "unknown": 3}
    feasible = [path for path in paths if path.get("feasible_now")]
    candidates = feasible or paths
    if not candidates:
        return None
    return min(candidates, key=lambda item: (rank_fit.get(str(item.get("fit")), 3), rank_path.get(str(item.get("path")), 9)))


def _upgrade_deltas(paths: list[dict[str, Any]], hardware: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        recommended = path.get("recommended") if isinstance(path.get("recommended"), dict) else {}
        if path.get("path") == "gpu":
            current = _float_or_none(hardware.get("gpu_vram_free_gib")) or 0.0
            target = _float_or_none(recommended.get("vram_gib"))
            if target is not None and current < target:
                result.append(
                    {
                        "path": "gpu",
                        "resource": "vram",
                        "add_gib": round(target - current, 2),
                        "target_gib": target,
                        "description": f"add about {round(target - current, 2)} GiB free VRAM for the recommended GPU path",
                    }
                )
        current_ram = _float_or_none(hardware.get("available_ram_gib")) or 0.0
        target_ram = _float_or_none(recommended.get("ram_gib"))
        if target_ram is not None and current_ram < target_ram:
            result.append(
                {
                    "path": path.get("path"),
                    "resource": "ram",
                    "add_gib": round(target_ram - current_ram, 2),
                    "target_gib": target_ram,
                    "description": f"add about {round(target_ram - current_ram, 2)} GiB available RAM for {path.get('path')}",
                }
            )
    return result


def _plan_next_commands(model_info: dict[str, Any], ctx_size: int, recommended: dict[str, Any] | None) -> list[str]:
    if recommended is None:
        return []
    model = _shell_quote(str(model_info.get("model") or ""))
    if not model:
        return []
    if recommended.get("path") == "gpu":
        return [f"omniinfer advisor fit {model} --ctx-size {ctx_size}", f"omniinfer load -m {model} --ctx-size {ctx_size}"]
    if recommended.get("path") == "cpu_offload":
        return [f"omniinfer advisor fit {model} --ctx-size {ctx_size}", f"omniinfer load -m {model} --ctx-size {ctx_size}"]
    return [f"omniinfer backend select llama.cpp-linux && omniinfer load -m {model} --ctx-size {ctx_size}"]


def _relative_speed(path: str, cpu_cores: int) -> str:
    if path == "gpu":
        return "fast"
    if path == "cpu_offload":
        return "medium"
    return "slow" if cpu_cores < 16 else "medium-slow"


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _backend_fit_payload(
    backend: BackendSpec,
    platform_obj: HostPlatform,
    model_info: dict[str, Any],
    estimate: dict[str, Any],
    ctx_size: int,
) -> dict[str, Any]:
    compatible, reasons = _backend_model_compatible(backend, model_info)
    hardware_ok = platform_obj.is_hardware_compatible(backend)
    available_gib = _available_memory_for_backend(backend)
    memory_kind = "gpu" if _is_gpu_backend(backend) else "ram"
    required_gib = estimate["estimated_gpu_memory_gib"] if memory_kind == "gpu" else estimate["estimated_ram_gib"]
    breakdown = _memory_breakdown_for_backend(estimate, memory_kind)
    margin_gib = GPU_MEMORY_MARGIN_GIB if _is_gpu_backend(backend) else CPU_MEMORY_MARGIN_GIB
    fit_level = _fit_level(required_gib, available_gib, margin_gib)
    launch_args: list[str] = []
    if backend.supports_ctx_size:
        launch_args.extend(["--ctx-size", str(ctx_size)])
    if _is_gpu_backend(backend) and backend.default_args:
        launch_args.extend(backend.default_args)
    if model_info.get("mmproj") and backend.supports_mmproj:
        launch_args.extend(["--mmproj", str(model_info["mmproj"])])
    notes = list(reasons)
    if not backend.binary_exists:
        notes.append("backend runtime is not installed")
    if not hardware_ok:
        notes.append("backend hardware probe did not pass")
    if estimate["confidence"] == "low":
        notes.append("memory estimate is based on file size only")
    evidence = _candidate_evidence(backend, model_info, estimate, compatible, hardware_ok)
    return {
        "backend": backend.id,
        "label": backend.label,
        "family": backend.family,
        "installed": backend.binary_exists,
        "hardware_compatible": hardware_ok,
        "compatible": compatible and hardware_ok,
        "fit": fit_level,
        "memory_required_gib": required_gib,
        "memory_available_gib": available_gib,
        "memory_margin_gib": margin_gib,
        "memory_kind": memory_kind,
        "memory_breakdown": breakdown,
        "launch_args": launch_args,
        "priority": BACKEND_PRIORITY.get(backend.id, 999),
        "evidence": evidence,
        "recommendation_confidence": evidence["confidence"],
        "notes": notes,
        "why_not": _why_not_candidate(backend, compatible, hardware_ok, fit_level, notes),
    }


def _backend_model_compatible(backend: BackendSpec, model_info: dict[str, Any]) -> tuple[bool, list[str]]:
    fmt = str(model_info.get("format") or "")
    artifact_kind = str(model_info.get("artifact_kind") or "")
    notes: list[str] = []
    if fmt == "gguf":
        if backend.family in {"llama.cpp", "turboquant"} and backend.model_artifact in {"file", "path"}:
            return True, notes
        return False, ["GGUF models require a llama.cpp-compatible backend"]
    if fmt == "hf-reference":
        if backend.family == "vllm":
            return True, notes
        return False, ["HF references require vLLM or an explicit local model path"]
    if fmt == "directory":
        if backend.model_artifact in {"directory", "path"}:
            return True, notes
        return False, ["directory models require a directory/path backend"]
    if artifact_kind == "file":
        if backend.model_artifact in {"file", "path"}:
            return True, notes
    return backend.model_artifact == "reference", ["unknown model format; only reference backends are considered safe"]


def _recommended_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    def rank(candidate: dict[str, Any]) -> tuple[int, int, int, float, int, str]:
        fit_rank = {"good": 0, "marginal": 1, "too_tight": 2, "unknown": 3}.get(str(candidate.get("fit")), 3)
        installed_rank = 0 if candidate.get("installed") else 1
        priority = int(candidate.get("priority", 999))
        family_variant = 1 if str(candidate.get("backend", "")).startswith("ik_llama.cpp") else 0
        available = float(candidate.get("memory_available_gib") or 0)
        return (fit_rank, installed_rank, priority, family_variant, -available, 0 if candidate.get("hardware_compatible") else 1, str(candidate["backend"]))

    return min(candidates, key=rank)


def _candidate_evidence(
    backend: BackendSpec,
    model_info: dict[str, Any],
    estimate: dict[str, Any],
    compatible: bool,
    hardware_ok: bool,
) -> dict[str, Any]:
    model_path = model_info.get("model_path")
    exists = bool(model_info.get("exists"))
    fmt = str(model_info.get("format") or "")
    confidence = str(estimate.get("confidence") or "low")
    sources: list[str] = []
    if model_path and exists:
        sources.append("local_model_file")
    if backend.binary_exists:
        sources.append("installed_backend")
    if hardware_ok:
        sources.append("hardware_probe")
    if estimate.get("estimate_source"):
        sources.append(str(estimate["estimate_source"]))

    if not compatible or not hardware_ok:
        level = "none"
        confidence_label = "low"
    elif model_path and exists and backend.binary_exists and hardware_ok and confidence in {"medium", "high"}:
        level = "direct"
        confidence_label = "high"
    elif fmt == "hf-reference" and backend.family == "vllm":
        level = "self_reported"
        confidence_label = "medium" if backend.binary_exists and hardware_ok else "low"
    elif fmt in {"gguf", "directory"} and compatible:
        level = "variant"
        confidence_label = "medium" if hardware_ok else "low"
    else:
        level = "none"
        confidence_label = "low"

    return {
        "level": level,
        "confidence": confidence_label,
        "sources": sorted(set(sources)),
        "notes": _evidence_notes(level, model_info, backend, estimate),
    }


def _evidence_notes(level: str, model_info: dict[str, Any], backend: BackendSpec, estimate: dict[str, Any]) -> list[str]:
    if level == "direct":
        return [
            "local model artifact exists",
            f"{backend.id} runtime is installed and hardware-compatible",
            f"memory estimate confidence is {estimate.get('confidence') or 'unknown'}",
        ]
    if level == "variant":
        return [
            "model format is compatible with backend family",
            "estimate is inferred from local artifact metadata rather than a measured run",
        ]
    if level == "self_reported":
        return [
            "model is a remote or external reference accepted by the backend",
            "local artifact size is unavailable unless the backend downloads or resolves it",
        ]
    return ["insufficient compatible local evidence"]


def _why_recommended(candidate: dict[str, Any], model_info: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    fit = str(candidate.get("fit") or "unknown")
    backend = str(candidate.get("backend") or "backend")
    reasons.append(f"{backend} has the best ranked fit among compatible backends ({fit})")
    if candidate.get("installed"):
        reasons.append("runtime is already installed")
    if candidate.get("hardware_compatible"):
        reasons.append("hardware probe passed")
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
    level = evidence.get("level")
    if level:
        reasons.append(f"recommendation evidence level is {level}")
    memory_required = candidate.get("memory_required_gib")
    memory_available = candidate.get("memory_available_gib")
    if memory_required is not None and memory_available is not None:
        reasons.append(f"estimated memory {memory_required} GiB fits available {memory_available} GiB")
    capabilities = ", ".join(model_info.get("capabilities") or [])
    if capabilities:
        reasons.append(f"model capabilities: {capabilities}")
    return reasons


def _why_not_candidate(
    backend: BackendSpec,
    compatible: bool,
    hardware_ok: bool,
    fit_level: str,
    notes: list[str],
) -> list[str]:
    reasons: list[str] = []
    if not compatible:
        reasons.extend(notes or ["model format is not compatible with this backend"])
    if not backend.binary_exists:
        reasons.append("runtime is not installed")
    if not hardware_ok:
        reasons.append("hardware probe did not pass")
    if fit_level == "too_tight":
        reasons.append("estimated memory requirement exceeds available memory")
    if fit_level == "marginal":
        reasons.append("estimated memory fits but has little headroom")
    return reasons


def _with_why_not(candidate: dict[str, Any], recommended: dict[str, Any] | None, model_info: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    why_not = list(result.get("why_not") or [])
    if recommended is not None and not why_not:
        if result.get("fit") != recommended.get("fit"):
            why_not.append(f"recommended backend has better fit ({recommended.get('fit')})")
        elif result.get("installed") != recommended.get("installed"):
            why_not.append("recommended backend is installed")
        elif int(result.get("priority", 999)) > int(recommended.get("priority", 999)):
            why_not.append("recommended backend has higher product priority")
        else:
            why_not.append("ranked below the recommended backend by tie-breakers")
    if not why_not and model_info.get("format"):
        why_not.append(f"compatible with {model_info.get('format')}, but not the top-ranked option")
    result["why_not"] = why_not
    return result


def _recommended_backend_id(backends: list[BackendSpec]) -> str | None:
    if not backends:
        return None
    return min(
        backends,
        key=lambda backend: (
            BACKEND_PRIORITY.get(backend.id, 999),
            1 if backend.id.startswith("ik_llama.cpp") else 0,
            backend.id,
        ),
    ).id


def _next_load_command(model_info: dict[str, Any], recommended: dict[str, Any] | None, ctx_size: int) -> str | None:
    if recommended is None:
        return None
    select_parts = ["omniinfer", "backend", "select", _shell_quote(str(recommended["backend"]))]
    parts = ["omniinfer", "model", "load", "-m", _shell_quote(str(model_info["model"]))]
    if model_info.get("mmproj"):
        parts.extend(["--mmproj", _shell_quote(str(model_info["mmproj"]))])
    if ctx_size and _supports_ctx_in_command(recommended):
        parts.extend(["--ctx-size", str(ctx_size)])
    skip_next = False
    for arg in recommended.get("launch_args", []):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--ctx-size", "--mmproj"}:
            skip_next = True
            continue
        parts.append(_shell_quote(str(arg)))
    return " ".join(select_parts) + " && " + " ".join(parts)


def _supports_ctx_in_command(candidate: dict[str, Any]) -> bool:
    return "--ctx-size" in list(candidate.get("launch_args", []))


def _resolve_model_artifact(model: str, mmproj: str | None) -> tuple[Path | str, Path | None, str, list[str]]:
    text = model.strip()
    warnings: list[str] = []
    if not text:
        raise ValueError("model reference must not be empty")
    path = Path(os.path.abspath(os.path.expanduser(text)))
    resolved_mmproj = Path(os.path.abspath(os.path.expanduser(mmproj))).resolve() if mmproj else None
    if path.exists():
        path = path.resolve()
        if path.is_dir():
            try:
                model_path, auto_mmproj = discover_llama_cpp_model_artifacts(path)
            except (FileNotFoundError, ValueError):
                return path, resolved_mmproj, "directory", warnings
            if resolved_mmproj is None and auto_mmproj:
                resolved_mmproj = Path(auto_mmproj).resolve()
            return Path(model_path).resolve(), resolved_mmproj, "file", warnings
        if path.is_file() and path.suffix.lower() == ".gguf" and resolved_mmproj is None:
            auto_mmproj = maybe_auto_mmproj(str(path.parent), str(path))
            if auto_mmproj:
                resolved_mmproj = Path(auto_mmproj).resolve()
        return path, resolved_mmproj, "file", warnings
    if any(sep in text for sep in ("/", ":")) and not text.lower().endswith(".gguf"):
        return text, resolved_mmproj, "reference", warnings
    warnings.append(f"model path does not exist locally: {path}")
    return path, resolved_mmproj, "missing", warnings


def _model_format(artifact_kind: str, model_path: Path | None, model_ref: str) -> str:
    if artifact_kind == "reference":
        return "hf-reference"
    if model_path and model_path.is_dir():
        return "directory"
    if model_path and model_path.suffix.lower() == ".gguf":
        return "gguf"
    if model_ref.lower().endswith(".gguf"):
        return "gguf"
    if artifact_kind == "directory":
        return "directory"
    return "unknown"


def _memory_estimate(
    size_gib: float | None,
    mmproj_size_gib: float | None,
    params_b: float | None,
    ctx_size: int,
) -> dict[str, Any]:
    if size_gib is None:
        return {
            "estimated_gpu_memory_gib": None,
            "estimated_ram_gib": None,
            "estimated_kv_cache_gib": None,
            "breakdown": _unknown_memory_breakdown(ctx_size),
            "estimate_source": "unknown",
            "confidence": "low",
            "notes": ["local model size is unknown; fit cannot be estimated safely"],
        }
    base = float(size_gib) + float(mmproj_size_gib or 0.0)
    ctx_factor = max(ctx_size, 1) / DEFAULT_CONTEXT_SIZE
    param_factor = params_b if params_b is not None else max(base * 2.0, 1.0)
    weights = round(float(size_gib), 2)
    mmproj_weights = round(float(mmproj_size_gib or 0.0), 2)
    kv_cache = round(max(0.25, param_factor * 0.03 * ctx_factor), 2)
    activation = round(max(0.12, param_factor * 0.01 * min(ctx_factor, 4.0)), 2)
    framework_overhead = round(max(0.35, base * 0.08), 2)
    allocator_slack = round(max(0.15, base * 0.04), 2)
    runtime_overhead = round(framework_overhead + allocator_slack, 2)
    required = round(weights + mmproj_weights + kv_cache + activation + runtime_overhead, 2)
    confidence = "medium" if params_b is not None else "low"
    breakdown = {
        "weights_gib": weights,
        "mmproj_gib": mmproj_weights,
        "kv_cache_gib": kv_cache,
        "activation_gib": activation,
        "framework_overhead_gib": framework_overhead,
        "allocator_slack_gib": allocator_slack,
        "runtime_overhead_gib": runtime_overhead,
        "total_gib": required,
        "context_size": ctx_size,
        "assumptions": [
            "weights are approximated from local artifact file size",
            "KV cache is estimated from inferred parameter count and requested context",
            "activation and framework overhead include conservative runtime buffers and allocator slack",
        ],
    }
    return {
        "estimated_gpu_memory_gib": required,
        "estimated_ram_gib": required,
        "estimated_kv_cache_gib": kv_cache,
        "weight_and_projector_gib": round(weights + mmproj_weights, 2),
        "activation_gib": activation,
        "framework_overhead_gib": framework_overhead,
        "allocator_slack_gib": allocator_slack,
        "overhead_gib": runtime_overhead,
        "breakdown": breakdown,
        "context_size": ctx_size,
        "estimate_source": "file_size_heuristic",
        "confidence": confidence,
        "notes": [
            "Estimate uses local file size plus KV cache, activation, framework overhead, and allocator slack; backend logs or benchmark results are authoritative.",
        ],
    }


def _unknown_memory_breakdown(ctx_size: int) -> dict[str, Any]:
    return {
        "weights_gib": None,
        "mmproj_gib": None,
        "kv_cache_gib": None,
        "activation_gib": None,
        "framework_overhead_gib": None,
        "allocator_slack_gib": None,
        "runtime_overhead_gib": None,
        "total_gib": None,
        "context_size": ctx_size,
        "assumptions": ["local model size is unknown"],
    }


def _memory_breakdown_for_backend(estimate: dict[str, Any], memory_kind: str) -> dict[str, Any]:
    source = estimate.get("breakdown") if isinstance(estimate.get("breakdown"), dict) else {}
    result = dict(source)
    result["memory_kind"] = memory_kind
    result["total_gib"] = estimate.get("estimated_gpu_memory_gib") if memory_kind == "gpu" else estimate.get("estimated_ram_gib")
    return result


def _available_memory_for_backend(backend: BackendSpec) -> float | None:
    caps = set(backend.capabilities)
    if "cuda" in caps:
        devices = _filter_visible_cuda_devices(
            _query_cuda_devices(),
            os.environ.get("OMNIINFER_CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if devices:
            return max(float(device["free_gib"]) for device in devices)
        return None
    if "rocm" in caps or "hip" in caps:
        free_bytes = get_available_rocm_memory_bytes()
        return bytes_to_gib(free_bytes) if free_bytes is not None else None
    if _is_gpu_backend(backend) and "shared-memory" not in caps:
        return None
    return _available_ram_gib()


def _fit_level(required_gib: float | None, available_gib: float | None, margin_gib: float) -> str:
    if required_gib is None or available_gib is None:
        return "unknown"
    if required_gib + margin_gib <= available_gib:
        return "good"
    if required_gib <= available_gib:
        return "marginal"
    return "too_tight"


def _is_gpu_backend(backend: BackendSpec) -> bool:
    caps = set(backend.capabilities)
    return "gpu" in caps or bool(caps.intersection({"cuda", "rocm", "vulkan", "metal", "hip", "sycl"}))


def _path_size_gib(path: Path | None) -> float | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return round(path.stat().st_size / float(1024 ** 3), 2)


def _infer_quantization(text: str) -> str | None:
    upper = Path(text).name.upper()
    for quant in QUANT_PATTERNS:
        if quant in upper:
            return quant
    return None


def _infer_params_b(text: str) -> float | None:
    name = Path(text).name
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([BbMm])(?![A-Za-z0-9])", name)
    values: list[float] = []
    for raw, suffix in matches:
        value = float(raw)
        if suffix.lower() == "m":
            value /= 1000.0
        values.append(value)
    return max(values) if values else None


def _infer_model_capabilities(text: str, mmproj_path: Path | None) -> list[str]:
    lower = Path(text).name.lower()
    capabilities = ["chat"]
    if mmproj_path is not None or any(token in lower for token in ("vl", "vision", "mmproj", "multimodal")):
        capabilities.append("vision")
    if any(token in lower for token in ("embed", "embedding", "bge", "nomic")):
        capabilities.append("embedding")
    return sorted(set(capabilities))


def _query_cuda_devices() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=True,
            **hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, total_mib, free_mib, used_mib, util = parts
        try:
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "total_gib": round(int(total_mib) / 1024.0, 2),
                    "free_gib": round(int(free_mib) / 1024.0, 2),
                    "used_gib": round(int(used_mib) / 1024.0, 2),
                    "utilization_pct": int(util),
                }
            )
        except ValueError:
            continue
    return devices


def _filter_visible_cuda_devices(devices: list[dict[str, Any]], visible_filter: str | None) -> list[dict[str, Any]]:
    if not visible_filter:
        return devices
    requested = {item.strip() for item in visible_filter.split(",") if item.strip()}
    if not requested:
        return devices
    return [device for device in devices if str(device.get("index")) in requested]


def _best_cuda_device(devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not devices:
        return None
    return max(devices, key=lambda device: (float(device.get("free_gib") or 0), -int(device.get("utilization_pct") or 0)))


def _available_ram_gib() -> float | None:
    try:
        return bytes_to_gib(get_available_memory_bytes())
    except (OSError, ValueError):
        return None


def _total_ram_gib() -> float | None:
    total = get_total_memory_bytes()
    if total is None:
        return None
    return bytes_to_gib(total)


def _candidate_model_dirs(backends: dict[str, BackendSpec]) -> list[Path]:
    dirs: list[Path] = []
    for backend in backends.values():
        if backend.models_dir:
            path = Path(backend.models_dir).expanduser()
            if path.is_dir() and path not in dirs:
                dirs.append(path)
    return dirs


def _iter_local_models(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.gguf") if path.is_file() and "mmproj" not in path.name.lower())
    return files


def _task_matches_model(task: str, model_info: dict[str, Any]) -> bool:
    normalized = task.strip().lower()
    capabilities = set(model_info.get("capabilities") or [])
    model_name = str(model_info.get("model") or "").lower()
    if normalized in {"any", "chat", "general"}:
        return True
    if normalized in {"vision", "multimodal"}:
        return "vision" in capabilities
    if normalized in {"embedding", "embeddings"}:
        return "embedding" in capabilities
    if normalized == "coding":
        return any(token in model_name for token in ("coder", "code", "deepseek", "qwen"))
    return True


def _recommendation_score(candidate: dict[str, Any], model_info: dict[str, Any]) -> float:
    fit_score = {"good": 100.0, "marginal": 65.0, "too_tight": 10.0, "unknown": 20.0}.get(str(candidate.get("fit")), 20.0)
    installed_bonus = 10.0 if candidate.get("installed") else 0.0
    priority_penalty = float(candidate.get("priority") or 0) * 2.0
    size = float(model_info.get("size_gib") or 0.0)
    size_bonus = min(size, 30.0) / 3.0
    return round(fit_score + installed_bonus + size_bonus - priority_penalty, 2)


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
