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
    get_available_rocm_memory_bytes,
    hidden_subprocess_kwargs,
)


DEFAULT_CONTEXT_SIZE = 8192
GPU_MEMORY_MARGIN_GIB = 0.5
CPU_MEMORY_MARGIN_GIB = 1.0
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
    installed = [backend.id for backend in backends.values() if backend.binary_exists]
    compatible = [
        backend.id
        for backend in backends.values()
        if platform_obj.is_hardware_compatible(backend)
    ]
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
            _backend_payload(backend, platform_obj, installed=set(installed))
            for backend in sorted(backends.values(), key=lambda item: BACKEND_PRIORITY.get(item.id, 999))
        ],
        "summary": {
            "installed_backends": installed,
            "compatible_backends": compatible,
            "recommended_installed_backend": _recommended_backend_id(
                [backend for backend in backends.values() if backend.binary_exists and backend.id in compatible]
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
        "alternatives": [candidate for candidate in compatible if recommended is None or candidate["backend"] != recommended["backend"]],
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
            candidates.append(
                {
                    "score": score,
                    "model": fit["model"],
                    "recommended": recommended,
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


def _backend_payload(backend: BackendSpec, platform_obj: HostPlatform, *, installed: set[str]) -> dict[str, Any]:
    return {
        "id": backend.id,
        "label": backend.label,
        "family": backend.family,
        "installed": backend.id in installed,
        "hardware_compatible": platform_obj.is_hardware_compatible(backend),
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
    required_gib = estimate["estimated_gpu_memory_gib"] if _is_gpu_backend(backend) else estimate["estimated_ram_gib"]
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
        "launch_args": launch_args,
        "priority": BACKEND_PRIORITY.get(backend.id, 999),
        "notes": notes,
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
            "estimate_source": "unknown",
            "confidence": "low",
            "notes": ["local model size is unknown; fit cannot be estimated safely"],
        }
    base = float(size_gib) + float(mmproj_size_gib or 0.0)
    ctx_factor = max(ctx_size, 1) / DEFAULT_CONTEXT_SIZE
    param_factor = params_b if params_b is not None else max(base * 2.0, 1.0)
    kv_cache = round(max(0.25, param_factor * 0.03 * ctx_factor), 2)
    overhead = round(max(0.5, base * 0.12), 2)
    required = round(base + kv_cache + overhead, 2)
    confidence = "medium" if params_b is not None else "low"
    return {
        "estimated_gpu_memory_gib": required,
        "estimated_ram_gib": required,
        "estimated_kv_cache_gib": kv_cache,
        "weight_and_projector_gib": round(base, 2),
        "overhead_gib": overhead,
        "context_size": ctx_size,
        "estimate_source": "file_size_heuristic",
        "confidence": confidence,
        "notes": [
            "Estimate uses local file size plus conservative overhead; backend logs or benchmark results are authoritative.",
        ],
    }


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
    try:
        if os.name == "nt":
            return None
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round((page_size * pages) / float(1024 ** 3), 2)
    except (AttributeError, OSError, ValueError):
        return None


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
