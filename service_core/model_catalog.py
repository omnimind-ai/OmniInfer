from __future__ import annotations

import copy
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from service_core.backends import BACKEND_PRIORITY
from service_core.platforms import HostPlatform, SYSTEM_MODEL_LIST_URLS, parse_size_gib


class SupportedModelCatalog:
    def __init__(self, platform: HostPlatform, available_backend_ids: Iterable[str]) -> None:
        self.platform = platform
        self.available_backend_ids = set(available_backend_ids)

    def list_supported_models(self, system_name: str) -> dict[str, Any]:
        return self._annotated_system_catalog(system_name)

    def list_supported_models_best(self, system_name: str) -> dict[str, Any]:
        return self._merge_best_supported_models(self._annotated_system_catalog(system_name))

    def auto_select_backend_for_model(self, model_path: str, mmproj_path: str | None) -> str:
        system_name = self.platform.system_name
        annotated_catalog = self._annotated_system_catalog(system_name)
        model_basename = Path(model_path).name.lower()
        mmproj_basename = Path(mmproj_path).name.lower() if mmproj_path else None

        candidates = [
            candidate
            for candidate in self._flatten_catalog_candidates(annotated_catalog)
            if candidate["model_basename"] == model_basename
            and (mmproj_basename is None or candidate["mmproj_basename"] in (None, mmproj_basename))
        ]
        if not candidates:
            raise ValueError(
                f"model not found in the {system_name} catalog for any installed backend: "
                f"{Path(model_path).name}"
            )

        best_candidate = min(candidates, key=self._candidate_rank)
        if not best_candidate["suitable"]:
            raise RuntimeError(
                "no suitable backend was found for this model on the current device; "
                f"best candidate is {best_candidate['backend']} and requires about "
                f"{best_candidate['required_memory_gib']} GiB of memory"
            )
        return str(best_candidate["backend"])

    def _fetch_system_catalog(self, system_name: str) -> dict[str, Any]:
        system_key = (system_name or "").strip().lower()
        if system_key not in SYSTEM_MODEL_LIST_URLS:
            raise ValueError("field 'system' must be one of: windows, mac, linux")
        req = urllib.request.Request(
            SYSTEM_MODEL_LIST_URLS[system_key],
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8-sig"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid model catalog payload for system: {system_key}")
        return payload

    def _annotate_supported_models(self, payload: Any, available_memory_gib: float, safety_margin_gib: float) -> Any:
        if isinstance(payload, dict):
            quantizations = payload.get("quantization")
            if isinstance(quantizations, dict):
                vision = payload.get("vision")
                vision_size_gib = parse_size_gib(vision.get("size")) if isinstance(vision, dict) else 0.0
                annotated: dict[str, Any] = {}
                for key, value in payload.items():
                    if key != "quantization":
                        annotated[key] = self._annotate_supported_models(value, available_memory_gib, safety_margin_gib)
                        continue

                    quantization_payload: dict[str, Any] = {}
                    for quant_name, quant_info in value.items():
                        if not isinstance(quant_info, dict):
                            quantization_payload[quant_name] = quant_info
                            continue
                        required_memory_gib = round(parse_size_gib(quant_info.get("size")) + vision_size_gib, 2)
                        quantization_payload[quant_name] = {
                            **quant_info,
                            "required_memory_gib": required_memory_gib,
                            "suitable": available_memory_gib >= round(required_memory_gib + safety_margin_gib, 2),
                        }
                    annotated[key] = quantization_payload
                return annotated

            return {
                key: self._annotate_supported_models(value, available_memory_gib, safety_margin_gib)
                for key, value in payload.items()
            }

        if isinstance(payload, list):
            return [self._annotate_supported_models(item, available_memory_gib, safety_margin_gib) for item in payload]

        return payload

    def _annotated_system_catalog(self, system_name: str) -> dict[str, Any]:
        catalog = self._fetch_system_catalog(system_name)
        annotated: dict[str, Any] = {}
        for backend_name, backend_payload in catalog.items():
            annotated[backend_name] = self._annotate_supported_models(
                backend_payload,
                self.platform.available_memory_gib_for_backend(backend_name),
                self.platform.safety_margin_gib_for_backend(backend_name),
            )
        return annotated

    def _candidate_rank(self, candidate: dict[str, Any]) -> tuple[int, int]:
        return (
            0 if candidate.get("suitable") else 1,
            BACKEND_PRIORITY.get(str(candidate.get("backend")), 999),
        )

    def _merge_best_supported_models(self, annotated_catalog: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        quantization_candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

        for backend_name, backend_payload in annotated_catalog.items():
            runtime_backend_id = self.platform.resolve_catalog_backend_id(backend_name)
            if runtime_backend_id not in self.available_backend_ids:
                continue
            if not isinstance(backend_payload, dict):
                continue
            for family_name, family_models in backend_payload.items():
                if not isinstance(family_models, dict):
                    continue
                target_family = merged.setdefault(family_name, {})
                for model_name, model_info in family_models.items():
                    if not isinstance(model_info, dict):
                        continue
                    target_model = target_family.setdefault(model_name, {})
                    for key, value in model_info.items():
                        if key == "quantization":
                            continue
                        if key not in target_model:
                            target_model[key] = copy.deepcopy(value)

                    quantizations = model_info.get("quantization")
                    if not isinstance(quantizations, dict):
                        continue
                    target_quantizations = target_model.setdefault("quantization", {})
                    for quant_name, quant_info in quantizations.items():
                        if not isinstance(quant_info, dict):
                            continue
                        candidate = {
                            "backend": runtime_backend_id,
                            "payload": copy.deepcopy(quant_info),
                            "required_memory_gib": quant_info.get("required_memory_gib"),
                            "suitable": bool(quant_info.get("suitable")),
                        }
                        candidate_key = (family_name, model_name, quant_name)
                        quantization_candidates.setdefault(candidate_key, []).append(candidate)
                        target_quantizations.setdefault(quant_name, copy.deepcopy(quant_info))

        for (family_name, model_name, quant_name), candidates in quantization_candidates.items():
            target_quant = merged[family_name][model_name]["quantization"][quant_name]
            suitable_candidates = [candidate for candidate in candidates if candidate["suitable"]]
            if suitable_candidates:
                best_candidate = min(suitable_candidates, key=self._candidate_rank)
                target_quant.clear()
                target_quant.update(best_candidate["payload"])
                target_quant["backend"] = str(best_candidate["backend"])
                continue

            best_candidate = min(candidates, key=self._candidate_rank)
            target_quant.clear()
            target_quant.update(best_candidate["payload"])
            target_quant["backend"] = ""

        return merged

    def _flatten_catalog_candidates(self, annotated_catalog: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for backend_name, backend_payload in annotated_catalog.items():
            runtime_backend_id = self.platform.resolve_catalog_backend_id(backend_name)
            if runtime_backend_id not in self.available_backend_ids or not isinstance(backend_payload, dict):
                continue
            for family_name, family_models in backend_payload.items():
                if not isinstance(family_models, dict):
                    continue
                for model_name, model_info in family_models.items():
                    if not isinstance(model_info, dict):
                        continue
                    quantizations = model_info.get("quantization")
                    if not isinstance(quantizations, dict):
                        continue
                    vision = model_info.get("vision")
                    mmproj_download = vision.get("download") if isinstance(vision, dict) else None
                    mmproj_basename = Path(str(mmproj_download)).name.lower() if mmproj_download else None
                    for quant_name, quant_info in quantizations.items():
                        if not isinstance(quant_info, dict):
                            continue
                        download = quant_info.get("download")
                        if not download:
                            continue
                        candidates.append(
                            {
                                "backend": runtime_backend_id,
                                "family": family_name,
                                "model_name": model_name,
                                "quantization": quant_name,
                                "model_basename": Path(str(download)).name.lower(),
                                "mmproj_basename": mmproj_basename,
                                "required_memory_gib": quant_info.get("required_memory_gib"),
                                "suitable": bool(quant_info.get("suitable")),
                            }
                        )
        return candidates
