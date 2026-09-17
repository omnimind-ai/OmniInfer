#!/usr/bin/env python3
"""Benchmark OmniInfer-VLA-Fast and OmniInfer-compatible Pi0.5 inputs on Thor.

The OIVF project is an optional git submodule.  This benchmark deliberately
uses the same workload as ``benchmark_omniinfer_vla_pi05.py``: three F32 RGB
frames at 480x640, the native task string, an 8-D state vector, 48-ish
PaliGemma tokens, and a fixed 50x32 diffusion latent.  It measures the complete
OIVF policy call (resize + tokenizer + model + unnormalize), so the result is
comparable with the native-Processor OmniInfer server measurement.

The standard BF16 variant is the lossless JSON-tactics path:

``on``
    BF16 with OIVF's validated Thor SM110 tactics JSON.  This changes only
    algorithm selection; it does not quantize weights or activations.

The provider-default ``off`` path is retained only for an explicit diagnostic
comparison and is not run by the standard shell benchmark.

``--mode off`` and ``--mode both`` remain available for an explicit diagnostic
comparison, but are not used by the standard shell benchmark.  The same
explicit noise is passed to every call, making the comparison deterministic
apart from GPU scheduling.

Passing ``--num-flow-steps 1 --flow-start-time 0.5 --warm-start`` enables the
actual OIVF one-step action-generation pruning path: it reduces the diffusion
solver from ten steps to one and uses the previous normalized action chunk as a
warm-start.  This is intentionally a separate, potentially lossy comparison
from the BF16 tactics switch above.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = pathlib.Path.home() / "models" / "pi05_libero_finetuned_v044"
DEFAULT_TOKENIZER = pathlib.Path.home() / "models" / "paligemma-3b-pt-224"


def _default_oivf_root() -> pathlib.Path:
    configured = os.environ.get("OMNIINFER_VLA_FAST_ROOT")
    if configured:
        return pathlib.Path(configured).expanduser().resolve()
    return (ROOT / "framework" / "OmniInfer-VLA-Fast").resolve()


def _add_import_paths(root: pathlib.Path) -> None:
    """Make a source checkout usable; an installed wheel still wins."""
    candidates = [root / "src", root / "oivf" / "python" / "oivf"]
    # A maturin build leaves the extension under target/.  This fallback keeps
    # the benchmark convenient on a development board without requiring a
    # second package install, while normal deployments should install the wheel.
    target = root / "oivf" / "target"
    if target.is_dir():
        candidates.extend(path.parent for path in target.rglob("oivf_py*.so"))
    for candidate in candidates:
        text = str(candidate)
        if candidate.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "p50_ms": ordered[len(ordered) // 2],
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": len(ordered),
    }


def _workload(task: str, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    """Build the exact raw observation used by OmniInfer's native benchmark."""
    import torch

    torch.manual_seed(20260827)
    raw_images = torch.rand(1, 3, 3, 480, 640, dtype=torch.float32)
    images = {
        f"view_{index}": np.ascontiguousarray(
            raw_images[0, index].permute(1, 2, 0).numpy()
        )
        for index in range(3)
    }
    state = (torch.rand(1, 8, dtype=torch.float32) * 2.0 - 1.0)[0].numpy()
    noise = np.ascontiguousarray(
        np.random.default_rng(seed).standard_normal((50, 32), dtype=np.float32)
    )
    observation: dict[str, Any] = {
        **images,
        "state": np.ascontiguousarray(state, dtype=np.float32),
        "prompt": task,
    }
    return observation, noise


def _build_policy(
    oivf_root: pathlib.Path,
    model_dir: pathlib.Path,
    tokenizer: pathlib.Path,
    mode: str,
    seed: int,
    *,
    num_flow_steps: int | None = None,
    flow_start_time: float | None = None,
):
    """Load one OIVF policy, explicitly controlling the tactics path."""
    _add_import_paths(oivf_root)
    import oivf_py
    from oivf import Pi05Policy
    from omniinfer_vla_fast.engine import resolve_tactics

    tactics = None
    if mode == "on":
        tactics = resolve_tactics(
            "cuda:0", "bf16", model_dir=model_dir, allow_missing=False
        )
        if tactics is None:
            raise FileNotFoundError(
                "OIVF did not resolve a Thor SM110 BF16 tactics JSON"
            )

    # Passing tactics=None is intentional: unlike Pi05Policy.from_pretrained's
    # convenience loader, the low-level binding then leaves GEMM selection to
    # the CUDA provider.  For the enabled mode we pass the exact same checkpoint
    # with only the validated tactics file changed.
    handle = oivf_py.Model.load(
        "pi05",
        str(model_dir / "model.safetensors"),
        device="cuda:0",
        precision="bf16",
        tactics=str(tactics) if tactics is not None else None,
        num_flow_steps=num_flow_steps,
        flow_start_time=flow_start_time,
        sampling_seed=seed,
    )
    policy = Pi05Policy.from_pretrained(
        model_dir,
        model=handle,
        tokenizer_path=tokenizer,
        image_keys=("view_0", "view_1", "view_2"),
        state_key="state",
        discrete_state=True,
        action_dim=7,
        prompt_key="prompt",
        seed=seed,
    )
    return policy, tactics


def _warm_start_noise(
    previous: np.ndarray,
    epsilon: np.ndarray,
    *,
    alpha: float,
    replan: int,
) -> np.ndarray:
    if previous.shape != epsilon.shape:
        raise ValueError(
            f"warm-start shape mismatch: {previous.shape} vs {epsilon.shape}"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"warm-start alpha must be in (0, 1), got {alpha}")
    if not 0 < replan <= previous.shape[0]:
        raise ValueError(f"replan must be in 1..{previous.shape[0]}, got {replan}")
    shifted = np.empty_like(previous)
    shifted[:-replan] = previous[replan:]
    shifted[-replan:] = previous[-1]
    return np.ascontiguousarray(
        alpha * shifted + (1.0 - alpha) * epsilon, dtype=np.float32
    )


def _run(
    policy,
    observation: dict[str, Any],
    noise: np.ndarray,
    warmup: int,
    timed: int,
    *,
    warm_start: bool = False,
    warm_start_alpha: float = 0.5,
    replan: int = 5,
):
    previous_normalized: np.ndarray | None = None
    noise_rng = np.random.default_rng(20260828)

    def infer_once() -> dict[str, Any]:
        nonlocal previous_normalized
        selected_noise = noise
        if warm_start and previous_normalized is not None:
            epsilon = np.ascontiguousarray(
                noise_rng.standard_normal(previous_normalized.shape), dtype=np.float32
            )
            selected_noise = _warm_start_noise(
                previous_normalized,
                epsilon,
                alpha=warm_start_alpha,
                replan=replan,
            )
        result = policy.infer(observation, noise=selected_noise)
        if warm_start:
            previous_normalized = np.asarray(
                result["normalized_actions"], dtype=np.float32
            ).copy()
        return result

    for _ in range(warmup):
        infer_once()
    values: list[float] = []
    actions: list[np.ndarray] = []
    model_values: list[float] = []
    for _ in range(timed):
        started = time.perf_counter()
        result = infer_once()
        # Policy timing is model-only vs. the complete call.  The wall clock
        # around the call is retained as a sanity check for host scheduling.
        values.append(float(result["timing"]["total_ms"]))
        model_values.append(float(result["timing"]["model_ms"]))
        actions.append(np.asarray(result["actions"], dtype=np.float32).copy())
        elapsed = (time.perf_counter() - started) * 1000.0
        if not np.isfinite(elapsed):
            raise RuntimeError("non-finite benchmark timing")
    return {
        "policy_total": _stats(values),
        "model": _stats(model_values),
        "actions": actions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oivf-root", type=pathlib.Path, default=_default_oivf_root())
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--mode", choices=("off", "on", "both"), default="on")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timed", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--num-flow-steps",
        type=int,
        default=None,
        help="override Pi0.5 flow steps; use 1 for OIVF one-step pruning",
    )
    parser.add_argument(
        "--flow-start-time",
        type=float,
        default=None,
        help="override reverse-flow start time; warm-start pruning uses 0.5",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="blend shifted previous normalized actions into one-step noise",
    )
    parser.add_argument("--warm-start-alpha", type=float, default=0.5)
    parser.add_argument("--replan", type=int, default=5)
    parser.add_argument(
        "--task",
        default="pick up the cup from table and place it in the bowl carefully",
    )
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.timed < 1:
        raise SystemExit("--warmup must be >= 0 and --timed must be >= 1")
    if args.num_flow_steps is not None and args.num_flow_steps < 1:
        raise SystemExit("--num-flow-steps must be >= 1")
    if args.flow_start_time is not None and not 0.0 < args.flow_start_time <= 1.0:
        raise SystemExit("--flow-start-time must be in (0, 1]")
    if args.warm_start and args.num_flow_steps != 1:
        raise SystemExit("--warm-start is intended for --num-flow-steps 1")
    model_dir = args.model_dir.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    tokenizer_file = tokenizer / "tokenizer.model" if tokenizer.is_dir() else tokenizer
    oivf_root = args.oivf_root.expanduser().resolve()
    for path in (model_dir / "model.safetensors", tokenizer_file):
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")
    if (
        not (oivf_root / "src").is_dir()
        or not (oivf_root / "oivf" / "python" / "oivf").is_dir()
    ):
        raise SystemExit(f"invalid OmniInfer-VLA-Fast checkout: {oivf_root}")

    observation, noise = _workload(args.task, args.seed)
    modes = ("off", "on") if args.mode == "both" else (args.mode,)
    report: dict[str, Any] = {
        "schema": "omniinfer.omniinfer_vla_fast.pi05.benchmark.v1",
        "model_dir": str(model_dir),
        "omniinfer_vla_fast_root": str(oivf_root),
        "precision": "bf16",
        "workload": {
            "images": 3,
            "raw_image_shape": [480, 640, 3],
            "action_shape": [50, 7],
            "state_dim": 8,
            "seed": args.seed,
            "task": args.task,
        },
        "warmup": args.warmup,
        "timed": args.timed,
        "num_flow_steps": args.num_flow_steps,
        "flow_start_time": args.flow_start_time,
        "warm_start": args.warm_start,
        "warm_start_alpha": args.warm_start_alpha if args.warm_start else None,
        "replan": args.replan if args.warm_start else None,
        "variant_definition": {
            "lossless_json": "BF16 validated OIVF Thor SM110 tactics JSON; no quantization",
            "lossy_pruning": "one-step diffusion with warm-start; changes model output",
        },
        "modes": {},
    }
    for mode in modes:
        variant = "lossless-json" if mode == "on" else "provider-default"
        print(f"\nOmniInfer-VLA-Fast Pi0.5 | variant={variant} | BF16", flush=True)
        policy, tactics = _build_policy(
            oivf_root,
            model_dir,
            tokenizer_file,
            mode,
            args.seed,
            num_flow_steps=args.num_flow_steps,
            flow_start_time=args.flow_start_time,
        )
        try:
            result = _run(
                policy,
                observation,
                noise,
                args.warmup,
                args.timed,
                warm_start=args.warm_start,
                warm_start_alpha=args.warm_start_alpha,
                replan=args.replan,
            )
        finally:
            policy.close()
        actions = result.pop("actions")
        report["modes"][mode] = {
            **result,
            "tactics": str(tactics) if tactics is not None else None,
            "action_reference": actions[0].tolist(),
        }
        print(
            f"policy total p50={result['policy_total']['p50_ms']:.2f} ms "
            f"mean={result['policy_total']['mean_ms']:.2f} ms | "
            f"model p50={result['model']['p50_ms']:.2f} ms",
            flush=True,
        )

    if "off" in report["modes"] and "on" in report["modes"]:
        off = np.asarray(report["modes"]["off"]["action_reference"], dtype=np.float32)
        on = np.asarray(report["modes"]["on"]["action_reference"], dtype=np.float32)
        delta = np.abs(off - on)
        report["output_delta_off_vs_on"] = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(off - on)))),
            "array_equal": bool(np.array_equal(off, on)),
        }
        print(
            "output delta off->on: "
            f"max_abs={report['output_delta_off_vs_on']['max_abs']:.3g} "
            f"MAE={report['output_delta_off_vs_on']['mean_abs']:.3g}",
            flush=True,
        )
    if args.output:
        args.output = args.output.expanduser()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
