#!/usr/bin/env python3
"""Serve OmniInfer-VLA-Fast Pi0.5 through OmniInfer's existing ZMQ protobuf wire.

This is intentionally a small protocol bridge, not a second model runtime:
OIVF owns resize/tokenization/inference/postprocess while this process owns
only the same ``vla.proto`` request/response contract used by
``omniinfer_server.py``.  It makes a fair server-total/ZMQ-RTT comparison
possible without changing the production OmniInfer backend.
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_HOME = pathlib.Path(
    os.environ.get("OMNIINFER_VLA_RUNTIME_HOME", ROOT / "framework" / "OmniInfer-VLA")
).expanduser()


def _decode_image(image):
    from PIL import Image

    height, width, encoding = int(image.height), int(image.width), int(image.encoding)
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    if encoding == 2:  # F32_RGB_01
        values = np.frombuffer(image.data, dtype=np.float32)
        expected = height * width * 3
        if values.size != expected:
            raise ValueError(f"F32 image has {values.size} values; expected {expected}")
        return values.reshape(height, width, 3)
    if encoding == 1:  # RGB_U8
        values = np.frombuffer(image.data, dtype=np.uint8)
        expected = height * width * 3
        if values.size != expected:
            raise ValueError(
                f"RGB_U8 image has {values.size} values; expected {expected}"
            )
        return values.reshape(height, width, 3)
    if encoding == 0:  # JPEG
        with Image.open(io.BytesIO(image.data)) as decoded:
            return np.asarray(decoded.convert("RGB"))
    raise ValueError(f"unsupported image encoding enum value: {encoding}")


def _load_proto():
    runtime_text = str(RUNTIME_HOME)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    import vla_pb2

    return vla_pb2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--oivf-root", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--tactics-mode", choices=("off", "on"), default="off")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--num-flow-steps",
        type=int,
        default=None,
        help="override Pi0.5 flow steps; use 1 with --warm-start for pruning",
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # The benchmark module contains the one loader used by both in-process and
    # wire tests, so a mode cannot accidentally change checkpoint/tokenizer
    # handling between the two entry points.
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from benchmark_omniinfer_vla_fast_pi05 import _build_policy, _default_oivf_root

    oivf_root = (
        args.oivf_root.expanduser().resolve()
        if args.oivf_root is not None
        else _default_oivf_root()
    )
    tokenizer = args.tokenizer.expanduser().resolve()
    tokenizer_file = tokenizer / "tokenizer.model" if tokenizer.is_dir() else tokenizer
    if args.num_flow_steps is not None and args.num_flow_steps < 1:
        raise SystemExit("--num-flow-steps must be >= 1")
    if args.flow_start_time is not None and not 0.0 < args.flow_start_time <= 1.0:
        raise SystemExit("--flow-start-time must be in (0, 1]")
    if args.warm_start and args.num_flow_steps != 1:
        raise SystemExit("--warm-start is intended for --num-flow-steps 1")
    if args.warm_start and not 0.0 < args.warm_start_alpha < 1.0:
        raise SystemExit("--warm-start-alpha must be in (0, 1)")
    if args.warm_start and not 1 <= args.replan <= 50:
        raise SystemExit("--replan must be in 1..50")

    policy, _tactics = _build_policy(
        oivf_root,
        args.checkpoint.expanduser().resolve(),
        tokenizer_file,
        args.tactics_mode,
        args.seed,
        num_flow_steps=args.num_flow_steps,
        flow_start_time=args.flow_start_time,
    )
    pb = _load_proto()
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, 64 * 1024 * 1024)
    socket.bind(args.bind)
    print(
        f"omniinfer-vla-fast-server: bound to {args.bind}. ready. "
        f"tactics={args.tactics_mode} checkpoint={args.checkpoint}",
        flush=True,
    )
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    previous_normalized = None
    noise_rng = np.random.default_rng(args.seed + 1)
    running = True
    try:
        while running:
            if not poller.poll(200):
                continue
            body = socket.recv()
            request = pb.PredictRequest()
            response = pb.PredictResponse()
            try:
                if not request.ParseFromString(body):
                    raise ValueError("request protobuf parse failed")
                response.request_id = request.request_id
                if not request.task_text:
                    raise ValueError(
                        "OIVF bridge requires task_text (native tokenizer path); "
                        "send --native requests"
                    )
                if len(request.images) != 3:
                    raise ValueError(
                        f"expected 3 image views, got {len(request.images)}"
                    )
                observation = {
                    f"view_{index}": _decode_image(image)
                    for index, image in enumerate(request.images)
                }
                observation["state"] = np.asarray(request.state, dtype=np.float32)
                observation["prompt"] = str(request.task_text)
                noise = None
                if request.noise:
                    values = np.asarray(request.noise, dtype=np.float32)
                    if values.size != 50 * 32:
                        raise ValueError(
                            f"noise length {values.size} != expected {50 * 32}"
                        )
                    noise = values.reshape(50, 32)
                if args.warm_start and previous_normalized is not None:
                    # Warm-start uses fresh epsilon at every replan.  The
                    # client still sends a fixed latent for the baseline wire
                    # contract; once a previous chunk exists, replace it with
                    # deterministic per-request epsilon before blending.
                    noise = np.ascontiguousarray(
                        noise_rng.standard_normal(previous_normalized.shape),
                        dtype=np.float32,
                    )
                    shifted = np.empty_like(previous_normalized)
                    shifted[: -args.replan] = previous_normalized[args.replan :]
                    shifted[-args.replan :] = previous_normalized[-1]
                    noise = np.ascontiguousarray(
                        args.warm_start_alpha * shifted
                        + (1.0 - args.warm_start_alpha) * noise,
                        dtype=np.float32,
                    )
                started = time.perf_counter()
                result = policy.infer(observation, noise=noise)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.shape != (50, 7) or not np.isfinite(actions).all():
                    raise ValueError(f"bad action shape/values: {actions.shape}")
                if args.warm_start:
                    previous_normalized = np.asarray(
                        result["normalized_actions"], dtype=np.float32
                    ).copy()
                model_ms = float(result["timing"]["model_ms"])
                total_ms = float(result["timing"]["total_ms"])
                response.chunk_size = 50
                response.action_dim = 7
                response.action_chunk.extend(
                    float(value) for value in actions.reshape(-1)
                )
                response.latency_ms_total = total_ms
                response.latency_ms_inference = model_ms
                # OIVF's policy timing groups its complete pre/post chain
                # together.  Preserve the wire fields by charging that delta
                # to processor and leaving postprocess at zero.
                response.latency_ms_processor = max(0.0, total_ms - model_ms)
                response.latency_ms_postprocess = 0.0
                print(
                    f"request={request.request_id} total={total_ms:.2f} ms "
                    f"model={model_ms:.2f} ms wall={elapsed_ms:.2f} ms",
                    flush=True,
                )
            except (
                Exception
            ) as error:  # noqa: BLE001 - return protocol errors per request
                response.error = f"{type(error).__name__}: {error}"
                print(
                    f"omniinfer-vla-fast-server: request failed: {response.error}",
                    file=sys.stderr,
                    flush=True,
                )
            socket.send(response.SerializeToString())
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        context.term()
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
