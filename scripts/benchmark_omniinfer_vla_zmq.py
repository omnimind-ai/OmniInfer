#!/usr/bin/env python3
"""Quick fixed-input latency benchmark for an OmniInfer VLA Runtime ZMQ server."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime_home() -> Path:
    configured = os.environ.get("OMNIINFER_VLA_RUNTIME_HOME")
    if configured:
        return Path(configured).expanduser()
    for candidate in sorted((REPO_ROOT / "framework").iterdir()):
        if (candidate / "omniinfer_server.py").is_file():
            return candidate
    raise RuntimeError("set OMNIINFER_VLA_RUNTIME_HOME to the provisioned VLA runtime home")


RUNTIME_HOME = _runtime_home()
PROTO_OVERRIDE = os.environ.get("OMNIINFER_VLA_PROTO")
PROTO = PROTO_OVERRIDE or str(RUNTIME_HOME / "vla.proto")
ARCH_DEFAULTS = {
    "pi05": {"num_images": 3, "image_size": 224},
    "gr00t_n17": {"num_images": 2, "image_size": 256},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", required=True, help="tcp://127.0.0.1:<port>")
    parser.add_argument("--arch", choices=tuple(ARCH_DEFAULTS), default="pi05")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timed", type=int, default=3)
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Use the same value used to start the OmniInfer VLA Runtime server; defaults per arch",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lang-len", type=int, default=None)
    parser.add_argument(
        "--native",
        action="store_true",
        help="Send raw images/state/task_text and use the server's native model processor.",
    )
    parser.add_argument(
        "--task",
        default="pick up the cup from table and place it in the bowl carefully",
        help="Raw task text used with --native.",
    )
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_proto():
    generated = RUNTIME_HOME / "vla_pb2.py"
    if not PROTO_OVERRIDE and generated.is_file():
        runtime_home_text = str(RUNTIME_HOME)
        if runtime_home_text not in sys.path:
            sys.path.insert(0, runtime_home_text)
        import vla_pb2

        return vla_pb2

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            [
                "protoc",
                f"--proto_path={os.path.dirname(PROTO)}",
                f"--python_out={temp_dir}",
                PROTO,
            ],
            check=True,
        )
        sys.path.insert(0, temp_dir)
        import vla_pb2

        # Keep the generated module alive after TemporaryDirectory cleanup.
        return vla_pb2


def main() -> int:
    args = parse_args()
    defaults = ARCH_DEFAULTS[args.arch]
    if args.num_images is None:
        args.num_images = defaults["num_images"]
    if args.image_size is None:
        args.image_size = defaults["image_size"]
    if args.arch == "pi05":
        state_dim, action_horizon, action_dim = (8 if args.native else 32), 50, 32
        if args.lang_len is None:
            args.lang_len = 48
    else:
        # Qwen3-VL / GR00T uses 64 placeholders per 256x256 image. A
        # non-image token between views is required by the M-RoPE layout.
        # LIBERO's native processor schema has 7 state keys but the gripper
        # key is two-dimensional, hence 8 raw state scalars in total.
        state_dim, action_horizon, action_dim = (8 if args.native else 132), 40, 132
        if args.lang_len is None:
            args.lang_len = args.num_images * 64 + 28
    if args.num_images < 1 or args.lang_len < 1:
        raise SystemExit("--num-images and --lang-len must be positive")
    if args.warmup < 0 or args.timed < 1:
        raise SystemExit("--warmup must be >= 0 and --timed must be >= 1")

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import zmq

    pb = load_proto()
    if args.image_height is None:
        args.image_height = 480 if args.native and args.arch == "pi05" else args.image_size
    if args.image_width is None:
        args.image_width = 640 if args.native and args.arch == "pi05" else args.image_size
    if args.native and args.arch == "pi05" and args.num_images != 3:
        raise SystemExit("native Pi0.5 benchmark requires exactly 3 image views")
    # Match pi05_fair_benchmark.py's raw input construction exactly for the
    # native Pi0.5 path.  Its fair engine-only benchmark times after this
    # processing stage; here the server intentionally runs that stage so an
    # OmniInfer deployment can submit real task/state/image observations.
    if args.native and args.arch == "pi05":
        import torch

        torch.manual_seed(20260827)
        raw_images = torch.rand(1, 3, 3, 480, 640, dtype=torch.float32)
        native_images = [
            np.ascontiguousarray(raw_images[0, view].permute(1, 2, 0).numpy())
            for view in range(3)
        ]
        native_state = (torch.rand(1, 8, dtype=torch.float32) * 2.0 - 1.0)[0]
        state_payload = native_state.tolist()
        noise_payload = np.random.default_rng(20260827).standard_normal(
            (50, 32), dtype=np.float32
        ).reshape(-1).tolist()
        input_source = "pi05_fair_benchmark.py seed=20260827"
    else:
        native_images = None
        state_payload = [0.0] * state_dim
        noise_payload = [0.0] * (action_horizon * action_dim)
        input_source = "fixed_zero_observation"
    image_data = b"\x00" * (args.image_height * args.image_width * 3 * 4)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(args.addr)
    socket.setsockopt(zmq.RCVTIMEO, 120_000)

    server_values: list[float] = []
    processor_values: list[float] = []
    engine_values: list[float] = []
    postprocess_values: list[float] = []
    wall_values: list[float] = []
    total = args.warmup + args.timed
    try:
        for index in range(total):
            if args.native:
                language_tokens = []
            elif args.arch == "pi05":
                language_tokens = [1] * args.lang_len
            else:
                image_token_id = 151655
                language_tokens = []
                for view_index in range(args.num_images):
                    language_tokens.extend([image_token_id] * 64)
                    if view_index + 1 < args.num_images:
                        language_tokens.append(1)
                language_tokens.extend(
                    [0] * (args.lang_len - len(language_tokens))
                )
            request = pb.PredictRequest(
                request_id=index + 1,
                lang_tokens=language_tokens,
                state=state_payload,
                noise=noise_payload,
            )
            if args.native:
                request.task_text = args.task
            for view in range(args.num_images):
                data = (
                    native_images[view].tobytes()
                    if native_images is not None
                    else image_data
                )
                request.images.add(
                    encoding=pb.Image.F32_RGB_01,
                    height=args.image_height,
                    width=args.image_width,
                    data=data,
                )

            started = time.perf_counter()
            socket.send(request.SerializeToString())
            response = pb.PredictResponse.FromString(socket.recv())
            wall_ms = (time.perf_counter() - started) * 1000.0
            if response.error:
                raise RuntimeError(response.error)

            print(
                f"run={index + 1} "
                f"{'warmup' if index < args.warmup else 'timed'} "
                f"server={response.latency_ms_total:.2f} ms "
                f"processor={response.latency_ms_processor:.2f} ms "
                f"engine={response.latency_ms_inference:.2f} ms "
                f"postprocess={response.latency_ms_postprocess:.2f} ms "
                f"wall={wall_ms:.2f} ms "
                f"shape={response.chunk_size}x{response.action_dim}",
                flush=True,
            )
            if index >= args.warmup:
                server_values.append(float(response.latency_ms_total))
                processor_values.append(float(response.latency_ms_processor))
                engine_values.append(float(response.latency_ms_inference))
                postprocess_values.append(float(response.latency_ms_postprocess))
                wall_values.append(wall_ms)
    finally:
        socket.close(0)
        context.term()

    result = {
        "address": args.addr,
        "num_images": args.num_images,
        "image_size": args.image_size,
        "lang_len": args.lang_len,
        "input_mode": "native_processor" if args.native else "prepared_tokens",
        "task": args.task if args.native else None,
        "input_source": input_source,
        "measurement": {
            "processor": "native PI05Processor.preprocess or GR00TProcessor.process_observation, CUDA-synchronized",
            "engine": "OmniInfer VLA Runtime engine step, CUDA-synchronized",
            "postprocess": "native action postprocess/decode, CUDA-synchronized",
            "server_total": "wire/input adaptation + processor + engine + postprocess",
            "wall": "client ZeroMQ round-trip including protobuf serialization",
        },
        "image_height": args.image_height,
        "image_width": args.image_width,
        "warmup": args.warmup,
        "timed": args.timed,
        "server_mean_ms": statistics.fmean(server_values),
        "server_median_ms": statistics.median(server_values),
        "processor_mean_ms": statistics.fmean(processor_values),
        "processor_median_ms": statistics.median(processor_values),
        "engine_mean_ms": statistics.fmean(engine_values),
        "engine_median_ms": statistics.median(engine_values),
        "postprocess_mean_ms": statistics.fmean(postprocess_values),
        "postprocess_median_ms": statistics.median(postprocess_values),
        "wall_mean_ms": statistics.fmean(wall_values),
        "wall_median_ms": statistics.median(wall_values),
    }
    print(f"server_mean={result['server_mean_ms']:.2f} ms")
    print(f"server_median={result['server_median_ms']:.2f} ms")
    print(f"processor_mean={result['processor_mean_ms']:.2f} ms")
    print(f"processor_median={result['processor_median_ms']:.2f} ms")
    print(f"engine_mean={result['engine_mean_ms']:.2f} ms")
    print(f"engine_median={result['engine_median_ms']:.2f} ms")
    print(f"postprocess_mean={result['postprocess_mean_ms']:.2f} ms")
    print(f"postprocess_median={result['postprocess_median_ms']:.2f} ms")
    print(f"wall_mean={result['wall_mean_ms']:.2f} ms")
    print(f"wall_median={result['wall_median_ms']:.2f} ms")
    if args.output:
        import json
        from pathlib import Path

        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
