"""Matched LIBERO rollouts for OmniInfer-VLA and Fast one-step pruning.

Uses the LeRobot v044 contract: two oriented cameras plus a black third view,
8-D state (both finger joints), 50x32 normalized actions and 50x7 decoded actions.
Both backends use one flow step at t=0.5 and identical warm-start noise seeds.
The caller must provide a working LIBERO installation/configuration.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _fix_numpy_enum_addresses():
    """Handle NumPy 2.x / pybind enum membership in old robosuite bindings.

    This only translates joint types to their qpos/qvel widths; it does not
    alter the model, dynamics, observations, actions or success conditions.
    """
    import mujoco
    from robosuite.utils.binding_utils import MjModel

    if np.int32(int(mujoco.mjtJoint.mjJNT_HINGE)) in (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
    ):
        return

    def address(model, name, velocity=False):
        joint = model.joint_name2id(name)
        kind = int(model.jnt_type[joint])
        widths = {
            int(mujoco.mjtJoint.mjJNT_FREE): 6 if velocity else 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 3 if velocity else 4,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        }
        width = widths[kind]
        start = int((model.jnt_dofadr if velocity else model.jnt_qposadr)[joint])
        return start if width == 1 else (start, start + width)

    MjModel.get_joint_qpos_addr = lambda self, name: address(self, name)
    MjModel.get_joint_qvel_addr = lambda self, name: address(self, name, True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument(
        "--trial-start",
        type=int,
        default=0,
        help="first trial index (inclusive), useful for resuming a long run",
    )
    p.add_argument(
        "--trial-end",
        type=int,
        default=None,
        help="last trial index (exclusive); defaults to --trials",
    )
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "models/pi05_libero_finetuned_v044",
    )
    p.add_argument(
        "--tokenizer", type=Path, default=Path.home() / "models/paligemma-3b-pt-224"
    )
    a = p.parse_args()
    trial_end = a.trials if a.trial_end is None else a.trial_end
    if not 0 <= a.trial_start < trial_end:
        p.error("require 0 <= --trial-start < --trial-end/--trials")
    if trial_end > 50:
        p.error("trial end must be <= 50 for the LIBERO init-state set")
    if a.output.exists():
        p.error("use a new output path to preserve previous results")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "framework/OmniInfer-VLA"))
    import omniinfer_server as server
    from benchmark_omniinfer_vla_fast_pi05 import _build_policy, _warm_start_noise
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from omniinfer_vla_fast.envs.libero import quat_to_axis_angle

    _fix_numpy_enum_addresses()

    old_argv = sys.argv
    sys.argv = [
        "eval",
        "--bind",
        "tcp://127.0.0.1:0",
        "--checkpoint",
        str(a.checkpoint),
        "--arch",
        "pi05",
        "--num-images",
        "3",
        "--processor-mode",
        "native",
        "--pi05-tokenizer",
        str(a.tokenizer),
        "--warm-start",
    ]
    try:
        args = server._parse_args()
    finally:
        sys.argv = old_argv
    pb = server._load_proto()
    model = server.OmniInferVlaModelServer(args, pb)
    policy, _ = _build_policy(
        root / "framework/OmniInfer-VLA-Fast",
        a.checkpoint,
        a.tokenizer / "tokenizer.model",
        "on",
        args.seed,
        num_flow_steps=1,
        flow_start_time=0.5,
    )
    suite = benchmark.get_benchmark_dict()[a.suite]()
    records = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for task_id in map(int, a.tasks.split(",")):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            for trial in range(a.trial_start, trial_end):
                for name in ("fast", "omni"):
                    env = OffScreenRenderEnv(
                        bddl_file_name=str(
                            Path(get_libero_path("bddl_files"))
                            / task.problem_folder
                            / task.bddl_file
                        ),
                        camera_heights=256,
                        camera_widths=256,
                    )
                    env.seed(a.seed + task_id * 1000 + trial)
                    model.reset_warm_start()
                    previous = None
                    rng = np.random.default_rng(args.seed + 1)
                    first_noise = (
                        np.random.default_rng(a.seed + task_id * 1000 + trial)
                        .standard_normal((50, 32))
                        .astype(np.float32)
                    )
                    timings = []
                    replans = 0
                    success = False
                    started = time.perf_counter()
                    try:
                        env.reset()
                        obs = env.set_init_state(initial_states[trial])
                        for _ in range(10):
                            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                        for step in range(0, a.max_steps, 5):
                            images = [
                                np.ascontiguousarray(obs[key][::-1, ::-1]).astype(
                                    np.float32
                                )
                                / 255.0
                                for key in (
                                    "agentview_image",
                                    "robot0_eye_in_hand_image",
                                )
                            ]
                            images.append(np.zeros_like(images[0]))
                            state = np.concatenate(
                                [
                                    obs["robot0_eef_pos"],
                                    quat_to_axis_angle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                ]
                            ).astype(np.float32)
                            tick = time.perf_counter()
                            if name == "fast":
                                noise = (
                                    first_noise
                                    if previous is None
                                    else _warm_start_noise(
                                        previous,
                                        rng.standard_normal(previous.shape).astype(
                                            np.float32
                                        ),
                                        alpha=0.5,
                                        replan=5,
                                    )
                                )
                                result = policy.infer(
                                    {
                                        **{
                                            f"view_{i}": im
                                            for i, im in enumerate(images)
                                        },
                                        "state": state,
                                        "prompt": task.language,
                                    },
                                    noise=noise,
                                )
                                actions = np.asarray(result["actions"])
                                previous = np.asarray(
                                    result["normalized_actions"], dtype=np.float32
                                ).copy()
                            else:
                                req = pb.PredictRequest(
                                    request_id=replans + 1,
                                    task_text=task.language,
                                    state=state.tolist(),
                                    noise=first_noise.ravel().tolist(),
                                )
                                for im in images:
                                    req.images.add(
                                        encoding=pb.Image.F32_RGB_01,
                                        height=256,
                                        width=256,
                                        data=im.tobytes(),
                                    )
                                actions, _, _ = model.predict(req)
                            timings.append((time.perf_counter() - tick) * 1000)
                            if (
                                actions.shape != (50, 7)
                                or not np.isfinite(actions).all()
                            ):
                                raise RuntimeError("invalid actions")
                            replans += 1
                            for action in actions[: min(5, a.max_steps - step)]:
                                obs, _, done, _ = env.step(action.tolist())
                                if done:
                                    success = True
                                    break
                            if success:
                                break
                        row = {
                            "backend": name,
                            "suite": a.suite,
                            "task": task_id,
                            "trial": trial,
                            "success": success,
                            "replans": replans,
                            "policy_mean_ms": float(np.mean(timings)),
                            "elapsed_s": time.perf_counter() - started,
                        }
                        records.append(row)
                        with a.output.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        print("RESULT " + json.dumps(row), flush=True)
                    finally:
                        env.close()
    finally:
        model.close()
        policy.close()
    print(
        json.dumps(
            {
                name: {
                    "success": sum(
                        r["success"] for r in records if r["backend"] == name
                    ),
                    "episodes": sum(r["backend"] == name for r in records),
                }
                for name in ("fast", "omni")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
