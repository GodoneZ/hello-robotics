"""Closed-loop evaluation of FastWAM on the G2 pick-and-place task in Isaac Sim.

Fast-WAM predicts a 32-step action chunk per call; following the official
RoboTwin protocol (sim_robotwin.yaml: replan_steps=24), the evaluator executes
the first 24 steps of each chunk before observing again.  The G2 scene, success
criterion and execution layer (control.py) are shared with Chapter 16.

To compare the Optional IDM modes, start two serve_policy servers with
--action-infer-mode first_frame / idm and run this script against each port —
reproducing the paper's "is test-time imagination necessary" experiment on G2.

Usage:
    python evaluate.py --port 8622 --episodes-per-color 10
    python evaluate.py --port 8623 --episodes-per-color 10 --output results/eval_idm.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from control import GraspReleaseGuard, clip_joint_limits, rate_limit_target
from environment import COLORS, FastWAMSimulationConfig, G2Robot, G2Simulation, closed_fraction
from policy_rpc import RemotePolicy
from settings import ACTION_CHUNK_SIZE, OUTPUT_ROOT, TASK_TEXT

OFFICIAL_REPLAN_STEPS = 24  # replan_steps in the official sim_robotwin.yaml


def keep_simulation_alive(sim, robot, hold_target: np.ndarray) -> None:
    """Advance/render Isaac Sim while holding the request-time robot posture."""
    hold_target = np.asarray(hold_target, np.float32)
    applied = robot.apply(hold_target)
    sim.task.update(closed_fraction(applied[15]))
    sim.step(True)


def wait_for_prediction(
    sim, robot, future: Future, hold_target: np.ndarray,
    timeout: float | None = None,
):
    """Wait for FastWAM in the RPC thread while Isaac Sim keeps rendering."""
    started = time.monotonic()
    frame_period = 1.0 / max(float(sim.cfg.render_hz), 1.0)
    while not future.done():
        tick = time.monotonic()
        keep_simulation_alive(sim, robot, hold_target)
        if timeout is not None and time.monotonic() - started > timeout:
            raise TimeoutError(f"FastWAM inference exceeded {timeout:.1f}s")
        remaining = frame_period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    return future.result()


def execute_target(sim, robot, target: np.ndarray, seconds: float = 0.1) -> None:
    start = robot.state().astype(np.float32)
    steps = max(1, round(seconds * sim.cfg.physics_hz))
    for index in range(1, steps + 1):
        alpha = index / steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        applied = robot.apply(start + alpha * (target - start))
        sim.task.update(closed_fraction(applied[15]))
        sim.step(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="closed-loop FastWAM evaluation in Isaac Sim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8622)
    parser.add_argument("--colors", nargs="*", choices=COLORS, default=list(COLORS))
    parser.add_argument("--episodes-per-color", type=int, default=10)
    parser.add_argument("--execution-mode", choices=("official", "receding"), default="official")
    parser.add_argument(
        "--execute-steps", type=int, default=None,
        help=f"actions per prediction; defaults to {OFFICIAL_REPLAN_STEPS} (official) or 4 (receding)",
    )
    parser.add_argument("--max-replans", type=int, default=20)
    parser.add_argument("--position-noise", type=float, default=0.01)
    parser.add_argument("--max-arm-step", type=float, default=0.40)
    parser.add_argument("--max-gripper-step", type=float, default=0.80)
    parser.add_argument("--action-seconds", type=float, default=1.0 / 30.0)
    parser.add_argument("--seed", type=int, default=16017)
    parser.add_argument(
        "--enable-grasp-guard", action="store_true",
        help="optional non-official right-gripper state machine (off by default)",
    )
    parser.add_argument(
        "--inference-timeout", type=float, default=900.0,
        help="maximum wait for one FastWAM RPC request while Isaac Sim keeps stepping",
    )
    parser.add_argument(
        "--pause-during-inference", action="store_true",
        help="block Isaac Sim while waiting; default keeps the display running",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "fastwam_evaluation.json")
    args = parser.parse_args()

    if args.episodes_per_color <= 0:
        parser.error("--episodes-per-color must be positive")
    if args.max_replans <= 0:
        parser.error("--max-replans must be positive")
    if args.position_noise < 0:
        parser.error("--position-noise must be non-negative")
    if args.action_seconds <= 0 or args.inference_timeout <= 0:
        parser.error("--action-seconds and --inference-timeout must be positive")
    if args.execution_mode == "official" and args.enable_grasp_guard:
        parser.error("--enable-grasp-guard is only available with --execution-mode receding")

    execute_steps = args.execute_steps
    if execute_steps is None:
        execute_steps = OFFICIAL_REPLAN_STEPS if args.execution_mode == "official" else 4
    if not 1 <= execute_steps <= ACTION_CHUNK_SIZE:
        parser.error(f"--execute-steps must be in [1,{ACTION_CHUNK_SIZE}]")

    policy = RemotePolicy(args.host, args.port, timeout=args.inference_timeout)
    rng = np.random.default_rng(args.seed)
    use_async_rpc = not args.pause_during_inference
    inference_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="fastwam-rpc")
        if use_async_rpc else None
    )
    print(
        "evaluation protocol: "
        + (f"FastWAM official action protocol (async_display={use_async_rpc}, "
             f"first {execute_steps} of each 32-action chunk, no guard/limiter)"
           if args.execution_mode == "official"
           else f"chapter receding (execute_steps={execute_steps}, async_rpc={use_async_rpc}, "
                f"grasp_guard={args.enable_grasp_guard})"),
        flush=True,
    )
    rows = []
    sim = G2Simulation(FastWAMSimulationConfig(headless=args.headless))
    try:
        robot = G2Robot(sim.articulation)
        for color in args.colors:
            for episode in range(args.episodes_per_color):
                robot.reset()
                sim.task.randomize(rng, args.position_noise)
                for _ in range(30):
                    sim.step(True)
                started = time.monotonic()
                limited = 0
                reason = "max_replans"
                replans = 0
                executed = 0
                max_raw_arm_delta = 0.0
                max_raw_gripper_delta = 0.0
                right_gripper_min = float("inf")
                right_gripper_max = float("-inf")
                grasp_guard = GraspReleaseGuard(color) if args.enable_grasp_guard else None
                if grasp_guard is not None:
                    grasp_guard.reset(color)
                try:
                    for replan in range(args.max_replans):
                        replans = replan + 1
                        if sim.task.success(color):
                            reason = "success"
                            break
                        images = sim.cameras.capture()
                        state_snapshot = robot.state().copy()
                        instruction = TASK_TEXT[color]
                        if not use_async_rpc:
                            chunk = policy.predict(images, state_snapshot, instruction)
                        else:
                            future = inference_pool.submit(
                                policy.predict, images, state_snapshot, instruction
                            )
                            chunk = wait_for_prediction(
                                sim, robot, future, state_snapshot,
                                args.inference_timeout,
                            )
                        if chunk.shape != (ACTION_CHUNK_SIZE, 16):
                            raise ValueError(
                                f"expected action chunk ({ACTION_CHUNK_SIZE},16), got {chunk.shape}"
                            )
                        for raw in chunk[:execute_steps]:
                            current = robot.state()
                            if args.execution_mode == "official":
                                target, report = clip_joint_limits(raw, current)
                            else:
                                target, report = rate_limit_target(
                                    raw, current, args.max_arm_step, args.max_gripper_step
                                )
                            guard_report = {}
                            if grasp_guard is not None:
                                target, guard_report = grasp_guard.filter_target(sim, robot, target)
                            limited += int(report["limited"])
                            max_raw_arm_delta = max(max_raw_arm_delta, report["raw_max_arm_delta"])
                            max_raw_gripper_delta = max(max_raw_gripper_delta, report["raw_max_gripper_delta"])
                            right_gripper_min = min(right_gripper_min, float(target[15]))
                            right_gripper_max = max(right_gripper_max, float(target[15]))
                            execute_target(sim, robot, target, args.action_seconds)
                            executed += 1
                            # Official mode always finishes the planned prefix of
                            # the chunk; receding mode may stop early on success.
                            if args.execution_mode != "official" and sim.task.success(color):
                                reason = "success"
                                break
                        if args.execution_mode != "official" and reason == "success":
                            break
                except Exception as exc:
                    reason = f"error:{type(exc).__name__}:{exc}"
                success = sim.task.success(color)
                final_state = robot.state()
                final_block = np.asarray(sim.task.block_position(color), dtype=float)
                row = {
                    "color": color,
                    "episode": episode,
                    "success": bool(success),
                    "reason": "success" if success else reason,
                    "execution_mode": args.execution_mode,
                    "protocol": (
                        f"fastwam_official_replan{execute_steps}"
                        if args.execution_mode == "official"
                        else "fastwam_receding"
                    ),
                    "execute_steps": execute_steps,
                    "replans": replans,
                    "executed_actions": executed,
                    "limited_actions": limited,
                    "limited_fraction": limited / max(executed, 1),
                    "max_raw_arm_delta": max_raw_arm_delta,
                    "max_raw_gripper_delta": max_raw_gripper_delta,
                    "right_gripper_target_range": [
                        None if not np.isfinite(right_gripper_min) else right_gripper_min,
                        None if not np.isfinite(right_gripper_max) else right_gripper_max,
                    ],
                    "final_right_gripper_state": float(final_state[15]),
                    "grasp_guard": None if grasp_guard is None else {
                        "phase": grasp_guard.phase,
                        "locked_actions": grasp_guard.locked_actions,
                        "release_actions": grasp_guard.release_actions,
                    },
                    "final_block_position": final_block.tolist(),
                    "inside_box": bool(sim.task.inside_box(color)),
                    "seconds": time.monotonic() - started,
                }
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                print(row, flush=True)
    finally:
        if inference_pool is not None:
            inference_pool.shutdown(wait=True, cancel_futures=True)
        sim.close()

    total = len(rows)
    successes = sum(int(r["success"]) for r in rows)
    print(f"\n[summary] success rate {successes}/{total} = {successes / max(total, 1):.1%}", flush=True)


if __name__ == "__main__":
    main()
