"""Serve a FastWAM checkpoint over a socket for Isaac Sim closed-loop rollout.

Same policy_rpc protocol as Chapter 16: the client sends head/wrist images,
the 16D state and the instruction; the server returns a [32, 16] action chunk
(Fast-WAM predicts 32 steps per call; the evaluator consumes replan_steps).

Usage:
    python serve_policy.py \
        --checkpoint FastWAM/runs/<run>/checkpoints/weights/step_007500.pt \
        --dataset-stats FastWAM/runs/<run>/dataset_stats.json --port 8622
    # Optional IDM dual mode:
    python serve_policy.py --checkpoint ... --dataset-stats ... --action-infer-mode idm
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import traceback
from pathlib import Path

import numpy as np

from fastwam_runtime import FastWAMPolicy
from policy_rpc import pack, receive, send, unpack
from settings import ACTION_DIM

# policy_rpc request field -> LeRobot/FastWAM camera key
REQUEST_CAMERAS = {
    "head": "cam_head",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}


def save_predicted_video(frames, output: Path, fps: float = 7.5) -> None:
    """Encode the decoded future frames returned by infer_joint as H.264."""
    if not frames:
        raise ValueError("joint inference returned no video frames")
    arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    height, width = arrays[0].shape[:2]
    if any(frame.shape != (height, width, 3) for frame in arrays):
        raise ValueError("predicted video frames have inconsistent shapes")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    for frame in arrays:
        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM G2 socket inference server")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8622)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--inference-mode",
        default="action",
        choices=["action", "joint"],
        help="action: action-only Fast-WAM; joint: jointly generate future video and actions",
    )
    parser.add_argument("--measure-model-latency", action="store_true")
    parser.add_argument("--save-predicted-video-dir", type=Path, default=None)
    parser.add_argument("--save-predicted-video-limit-per-prompt", type=int, default=0)
    parser.add_argument("--seed", type=int, default=16017)
    parser.add_argument("--action-infer-mode", default=None, choices=["idm", "first_frame"],
                        help="inference mode of an Optional IDM checkpoint; empty for plain Fast-WAM")
    args = parser.parse_args()

    policy = FastWAMPolicy(
        args.checkpoint,
        args.dataset_stats,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        inference_mode=args.inference_mode,
        measure_model_latency=args.measure_model_latency,
        seed=args.seed,
        action_infer_mode=args.action_infer_mode,
    )
    saved_by_prompt: dict[str, int] = {}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(2)
        print(f"FastWAM policy ready at {args.host}:{args.port}", flush=True)
        while True:
            connection, address = server.accept()
            with connection:
                try:
                    request = unpack(receive(connection))
                    images = {
                        camera: request[name] for name, camera in REQUEST_CAMERAS.items()
                    }
                    instruction = str(request["instruction"].item())
                    actions = policy.infer(
                        images,
                        request["state"],
                        instruction,
                    )
                    if policy.last_model_latency_s is not None:
                        print(
                            f"MODEL_LATENCY_S={policy.last_model_latency_s:.6f} "
                            f"MODE={policy.inference_mode}",
                            flush=True,
                        )
                    if (
                        args.save_predicted_video_dir is not None
                        and policy.last_video_frames is not None
                        and saved_by_prompt.get(instruction, 0)
                        < args.save_predicted_video_limit_per_prompt
                    ):
                        index = saved_by_prompt.get(instruction, 0) + 1
                        saved_by_prompt[instruction] = index
                        label_match = re.search(r"\b(red|green|blue)\b", instruction.lower())
                        label = label_match.group(1) if label_match else "task"
                        video_path = args.save_predicted_video_dir / f"predicted_{label}_{index:02d}.mp4"
                        save_predicted_video(policy.last_video_frames, video_path)
                        metadata = {
                            "instruction": instruction,
                            "frames": len(policy.last_video_frames),
                            "fps": 7.5,
                            "model_latency_s": policy.last_model_latency_s,
                            "checkpoint": str(args.checkpoint),
                            "inference_mode": policy.inference_mode,
                        }
                        video_path.with_suffix(".json").write_text(
                            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
                        )
                        print(f"SAVED_PREDICTED_VIDEO={video_path}", flush=True)
                        policy.last_video_frames = None
                    response = pack(actions=actions, error=np.asarray(""))
                except Exception as exc:
                    traceback.print_exc()
                    response = pack(
                        actions=np.empty((0, ACTION_DIM), np.float32),
                        error=np.asarray(f"{type(exc).__name__}: {exc}"),
                    )
                send(connection, response)


if __name__ == "__main__":
    main()
