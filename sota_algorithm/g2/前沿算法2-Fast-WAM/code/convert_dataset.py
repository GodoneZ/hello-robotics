"""Convert collected NPZ episodes into a FastWAM-readable LeRobot 2.1 dataset.

Input: raw episodes from collect_data.py (data/raw/{split}/{task}/episode_*.npz).
Output: a LeRobot dataset matching configs/data/g2_3cam.yaml — three cameras
(cam_head/cam_left_wrist/cam_right_wrist at 240x320), 16D observation.state
(measured qpos) and 16D action (expert command).
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from settings import (
    ACTION_DIM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEROBOT_ROOT,
    NUM_FRAMES,
    RAW_FPS,
    RAW_ROOT,
    STATE_DIM,
)

REPO_ID = "g2_pick_block_lerobot"

FRAME_KEYS = ("head_image", "left_image", "right_image", "qpos", "command")
IMAGE_KEYS = FRAME_KEYS[:3]
EXPECTED_IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

# LeRobot feature key -> NPZ field
CAMERA_FEATURES = {
    "cam_head": "head_image",
    "cam_left_wrist": "left_image",
    "cam_right_wrist": "right_image",
}

JOINT_NAMES = [
    "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4",
    "left_arm_5", "left_arm_6", "left_arm_7",
    "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
    "right_arm_5", "right_arm_6", "right_arm_7",
    "left_gripper", "right_gripper",
]


def validate_episode(data, *, path: Path) -> None:
    """Fully check one successful episode before it enters video encoding."""
    missing = [
        key
        for key in (*FRAME_KEYS, "raw_fps", "instruction", "color", "success")
        if key not in data
    ]
    if missing:
        raise ValueError(f"{path.name} missing fields: {missing}")
    if not bool(data["success"]):
        raise ValueError(f"{path.name} is not a successful trajectory")
    if int(data["raw_fps"]) != RAW_FPS:
        raise ValueError(
            f"{path.name} recorded at {int(data['raw_fps'])} Hz, expected {RAW_FPS} Hz"
        )

    frames = int(data["qpos"].shape[0])
    if frames < NUM_FRAMES:
        raise ValueError(f"{path.name} has {frames} frames, below the {NUM_FRAMES}-frame window")
    lengths = {key: int(data[key].shape[0]) for key in FRAME_KEYS}
    if set(lengths.values()) != {frames}:
        raise ValueError(f"{path.name} frame-count mismatch: {lengths}")

    for key, dim in (("qpos", STATE_DIM), ("command", ACTION_DIM)):
        value = np.asarray(data[key])
        if value.shape != (frames, dim) or not np.isfinite(value).all():
            raise ValueError(f"{path.name} bad {key} shape/values: {value.shape}")
    for key in IMAGE_KEYS:
        value = np.asarray(data[key])
        if value.shape != (frames, *EXPECTED_IMAGE_SHAPE) or value.dtype != np.uint8:
            raise ValueError(
                f"{path.name} bad {key} image format: {value.shape}/{value.dtype}"
            )

    if not str(data["instruction"]).strip():
        raise ValueError(f"{path.name} has an empty instruction")


def resolve_dataset_output(lerobot_home: Path, repo_id: str) -> Path:
    """Confine the dataset output inside the LeRobot root so --overwrite is safe."""
    root = Path(lerobot_home).expanduser().resolve()
    relative = Path(repo_id)
    if relative.is_absolute():
        raise ValueError("--repo-id must be a repository name relative to --lerobot-home")
    output = (root / relative).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError(f"dataset output must stay inside {root}, got {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZ -> LeRobot 2.1 (FastWAM G2)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--lerobot-home", type=Path, default=LEROBOT_ROOT)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for path in sorted(args.raw_dir.glob("*/*/episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "success" in data and bool(data["success"]):
                episodes.append(path)
            else:
                print(f"[skip] {path.relative_to(args.raw_dir)} not successful")
    if not episodes:
        raise RuntimeError("no convertible episodes; run collect_data.py first")

    task_counts = Counter()
    for path in episodes:
        with np.load(path, allow_pickle=False) as data:
            validate_episode(data, path=path)
        task_counts[path.parent.name] += 1
    print(f"[stats] successful episodes per task: {dict(task_counts)}, total {len(episodes)}")

    output = resolve_dataset_output(args.lerobot_home, args.repo_id)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to rebuild")
        shutil.rmtree(output)

    features = {
        "observation.state": {
            "dtype": "float32", "shape": (STATE_DIM,), "names": [JOINT_NAMES],
        },
        "action": {
            "dtype": "float32", "shape": (ACTION_DIM,), "names": [JOINT_NAMES],
        },
    }
    for camera in CAMERA_FEATURES:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": EXPECTED_IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        robot_type="G2_omnipicker",
        fps=RAW_FPS,
        features=features,
        use_videos=True,
        image_writer_threads=8,
    )
    for path in episodes:
        with np.load(path, allow_pickle=False) as data:
            frame = {
                "observation.state": data["qpos"],
                "action": data["command"],
            }
            images = {
                f"observation.images.{camera}": data[key]
                for camera, key in CAMERA_FEATURES.items()
            }
            instruction = str(data["instruction"])
            for index in range(len(data["qpos"])):
                dataset.add_frame(
                    {
                        **{key: value[index] for key, value in images.items()},
                        "observation.state": frame["observation.state"][index],
                        "action": frame["action"][index],
                        "task": instruction,
                    }
                )
            dataset.save_episode()
    print(f"[done] LeRobot dataset: {output}, {len(episodes)} episodes")


if __name__ == "__main__":
    main()
