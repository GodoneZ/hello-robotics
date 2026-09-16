"""Inspect raw/converted G2 episodes before training.

Validates every successful NPZ against the data contract, prints aggregate
statistics, and exports three-camera preview grids for visual confirmation.

Usage:
    python inspect_data.py                      # audit all NPZ under data/raw
    python inspect_data.py --preview 3          # also export 3 preview grids
    python inspect_data.py --lerobot data/lerobot/g2_pick_block_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from convert_dataset import CAMERA_FEATURES, validate_episode
from settings import (
    ACTION_DIM,
    CAMERA_KEYS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEROBOT_ROOT,
    NUM_FRAMES,
    OUTPUT_ROOT,
    RAW_ROOT,
    STATE_DIM,
)

PREVIEW_DIR = OUTPUT_ROOT / "inspect"


def episode_summary(data, path: Path) -> dict:
    """Return the key statistics of one episode."""
    frames = len(data["qpos"])
    return {
        "path": str(path),
        "split": str(data["split"]),
        "task": path.parent.name,
        "color": str(data["color"]),
        "frames": frames,
        "seconds": round(frames / int(data["raw_fps"]), 2),
        "success": bool(data["success"]),
        "instruction": str(data["instruction"]),
    }


def make_preview(data, path: Path) -> Path:
    """Tile the first/middle/last frames of all three cameras into one grid."""
    frames = len(data["qpos"])
    indices = sorted({0, frames // 2, frames - 1})
    rows = []
    for index in indices:
        row = np.concatenate(
            [np.asarray(data[key][index]) for key in CAMERA_FEATURES.values()], axis=1
        )
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / f"{path.parent.parent.name}_{path.parent.name}_{path.stem}.png"
    Image.fromarray(grid).save(out)
    return out


def inspect_raw(raw_root: Path, num_preview: int) -> dict:
    paths = sorted(raw_root.glob("*/*/episode_*.npz"))
    if not paths:
        raise RuntimeError(f"no episode_*.npz under {raw_root}; run collect_data.py first")

    successes, failures, total_frames = 0, 0, 0
    state_min = np.full(STATE_DIM, np.inf)
    state_max = np.full(STATE_DIM, -np.inf)
    command_lag = []
    per_group: dict[str, int] = {}
    previews = []
    sample_rows = []

    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            ok = "success" in data and bool(data["success"])
            if not ok:
                failures += 1
                continue
            validate_episode(data, path=path)
            successes += 1
            total_frames += len(data["qpos"])
            state_min = np.minimum(state_min, data["qpos"].min(axis=0))
            state_max = np.maximum(state_max, data["qpos"].max(axis=0))
            command_lag.append(np.abs(data["command"] - data["qpos"]))
            group = f"{data['split']}/{path.parent.name}"
            per_group[group] = per_group.get(group, 0) + 1
            if len(sample_rows) < 5:
                sample_rows.append(episode_summary(data, path))
            if len(previews) < num_preview and i % max(1, len(paths) // max(1, num_preview)) == 0:
                previews.append(str(make_preview(data, path)))

    if not command_lag:
        raise RuntimeError(f"no successful episodes under {raw_root}")
    lag = np.concatenate(command_lag)
    return {
        "raw_root": str(raw_root),
        "episodes_total": len(paths),
        "episodes_success": successes,
        "episodes_failed": failures,
        "frames_30hz": total_frames,
        "hours": round(total_frames / 30 / 3600, 2),
        "per_group_successes": per_group,
        "state_min": np.round(state_min, 4).tolist(),
        "state_max": np.round(state_max, 4).tolist(),
        "mean_abs_command_minus_state": round(float(lag.mean()), 6),
        "max_abs_command_minus_state": round(float(lag.max()), 6),
        "sample_episodes": sample_rows,
        "previews": previews,
    }


def inspect_lerobot(root: Path, num_preview: int) -> dict:
    """Read the LeRobot dataset back and check the feature contract."""
    os_env_root = root.resolve()
    if not os_env_root.exists():
        raise FileNotFoundError(f"{os_env_root} missing; run convert_dataset.py first")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=root.name, root=os_env_root)
    meta = dataset.meta
    info = {
        "root": str(os_env_root),
        "repo_id": meta.repo_id,
        "fps": meta.fps,
        "total_episodes": meta.total_episodes,
        "total_frames": meta.total_frames,
        "features": {k: v.get("shape") for k, v in meta.features.items()},
    }
    expected = [f"observation.images.{c}" for c in CAMERA_KEYS]
    missing = [k for k in (*expected, "observation.state", "action") if k not in meta.features]
    if missing:
        raise ValueError(f"LeRobot dataset missing features: {missing}")
    for key in expected:
        shape = tuple(meta.features[key]["shape"])
        if shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise ValueError(f"{key} shape {shape} != expected {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}")
    for key, dim in (("observation.state", STATE_DIM), ("action", ACTION_DIM)):
        shape = tuple(meta.features[key]["shape"])
        if shape != (dim,):
            raise ValueError(f"{key} shape {shape} != expected {dim}")

    # Decode the first frame of the first/last episode and export previews.
    previews = []
    for ep_index in sorted({0, meta.total_episodes - 1})[:num_preview]:
        start = int(dataset.episode_data_index["from"][ep_index])
        frame = dataset[start]
        row = np.concatenate(
            [(frame[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for k in expected],
            axis=1,
        )
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out = PREVIEW_DIR / f"lerobot_ep{ep_index:04d}.png"
        Image.fromarray(row).save(out)
        previews.append(str(out))
        state = frame["observation.state"].numpy()
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"episode {ep_index} first-frame state abnormal: {state.shape}")
    info["previews"] = previews
    info["window_check"] = f"Fast-WAM window needs {NUM_FRAMES} frames (9 video + 32 action)"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM G2 data audit")
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--lerobot", type=Path, default=None,
                        help="audit this LeRobot dataset instead of raw NPZ")
    parser.add_argument("--preview", type=int, default=2, help="number of preview grids")
    args = parser.parse_args()
    if args.preview < 0:
        parser.error("--preview must be non-negative")

    target = args.lerobot
    if target is None and not args.raw_dir.exists() and LEROBOT_ROOT.exists():
        candidates = sorted(p for p in LEROBOT_ROOT.iterdir() if p.is_dir())
        target = candidates[-1] if candidates else None

    if target is not None:
        report = inspect_lerobot(target, args.preview)
    else:
        report = inspect_raw(args.raw_dir, args.preview)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
