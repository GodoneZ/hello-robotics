"""Offline unit tests: no Isaac Sim, GPU or lerobot package required.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_data import balanced_quotas  # noqa: E402
from convert_dataset import resolve_dataset_output, validate_episode  # noqa: E402
from image_layout import concat_robotwin_layout  # noqa: E402
from settings import ACTION_DIM, CAMERA_KEYS, NUM_FRAMES, STATE_DIM  # noqa: E402


def _synthetic_episode(frames: int = 40) -> dict:
    rng = np.random.default_rng(0)
    return {
        "head_image": rng.integers(0, 255, (frames, 240, 320, 3), np.uint8),
        "left_image": rng.integers(0, 255, (frames, 240, 320, 3), np.uint8),
        "right_image": rng.integers(0, 255, (frames, 240, 320, 3), np.uint8),
        "qpos": rng.standard_normal((frames, STATE_DIM)).astype(np.float32),
        "command": rng.standard_normal((frames, ACTION_DIM)).astype(np.float32),
        "instruction": np.asarray("Pick up the red block and place it into the empty box."),
        "color": np.asarray("red"),
        "split": np.asarray("clean"),
        "success": np.asarray(True),
        "raw_fps": np.asarray(30, np.int64),
    }


def test_balanced_quotas_exact_split():
    assert balanced_quotas(150, ["red", "green", "blue"]) == {
        "red": 50, "green": 50, "blue": 50,
    }


def test_balanced_quotas_remainder_deterministic():
    quotas = balanced_quotas(151, ["red", "green", "blue"])
    assert sum(quotas.values()) == 151
    assert quotas["red"] == 51 and quotas["green"] == 50


def test_validate_episode_accepts_synthetic():
    validate_episode(_synthetic_episode(), path=Path("episode_000000.npz"))


def test_validate_episode_rejects_short_window():
    data = _synthetic_episode(frames=NUM_FRAMES - 1)
    try:
        validate_episode(data, path=Path("episode_000000.npz"))
    except ValueError as exc:
        assert "window" in str(exc)
    else:
        raise AssertionError("episodes shorter than the Fast-WAM window must be rejected")


def test_validate_episode_rejects_wrong_fps():
    data = _synthetic_episode()
    data["raw_fps"] = np.asarray(15, np.int64)
    try:
        validate_episode(data, path=Path("episode_000000.npz"))
    except ValueError as exc:
        assert "Hz" in str(exc)
    else:
        raise AssertionError("mismatched recording fps must be rejected")


def test_robotwin_concat_layout_shape():
    rng = np.random.default_rng(1)
    images = {key: rng.integers(0, 255, (240, 320, 3), np.uint8) for key in CAMERA_KEYS}
    frame = concat_robotwin_layout(images)
    assert frame.shape == (384, 320, 3)
    assert frame.dtype == np.uint8


def test_resolve_dataset_output_confined(tmp_path):
    out = resolve_dataset_output(tmp_path, "g2_pick_block_lerobot")
    assert out == (tmp_path / "g2_pick_block_lerobot").resolve()
    for bad in ("/abs/path", "../escape"):
        try:
            resolve_dataset_output(tmp_path, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"repo_id={bad} must be rejected")
