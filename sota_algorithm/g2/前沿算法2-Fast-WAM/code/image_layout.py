"""Three-camera RoboTwin layout stitching (pure NumPy/PIL, shared by train/deploy).

Official layout: the head camera resized to 256x320 on top, the two wrist
cameras resized to 128x160 side by side below, forming one 384x320 frame —
the same geometry as FastWAM's training-side concat_multi_camera="robotwin".
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from settings import CAMERA_KEYS

HEAD_SIZE_WH = (320, 256)
WRIST_SIZE_WH = (160, 128)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil_image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def concat_robotwin_layout(images: dict[str, np.ndarray]) -> np.ndarray:
    """Three HWC uint8 camera frames -> one 384x320x3 frame (robotwin layout)."""
    missing = [key for key in CAMERA_KEYS if key not in images]
    if missing:
        raise KeyError(f"missing camera images: {missing}")
    head = _resize_rgb(images["cam_head"], HEAD_SIZE_WH)
    left = _resize_rgb(images["cam_left_wrist"], WRIST_SIZE_WH)
    right = _resize_rgb(images["cam_right_wrist"], WRIST_SIZE_WH)
    bottom = np.concatenate([left, right], axis=1)
    return np.concatenate([head, bottom], axis=0)
