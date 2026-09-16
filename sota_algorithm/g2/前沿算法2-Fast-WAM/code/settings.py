"""Chapter paths, task definitions and the Fast-WAM data contract for G2.

The G2 embodiment interface (16D state/action) and the three-color pick-and-place
task are shared with Chapter 15/16.  Timing, camera layout and dataset format
follow FastWAM's official configs (configs/data/robotwin.yaml); the LeRobot
dataset format stays on v2.1, same as Chapter 14 and the ACoT-VLA chapter.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FASTWAM_ROOT = Path(os.getenv("FASTWAM_ROOT", ROOT / "FastWAM")).expanduser().resolve()
WEIGHTS_ROOT = Path(os.getenv("FASTWAM_G2_WEIGHTS", ROOT / "weights")).expanduser().resolve()
DATA_ROOT = Path(os.getenv("FASTWAM_G2_DATA", ROOT / "data")).expanduser().resolve()
RAW_ROOT = DATA_ROOT / "raw"
LEROBOT_ROOT = DATA_ROOT / "lerobot"
TEXT_CACHE_ROOT = DATA_ROOT / "text_embeds_cache"
CHECKPOINT_ROOT = ROOT / "checkpoints"
OUTPUT_ROOT = ROOT / "results"
LOG_ROOT = OUTPUT_ROOT / "logs"
TASK_RUNTIME_ROOT = ROOT / "task_runtime"

# Wan2.2-TI2V-5B is shared with Chapter 16 and can be reused via symlink.
WAN_ROOT = WEIGHTS_ROOT / "Wan2.2-TI2V-5B"
ACTION_DIT_INIT = CHECKPOINT_ROOT / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

# ---- Task definition (same as Chapter 15/16: G2 Omnipicker three-color pick-place) ----
COLORS = ("red", "green", "blue")
TASK_NAMES = {color: f"pick_{color}_block" for color in COLORS}
TASK_TEXT = {
    color: f"Pick up the {color} block and place it into the empty box."
    for color in COLORS
}

# ---- G2 embodiment interface (14 arm joints + 2 grippers, same as Chapter 16) ----
STATE_DIM = 16
ACTION_DIM = 16
RAW_FPS = 30

# ---- Fast-WAM temporal structure (official: 33 obs frames = 9 video + 32 action) ----
NUM_FRAMES = 33
ACTION_VIDEO_FREQ_RATIO = 4
NUM_VIDEO_FRAMES = 9
ACTION_CHUNK_SIZE = 32
GLOBAL_SAMPLE_STRIDE = 1

# ---- Three-camera layout (official RoboTwin: head + wrists -> 384x320) ----
CAMERA_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
VIDEO_SIZE = (384, 320)
CONCAT_MULTI_CAMERA = "robotwin"

INFERENCE_STEPS = 10

# ---- Collection quotas (single task family, following Chapter 16) ----
CLEAN_EPISODES = 150
RANDOMIZED_EPISODES = 1_500
POSITION_NOISE = 0.01


def ensure_dirs() -> None:
    for path in (
        RAW_ROOT,
        LEROBOT_ROOT,
        TEXT_CACHE_ROOT,
        CHECKPOINT_ROOT,
        OUTPUT_ROOT,
        LOG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
