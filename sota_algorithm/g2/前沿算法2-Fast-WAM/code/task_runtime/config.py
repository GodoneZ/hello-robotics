"""G2 三色物块任务的仿真与机器人配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[4]
ASSETS_ROOT = Path(
    os.getenv("G2_ASSETS_ROOT", REPO_ROOT / "code/assets")
).expanduser().resolve()

DATASET_FPS = 30
POSITION_NOISE = 0.01
COLORS = ("red", "green", "blue")
COLOR_RGB = {
    "red": (0.90, 0.05, 0.05),
    "green": (0.05, 0.75, 0.12),
    "blue": (0.05, 0.30, 0.92),
}
TASK_TEMPLATE = "Pick up the {color} block and place it into the empty box."
STATE_DIM = ACTION_DIM = 16
SEED = 1000

ROBOT_USD = ASSETS_ROOT / "robot/G2_omnipicker/robot.usda"
ROBOT_PRIM = "/genie"
ARM_BASE_PRIM = f"{ROBOT_PRIM}/arm_base_link"
CAMERA_PRIMS = {
    "head": f"{ROBOT_PRIM}/head_link3/head_front_Camera",
    "left": f"{ROBOT_PRIM}/gripper_l_base_link/Left_Camera",
    "right": f"{ROBOT_PRIM}/gripper_r_base_link/Right_Camera",
}
LEFT_ARM_JOINTS = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
WAIST_JOINTS = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
LEFT_GRIPPER_JOINTS = ("idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1")
RIGHT_GRIPPER_JOINTS = ("idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1")
ARM_LOWER_7 = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])
ARM_UPPER_7 = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708])
ARM_LOWER_14, ARM_UPPER_14 = np.tile(ARM_LOWER_7, 2), np.tile(ARM_UPPER_7, 2)
HOME_ARMS_14 = np.array(
    [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.278110,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.278110,
        -0.925845,
        0.839257,
    ]
)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.785, 0.0


@dataclass(frozen=True)
class SimulationConfig:
    physics_hz: int = 120
    render_hz: int = 30
    image_size: tuple[int, int] = (160, 120)
    headless: bool = False
    renderer: str = "RaytracedLighting"
    warmup_steps: int = 90

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def record_every(self) -> int:
        return max(1, self.physics_hz // DATASET_FPS)


@dataclass(frozen=True)
class TaskConfig:
    table_center: tuple[float, float, float] = (0.72, 0.610, -0.28)
    table_size: tuple[float, float, float] = (0.95, 0.08, 1.15)
    block_size: float = 0.050
    block_positions: tuple[tuple[float, float, float], ...] = (
        (0.56, 0.545, -0.43),
        (0.56, 0.545, -0.18),
        (0.56, 0.545, 0.04),
    )
    box_position: tuple[float, float, float] = (0.59, 0.535, -0.63)
    box_inner_size: tuple[float, float] = (0.24, 0.20)
    box_wall_height: float = 0.11
    box_wall_thickness: float = 0.018
    box_floor_thickness: float = 0.024
    block_mass: float = 0.025
    contact_offset: float = 0.002
    static_friction: float = 1.8
    dynamic_friction: float = 1.5
    pregrasp_clearance: float = 0.18
    grasp_offsets: tuple[float, float, float] = (0.010, 0.000, -0.010)
    place_clearance: float = 0.13

    @property
    def table_top_y(self) -> float:
        return self.table_center[1] - self.table_size[1] / 2
