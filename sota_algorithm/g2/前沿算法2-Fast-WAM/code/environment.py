"""Load the G2 pick/place scene from this chapter's local task runtime."""

from __future__ import annotations

from dataclasses import dataclass
import sys

from settings import IMAGE_HEIGHT, IMAGE_WIDTH, RAW_FPS, TASK_RUNTIME_ROOT

if str(TASK_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_RUNTIME_ROOT))

from auto_expert import AutoExpert  # noqa: E402
from config import COLORS, POSITION_NOISE, SEED, TASK_TEMPLATE, SimulationConfig  # noqa: E402
from robot import G2Robot, closed_fraction  # noqa: E402
from simulation import G2Simulation  # noqa: E402


@dataclass(frozen=True)
class FastWAMSimulationConfig(SimulationConfig):
    """G2 scene at Fast-WAM's native 320x240 camera resolution, 30 Hz recording."""

    image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)

    @property
    def record_every(self) -> int:
        return max(1, self.physics_hz // RAW_FPS)


__all__ = [
    "AutoExpert",
    "COLORS",
    "FastWAMSimulationConfig",
    "G2Robot",
    "G2Simulation",
    "POSITION_NOISE",
    "SEED",
    "TASK_TEMPLATE",
    "closed_fraction",
]
