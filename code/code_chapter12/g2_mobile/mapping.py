"""与第十章一致的 2D 占据栅格/合成激光；地图加入目标桌面占地。"""

from pathlib import Path
import math
import numpy as np
from g2_mobile.common import load_config

RES, ORIGIN, SIZE = 0.05, -4.75, 190


def boxes(cfg):
    # 边界实体墙由模拟器使用同一份定义创建。
    walls = [
        [0, -4.69, 1, 9.5, 0.12, 2],
        [0, 4.69, 1, 9.5, 0.12, 2],
        [-4.69, 0, 1, 0.12, 9.5, 2],
        [4.69, 0, 1, 0.12, 9.5, 2],
    ]
    return cfg["obstacles"] + [cfg["source_table"], cfg["table"]] + walls


def grid(cfg):
    data = np.zeros((SIZE, SIZE), dtype=bool)
    y, x = np.mgrid[:SIZE, :SIZE]
    x, y = ORIGIN + (x + 0.5) * RES, ORIGIN + (y + 0.5) * RES
    for cx, cy, _, sx, sy, _ in boxes(cfg):
        data |= (abs(x - cx) <= sx / 2) & (abs(y - cy) <= sy / 2)
    return data


def raycast(data, pose, count=360):
    """沿地图射线采样，非 RTX/真实激光。保留第十章可复现传感器模型。"""
    angles = np.linspace(-math.pi, math.pi, count) + pose[2]
    distances = np.arange(0.12, 8.01, RES / 2)
    x = pose[0] + np.cos(angles[:, None]) * distances
    y = pose[1] + np.sin(angles[:, None]) * distances
    ix, iy = np.floor((x - ORIGIN) / RES).astype(int), np.floor((y - ORIGIN) / RES).astype(int)
    outside = (ix < 0) | (iy < 0) | (ix >= SIZE) | (iy >= SIZE)
    hit = outside | data[np.clip(iy, 0, SIZE - 1), np.clip(ix, 0, SIZE - 1)]
    return np.where(hit.any(axis=1), distances[np.argmax(hit, axis=1)], 8.0).tolist()


def save(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    image = np.where(grid(load_config())[::-1], 0, 254).astype(np.uint8)
    (path / "room.pgm").write_bytes(f"P5\n{SIZE} {SIZE}\n255\n".encode() + image.tobytes())
    (path / "room.yaml").write_text(
        f"image: room.pgm\nresolution: {RES}\norigin: [{ORIGIN}, {ORIGIN}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )


if __name__ == "__main__":
    save(Path(__file__).resolve().parents[1] / "maps")
