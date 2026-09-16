"""无 ROS/Isaac 依赖的配置、几何和轨迹校验；供两个 Python 运行时共享。"""

from pathlib import Path
import math
import numpy as np
import yaml

ARM = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
GRIPPER = "idx81_gripper_r_outer_joint1"
LOWER = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])
UPPER = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708])


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config/task.yaml"
        if not path.is_file():
            from ament_index_python.packages import get_package_share_directory

            path = Path(get_package_share_directory("g2_chapter12")) / "config/task.yaml"
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    for key, n in [
        ("start", 3),
        ("dock", 3),
        ("table", 6),
        ("tray", 6),
        ("home", 7),
        ("left_home", 7),
    ]:
        a = np.asarray(cfg[key], dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"非法配置 {key}")
        if n == 6 and np.any(a[3:] <= 0):
            raise ValueError(f"{key} 尺寸必须为正")
    return cfg


def rotation(q):
    """ROS xyzw 四元数 → 旋转矩阵。"""
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("非法四元数")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform(points, translation, quaternion):
    return np.asarray(points) @ rotation(quaternion).T + translation


def backproject(depth, K, stride=3):
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)


def grasp_candidates(points, table_top, tilt=0.65):
    """传统几何抓取：YOLO 点云 PCA、多偏航、朝桌内倾斜的俯抓。返回 xyz+xyzw。"""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 30 or not np.isfinite(p).all():
        return []
    lo, hi = np.percentile(p, [2, 98], axis=0)
    height = hi[2] - table_top
    if not 0.020 < height < 0.10 or max(hi[:2] - lo[:2]) > 0.10:
        return []
    center = (lo + hi) / 2
    center[2] = max(table_top + height * 0.60, table_top + 0.025)
    _, axes = np.linalg.eigh(np.cov(p[:, :2].T))
    yaw = math.atan2(axes[1, 0], axes[0, 0])
    result = []
    # 球形苹果的 PCA 轴对遮挡很敏感；先试桌边横向夹持，再试 PCA。
    for angle in [math.pi / 2, 3 * math.pi / 2, *(yaw + np.arange(4) * math.pi / 2)]:
        x = np.array([math.cos(angle), math.sin(angle), 0])
        width = np.ptp(p @ x) + 0.006
        if 0.014 < width < 0.074:
            # R = Rz(yaw) Rx(pi)，TCP +Z 朝下，X 为闭合轴。
            # Ry(-tilt) Rz(angle) Rx(pi)：腕部位于目标靠机器人一侧，
            # 不再强迫腕部与 TCP 同 XY（该约束在桌边超过臂展）。
            c, s = math.cos(tilt / 2), math.sin(tilt / 2)
            a, b = math.cos(angle / 2), math.sin(angle / 2)
            result.append([*center, c*a, c*b, s*a, s*b])
    return result


def validate_trajectory(names, points, times):
    p, t = np.asarray(points, dtype=float), np.asarray(times, dtype=float)
    if tuple(names) != ARM or p.ndim != 2 or p.shape[1] != 7 or len(p) != len(t) or not len(t):
        raise ValueError("关节顺序/轨迹维度错误")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("轨迹包含 NaN/Inf")
    if t[0] < 0 or t[-1] <= 0 or t[-1] > 120 or np.any(np.diff(t) <= 0):
        raise ValueError("轨迹时间必须严格递增，持续时间 (0,120] 秒")
    if np.any(p < LOWER) or np.any(p > UPPER):
        i, j = np.argwhere((p < LOWER) | (p > UPPER))[0]
        raise ValueError(f"关节越界: point={i} joint={ARM[j]} q={p[i,j]:.9f} "
                         f"limits=[{LOWER[j]}, {UPPER[j]}]")
    if len(t) > 1 and np.any(abs(np.diff(p, axis=0) / np.diff(t)[:, None]) > 3.1416 + 0.01):
        raise ValueError("轨迹超出关节速度上限")
    return p, t


def sample_trajectory(points, times, initial, elapsed):
    """第一点在未来时从实际关节插值，禁止瞬间跳到第一点。"""
    p, t = np.asarray(points), np.asarray(times)
    if t[0] > 0:
        p, t = np.vstack([initial, p]), np.r_[0.0, t]
    return np.array([np.interp(elapsed, t, p[:, i]) for i in range(7)])


def stopped(velocity):
    return (
        np.isfinite(velocity).all()
        and np.linalg.norm(velocity[:2]) < 0.015
        and abs(velocity[2]) < 0.025
    )
