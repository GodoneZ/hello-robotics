"""第五/九章旋转矩阵转换；只保留本章需要的数学函数。"""

import math
import numpy as np


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵转 [w, x, y, z] 四元数。"""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation 必须是 3x3 矩阵")

    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array(
                [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ]
            )
        elif index == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array(
                [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ]
            )
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array(
                [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ]
            )
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q
