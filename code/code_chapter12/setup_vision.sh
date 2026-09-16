#!/usr/bin/env bash
# 检查系统 Python 的 pip 推理库和本章权重；不自动安装或升级依赖。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
unset PYTHONNOUSERSITE PYTHONHOME PYTHONPATH
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT/.ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
/usr/bin/python3 - <<'PY'
import numpy as np
import sys
import ultralytics
from ultralytics import YOLO
if ultralytics.__version__ != "8.3.203":
    raise RuntimeError("本章已验证版本为 ultralytics==8.3.203，请安装固定版本")
print("Python:", sys.executable)
print("Ultralytics:", ultralytics.__version__, ultralytics.__file__)
from pathlib import Path
p=Path('models/apple_yolo.pt')
if not p.is_file():
    raise FileNotFoundError(p)
m=YOLO(str(p))
print('YOLO ready:', p.resolve())
print('classes:', m.names)
# Do one inference to catch torch/OpenCV/runtime problems before launching ROS.
result = m.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.4,
                   device='cpu', verbose=False)
print('inference smoke test: OK, frames=', len(result))
PY
