#!/usr/bin/env bash
# Install the official FastWAM environment (Python 3.10 + PyTorch 2.7.1).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/FastWAM"

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -Fxq fastwam; then
  echo "Reusing existing conda environment: fastwam"
else
  conda create -n fastwam python=3.10 -y
fi
conda activate fastwam

pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# LeRobot v2.1 dataset writer (used by convert_dataset.py only), pinned to the
# same commit as openpi in Chapter 14.  --no-deps: every runtime dependency
# (datasets/pyarrow/av/imageio/...) is already provided by FastWAM.
pip install --no-deps \
  "git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# The pinned lerobot predates `datasets` 4.x, where hf_dataset["col"] returns a
# Column instead of a list.  A naive list(...) wrap is correct but iterates full
# rows through set_transform, decoding every image frame (~1h CPU on 1650
# episodes); read the underlying arrow table directly instead.  draccus is
# imported by the dataset writer at module load.
pip install -q draccus "pytest>=7.4"
_LEROBOT_DS="$CONDA_PREFIX/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py"
python3 - "$_LEROBOT_DS" <<'PYEOF'
import re, sys
from pathlib import Path
f = Path(sys.argv[1])
src = f.read_text()
pattern = re.compile(
    r"        timestamps = torch\.stack\((?:list\()?self\.hf_dataset\[\"timestamp\"\]\)?\)\.numpy\(\)\n"
    r"        episode_indices = torch\.stack\((?:list\()?self\.hf_dataset\[\"episode_index\"\]\)?\)\.numpy\(\)"
)
replacement = (
    "        # datasets 4.x: hf_dataset[\"col\"] returns a Column whose iteration walks\n"
    "        # full rows through set_transform (decoding every image frame); read the\n"
    "        # underlying arrow table directly instead.\n"
    "        _arrow_table = self.hf_dataset.data\n"
    "        timestamps = torch.tensor(_arrow_table.column(\"timestamp\").to_pylist()).numpy()\n"
    "        episode_indices = torch.tensor(_arrow_table.column(\"episode_index\").to_pylist()).numpy()"
)
if "_arrow_table" not in src:
    src, n = pattern.subn(replacement, src)
    assert n == 1, "lerobot timestamp patch did not apply"
    f.write_text(src)
    print(f"patched lerobot datasets-4.x compatibility: {f}")
PYEOF

echo "FastWAM environment ready. The Isaac Sim side still uses the shared tutorial environment."
