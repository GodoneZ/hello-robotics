#!/usr/bin/env bash
# Wire this chapter's configs/data/checkpoints into the FastWAM repo (idempotent).
# FastWAM's Hydra configs and relative paths (./data, ./checkpoints) resolve
# from the FastWAM/ root, so before training/preprocessing we:
#   1. copy this chapter's configs into FastWAM/configs/;
#   2. install the optional LoRA extension;
#   3. symlink FastWAM/data and FastWAM/checkpoints to this chapter's directories.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_DIR="$ROOT/FastWAM"
PINNED_COMMIT="7faa71108368fbb3b6885649f112af607427a2d4"

if [ ! -d "$FASTWAM_DIR/configs" ]; then
  echo "Missing $FASTWAM_DIR/configs. Clone FastWAM first:" >&2
  echo "  git clone https://github.com/yuantianyuan01/FastWAM.git FastWAM" >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "$FASTWAM_DIR" rev-parse HEAD 2>/dev/null || true)"
if [ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]; then
  echo "FastWAM revision mismatch: expected $PINNED_COMMIT, got ${ACTUAL_COMMIT:-unknown}." >&2
  echo "Run: git -C FastWAM checkout $PINNED_COMMIT" >&2
  exit 1
fi

cp "$ROOT/configs/data/g2_3cam.yaml" "$FASTWAM_DIR/configs/data/"
cp "$ROOT/configs/task/g2_uncond_3cam384_1e-4.yaml" "$FASTWAM_DIR/configs/task/"
cp "$ROOT/configs/task/g2_uncond_3cam384_1e-4_lora.yaml" "$FASTWAM_DIR/configs/task/"

cp "$ROOT/lora.py" "$FASTWAM_DIR/src/fastwam/lora.py"
TRAINER="$FASTWAM_DIR/src/fastwam/trainer.py"
if ! grep -q 'FASTWAM_LORA_RANK' "$TRAINER"; then
  if git -C "$FASTWAM_DIR" apply --check "$ROOT/patches/fastwam_lora.patch"; then
    git -C "$FASTWAM_DIR" apply "$ROOT/patches/fastwam_lora.patch"
  else
    echo "FastWAM trainer differs from the tutorial's pinned revision." >&2
    echo "Checkout commit 7faa71108368fbb3b6885649f112af607427a2d4, then rerun." >&2
    exit 1
  fi
fi

mkdir -p "$ROOT/data" "$ROOT/checkpoints"
for name in data checkpoints; do
  target="$FASTWAM_DIR/$name"
  if [ -L "$target" ] || [ ! -e "$target" ]; then
    ln -sfn "$ROOT/$name" "$target"
  else
    echo "warning: $target exists and is not a symlink; leaving it untouched" >&2
  fi
done

echo "Ready: G2 configs and LoRA extension installed; data/checkpoints linked."
