#!/usr/bin/env bash
# Prepare FastWAM model weights:
#   1. Wan2.2-TI2V-5B video backbone (shared with Chapter 16, symlink to reuse)
#   2. ActionDiT initial weights interpolated from Wan2.2 (required before training)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/FastWAM"

bash "$ROOT/link_dirs.sh"

mkdir -p "$ROOT/weights" "$ROOT/checkpoints"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/weights"

# If another chapter already downloaded Wan2.2-TI2V-5B, set its directory and
# reuse it via symlink:
#   WAN22_SOURCE="../shared_weights/Wan2.2-TI2V-5B"
#   ln -s "$(realpath "$WAN22_SOURCE")" "$ROOT/weights/Wan2.2-TI2V-5B"
# Otherwise the model is downloaded automatically on first use.

# NOTE: run on CPU — loading the 5B backbone on a 24 GB GPU OOMs and the
# interpolation itself is cheap weight averaging.
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output "$ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  --device cpu \
  --dtype bfloat16

echo "Weights ready."
