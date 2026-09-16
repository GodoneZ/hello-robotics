#!/usr/bin/env bash
# FastWAM G2 fine-tuning entry (DeepSpeed ZeRO-1).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/FastWAM"

bash "$ROOT/link_dirs.sh"

NUM_GPUS="${1:-8}"
shift || true
TASK="${FASTWAM_TASK:-g2_uncond_3cam384_1e-4}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/weights"

bash scripts/train_zero1.sh "$NUM_GPUS" task="$TASK" "$@"

# Training objective examples:
#   bash train.sh 8 model.loss.lambda_video=1
#   bash train.sh 8 model.loss.lambda_video=0

# Single-GPU smoke run on 24 GB (batch 1, gradient checkpointing, 20 steps):
#   bash train.sh 1 batch_size=1 model.mot_checkpoint_mixed_attn=true \
#     max_steps=20 save_every=20 eval_every=20

# Single/few-GPU debugging: run the official dryrun first:
#   python scripts/dryrun_fastwam.py task=g2_uncond_3cam384_1e-4
