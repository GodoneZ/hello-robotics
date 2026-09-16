#!/usr/bin/env bash
# Fast-WAM G2 single-GPU LoRA fine-tuning for a 24 GB GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT/link_dirs.sh"
cd "$ROOT/FastWAM"

export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/weights"
export FASTWAM_LORA_RANK="${FASTWAM_LORA_RANK:-16}"
export FASTWAM_LORA_ALPHA="${FASTWAM_LORA_ALPHA:-32}"
export FASTWAM_LORA_DROPOUT="${FASTWAM_LORA_DROPOUT:-0}"
export FASTWAM_LORA_QUANTIZE_BASE="${FASTWAM_LORA_QUANTIZE_BASE:-1}"

TASK="${FASTWAM_TASK:-g2_uncond_3cam384_1e-4_lora}"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero0.yaml \
  --num_processes 1 scripts/train.py \
  task="$TASK" "$@"
