#!/usr/bin/env bash
# Precompute T5 text-embedding caches for the G2 instructions (run once before training).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/FastWAM"

bash "$ROOT/link_dirs.sh"

export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/weights"

python scripts/precompute_text_embeds.py \
  task=g2_uncond_3cam384_1e-4

echo "Text embedding cache: $ROOT/data/text_embeds_cache/g2"
