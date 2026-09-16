"""Merge a LoRA-trained FastWAM checkpoint back into a plain state dict.

Training with FASTWAM_LORA_RANK>0 saves checkpoints whose ``mot`` payload
contains ``lora_A``/``lora_B`` pairs.  The inference runtime (and the official
deploy scripts) load plain FastWAM weights, so fold the adapters back:

    python merge_lora.py \
      --checkpoint FastWAM/runs/<run>/checkpoints/weights/step_xxxxxx.pt \
      --output     FastWAM/runs/<run>/checkpoints/weights/step_xxxxxx_merged.pt \
      --rank 16 --alpha 32

``--alpha`` must match training (default: 2 * rank).  The merged checkpoint
keeps the same payload layout ({mot, proprio_encoder, ...}) and can be passed
directly to serve_policy.py / fastwam_runtime.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lora import merge_lora_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into a FastWAM checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=None, help="default: 2 * rank")
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    payload = torch.load(args.checkpoint, map_location="cpu")
    if "mot" not in payload:
        raise KeyError(f"{args.checkpoint} has no 'mot' payload; nothing to merge")
    payload["mot"] = merge_lora_state_dict(payload["mot"], alpha_over_rank=alpha / args.rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"merged checkpoint saved: {args.output}")


if __name__ == "__main__":
    main()
