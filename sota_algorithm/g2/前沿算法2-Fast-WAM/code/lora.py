"""LoRA fine-tuning for FastWAM on small GPUs (pattern from Chapter 16).

The official FastWAM trainer only supports full-MoT fine-tuning, which does not
fit on a 24 GB GPU.  This module injects rank-r adapters into every attention
and FFN linear of both MoT experts (video + action), freezes the base weights,
and keeps the G2-specific new layers fully trainable:

- ``action_encoder`` / ``head`` of the action expert (16-D in/out, randomly
  initialized by the Wan2.2 interpolation preprocessing);
- ``proprio_encoder`` (16-D state input, new for G2);
- per-block ``modulation`` vectors and affine norms (cheap, useful).

Everything is opt-in via environment variables (see Trainer patch), so the
official full fine-tuning path is untouched.  For deployment,
``merge_lora_state_dict`` folds adapters back into a plain FastWAM checkpoint,
so the inference runtime needs no changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameter-name fragments that stay fully trainable even with LoRA enabled.
FULL_TRAIN_FRAGMENTS = ("action_encoder.", "head.", "proprio_encoder.")
# Cheap per-block parameters that are worth updating (ch16 recipe).
SMALL_TRAIN_SUFFIXES = (".modulation",)


class _FrozenInt8Linear(torch.autograd.Function):
    """F.linear with an int8 base weight that needs no grad.

    Only the int8 tensor and scale are saved for backward; the bf16 copy is a
    per-call transient, and grad_weight is never computed (weight is frozen).
    """

    @staticmethod
    def forward(ctx, value, weight_int8, weight_scale, bias):
        ctx.save_for_backward(weight_int8, weight_scale)
        ctx.input_shape = value.shape
        weight = weight_int8.to(value.dtype) * weight_scale
        return F.linear(value, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        weight_int8, weight_scale = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        weight = weight_int8.to(grad_out.dtype) * weight_scale
        grad_value = grad_out.reshape(-1, grad_out.shape[-1]) @ weight
        return grad_value.view(ctx.input_shape), None, None, None


class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement that preserves the original key layout.

    With ``quantize_base=True`` the frozen base weight is stored as int8 with a
    per-output-channel scale (LLM.int8 style), halving the base-model memory so
    LoRA training fits on a 24 GB GPU; dequantization happens inside forward.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        quantize_base: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.quantize_base = bool(quantize_base)
        if self.quantize_base:
            weight = base.weight.detach()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            self.register_buffer("weight_int8", torch.round(weight / scale).to(torch.int8))
            self.register_buffer("weight_scale", scale)
        else:
            self.weight = base.weight
            self.weight.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.quantize_base:
            base = _FrozenInt8Linear.apply(value, self.weight_int8, self.weight_scale, self.bias)
        else:
            base = F.linear(value, self.weight, self.bias)
        update = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return base + update * self.scaling


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _is_target_linear(name: str) -> bool:
    """Attention q/k/v/o and FFN linears inside MoT expert blocks."""
    if "blocks" not in name.split("."):
        return False
    return any(
        fragment in name
        for fragment in (".self_attn.", ".cross_attn.", ".ffn.")
    )


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    quantize_base: bool = False,
) -> dict:
    """Freeze the FastWAM MoT and inject LoRA adapters into both experts."""
    alpha = float(alpha if alpha is not None else 2 * rank)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            if _is_target_linear(name):
                _set_submodule(
                    model, name, LoRALinear(module, rank, alpha, dropout, quantize_base)
                )
                wrapped.append(name)

    for name, parameter in model.named_parameters():
        if any(fragment in name for fragment in FULL_TRAIN_FRAGMENTS):
            parameter.requires_grad_(True)
        elif name.endswith(SMALL_TRAIN_SUFFIXES):
            parameter.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "wrapped_modules": len(wrapped),
        "trainable_params": trainable,
        "total_params": total,
    }
    print(
        f"FastWAM-LoRA: rank={rank} alpha={alpha} wrapped={len(wrapped)} "
        f"quantize_base={quantize_base} "
        f"trainable={trainable / 1e6:.1f}M total={total / 1e9:.2f}B "
        f"({100.0 * trainable / total:.3f}%)",
        flush=True,
    )
    return info


def merge_lora_state_dict(
    state: dict[str, torch.Tensor],
    alpha_over_rank: float | None = None,
) -> dict[str, torch.Tensor]:
    """Fold ``lora_A``/``lora_B`` pairs back into their base ``weight`` tensors.

    Works on any state dict produced by a LoRA-wrapped model (e.g. the ``mot``
    payload of a training checkpoint).  If ``alpha_over_rank`` is not given it
    is recovered per module, assuming alpha = 2 * rank (the apply_lora default).
    """
    merged: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            skipped += 1
            continue
        merged[key] = value
    folded = 0
    for key in list(state):
        if not key.endswith(".lora_A"):
            continue
        prefix = key[: -len(".lora_A")]
        a = state[key]
        b = state.get(prefix + ".lora_B")
        base = merged.get(prefix + ".weight")
        if base is None and prefix + ".weight_int8" in merged:
            base = (
                merged.pop(prefix + ".weight_int8").to(torch.float32)
                * merged.pop(prefix + ".weight_scale").to(torch.float32)
            )
        if b is None or base is None:
            raise KeyError(f"incomplete LoRA triple for {prefix}")
        rank = a.shape[0]
        scaling = alpha_over_rank if alpha_over_rank is not None else 2.0
        out_dtype = torch.bfloat16
        merged[prefix + ".weight"] = (
            base.to(torch.float32) + scaling * (b.to(torch.float32) @ a.to(torch.float32))
        ).to(out_dtype)
        folded += 1
    print(f"merge_lora_state_dict: folded={folded} dropped={skipped}", flush=True)
    return merged
