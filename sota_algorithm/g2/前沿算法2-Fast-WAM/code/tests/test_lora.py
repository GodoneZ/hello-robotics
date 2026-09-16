"""LoRA contract tests; skipped when the lightweight test environment has no torch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
    from torch import nn
except ImportError:  # Lightweight CI can still run the non-torch test suite.
    torch = None
    nn = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from lora import (  # noqa: E402
        LoRALinear,
        apply_lora,
        merge_lora_state_dict,
        restore_lora_trainability,
    )


if nn is not None:
    class _Expert(nn.Module):
        def __init__(self, *, action: bool) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "self_attn": nn.ModuleDict(
                                {"q": nn.Linear(4, 4), "o": nn.Linear(4, 4)}
                            ),
                            "ffn": nn.Sequential(
                                nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)
                            ),
                        }
                    )
                ]
            )
            self.head = nn.Linear(4, 4)
            if action:
                self.action_encoder = nn.Linear(4, 4)


    class _FastWAMStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.video_expert = _Expert(action=False)
            self.action_expert = _Expert(action=True)
            self.mot = nn.Module()
            self.mot.mixtures = nn.ModuleDict(
                {"video": self.video_expert, "action": self.action_expert}
            )
            self.dit = self.mot
            self.proprio_encoder = nn.Linear(4, 4)


@pytest.mark.skipif(torch is None, reason="torch is installed by setup_env.sh")
def test_lora_keeps_only_adapters_and_g2_layers_trainable():
    model = _FastWAMStub()
    info = apply_lora(model, rank=2, alpha=4, quantize_base=True)

    assert info["wrapped_modules"] == 8
    assert isinstance(model.video_expert.blocks[0]["self_attn"]["q"], LoRALinear)
    assert not model.video_expert.head.weight.requires_grad
    assert model.action_expert.head.weight.requires_grad
    assert model.action_expert.action_encoder.weight.requires_grad
    assert model.proprio_encoder.weight.requires_grad

    # Reproduce the upstream trainer's mode reset, then restore the LoRA mask.
    model.dit.requires_grad_(True)
    restore_lora_trainability(model)
    assert not model.video_expert.head.weight.requires_grad
    assert model.action_expert.head.weight.requires_grad
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith((".lora_A", ".lora_B"))
    )


@pytest.mark.skipif(torch is None, reason="torch is installed by setup_env.sh")
def test_quantized_lora_checkpoint_merges_to_plain_weights():
    model = _FastWAMStub()
    apply_lora(model, rank=2, alpha=4, quantize_base=True)

    merged = merge_lora_state_dict(model.mot.state_dict(), alpha_over_rank=2.0)

    assert not any(
        key.endswith((".lora_A", ".lora_B", ".weight_int8", ".weight_scale"))
        for key in merged
    )
    assert "mixtures.video.blocks.0.self_attn.q.weight" in merged
    assert "mixtures.action.blocks.0.ffn.2.weight" in merged
