"""Self-contained Visual LoRA (Low-Rank Adaptation) injection for RemoteCLIP ViT-L-14."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualLoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper for a Linear layer."""

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def inject_visual_lora(
    vit_model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("c_fc", "c_proj"),
) -> List[str]:
    """Inject VisualLoRALayer into MLP projection layers of a ViT model.
    
    Targets c_fc and c_proj across all transformer blocks (48 linear layers in ViT-L-14)
    without interfering with PyTorch C++ MultiheadAttention internals.
    
    Returns:
        replaced_names: List of module names that were wrapped with LoRA.
    """
    replaced_names = []
    for name, module in vit_model.named_modules():
        if isinstance(module, nn.Linear):
            # Only match exact leaf attribute names (e.g., c_fc or c_proj)
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name in target_modules:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                parent = vit_model.get_submodule(parent_name) if parent_name else vit_model

                lora_wrapper = VisualLoRALayer(
                    original_layer=module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, leaf_name, lora_wrapper)
                replaced_names.append(name)

    print(f"[CodeAug] Injected Visual LoRA (r={r}, alpha={alpha}) into {len(replaced_names)} ViT linear layers.")
    return replaced_names
