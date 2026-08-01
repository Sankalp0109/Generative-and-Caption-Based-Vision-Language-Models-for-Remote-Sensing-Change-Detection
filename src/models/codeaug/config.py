"""Configuration for the CodeAug Indian Urban & Seasonal RSICC architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CodeAugConfig:
    """Hyperparameters and architectural settings for CodeAug RSICC."""

    # ── Resolution & Patch Math ──────────────────────────────────────────────
    img_size: Tuple[int, int] = (252, 252)
    """Input resolution resized to clean 18x18 = 324 ViT-L-14 patch tokens."""

    patch_size: int = 14
    """RemoteCLIP ViT-L-14 patch size."""

    grid_size: Tuple[int, int] = (18, 18)
    """2D patch grid dimensions (252 / 14 = 18)."""

    # ── Visual Backbone & LoRA ───────────────────────────────────────────────
    vision_backbone: str = "ViT-L-14"
    """OpenCLIP pretrained vision backbone."""

    fusion_dim: int = 1024
    """Output dimension of RemoteCLIP ViT-L-14."""

    lora_r: int = 16
    """LoRA rank injected into ViT attention projection layers."""

    lora_alpha: int = 32
    """LoRA scaling alpha factor."""

    lora_dropout: float = 0.05
    """LoRA dropout probability."""

    # ── Token Compressor (Q-Former) ──────────────────────────────────────────
    num_queries: int = 64
    """Number of learnable latent query tokens in Q-Former (compresses 324 -> 64 tokens, 61.3% reduction)."""

    qformer_heads: int = 8
    """Number of cross-attention heads in Q-Former."""

    qformer_layers: int = 2
    """Number of transformer blocks in Q-Former."""

    # ── Pretrained Causal LLM Decoder ────────────────────────────────────────
    llm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    """Pretrained causal LLM decoder repository ID."""

    use_4bit_quantization: bool = True
    """Use 4-bit NormalFloat4 (NF4) quantization via bitsandbytes."""

    bnb_4bit_compute_dtype: str = "float16"
    """Compute dtype for NF4 math (torch.float16 for Pascal CC 6.1 gnode004 compatibility)."""

    discard_native_vision_tower: bool = True
    """Explicitly discard Qwen2-VL's native vision tower to conserve VRAM."""

    # ── Data Pipeline & Imbalance Control ────────────────────────────────────
    tau_gli: float = 0.05
    """Green Leaf Index vegetation mask threshold."""

    jitter_mag: float = 0.25
    """Hue/saturation perturbation magnitude for bi-temporal union GLI jitter."""

    sampler_ratio: float = 2.0
    """Changed-to-unchanged ratio enforced by WeightedRandomSampler (2:1)."""

    ce_gamma: float = 1.0
    """Unweighted Cross-Entropy loss scaling factor (gamma=1.0)."""

    # ── Learning Rates ───────────────────────────────────────────────────────
    lr_qformer: float = 1e-4
    """Learning rate for Q-Former queries and cross-attention blocks."""

    lr_lora: float = 5e-5
    """Learning rate for ViT-L-14 Visual LoRA adapters."""
