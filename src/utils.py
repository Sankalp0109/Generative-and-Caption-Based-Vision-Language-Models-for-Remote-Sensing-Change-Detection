"""Shared helper utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def setup_project_path() -> Path:
    """Ensure the project root is on sys.path when running from notebooks."""
    import sys

    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True):
    """Return the best available torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stack_image_pair(before_images, after_images):
    """
    Stack before/after tensors into model input format.

    Args:
        before_images: (B, 3, H, W)
        after_images: (B, 3, H, W)

    Returns:
        images: (B, 2, 3, H, W)
    """
    return torch.stack([before_images, after_images], dim=1)


def denormalize_image(
    image_tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """Convert a normalized CHW tensor back to HWC numpy for plotting."""
    mean_t = torch.tensor(mean, dtype=image_tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image_tensor.dtype).view(3, 1, 1)
    image = (image_tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()
