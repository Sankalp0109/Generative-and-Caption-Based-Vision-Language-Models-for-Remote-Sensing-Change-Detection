"""Shared helper utilities."""

from __future__ import annotations

import random
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional

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


def cuda_device_is_compatible() -> bool:
    """Return whether this PyTorch build can execute kernels on the GPU.

    ``torch.cuda.is_available()`` only confirms that CUDA can initialize. It
    does not guarantee that the wheel contains kernels for the allocated GPU's
    compute capability.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Exercise the same kernel family that previously failed inside the
        # image encoder, then synchronize so asynchronous errors surface here.
        image = torch.zeros((1, 1, 4, 4), device="cuda")
        kernel = torch.ones((1, 1, 1, 1), device="cuda")
        torch.nn.functional.conv2d(image, kernel)
        torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        warnings.warn(
            "CUDA initialized but cannot execute kernels on this GPU; "
            f"falling back to CPU. Original error: {exc}",
            RuntimeWarning,
        )
        return False


def get_device(prefer_cuda: bool = True):
    """Return CUDA only when the installed wheel supports the allocated GPU."""
    force_cpu = os.environ.get("RSICC_FORCE_CPU", "0").strip().lower()
    if force_cpu in {"1", "true", "yes"}:
        return torch.device("cpu")
    if prefer_cuda and cuda_device_is_compatible():
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


REMOTECLIP_REPO_ID = "chendelong/RemoteCLIP"
SUPPORTED_MODELS = {"RN50", "ViT-B-32", "ViT-L-14"}


def resolve_remoteclip_checkpoint(
    checkpoint_path: Optional[Path],
    model_name: str,
    download_if_missing: bool = False,
    repo_id: str = REMOTECLIP_REPO_ID,
) -> Path:
    """Return a local RemoteCLIP checkpoint, optionally downloading it."""
    if model_name not in SUPPORTED_MODELS:
        choices = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported RemoteCLIP model {model_name!r}. Choose one of: {choices}.")

    path = Path(checkpoint_path) if checkpoint_path is not None else None
    if path is not None and path.is_file():
        return path

    if not download_if_missing:
        expected = path or Path("checkpoints") / f"RemoteCLIP-{model_name}.pt"
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not found at {expected}. "
            "Download the official checkpoint there or set download_if_missing=True."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download RemoteCLIP checkpoints."
        ) from exc

    target = path or Path("checkpoints") / f"RemoteCLIP-{model_name}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"RemoteCLIP-{model_name}.pt",
            local_dir=str(target.parent),
        )
    )
    return downloaded

