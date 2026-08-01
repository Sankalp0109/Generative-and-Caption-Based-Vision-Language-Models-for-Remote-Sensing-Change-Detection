"""CodeAug Dataset & Data Pipeline extensions for Indian Urban & Seasonal RSICC.

Implements:
1. Exact 252x252 px image resizing (324 clean patch tokens, 18x18 grid for 14px ViT patches).
2. RGB Green Leaf Index (GLI) calculation: (2G - R - B) / (2G + R + B + eps).
3. Bi-Temporal Union GLI Masking & Spectral Jitter (M_veg = M_A ∪ M_B) to eliminate monsoon false positives.
4. Single-lever 2:1 WeightedRandomSampler for class imbalance control (SFPR < 5% target).
5. Static LEVIR-CC CIDEr IDF exporter/loader for anchored evaluation.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_codeaug_transforms(
    img_size: Tuple[int, int] = (252, 252),
    mean: Tuple[float, float, float] = OPENCLIP_MEAN,
    std: Tuple[float, float, float] = OPENCLIP_STD,
) -> transforms.Compose:
    """Preprocess images to exact 252x252 px resolution (18x18=324 clean patch tokens)."""
    return transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def compute_gli(image_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute RGB Green Leaf Index (GLI): (2G - R - B) / (2G + R + B + eps).
    
    Args:
        image_tensor: Tensor of shape (C, H, W) or (B, C, H, W) in RGB format.
                      Can be normalized or unnormalized (0-1).
    Returns:
        gli: Tensor of shape (H, W) or (B, H, W) with GLI values in [-1, 1].
    """
    if image_tensor.dim() == 3:
        r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    elif image_tensor.dim() == 4:
        r, g, b = image_tensor[:, 0], image_tensor[:, 1], image_tensor[:, 2]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got dim {image_tensor.dim()}")

    numerator = 2.0 * g - r - b
    denominator = 2.0 * g + r + b + eps
    return numerator / denominator


class BiTemporalUnionGLIJitter:
    """Applies spectral jitter to vegetation transition pixels across bi-temporal images.
    
    Computes M_A = (GLI(I_A) > tau_gli) and M_B = (GLI(I_B) > tau_gli).
    Union Mask M_veg = M_A ∪ M_B flags any pixel that is vegetation in either dry or monsoon frame.
    Applies asymmetric hue/saturation jitter to M_veg pixels in BOTH images to ensure invariance.
    """

    def __init__(
        self,
        tau_gli: float = 0.05,
        jitter_mag: float = 0.25,
        p: float = 0.5,
    ):
        self.tau_gli = tau_gli
        self.jitter_mag = jitter_mag
        self.p = p

    def __call__(
        self, before_img: Image.Image, after_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply bi-temporal union GLI jitter to a pair of PIL images."""
        if torch.rand(1).item() > self.p:
            return before_img, after_img

        # Convert PIL to Tensor (0..1)
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        t_before = to_tensor(before_img)
        t_after = to_tensor(after_img)

        # Compute GLI masks
        gli_a = compute_gli(t_before)
        gli_b = compute_gli(t_after)
        m_veg = ((gli_a > self.tau_gli) | (gli_b > self.tau_gli)).float()
        m_veg = m_veg.unsqueeze(0)  # (1, H, W)

        # Generate jittered versions (hue and saturation perturbation)
        hue_factor = float(
            torch.empty(1).uniform_(-self.jitter_mag * 0.5, self.jitter_mag * 0.5).item()
        )
        sat_factor = float(
            torch.empty(1).uniform_(max(0, 1.0 - self.jitter_mag), 1.0 + self.jitter_mag).item()
        )

        jit_before = TF.adjust_hue(before_img, hue_factor)
        jit_before = TF.adjust_saturation(jit_before, sat_factor)
        jit_after = TF.adjust_hue(after_img, hue_factor)
        jit_after = TF.adjust_saturation(jit_after, sat_factor)

        t_jit_before = to_tensor(jit_before)
        t_jit_after = to_tensor(jit_after)

        # Apply union mask: retain original where M_veg==0, use jittered where M_veg==1
        out_before = t_before * (1.0 - m_veg) + t_jit_before * m_veg
        out_after = t_after * (1.0 - m_veg) + t_jit_after * m_veg

        return to_pil(out_before.clamp(0, 1)), to_pil(out_after.clamp(0, 1))


def get_balanced_sampler(
    dataset: Dataset,
    changed_to_unchanged_ratio: float = 2.0,
) -> WeightedRandomSampler:
    """Create a single-lever WeightedRandomSampler enforcing a 2:1 changed-to-unchanged ratio.
    
    Prevents compounding change-seeking bias when paired with unweighted CE loss (gamma=1.0).
    """
    change_flags = []
    for idx in range(len(dataset)):
        # Extract changeflag from dataset item or underlying sample
        if hasattr(dataset, "samples") and idx < len(dataset.samples):
            cf = dataset.samples[idx]["changeflag"]
        else:
            item = dataset[idx]
            cf = item["changeflag"] if isinstance(item, dict) else 0
        change_flags.append(int(cf))

    num_total = len(change_flags)
    num_changed = sum(change_flags)
    num_unchanged = num_total - num_changed

    if num_changed == 0 or num_unchanged == 0:
        # Fallback to uniform sampling if all items are one class
        weights = [1.0] * num_total
    else:
        # Desired ratio: P(changed) / P(unchanged) = ratio
        # P(changed) = ratio / (ratio + 1), P(unchanged) = 1 / (ratio + 1)
        weight_changed = (changed_to_unchanged_ratio / (changed_to_unchanged_ratio + 1.0)) / num_changed
        weight_unchanged = (1.0 / (changed_to_unchanged_ratio + 1.0)) / num_unchanged

        weights = [
            weight_changed if cf == 1 else weight_unchanged for cf in change_flags
        ]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_total,
        replacement=True,
    )


def export_cider_idf(
    samples: List[Dict],
    save_path: Union[str, Path] = "checkpoints/levircc_cider_idf.pkl",
) -> Dict[str, float]:
    """Compute and export static LEVIR-CC document frequencies (IDF) for anchored CIDEr evaluation."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    doc_freq = Counter()
    num_docs = len(samples)

    for sample in samples:
        # Gather all reference caption words in this document
        doc_words = set()
        for sent_obj in sample["sentences"]:
            tokens = sent_obj["raw"].lower().strip().split()
            doc_words.update(tokens)
        for w in doc_words:
            doc_freq[w] += 1

    # Compute IDF weights: log( (N + eps) / (df + eps) )
    idf_dict = {}
    for word, df in doc_freq.items():
        idf_dict[word] = math.log(max(1.0, num_docs) / max(1.0, float(df)))

    with open(save_path, "wb") as f:
        pickle.dump(idf_dict, f)

    print(f"[CodeAug] Exported static LEVIR-CC CIDEr IDF ({len(idf_dict)} words) to {save_path}")
    return idf_dict


def load_cider_idf(
    load_path: Union[str, Path] = "checkpoints/levircc_cider_idf.pkl",
) -> Optional[Dict[str, float]]:
    """Load static LEVIR-CC CIDEr IDF weights from disk."""
    load_path = Path(load_path)
    if not load_path.exists():
        print(f"[CodeAug] WARNING: CIDEr IDF file not found at {load_path}. Will use dynamic IDF.")
        return None
    with open(load_path, "rb") as f:
        idf_dict = pickle.load(f)
    print(f"[CodeAug] Successfully loaded static CIDEr IDF ({len(idf_dict)} words) from {load_path}")
    return idf_dict
