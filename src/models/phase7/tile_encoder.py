"""Stage 2: RemoteCLIP Tile Encoder.

Encodes all tiles from all images in a batch with a *single* shared RemoteCLIP
ViT-B-32 backbone. Tiles are batched together for efficiency; the backbone
is loaded once and optionally frozen.

Input:  tiles  (B, N, 2, 3, tile_h, tile_w)   from TileExtractor
Output:
    before_features  (B, N, backbone_dim)
    after_features   (B, N, backbone_dim)
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

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


def _safe_torch_load(path: Path):
    """Load tensor-only checkpoints safely across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TileEncoder(nn.Module):
    """Encode a batch of tile pairs with a shared RemoteCLIP backbone.

    Args:
        model_name: Open-CLIP model string (e.g. ``"ViT-B-32"``).
        checkpoint_path: Path to the RemoteCLIP weight file.
        freeze_backbone: Whether to disable gradient computation through the
            backbone and keep it in eval mode.
        download_if_missing: Download weights from HuggingFace when not found.
        repo_id: HuggingFace repository identifier.
        backbone: Pre-instantiated Open-CLIP model (used for weight sharing
            across modules or unit tests).
        backbone_dim: Override the inferred feature dimension.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        checkpoint_path: Optional[Path] = Path("checkpoints/RemoteCLIP-ViT-B-32.pt"),
        freeze_backbone: bool = True,
        download_if_missing: bool = False,
        repo_id: str = REMOTECLIP_REPO_ID,
        backbone: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        if backbone is None:
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError(
                    "open-clip-torch is required for the Phase 7 tile encoder."
                ) from exc

            ckpt = resolve_remoteclip_checkpoint(
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                download_if_missing=download_if_missing,
                repo_id=repo_id,
            )
            backbone, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=None
            )
            state_dict = _safe_torch_load(ckpt)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            backbone.load_state_dict(state_dict, strict=True)

        self.backbone = backbone
        self.backbone_dim: int = backbone_dim or self._infer_backbone_dim(backbone)

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _infer_backbone_dim(backbone: nn.Module) -> int:
        visual = getattr(backbone, "visual", None)
        dim = getattr(visual, "output_dim", None)
        if dim is None:
            dim = getattr(backbone, "embed_dim", None)
        if dim is None:
            raise ValueError(
                "Could not infer the RemoteCLIP feature dimension. "
                "Pass backbone_dim explicitly."
            )
        return int(dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode_flat(self, images_flat: torch.Tensor) -> torch.Tensor:
        """Encode a flat batch of images (BN, C, H, W)."""
        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with ctx:
            features = self.backbone.encode_image(images_flat)
        if features.ndim != 2:
            features = features.flatten(start_dim=1)
        return features.float()

    def forward(
        self, tiles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all tiles with the shared backbone."""
        if tiles.ndim != 6 or tiles.size(2) != 2:
            raise ValueError(
                "TileEncoder expects tiles shaped (B, N, 2, 3, H, W). "
                f"Got shape {tuple(tiles.shape)}."
            )

        B, N, _, C, H, W = tiles.shape

        before_flat = tiles[:, :, 0].reshape(B * N, C, H, W)
        after_flat = tiles[:, :, 1].reshape(B * N, C, H, W)

        before_enc = self._encode_flat(before_flat)
        after_enc = self._encode_flat(after_flat)

        D = self.backbone_dim
        before_features = before_enc.view(B, N, D)
        after_features = after_enc.view(B, N, D)

        return before_features, after_features
