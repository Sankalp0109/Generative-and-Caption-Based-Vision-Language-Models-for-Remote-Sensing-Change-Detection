"""Phase 2: RemoteCLIP encoder with difference fusion.

This phase changes only the visual encoder. Caption decoding and the
cross-entropy training objective remain identical to the Phase 1 baseline.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..baseline.decoder import SimpleDecoder
from ..interface import ChangeCaptioningModel


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
    """Load tensor-only checkpoints safely across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RemoteCLIPEncoder(nn.Module):
    """Encode a temporal image pair with RemoteCLIP and difference fusion.

    Input:
        images: (batch, 2, 3, height, width)
    Output:
        fused representation: (batch, out_dim)
    """

    def __init__(
        self,
        out_dim: int = 512,
        model_name: str = "ViT-B-32",
        checkpoint_path: Optional[Path] = Path(
            "checkpoints/RemoteCLIP-ViT-B-32.pt"
        ),
        freeze_backbone: bool = True,
        download_if_missing: bool = False,
        repo_id: str = REMOTECLIP_REPO_ID,
        fusion_dropout: float = 0.1,
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
                    "open-clip-torch is required for the Phase 2 RemoteCLIP model."
                ) from exc

            checkpoint = resolve_remoteclip_checkpoint(
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                download_if_missing=download_if_missing,
                repo_id=repo_id,
            )
            backbone, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
            )
            state_dict = _safe_torch_load(checkpoint)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            backbone.load_state_dict(state_dict, strict=True)

        self.backbone = backbone
        self.backbone_dim = backbone_dim or self._infer_backbone_dim(backbone)

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.fusion_projection = nn.Sequential(
            nn.Linear(self.backbone_dim * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

    @staticmethod
    def _infer_backbone_dim(backbone: nn.Module) -> int:
        visual = getattr(backbone, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if output_dim is None:
            output_dim = getattr(backbone, "embed_dim", None)
        if output_dim is None:
            raise ValueError(
                "Could not infer the RemoteCLIP feature dimension. "
                "Pass backbone_dim explicitly."
            )
        return int(output_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            features = self.backbone.encode_image(image)
        if features.ndim != 2:
            features = features.flatten(start_dim=1)
        return features.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.size(1) != 2:
            raise ValueError(
                "RemoteCLIPEncoder expects images shaped (batch, 2, 3, height, width)."
            )

        before_features = self._encode_image(images[:, 0])
        after_features = self._encode_image(images[:, 1])
        difference_features = torch.abs(after_features - before_features)
        fused_features = torch.cat(
            [before_features, after_features, difference_features],
            dim=-1,
        )
        return self.fusion_projection(fused_features)


class RemoteCLIPDifferenceModel(ChangeCaptioningModel):
    """Phase 2 model: RemoteCLIP + difference fusion + baseline decoder."""

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_decoder_layers: int = 2,
        max_caption_len: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0,
        remoteclip_model_name: str = "ViT-B-32",
        remoteclip_checkpoint_path: Optional[Path] = Path(
            "checkpoints/RemoteCLIP-ViT-B-32.pt"
        ),
        freeze_remoteclip: bool = True,
        download_if_missing: bool = False,
        remoteclip_repo_id: str = REMOTECLIP_REPO_ID,
        fusion_dropout: float = 0.1,
        backbone: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = RemoteCLIPEncoder(
            out_dim=encoder_dim,
            model_name=remoteclip_model_name,
            checkpoint_path=remoteclip_checkpoint_path,
            freeze_backbone=freeze_remoteclip,
            download_if_missing=download_if_missing,
            repo_id=remoteclip_repo_id,
            fusion_dropout=fusion_dropout,
            backbone=backbone,
            backbone_dim=backbone_dim,
        )
        self.decoder = SimpleDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            max_len=max_caption_len,
            encoder_dim=encoder_dim,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def forward(self, images: torch.Tensor, caption_tokens: torch.Tensor):
        encoder_features = self.encoder(images)
        return self.decoder(encoder_features, caption_tokens)
