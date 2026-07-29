"""Phase 5 RemoteCLIP model with cross-attention fusion and contrastive alignment."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..baseline.decoder import SimpleDecoder
from ..interface import ChangeCaptioningModel
from ..variants.remoteclip_difference import REMOTECLIP_REPO_ID, resolve_remoteclip_checkpoint


def _safe_torch_load(path: Path):
    """Load tensor-only checkpoints safely across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RemoteCLIPBackboneEncoder(nn.Module):
    """Encode a before/after image pair with a frozen or trainable RemoteCLIP backbone."""

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
                    "open-clip-torch is required for the phase 5 RemoteCLIP model."
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

    @staticmethod
    def _infer_backbone_dim(backbone: nn.Module) -> int:
        visual = getattr(backbone, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if output_dim is None:
            output_dim = getattr(backbone, "embed_dim", None)
        if output_dim is None:
            raise ValueError(
                "Could not infer the RemoteCLIP feature dimension. Pass backbone_dim explicitly."
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

    def forward(self, images: torch.Tensor):
        if images.ndim != 5 or images.size(1) != 2:
            raise ValueError(
                "RemoteCLIPBackboneEncoder expects images shaped (batch, 2, 3, height, width)."
            )

        before_features = self._encode_image(images[:, 0])
        after_features = self._encode_image(images[:, 1])
        return before_features, after_features


class BidirectionalCrossAttentionFusion(nn.Module):
    """Fuse before/after RemoteCLIP features with bidirectional cross-attention."""

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int,
        num_heads: int = 4,
        token_count: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fusion_dim % num_heads != 0:
            raise ValueError("fusion_dim must be divisible by num_heads.")
        self.token_count = token_count
        self.fusion_dim = fusion_dim

        self.token_projection = nn.Linear(input_dim, token_count * fusion_dim)
        self.before_to_after = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.after_to_before = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_before = nn.LayerNorm(fusion_dim)
        self.norm_after = nn.LayerNorm(fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

    def _project_tokens(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.token_projection(features)
        return tokens.view(features.size(0), self.token_count, self.fusion_dim)

    def forward(self, before_features: torch.Tensor, after_features: torch.Tensor):
        before_tokens = self._project_tokens(before_features)
        after_tokens = self._project_tokens(after_features)

        before_context, _ = self.before_to_after(before_tokens, after_tokens, after_tokens)
        after_context, _ = self.after_to_before(after_tokens, before_tokens, before_tokens)

        before_enriched = self.norm_before(before_tokens + self.dropout(before_context))
        after_enriched = self.norm_after(after_tokens + self.dropout(after_context))
        fused_tokens = 0.5 * (before_enriched + after_enriched)
        fused_tokens = self.fusion_norm(fused_tokens + self.ffn(fused_tokens))
        fused_representation = fused_tokens.mean(dim=1)
        return fused_representation, fused_tokens


class CaptionContrastiveEncoder(nn.Module):
    """Encode caption tokens into the shared contrastive space."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        projection_dim: int,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, caption_tokens: torch.Tensor) -> torch.Tensor:
        if caption_tokens.ndim != 2:
            raise ValueError("CaptionContrastiveEncoder expects caption tokens shaped (batch, seq_len).")

        embedded = self.embedding(caption_tokens)
        mask = (caption_tokens != self.pad_idx).unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = (embedded * mask).sum(dim=1) / lengths
        return self.projection(pooled)


class Phase5ChangeEncoder(nn.Module):
    """Encode image pairs into the change representation used by the decoder."""

    def __init__(
        self,
        backbone_encoder: RemoteCLIPBackboneEncoder,
        fusion: BidirectionalCrossAttentionFusion,
        change_projection: nn.Module,
    ):
        super().__init__()
        self.backbone_encoder = backbone_encoder
        self.fusion = fusion
        self.change_projection = change_projection

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        before_features, after_features = self.backbone_encoder(images)
        fused_representation, fused_tokens = self.fusion(before_features, after_features)
        change_features = self.change_projection(fused_representation)

        if return_aux:
            return change_features, fused_tokens, before_features, after_features
        return change_features


class RemoteCLIPCrossAttentionModel(ChangeCaptioningModel):
    """Phase 5 model: RemoteCLIP + cross-attention fusion + contrastive caption alignment."""

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
        remoteclip_checkpoint_path: Optional[Path] = Path("checkpoints/RemoteCLIP-ViT-B-32.pt"),
        freeze_remoteclip: bool = True,
        download_if_missing: bool = False,
        remoteclip_repo_id: str = REMOTECLIP_REPO_ID,
        fusion_heads: int = 4,
        token_count: int = 4,
        contrastive_dim: int = 256,
        backbone: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()

        self.backbone_encoder = RemoteCLIPBackboneEncoder(
            model_name=remoteclip_model_name,
            checkpoint_path=remoteclip_checkpoint_path,
            freeze_backbone=freeze_remoteclip,
            download_if_missing=download_if_missing,
            repo_id=remoteclip_repo_id,
            backbone=backbone,
            backbone_dim=backbone_dim,
        )
        self.fusion = BidirectionalCrossAttentionFusion(
            input_dim=self.backbone_encoder.backbone_dim,
            fusion_dim=encoder_dim,
            num_heads=fusion_heads,
            token_count=token_count,
            dropout=dropout,
        )
        self.change_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
        self.caption_encoder = CaptionContrastiveEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            projection_dim=contrastive_dim,
            pad_idx=pad_idx,
            dropout=dropout,
        )
        self.encoder = Phase5ChangeEncoder(
            backbone_encoder=self.backbone_encoder,
            fusion=self.fusion,
            change_projection=self.change_projection,
        )
        self.image_projection = nn.Sequential(
            nn.Linear(encoder_dim, contrastive_dim),
            nn.LayerNorm(contrastive_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(contrastive_dim, contrastive_dim),
        )
        self.contrastive_dim = contrastive_dim
        self.pad_idx = pad_idx

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone_encoder.train(mode)
        self.fusion.train(mode)
        self.change_projection.train(mode)
        self.encoder.train(mode)
        return self

    def encode_images(self, images: torch.Tensor):
        change_features, fused_tokens, _, _ = self.encoder(images, return_aux=True)
        image_embeddings = self.image_projection(change_features)
        return change_features, image_embeddings, fused_tokens

    def encode_captions(self, caption_tokens: torch.Tensor):
        return self.caption_encoder(caption_tokens)

    def forward(
        self,
        images: torch.Tensor,
        caption_tokens: torch.Tensor,
        return_aux: bool = False,
    ):
        change_features, image_embeddings, fused_tokens = self.encode_images(images)
        logits = self.decoder(fused_tokens, caption_tokens)

        if not return_aux:
            return logits

        text_embeddings = self.encode_captions(caption_tokens)
        return {
            "logits": logits,
            "change_features": change_features,
            "fused_tokens": fused_tokens,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


Phase5RemoteCLIPModel = RemoteCLIPCrossAttentionModel