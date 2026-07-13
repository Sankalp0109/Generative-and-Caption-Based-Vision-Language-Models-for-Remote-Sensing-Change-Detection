"""Unit tests for the Phase 7 tile-based change captioning model."""

import unittest

import torch
import torch.nn as nn

from src.models.phase7 import (
    TileBasedChangeCaptioningModel,
    Phase7Config,
    phase7_total_loss,
)
from src.models.phase7.losses import contrastive_caption_loss


class FakeRemoteCLIP(nn.Module):
    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(3, output_dim, bias=False)
        self.embed_dim = output_dim

        class Visual:
            def __init__(self, output_dim):
                self.output_dim = output_dim
        self.visual = Visual(output_dim)

    def encode_image(self, images):
        pooled = images.mean(dim=(-2, -1))
        return self.projection(pooled)


class TileBasedModelTests(unittest.TestCase):
    def test_complete_model_returns_logits_and_auxiliary_embeddings(self):
        config = Phase7Config()
        config.remoteclip_model_name = "ViT-B-32"
        config.fusion_dim = 16
        config.global_dim = 16
        config.embed_dim = 16
        config.contrastive_dim = 12
        config.num_fusion_layers = 1
        config.num_decoder_layers = 1
        config.grid_size = 2
        config.tile_size = (16, 16)

        model = TileBasedChangeCaptioningModel(
            vocab_size=30,
            config=config,
            pad_idx=0,
            backbone=FakeRemoteCLIP(output_dim=8),
        )

        images = torch.randn(2, 2, 3, 32, 32)
        captions = torch.randint(0, 30, (2, 6))

        output = model(images, captions, return_aux=True)

        self.assertEqual(output["logits"].shape, (2, 6, 30))
        self.assertEqual(output["change_features"].shape, (2, 16))
        self.assertEqual(output["image_embeddings"].shape, (2, 12))
        self.assertEqual(output["text_embeddings"].shape, (2, 12))

    def test_contrastive_loss_returns_scalar(self):
        image_embeddings = torch.randn(4, 12)
        text_embeddings = torch.randn(4, 12)

        loss = contrastive_caption_loss(image_embeddings, text_embeddings)

        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_phase7_loss_combines_caption_and_contrastive_terms(self):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        logits = torch.randn(2, 5, 30, requires_grad=True)
        target_tokens = torch.randint(0, 30, (2, 5))
        image_embeddings = torch.randn(2, 12)
        text_embeddings = torch.randn(2, 12)

        total_loss, caption_loss_value, contrastive_loss_value = phase7_total_loss(
            logits=logits,
            target_tokens=target_tokens,
            criterion=criterion,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            contrastive_weight=0.2,
        )

        self.assertEqual(total_loss.dim(), 0)
        self.assertEqual(caption_loss_value.dim(), 0)
        self.assertEqual(contrastive_loss_value.dim(), 0)
        self.assertGreaterEqual(total_loss.item(), caption_loss_value.item())


if __name__ == "__main__":
    unittest.main()
