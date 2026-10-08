"""Unit tests for the phase 5 RemoteCLIP cross-attention model."""

import unittest

import torch
import torch.nn as nn

from src.models.final_model import (
    RemoteCLIPCrossAttentionModel,
    contrastive_caption_loss,
    phase5_total_loss,
)


class FakeRemoteCLIP(nn.Module):
    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(3, output_dim, bias=False)

    def encode_image(self, images):
        pooled = images.mean(dim=(-2, -1))
        return self.projection(pooled)


class RemoteCLIPCrossAttentionModelTests(unittest.TestCase):
    def test_complete_model_returns_logits_and_auxiliary_embeddings(self):
        model = RemoteCLIPCrossAttentionModel(
            vocab_size=30,
            encoder_dim=16,
            embed_dim=16,
            num_heads=4,
            num_decoder_layers=1,
            max_caption_len=12,
            dropout=0.0,
            fusion_heads=4,
            token_count=3,
            contrastive_dim=12,
            backbone=FakeRemoteCLIP(output_dim=8),
            backbone_dim=8,
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

    def test_phase5_loss_combines_caption_and_contrastive_terms(self):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        logits = torch.randn(2, 5, 30, requires_grad=True)
        target_tokens = torch.randint(0, 30, (2, 5))
        image_embeddings = torch.randn(2, 12)
        text_embeddings = torch.randn(2, 12)

        total_loss, caption_loss_value, contrastive_loss_value = phase5_total_loss(
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