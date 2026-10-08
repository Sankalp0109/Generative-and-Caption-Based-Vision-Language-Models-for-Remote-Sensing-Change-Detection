"""Unit tests for the Phase 2 model without downloading RemoteCLIP weights."""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from src.models.variants.remoteclip_difference import (
    RemoteCLIPDifferenceModel,
    RemoteCLIPEncoder,
)
from src.models.baseline import RSICCformerBaseline
from src.utils import get_device


class FakeRemoteCLIP(nn.Module):
    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.projection = nn.Linear(3, output_dim, bias=False)

    def encode_image(self, images):
        pooled = images.mean(dim=(-2, -1))
        return self.projection(pooled)


class RemoteCLIPEncoderTests(unittest.TestCase):
    @patch.dict("os.environ", {"RSICC_FORCE_CPU": "1"})
    def test_device_can_be_forced_to_cpu(self):
        self.assertEqual(get_device().type, "cpu")

    def test_baseline_directory_is_an_importable_package(self):
        self.assertEqual(
            RSICCformerBaseline.__module__,
            "src.models.baseline.baseline",
        )

    def test_encoder_output_shape(self):
        encoder = RemoteCLIPEncoder(
            out_dim=16,
            backbone=FakeRemoteCLIP(output_dim=8),
            backbone_dim=8,
            fusion_dropout=0.0,
        )
        images = torch.randn(2, 2, 3, 32, 32)

        output = encoder(images)

        self.assertEqual(output.shape, (2, 16))

    def test_frozen_backbone_stays_in_eval_mode(self):
        backbone = FakeRemoteCLIP()
        encoder = RemoteCLIPEncoder(
            backbone=backbone,
            backbone_dim=8,
            freeze_backbone=True,
        )

        encoder.train()

        self.assertFalse(backbone.training)
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))

    def test_invalid_pair_shape_is_rejected(self):
        encoder = RemoteCLIPEncoder(
            backbone=FakeRemoteCLIP(),
            backbone_dim=8,
        )

        with self.assertRaisesRegex(ValueError, "expects images shaped"):
            encoder(torch.randn(2, 3, 32, 32))

    def test_complete_model_matches_shared_interface(self):
        model = RemoteCLIPDifferenceModel(
            vocab_size=25,
            encoder_dim=16,
            embed_dim=16,
            num_heads=4,
            num_decoder_layers=1,
            max_caption_len=10,
            dropout=0.0,
            fusion_dropout=0.0,
            backbone=FakeRemoteCLIP(output_dim=8),
            backbone_dim=8,
        )
        images = torch.randn(2, 2, 3, 32, 32)
        captions = torch.randint(0, 25, (2, 6))

        logits = model(images, captions)

        self.assertEqual(logits.shape, (2, 6, 25))


if __name__ == "__main__":
    unittest.main()
