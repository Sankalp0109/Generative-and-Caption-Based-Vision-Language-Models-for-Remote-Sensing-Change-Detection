"""Unit tests for the generate_caption utility function in src/training.py."""

import unittest

import torch
import torch.nn as nn

from src.dataset import Vocabulary
from src.training import generate_caption
from src.models.phase7 import TileBasedChangeCaptioningModel, Phase7Config


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


class GenerateCaptionTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.word2idx = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
            "change": 4,
            "detected": 5,
        }
        self.vocab.idx2word = {idx: word for word, idx in self.vocab.word2idx.items()}
        self.vocab.next_idx = 6

        self.device = torch.device("cpu")
        self.before_image = torch.randn(3, 32, 32)
        self.after_image = torch.randn(3, 32, 32)

    def test_generate_caption_with_phase7_model(self):
        config = Phase7Config()
        config.remoteclip_model_name = "ViT-B-32"
        config.fusion_dim = 8
        config.global_dim = 8
        config.embed_dim = 8
        config.contrastive_dim = 8
        config.num_fusion_layers = 1
        config.num_decoder_layers = 1
        config.grid_size = 2
        config.tile_size = (16, 16)

        model = TileBasedChangeCaptioningModel(
            vocab_size=len(self.vocab.word2idx),
            config=config,
            pad_idx=self.vocab.pad_idx,
            backbone=FakeRemoteCLIP(output_dim=8),
        ).to(self.device)

        caption = generate_caption(
            model=model,
            before_image=self.before_image,
            after_image=self.after_image,
            vocab=self.vocab,
            device=self.device,
            max_len=5,
        )
        self.assertIsInstance(caption, str)


if __name__ == "__main__":
    unittest.main()
