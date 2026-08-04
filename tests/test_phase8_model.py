"""Test suite for Phase 8 stage 1a: RemoteCLIP -> DifferenceModule -> LightweightCaptionDecoder.

Offline-safe: RemoteCLIPEncoder loads the real checkpoint already vendored
at checkpoints/RemoteCLIP-ViT-B-32.pt, no network access required.
"""

import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.models.phase8 import (
    DifferenceModule,
    DifferenceRSICCModel,
    LightweightCaptionDecoder,
    Phase8Config,
    RemoteCLIPEncoder,
)

CHECKPOINT_PATH = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")


class DummyVocab:
    """Minimal vocab stand-in matching src.dataset.Vocabulary's interface."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "building": 4, "changed": 5}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    @property
    def pad_idx(self):
        return 0

    @property
    def start_idx(self):
        return 1

    @property
    def end_idx(self):
        return 2

    def __len__(self):
        return len(self.word2idx)

    def decode(self, indices, skip_special=True):
        specials = {"<PAD>", "<START>", "<END>", "<UNK>"}
        words = [self.idx2word.get(int(i), "<UNK>") for i in indices]
        if skip_special:
            words = [w for w in words if w not in specials]
        return " ".join(words)


@unittest.skipUnless(CHECKPOINT_PATH.is_file(), "RemoteCLIP checkpoint not vendored locally")
class TestRemoteCLIPEncoder(unittest.TestCase):
    def test_single_image_shape(self):
        encoder = RemoteCLIPEncoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 512))

    def test_patched_image_shape(self):
        encoder = RemoteCLIPEncoder()
        x = torch.randn(2, 4, 3, 256, 256)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 4, 512))

    def test_backbone_is_frozen(self):
        encoder = RemoteCLIPEncoder()
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))


class TestDifferenceModule(unittest.TestCase):
    def test_single_embedding_shape(self):
        diff = DifferenceModule(backbone_dim=512, fusion_dim=512, num_heads=4)
        before = torch.randn(2, 1, 512)
        after = torch.randn(2, 1, 512)
        out = diff(before, after)
        self.assertEqual(out.shape, (2, 1, 512))

    def test_patch_embedding_shape(self):
        diff = DifferenceModule(backbone_dim=512, fusion_dim=512, num_heads=4)
        before = torch.randn(2, 4, 512)
        after = torch.randn(2, 4, 512)
        out = diff(before, after)
        self.assertEqual(out.shape, (2, 4, 512))

    def test_mismatched_shapes_raise(self):
        diff = DifferenceModule(backbone_dim=512, fusion_dim=512, num_heads=4)
        before = torch.randn(2, 1, 512)
        after = torch.randn(2, 2, 512)
        with self.assertRaises(ValueError):
            diff(before, after)


class TestLightweightCaptionDecoder(unittest.TestCase):
    def test_forward_and_loss(self):
        vocab = DummyVocab()
        decoder = LightweightCaptionDecoder(vocab_size=len(vocab), pad_idx=vocab.pad_idx)
        memory = torch.randn(2, 1, 512)
        input_tokens = torch.tensor([[1, 4, 5], [1, 5, 0]])
        logits = decoder(memory, input_tokens)
        self.assertEqual(logits.shape, (2, 3, len(vocab)))

        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        target_tokens = torch.tensor([[4, 5, 2], [5, 2, 0]])
        loss = criterion(logits.reshape(-1, len(vocab)), target_tokens.reshape(-1))
        self.assertTrue(torch.isfinite(loss))


class TestDifferenceRSICCModelGradientFlow(unittest.TestCase):
    """Regression test for the original bug: gradients not reaching the diff module."""

    def _build_model(self):
        vocab = DummyVocab()
        config = Phase8Config(backbone_dim=64, fusion_dim=64, fusion_heads=4, embed_dim=32, num_decoder_heads=4)
        model = DifferenceRSICCModel(vocab=vocab, config=config)
        # Swap in a tiny fake encoder so this test needs no checkpoint/network.
        model.encoder = _FakeEncoder(dim=64)
        return model, vocab

    def test_gradients_reach_difference_module(self):
        model, vocab = self._build_model()
        before_images = torch.randn(2, 3, 32, 32)
        after_images = torch.randn(2, 3, 32, 32)
        input_tokens = torch.tensor([[1, 4, 5], [1, 5, 4]])
        target_tokens = torch.tensor([[4, 5, 2], [5, 4, 2]])

        logits = model(before_images, after_images, input_tokens)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        loss = criterion(logits.reshape(-1, len(vocab)), target_tokens.reshape(-1))
        loss.backward()

        diff_grad_norms = [
            p.grad.norm().item()
            for p in model.diff_module.parameters()
            if p.grad is not None
        ]
        self.assertTrue(len(diff_grad_norms) > 0, "DifferenceModule received no gradients at all")
        self.assertTrue(
            any(g > 0 for g in diff_grad_norms),
            "DifferenceModule gradients are all zero -- signal isn't reaching the diff stage",
        )

        # Frozen encoder must receive no gradients.
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_generate_caption_matches_shared_batched_contract(self):
        """model.generate_caption(before, after, max_new_tokens) -> List[str],
        the same contract src/metrics.py and src/training.py assume for every
        model variant (see CodeAugRSICCModel.generate_caption)."""
        model, vocab = self._build_model()
        before_images = torch.randn(3, 3, 32, 32)
        after_images = torch.randn(3, 3, 32, 32)

        captions = model.generate_caption(before_images, after_images, max_new_tokens=5)

        self.assertIsInstance(captions, list)
        self.assertEqual(len(captions), 3)
        self.assertTrue(all(isinstance(c, str) for c in captions))


class _FakeEncoder(nn.Module):
    """Tiny stand-in for RemoteCLIPEncoder so gradient-flow tests need no checkpoint."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            flat = images.reshape(B * N, C, H, W)
            pooled = self.proj(flat).mean(dim=[2, 3])
            return pooled.reshape(B, N, -1)
        pooled = self.proj(images).mean(dim=[2, 3])
        return pooled


if __name__ == "__main__":
    unittest.main()
