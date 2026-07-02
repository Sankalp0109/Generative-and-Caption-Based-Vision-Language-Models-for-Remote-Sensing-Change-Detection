"""Model interface shared by all RSICC architecture variants."""

from __future__ import annotations

import torch.nn as nn


class ChangeCaptioningModel(nn.Module):
    """
    Base interface for all change-captioning models in the ablation study.

    Every custom architecture should inherit from this class and implement:
        forward(images, caption_tokens) -> logits

    Expected shapes:
        images:         (batch, 2, 3, H, W)
        caption_tokens: (batch, seq_len)
        logits:         (batch, seq_len, vocab_size)
    """

    def forward(self, images, caption_tokens):
        raise NotImplementedError("Subclasses must implement forward().")

    @property
    def model_name(self) -> str:
        return self.__class__.__name__