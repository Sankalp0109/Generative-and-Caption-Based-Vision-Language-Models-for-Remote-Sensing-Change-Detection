"""Phase 8 stage 1a model: frozen RemoteCLIP -> trainable diff module -> trainable lightweight decoder.

This assembles the stage-1a curriculum described in the Phase 8 plan: only
DifferenceModule and LightweightCaptionDecoder are trained here, so the
gradient reaching the difference module is direct and strong. The eventual
Qwen2-VL-2B bridge (stage 1b) is a separate follow-up once this stage is
confirmed to actually train the difference module.

generate_caption follows the same (before_image, after_image, max_new_tokens)
-> List[str] batched contract as every other model variant in this repo (see
CodeAugRSICCModel.generate_caption), since src/metrics.py and src/training.py
already assume that shared interface. The vocabulary is stored on the model
itself (mirroring how CodeAugRSICCModel stores its tokenizer) so callers
never need to pass it in separately.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import Phase8Config
from .difference_module import DifferenceModule
from .lightweight_decoder import LightweightCaptionDecoder
from .remoteclip_encoder import RemoteCLIPEncoder


class DifferenceRSICCModel(nn.Module):
    """End-to-end stage-1a model: RemoteCLIP (frozen) + DifferenceModule + LightweightCaptionDecoder."""

    def __init__(self, vocab, config: Optional[Phase8Config] = None):
        super().__init__()
        self.config = config or Phase8Config()
        self.vocab = vocab

        self.encoder = RemoteCLIPEncoder(self.config)
        self.diff_module = DifferenceModule.from_config(self.config)
        self.decoder = LightweightCaptionDecoder(
            vocab_size=len(vocab.word2idx), pad_idx=vocab.pad_idx, config=self.config
        )

    def _encode_pair(self, before_image: torch.Tensor, after_image: torch.Tensor):
        before_feat = self.encoder(before_image)
        after_feat = self.encoder(after_image)
        # Normalize to (B, N, backbone_dim); a single whole image has N=1.
        if before_feat.dim() == 2:
            before_feat = before_feat.unsqueeze(1)
            after_feat = after_feat.unsqueeze(1)
        return before_feat, after_feat

    def forward(self, before_image: torch.Tensor, after_image: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            before_image: (B, 3, H, W) or (B, N, 3, H, W)
            after_image:  (B, 3, H, W) or (B, N, 3, H, W)
            input_tokens: (B, L) teacher-forcing input token ids
        Returns:
            logits: (B, L, vocab_size)
        """
        before_feat, after_feat = self._encode_pair(before_image, after_image)
        diff_embedding = self.diff_module(before_feat, after_feat)
        return self.decoder(diff_embedding, input_tokens)

    @torch.no_grad()
    def generate_caption(
        self,
        before_image: torch.Tensor,
        after_image: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> List[str]:
        """Greedy-decode a batch of before/after pairs into caption strings.

        Args:
            before_image: (B, 3, H, W) or (B, N, 3, H, W)
            after_image:  (B, 3, H, W) or (B, N, 3, H, W)
        Returns:
            List of B decoded caption strings.
        """
        self.eval()
        before_feat, after_feat = self._encode_pair(before_image, after_image)
        diff_embedding = self.diff_module(before_feat, after_feat)  # (B, N, fusion_dim)

        B = diff_embedding.size(0)
        device = diff_embedding.device
        vocab = self.vocab

        sequences: List[List[int]] = [[vocab.start_idx] for _ in range(B)]
        finished = [False] * B
        for _ in range(max_new_tokens):
            if all(finished):
                break
            max_len = max(len(seq) for seq in sequences)
            input_tokens = torch.full((B, max_len), vocab.pad_idx, dtype=torch.long, device=device)
            for i, seq in enumerate(sequences):
                input_tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

            logits = self.decoder(diff_embedding, input_tokens)
            for i in range(B):
                if finished[i]:
                    continue
                next_token = logits[i, len(sequences[i]) - 1, :].argmax(-1).item()
                sequences[i].append(next_token)
                if next_token == vocab.end_idx:
                    finished[i] = True

        return [vocab.decode(seq) for seq in sequences]
