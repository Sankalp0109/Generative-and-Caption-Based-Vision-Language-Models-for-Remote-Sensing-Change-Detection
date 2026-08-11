"""Baseline RSICCformer-style end-to-end model."""

from .decoder import SimpleDecoder
from .encoder import SimpleEncoder
from ..interface import ChangeCaptioningModel


class RSICCformerBaseline(ChangeCaptioningModel):
    """
    Default baseline model used as the reference point in the ablation study.

    Researchers can:
      1. use this class directly,
      2. replace encoder/decoder with custom modules,
      3. inherit from ChangeCaptioningModel and build a new architecture.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        encoder_dim: int = 512,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_decoder_layers: int = 2,
        max_caption_len: int = 100,
        dropout: float = 0.1,
        encoder_hidden_dim: int = 64,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.encoder = SimpleEncoder(
            in_channels=3,
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_dim,
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

    def forward(self, images, caption_tokens):
        encoder_features = self.encoder(images)
        return self.decoder(encoder_features, caption_tokens)
