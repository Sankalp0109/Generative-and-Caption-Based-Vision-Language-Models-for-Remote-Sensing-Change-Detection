"""
Example custom model template for ablation study researchers.

Copy this file and modify it for your own architecture, e.g.:
    src/models/variants/researcher2_remoteclip_encoder.py
"""

from src.models.baseline.baseline import RSICCformerBaseline
from src.models.baseline.encoder import SimpleEncoder
from src.models.interface import ChangeCaptioningModel


class CustomEncoderVariant(SimpleEncoder):
    """
    Example: inherit SimpleEncoder and override forward().

    Researcher 2 might replace CNN features with RemoteCLIP features here.
    """

    def forward(self, images):
        features = super().forward(images)
        # Example hook point:
        # features = self.extra_fusion_layer(features)
        return features


class CustomBaselineVariant(RSICCformerBaseline):
    """
    Example: swap only the encoder while keeping the shared decoder interface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_dim = kwargs.get("encoder_dim", 512)
        encoder_hidden_dim = kwargs.get("encoder_hidden_dim", 64)
        self.encoder = CustomEncoderVariant(
            in_channels=3,
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_dim,
        )


class FullyCustomModel(ChangeCaptioningModel):
    """
    Example: build a completely new architecture while keeping the same forward signature.
    """

    def __init__(self, vocab_size, encoder_dim=512, embed_dim=256, **kwargs):
        super().__init__()
        self.encoder = SimpleEncoder(out_dim=encoder_dim)
        from src.models.baseline.decoder import SimpleDecoder

        self.decoder = SimpleDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
        )

    def forward(self, images, caption_tokens):
        encoder_features = self.encoder(images)
        return self.decoder(encoder_features, caption_tokens)
