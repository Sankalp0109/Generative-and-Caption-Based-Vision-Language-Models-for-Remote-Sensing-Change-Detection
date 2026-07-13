"""Shared configuration for the RSICC ablation study."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """Paths and preprocessing settings shared by all researchers."""

    data_root: Path = Path("Levir-CC-dataset")
    caption_json: Path = field(init=False)
    image_root: Path = field(init=False)
    checkpoint_dir: Path = Path("checkpoints")
    vocab_path: Path = Path("checkpoints/shared_vocab.pkl")

    img_size: Tuple[int, int] = (256, 256)
    batch_size: int = 16
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    min_word_freq: int = 2
    caption_index: int = 0

    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.caption_json = self.data_root / "LevirCCcaptions.json"
        self.image_root = self.data_root / "images"
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.vocab_path = Path(self.vocab_path)
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size * 2


@dataclass
class ModelConfig:
    """Default baseline hyperparameters. Override per researcher in the notebook."""

    encoder_dim: int = 512
    embed_dim: int = 256
    num_heads: int = 4
    num_decoder_layers: int = 2
    max_caption_len: int = 100
    dropout: float = 0.3
    encoder_hidden_dim: int = 64


@dataclass
class RemoteCLIPConfig:
    """Phase 2 RemoteCLIP encoder settings."""

    model_name: str = "ViT-B-32"
    checkpoint_path: Path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    hf_repo_id: str = "chendelong/RemoteCLIP"
    encoder_dim: int = 512
    freeze_backbone: bool = True
    download_if_missing: bool = True
    fusion_dropout: float = 0.1

    # OpenCLIP's default preprocessing for RemoteCLIP ViT-B-32.
    image_size: Tuple[int, int] = (224, 224)
    image_mean: Tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    image_std: Tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    def __post_init__(self):
        self.checkpoint_path = Path(self.checkpoint_path)


@dataclass
class TrainConfig:
    """Default training settings."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 15
    grad_clip: float = 1.0
    scheduler_t_max: int = 10
    log_every: int = 10
    seed: int = 42
