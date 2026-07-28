"""Unified LEVIR-CC dataset loading for the RSICC ablation study."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms



SPECIAL_TOKENS = ("<PAD>", "<START>", "<END>", "<UNK>")


class Vocabulary:
    """Shared vocabulary for caption tokenization across all model variants."""

    def __init__(self, min_freq: int = 2):
        self.word2idx: Dict[str, int] = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }
        self.idx2word: Dict[int, str] = {idx: word for word, idx in self.word2idx.items()}
        self.word_freq = Counter()
        self.min_freq = min_freq
        self.next_idx = 4

    @property
    def pad_idx(self) -> int:
        return self.word2idx["<PAD>"]

    @property
    def start_idx(self) -> int:
        return self.word2idx["<START>"]

    @property
    def end_idx(self) -> int:
        return self.word2idx["<END>"]

    @property
    def unk_idx(self) -> int:
        return self.word2idx["<UNK>"]

    def build_vocab(self, captions_list: List[str]) -> None:
        """Build vocabulary from a list of caption strings."""
        for caption in captions_list:
            for token in caption.lower().split():
                self.word_freq[token] += 1

        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1

    def encode(self, caption: str):
        """Convert caption string to token indices with START/END tokens."""
        tokens = [self.start_idx]
        tokens.extend(self.word2idx.get(word, self.unk_idx) for word in caption.lower().split())
        tokens.append(self.end_idx)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, indices, skip_special: bool = True) -> str:
        """Convert token indices back to a caption string."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        words = []
        for idx in indices:
            word = self.idx2word.get(int(idx), "<UNK>")
            if skip_special and word in SPECIAL_TOKENS:
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word,
                    "word_freq": dict(self.word_freq),
                    "min_freq": self.min_freq,
                    "next_idx": self.next_idx,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        vocab = cls(min_freq=payload["min_freq"])
        vocab.word2idx = payload["word2idx"]
        vocab.idx2word = {int(k): v for k, v in payload["idx2word"].items()}
        vocab.word_freq = Counter(payload["word_freq"])
        vocab.next_idx = payload["next_idx"]
        return vocab


def load_levircc_annotations(caption_json: Path) -> Dict:
    """Load LEVIR-CC annotation JSON."""
    with open(caption_json, "r", encoding="utf-8") as f:
        return json.load(f)


def split_samples_by_split(annotations: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split annotation records into train/val/test lists."""
    images = annotations["images"]
    train_samples = [item for item in images if item["split"] == "train"]
    val_samples = [item for item in images if item["split"] == "val"]
    test_samples = [item for item in images if item["split"] == "test"]
    return train_samples, val_samples, test_samples


def build_vocabulary_from_annotations(
    annotations: Dict,
    min_freq: int = 2,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Vocabulary:
    """Build a shared vocabulary from selected dataset splits."""
    captions = []
    for sample in annotations["images"]:
        if sample["split"] in splits:
            captions.extend(sent["raw"].strip() for sent in sample["sentences"])

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(captions)
    return vocab


def build_vocabulary_from_multiple_annotations(
    annotations_list: List[Dict],
    min_freq: int = 2,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Vocabulary:
    """Build a single shared vocabulary from multiple annotation sources."""
    captions = []
    for annotations in annotations_list:
        for sample in annotations["images"]:
            if sample["split"] in splits:
                captions.extend(sent["raw"].strip() for sent in sample["sentences"])

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(captions)
    return vocab


def build_image_transforms(
    img_size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """Shared image preprocessing used by every researcher."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_remoteclip_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    ),
    std: Tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    ),
):
    """Preprocess images using the OpenCLIP evaluation convention."""
    return transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def extract_ordered_patches_from_pil(
    img: Image.Image,
    patch_size: int = 256,
    transforms_fn=None,
) -> torch.Tensor:
    """Extract non-overlapping patches of size patch_size x patch_size from a PIL image without resizing.

    If image dimensions are not divisible by patch_size, zero-pads right/bottom borders.
    Patches are extracted in strict row-major spatial order: patch0, patch1, ..., patchN-1.

    Returns:
        patches: Tensor of shape (N, C, patch_h, patch_w)
    """
    import math

    w, h = img.size
    padded_w = math.ceil(w / patch_size) * patch_size
    padded_h = math.ceil(h / patch_size) * patch_size

    if padded_w != w or padded_h != h:
        padded_img = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
        padded_img.paste(img, (0, 0))
        img = padded_img

    patch_tensors = []
    for r in range(0, padded_h, patch_size):
        for c in range(0, padded_w, patch_size):
            patch_crop = img.crop((c, r, c + patch_size, r + patch_size))
            if transforms_fn is not None:
                patch_tensor = transforms_fn(patch_crop)
            else:
                patch_tensor = transforms.ToTensor()(patch_crop)
            patch_tensors.append(patch_tensor)

    return torch.stack(patch_tensors, dim=0)


class LEVIRCCDataset(Dataset):
    """PyTorch dataset for before/after image pairs and captions with patch extraction."""

    def __init__(
        self,
        samples: List[Dict],
        image_root: Path,
        caption_index: int = 0,
        transforms_fn=None,
        patch_size: Optional[int] = 256,
    ):
        self.samples = samples
        self.image_root = Path(image_root)
        self.caption_index = caption_index
        self.transforms = transforms_fn
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        before_path = self.image_root / sample["filepath"] / "A" / sample["filename"]
        after_path = self.image_root / sample["filepath"] / "B" / sample["filename"]

        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")

        if self.patch_size is not None:
            before_img = extract_ordered_patches_from_pil(
                before_img, patch_size=self.patch_size, transforms_fn=self.transforms
            )
            after_img = extract_ordered_patches_from_pil(
                after_img, patch_size=self.patch_size, transforms_fn=self.transforms
            )
        elif self.transforms:
            before_img = self.transforms(before_img)
            after_img = self.transforms(after_img)

        caption = sample["sentences"][self.caption_index]["raw"].strip()
        return {
            "before_image": before_img,
            "after_image": after_img,
            "caption": caption,
            "changeflag": sample["changeflag"],
            "filename": sample["filename"],
        }


class SecondCCDataset(Dataset):
    """PyTorch dataset wrapper for the SECOND-CC-AUG annotation schema with patch extraction."""

    def __init__(
        self,
        samples: List[Dict],
        image_root: Path,
        caption_index: int = 0,
        transforms_fn=None,
        patch_size: Optional[int] = 256,
    ):
        self.samples = samples
        self.image_root = Path(image_root)
        self.caption_index = caption_index
        self.transforms = transforms_fn
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        before_path = self.image_root / sample["filepath"] / "rgb" / "A" / sample["filename"]
        after_path = self.image_root / sample["filepath"] / "rgb" / "B" / sample["filename"]

        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")

        if self.patch_size is not None:
            before_img = extract_ordered_patches_from_pil(
                before_img, patch_size=self.patch_size, transforms_fn=self.transforms
            )
            after_img = extract_ordered_patches_from_pil(
                after_img, patch_size=self.patch_size, transforms_fn=self.transforms
            )
        elif self.transforms:
            before_img = self.transforms(before_img)
            after_img = self.transforms(after_img)

        caption = sample["sentences"][self.caption_index]["raw"].strip()
        return {
            "before_image": before_img,
            "after_image": after_img,
            "caption": caption,
            "changeflag": sample["changeflag"],
            "filename": sample["filename"],
        }


class TestDataset(Dataset):
    """PyTorch test dataset wrapper with patch extraction."""

    def __init__(
        self,
        image_root: Path,
        transforms_fn=None,
        patch_size: Optional[int] = 256,
    ):
        self.image_root = Path(image_root)
        self.transforms = transforms_fn
        self.patch_size = patch_size
        self.num_samples = len(list((self.image_root / "before").glob("*.png")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        filename = f"Img{idx+1}.png"
        before_path = self.image_root / "before" / filename
        after_path = self.image_root / "after" / filename

        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")

        if self.patch_size is not None:
            before_img = extract_ordered_patches_from_pil(
                before_img, patch_size=self.patch_size, transforms_fn=self.transforms
            )
            after_img = extract_ordered_patches_from_pil(
                after_img, patch_size=self.patch_size, transforms_fn=self.transforms
            )
        elif self.transforms:
            before_img = self.transforms(before_img)
            after_img = self.transforms(after_img)

        return {
            "before_image": before_img,
            "after_image": after_img,
            "filename": filename,
        }


class CaptionCollate:
    """Collate function that produces a standardized batch dictionary.

    Every model variant receives the same batch keys:
        - before_images: (B, N, 3, H, W) or (B, 3, H, W)
        - after_images:  (B, N, 3, H, W) or (B, 3, H, W)
        - images:        (B, N, 2, 3, H, W) or (B, 2, 3, H, W)
        - caption_tokens:(B, L) padded token ids
        - captions:      list[str] raw reference captions
        - changeflags:   (B,)
        - filenames:     list[str]
    """

    def __init__(self, vocab: Vocabulary, device: str = "cpu"):
        self.vocab = vocab
        self.device = device

    def __call__(self, batch: List[Dict]) -> Dict:
        before_images = torch.stack([item["before_image"] for item in batch])
        after_images = torch.stack([item["after_image"] for item in batch])
        # Stack temporal pair: if 4D (N, C, H, W) -> stack at dim 2: (B, N, 2, C, H, W)
        # if 3D (C, H, W) -> stack at dim 1: (B, 2, C, H, W)
        temporal_dim = 2 if before_images.ndim == 5 else 1
        images = torch.stack([before_images, after_images], dim=temporal_dim)

        captions = [item["caption"] for item in batch]
        changeflags = torch.tensor([item["changeflag"] for item in batch], dtype=torch.long)
        filenames = [item["filename"] for item in batch]

        caption_tokens = [self.vocab.encode(caption) for caption in captions]
        max_len = max(len(tokens) for tokens in caption_tokens)
        padded_captions = torch.full(
            (len(batch), max_len),
            self.vocab.pad_idx,
            dtype=torch.long,
        )
        for i, tokens in enumerate(caption_tokens):
            padded_captions[i, : len(tokens)] = tokens

        return {
            "before_images": before_images,
            "after_images": after_images,
            "images": images,
            "caption_tokens": padded_captions,
            "captions": captions,
            "changeflags": changeflags,
            "filenames": filenames,
        }

def get_test_loaders(
        image_root: Path,
        device: str = "cpu",
        img_size: Tuple[int, int] = (256, 256),
        num_workers: int = 0,
        transforms_fn=None,
):

    image_transforms = transforms_fn or build_image_transforms(img_size=img_size)

    test_ds = TestDataset(
        image_root,
        transforms_fn=image_transforms,
    )
    val_batch_size=43

    loader_kwargs = {"num_workers": num_workers, "pin_memory": device != "cpu"}

    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return test_loader
    
    

def get_levircc_loaders(
    caption_json: Path,
    image_root: Path,
    vocab: Optional[Vocabulary] = None,
    batch_size: int = 8,
    val_batch_size: Optional[int] = None,
    device: str = "cpu",
    img_size: Tuple[int, int] = (256, 256),
    min_word_freq: int = 2,
    caption_index: int = 0,
    num_workers: int = 0,
    vocab_path: Optional[Path] = None,
    transforms_fn=None,
    patch_size: Optional[int] = 256,
):
    """
    Load LEVIR-CC train/val/test DataLoaders with a shared vocabulary.

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    annotations = load_levircc_annotations(caption_json)
    train_samples, val_samples, test_samples = split_samples_by_split(annotations)

    if vocab is not None:
        if vocab_path and not Path(vocab_path).exists():
            vocab.save(vocab_path)
    elif vocab_path and Path(vocab_path).exists():
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = build_vocabulary_from_annotations(annotations, min_freq=min_word_freq)
        if vocab_path:
            vocab.save(vocab_path)

    image_transforms = transforms_fn or build_image_transforms(img_size=img_size)
    collate_fn = CaptionCollate(vocab, device=device)
    val_batch_size = val_batch_size or batch_size * 2

    train_ds = LEVIRCCDataset(
        train_samples,
        image_root,
        caption_index=caption_index,
        transforms_fn=image_transforms,
        patch_size=patch_size,
    )
    val_ds = LEVIRCCDataset(
        val_samples,
        image_root,
        caption_index=caption_index,
        transforms_fn=image_transforms,
        patch_size=patch_size,
    )
    test_ds = LEVIRCCDataset(
        test_samples,
        image_root,
        caption_index=caption_index,
        transforms_fn=image_transforms,
        patch_size=patch_size,
    )

    loader_kwargs = {"num_workers": num_workers, "pin_memory": device != "cpu"}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, vocab


def get_secondcc_loaders(
    caption_json: Path,
    image_root: Path,
    vocab: Optional[Vocabulary] = None,
    batch_size: int = 8,
    val_batch_size: Optional[int] = None,
    device: str = "cpu",
    img_size: Tuple[int, int] = (256, 256),
    min_word_freq: int = 2,
    caption_index: int = 0,
    num_workers: int = 0,
    vocab_path: Optional[Path] = None,
    transforms_fn=None,
    patch_size: Optional[int] = 256,
):
    """
    Load SECOND-CC train/val/test DataLoaders with a shared vocabulary.

    This mirrors get_levircc_loaders so the final-phase notebook can train on
    LEVIR-CC first and then continue fine-tuning on SECOND-CC without changing
    the model interface.
    """
    annotations = load_levircc_annotations(caption_json)
    train_samples, val_samples, test_samples = split_samples_by_split(annotations)

    if vocab is not None:
        if vocab_path and not Path(vocab_path).exists():
            vocab.save(vocab_path)
    elif vocab_path and Path(vocab_path).exists():
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = build_vocabulary_from_annotations(annotations, min_freq=min_word_freq)
        if vocab_path:
            vocab.save(vocab_path)

    image_transforms = transforms_fn or build_image_transforms(img_size=img_size)
    collate_fn = CaptionCollate(vocab, device=device)
    val_batch_size = val_batch_size or batch_size * 2

    train_ds = SecondCCDataset(
        train_samples,
        image_root,
        caption_index=caption_index,
        transforms_fn=image_transforms,
        patch_size=patch_size,
    )
    val_ds = SecondCCDataset(
        val_samples,
        image_root,
        caption_index=caption_index,
        transforms_fn=image_transforms,
        patch_size=patch_size,
    )
    test_ds = SecondCCDataset(
        test_samples,
        image_root,
        caption_index=caption_index,
        transforms_fn=image_transforms,
        patch_size=patch_size,
    )

    loader_kwargs = {"num_workers": num_workers, "pin_memory": device != "cpu"}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, vocab
