# RSICC Modularized Pipeline for Remote Sensing Change Captioning

This repository contains a **modularized PyTorch framework** for Remote Sensing Image Change Captioning (RSICC) evaluated on the **LEVIR-CC** and **SECOND-CC** datasets. It includes base models, RemoteCLIP cross-attention models, and hierarchical tile-based models with spatial cross-attention.

---

## 📦 Project Structure

```text
.
├── Levir-CC-dataset/              # LEVIR-CC dataset (images and annotations)
│   ├── LevirCCcaptions.json      # Annotations with train/val/test splits
│   └── images/                   # Split folders: train/, val/, test/
│
├── src/                          # Modularized Python packages
│   ├── config.py                 # Dataclasses for Data, Model, RemoteCLIP & Training configs
│   ├── dataset.py                # Dataset loaders, Vocabulary, and transforms
│   ├── metrics.py                # Evaluation metrics (Sentence embedding, BLEU, CIDEr)
│   ├── training.py               # Training loops, loss calculation, checkpoint save/load
│   ├── utils.py                  # Reproducibility seeds and device setup
│   └── models/                   # Model architectures package
│       ├── baseline/             # Baseline encoder/decoder (SimpleEncoder, SimpleDecoder)
│       ├── final_model/          # Phase 5: RemoteCLIPCrossAttentionModel
│       ├── phase6/               # Phase 6: TileBasedChangeCaptioningModel
│       └── variants/             # Model variants & researcher templates
│
├── phase_final.ipynb             # 🚀 Training notebook for Phase 5 (RemoteCLIP Cross-Attention)
├── phase6.ipynb                  # 🚀 Training notebook for Phase 6 (Hierarchical Tile-Based Model)
├── execute.sh                    # ⚙️ SLURM batch execution script for phase_final.ipynb
├── execute_phase6.sh             # ⚙️ SLURM batch execution script for phase6.ipynb
├── check_pipeline.py             # 🧪 Automated dry-run pipeline compatibility test
├── test_remoteclip_diagnostics.py# 🔬 Local RemoteCLIP feature sensitivity diagnostic suite
├── README.md                     # This file
└── requirements.txt              # Project Python dependencies
```

---

## 🎯 Key Features & Architectures

### 1. Unified Dataset & Vocabulary Pipeline
- `src.dataset.get_levircc_loaders()`: Standardized dataloaders for LEVIR-CC with train/val/test splits.
- `src.dataset.Vocabulary`: Shared word-to-index encoding/decoding tokenization.
- `src.dataset.build_remoteclip_transforms()`: Normalization and transforms tuned for RemoteCLIP backbones.

### 2. Supported Model Architectures

| Model Architecture | Location | Key Design |
| :--- | :--- | :--- |
| **`RSICCformerBaseline`** | `src.models.baseline` | Two-stream CNN encoder + Transformer decoder baseline. |
| **`RemoteCLIPCrossAttentionModel`** | `src.models.final_model` | RemoteCLIP backbone + Bidirectional cross-attention + Contrastive caption alignment. |
| **`TileBasedChangeCaptioningModel`** | `src.models.phase6` | Spatial tile extractor + Shared RemoteCLIP + Bidirectional tile difference + Tile fusion transformer. |

### 3. Spatial Cross-Attention Decoder
All decoders (`SimpleDecoder`) support both 2D and 3D token sequences `(batch, num_tokens, embed_dim)`, allowing the language decoder to perform spatial cross-attention across all patch and tile regions rather than collapsing to a single 1D vector.

---

## 🚀 Quick Start & Local Execution

### Step 1: Environment Setup
```bash
# Activate your conda or venv environment
conda activate mlenv

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Pipeline Verification (Dry-Run Check)
Verify all models, dataloaders, loss functions, and forward/backward passes on your local machine without needing GPU clusters:
```bash
python check_pipeline.py
```

### Step 3: Run RemoteCLIP Feature Diagnostics
Test RemoteCLIP feature sensitivity and zero-shot caption alignment on your local machine:
```bash
python test_remoteclip_diagnostics.py
```

---

## ⚙️ Cluster Training (SLURM Execution)

To run long training jobs on high-performance compute clusters (e.g., ADA cluster):

```bash
# Execute Phase 5 (RemoteCLIP Cross-Attention)
sbatch execute.sh

# Execute Phase 6 (Hierarchical Tile-Based Model)
sbatch execute_phase6.sh
```

---

## 📚 Code Usage Examples

### Example 1: Loading Data and Vocabulary
```python
from src.config import DataConfig
from src.dataset import get_levircc_loaders

data_cfg = DataConfig(batch_size=16)
train_loader, val_loader, test_loader, vocab = get_levircc_loaders(
    caption_json=data_cfg.caption_json,
    image_root=data_cfg.image_root,
    batch_size=data_cfg.batch_size,
)
```

### Example 2: Instantiating Phase 5 Model
```python
from src.models.final_model import RemoteCLIPCrossAttentionModel

model = RemoteCLIPCrossAttentionModel(
    vocab_size=len(vocab),
    encoder_dim=512,
    embed_dim=256,
    num_heads=4,
    num_decoder_layers=2,
    remoteclip_model_name="ViT-B-32",
    download_if_missing=True,
)
```

### Example 3: Running Unit Tests
```python
python -m unittest discover tests
```

---

## 📖 Citation

If you use this codebase, please cite LEVIR-CC:
```bibtex
@article{hasan2021change,
  title={Change Detection in Satellite Imagery with ChangeNet},
  author={Hasan, Ali and Khan, Salman H. and Amir, Muhammad},
  journal={},
  year={2021}
}
```
