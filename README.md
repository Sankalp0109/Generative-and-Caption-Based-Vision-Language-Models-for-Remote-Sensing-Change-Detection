# RSICC Modularized Pipeline for Collaborative Ablation Study

This directory contains a **modularized** implementation of the Remote Sensing Image Change Captioning (RSICC) baseline for collaborative ablation study on the LEVIR-CC dataset.

## 📦 Project Structure

```
.
├── Levir-CC-dataset/              # LEVIR-CC dataset (images and annotations)
│   ├── LevirCCcaptions.json      # Annotations with train/val/test splits
│   └── images/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/                          # Modularized Python packages
│   ├── __init__.py
│   ├── dataset.py                # ✅ Shared: Vocabulary, LEVIRCCDataset, DataLoader helpers
│   └── models.py                 # ✅ Shared: SimpleEncoder, SimpleDecoder, RSICCformerBaseline
│
├── phase2_modular_pipeline.ipynb # 🎓 Template: Main training notebook for all researchers
├── checkpoints/                  # 📁 Save model checkpoints here
├── README.md                     # This file
└── SETUP.md                      # Environment and installation guide
```

## 🎯 Key Features

### ✅ Unified Dataset Loading
All researchers use **identical** data preprocessing and vocabulary:
- `src.dataset.get_levircc_loaders()` - Loads train/val/test with standard transforms
- `src.dataset.Vocabulary` - Shared tokenization and encoding/decoding
- `src.dataset.LEVIRCCDataset` - PyTorch Dataset class with before/after image pairs
- `src.dataset.CaptionCollate` - Batching function with automatic padding

### ✅ Base Model Architectures
- `src.models.SimpleEncoder` - Two-stream CNN with difference fusion
- `src.models.SimpleDecoder` - Transformer decoder with self/cross-attention
- `src.models.RSICCformerBaseline` - Complete end-to-end baseline

Researchers can:
1. **Use baseline as-is** for baseline experiments
2. **Inherit and modify** for ablation variants
3. **Create custom models** using the same interface

### ✅ Standardized Training Loop
`phase2_modular_pipeline.ipynb` provides:
- Data loading and vocabulary setup
- Model instantiation (baseline or custom)
- Training loop with gradient clipping
- Validation and loss tracking
- Checkpoint save/load utilities
- Inference and visualization functions

## 🚀 Quick Start for Each Researcher

### Step 1: Environment Setup
```bash
# If not already done, create clean conda environment
conda create -n rsicc-py311 python=3.11
conda activate rsicc-py311

# Install PyTorch (CPU or CUDA)
pip install torch torchvision

# Install other dependencies
pip install pillow numpy matplotlib
```

### Step 2: Get Your Copy
```bash
# Copy the template notebook with a unique name for you
cp phase2_modular_pipeline.ipynb researcher_NAME_ablation.ipynb

# Open in Jupyter
jupyter notebook researcher_NAME_ablation.ipynb
```

### Step 3: Customize Model (Cell 4)
Edit the model instantiation in Cell 4 of your notebook:

**Option A: Use Baseline**
```python
model = RSICCformerBaseline(
    vocab_size=len(vocab.word2idx),
    encoder_dim=512,
    embed_dim=256,
    num_heads=4,
    num_decoder_layers=2,
    max_caption_len=100,
    dropout=0.1
).to(device)
```

**Option B: Modify Encoder (e.g., add RemoteCLIP)**
```python
class EncoderWithRemoteCLIP(SimpleEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remoteclip = load_remoteclip_checkpoint(...)
    
    def forward(self, x):
        remoteclip_feat = self.remoteclip(x)
        cnn_feat = super().forward(x)
        return torch.cat([remoteclip_feat, cnn_feat], dim=1)

# Create model and replace encoder
model = RSICCformerBaseline(...)
model.encoder = EncoderWithRemoteCLIP(...).to(device)
```

**Option C: Modify Decoder (e.g., add cross-attention)**
```python
class DecoderWithCrossAttention(SimpleDecoder):
    def forward(self, encoder_features, captions):
        # Your custom logic here
        pass

# Create model and replace decoder
model = RSICCformerBaseline(...)
model.decoder = DecoderWithCrossAttention(...).to(device)
```

**Option D: Create Completely Custom Model**
```python
class MyCustomRSICCModel(nn.Module):
    def __init__(self, vocab_size, ...):
        super().__init__()
        # Your architecture
    
    def forward(self, images, captions):
        # Your forward pass
        pass

model = MyCustomRSICCModel(...).to(device)
```

### Step 4: Train and Validate
Run cells in order:
1. Cell 1-3: Setup, data loading, model instantiation
2. Cell 4: Train for 1 epoch (adjust num_epochs as needed)
3. Cell 5: Validate
4. Cell 6: Save checkpoint
5. Cell 7: Visualize predictions on test set

### Step 5: Save Results
```python
# Checkpoints are automatically saved to checkpoints/ directory
# Save with a unique identifier for your variant:
# RSICC Ablation Study (Modular)

Minimal shared setup for 3 researchers working on different model variants with the same data pipeline.

## What is shared

- Dataset and batching: [src/dataset.py](src/dataset.py)
- Model interfaces/baseline: [src/models](src/models)
- Training helpers: [src/training.py](src/training.py)
- Metrics (BLEU, METEOR, ROUGE-L, CIDEr, semantic similarity): [src/metrics.py](src/metrics.py)
- Main template notebook: [phase2_modular_pipeline.ipynb](phase2_modular_pipeline.ipynb)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure dataset exists:

```text
Levir-CC-dataset/
  LevirCCcaptions.json
  images/{train,val,test}/{A,B}/...
```

## Team workflow

1. Copy [phase2_modular_pipeline.ipynb](phase2_modular_pipeline.ipynb) per person.
2. Change only model section (Section 3).
3. Keep dataset/training/metrics sections unchanged.
4. Save checkpoints with unique names per researcher.

## Shared batch format

All models must consume:

- `images`: `(B, 2, 3, H, W)`
- `caption_tokens`: `(B, L)`

Additional keys available for analysis:

- `captions`, `before_images`, `after_images`, `filenames`, `changeflags`

## Evaluation outputs

Notebook evaluation section computes:

- 40-sample qualitative + metrics
- full test-set metrics

Saved JSON:

- `checkpoints/<RESEARCHER_NAME>_test_metrics.json`

## Notes

- Run notebook from project root so `src` imports work.
- Use consistent `RESEARCHER_NAME` to avoid checkpoint overwrite.
```python
