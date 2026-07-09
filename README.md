# RSICC Modularized Pipeline for Collaborative Ablation Study

This directory contains a **modularized** implementation of the Remote Sensing Image Change Captioning (RSICC) baseline for collaborative ablation study on the LEVIR-CC Dataset.

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
# Checkpoints are saved during training as rolling files:
# - researcher_NAME_current.pt after every epoch
# - researcher_NAME_best.pt when validation loss improves
RESEARCHER_NAME = 'researcher_NAME_variant'
save_checkpoint(
    model, optimizer, epoch=10, loss=val_loss, vocab=vocab,
    checkpoint_dir=CHECKPOINT_DIR,
    name=RESEARCHER_NAME,
    filename=f'{RESEARCHER_NAME}_current.pt',
)
```

## 📊 Comparison Protocol

All results are directly comparable because they share:
- ✅ Same **train/val/test splits** (from `LevirCCcaptions.json`)
- ✅ Same **vocabulary** (`src.dataset.Vocabulary`)
- ✅ Same **image preprocessing** (resize to 256×256, ImageNet normalization)
- ✅ Same **batch size** (default 16)
- ✅ Same **loss function** (CrossEntropyLoss)
- ✅ Same **evaluation protocol** (validation loss, test inference)

### Metrics to Report
```python
# For each checkpoint, compute and save:
{
    'model': 'baseline | your_variant_name',
    'epoch': int,
    'train_loss': float,
    'val_loss': float,
    'test_loss': float,
    'num_parameters': int,
    'checkpoint_path': str,
    'notes': 'Description of any modifications'
}
```

## 📚 Code Examples

### Example 1: Loading Data
```python
from src.dataset import get_levircc_loaders

train_loader, val_loader, test_loader, vocab = get_levircc_loaders(
    caption_json='Levir-CC-dataset/LevirCCcaptions.json',
    image_root='Levir-CC-dataset/images',
    batch_size=16,
    device=device,
    img_size=(256, 256)
)
```

### Example 2: Using Vocabulary
```python
from src.dataset import Vocabulary

# Encode a caption
caption_tokens = vocab.encode("A new building appeared")
# Output: tensor([1, 1234, 567, 890, 2])  # START, word, word, word, END

# Decode tokens back to text
caption_text = vocab.decode(caption_tokens)
# Output: "<START> a new building appeared <END>"
```

### Example 3: Inference on Single Image Pair
```python
def generate_caption(model, before_image, after_image, vocab, device):
    model.eval()
    with torch.no_grad():
        images = torch.stack([before_image, after_image]).unsqueeze(0).to(device)
        encoder_features = model.encoder(images)
        
        caption_tokens = [vocab.word2idx.get('<START>', 1)]
        for _ in range(100):
            cap_tensor = torch.tensor([caption_tokens], dtype=torch.long, device=device)
            logits = model.decoder(encoder_features, cap_tensor)
            next_token = logits[0, -1, :].argmax(-1).item()
            caption_tokens.append(next_token)
            if next_token == vocab.word2idx.get('<END>', 2):
                break
    
    return vocab.decode(torch.tensor(caption_tokens))
```

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're running the notebook from the project root directory
```bash
cd /path/to/RSICC/project
jupyter notebook phase2_modular_pipeline.ipynb
```

### Issue: "TORCH_AVAILABLE = False"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Issue: Dataset not found
**Solution**: Check that `Levir-CC-dataset/` exists in the project root
```bash
ls -la Levir-CC-dataset/
# Should show: LevirCCcaptions.json, images/
```

### Issue: "out of memory" or slow training
**Solution**: Reduce batch size in `get_levircc_loaders()`
```python
train_loader, val_loader, test_loader, vocab = get_levircc_loaders(
    ...,
    batch_size=8,  # Reduced from 16
    ...
)
```

## 📖 API Reference

### `src.dataset`

#### `get_levircc_loaders(caption_json, image_root, vocab=None, batch_size=8, device='cpu', img_size=(256, 256))`
Returns `(train_loader, val_loader, test_loader, vocab)` with all dataloaders ready to use.

#### `Vocabulary.build_vocab(captions_list)`
Builds vocabulary from a list of caption strings.

#### `Vocabulary.encode(caption)` → torch.Tensor
Converts caption string to token indices with START and END tokens.

#### `Vocabulary.decode(indices)` → str
Converts token indices back to caption string.

#### `LEVIRCCDataset(samples, image_root, caption_index=0, transforms_fn=None)`
PyTorch Dataset class. Returns dict with 'before_image', 'after_image', 'caption', 'changeflag', 'filename'.

### `src.models`

#### `SimpleEncoder(in_channels=3, hidden_dim=64, out_dim=512)`
Extracts features from before/after image pairs.
- **Input**: (batch, 2, 3, H, W)
- **Output**: (batch, out_dim)

#### `SimpleDecoder(vocab_size, embed_dim=256, num_heads=4, num_layers=2, max_len=100, encoder_dim=512, dropout=0.1)`
Generates captions from encoded features.
- **Input**: encoder_features (batch, encoder_dim), captions (batch, seq_len)
- **Output**: logits (batch, seq_len, vocab_size)

#### `RSICCformerBaseline(vocab_size, encoder_dim=512, embed_dim=256, num_heads=4, num_decoder_layers=2, max_caption_len=100, dropout=0.1)`
Complete end-to-end model combining encoder and decoder.
- **Input**: images (batch, 2, 3, H, W), captions (batch, seq_len)
- **Output**: logits (batch, seq_len, vocab_size)

## 📝 Citation

If you use this codebase, please cite LEVIR-CC:
```
@article{hasan2021change,
  title={Change Detection in Satellite Imagery with ChangeNet},
  author={Hasan, Ali and Khan, Salman H. and Amir, Muhammad},
  journal={},
  year={2021}
}
```

## 📞 Contact

For questions about the modularized pipeline, contact the research team.

---

**Last Updated**: 2024
**Python Version**: 3.11+
**PyTorch Version**: 2.0+
