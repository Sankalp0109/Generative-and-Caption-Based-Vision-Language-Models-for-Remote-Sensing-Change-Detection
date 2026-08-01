# RSICC Phase 7: CodeAug — Indian Urban & Seasonal Domain Adaptation

> [!IMPORTANT]
> **🚀 BRANCH `aug`: Complete Phase 7 Implementation (The CodeAug Protocol)**
> We have implemented and empirical-calibrated the complete **CodeAug RSICC Phase 7** architecture specifically tailored for Indian urban morphology and seasonal monsoonal vegetation shifts, running on ADA cluster GPUs (`gnode004`, GTX 1080 Ti 11 GB VRAM) and CPU compute nodes.
> * **📖 Comprehensive Guide:** See [README_CODEAUG.md](file:///home/rinkeshverma/Desktop/Projects/PJ/README_CODEAUG.md) for full architectural diagrams, derivations, pre-flight calibration instructions, and step-by-step training.
> * **📄 Formal PDF Reference:** See [CodeAug.pdf](file:///home/rinkeshverma/Desktop/Projects/PJ/CodeAug.pdf) (`13.8 KB`).
> * **⚡ Automated Training Script:** `sbatch execute_phase7.sh` (or interactively run [phase7.ipynb](file:///home/rinkeshverma/Desktop/Projects/PJ/phase7.ipynb)).

This repository contains our **Phase 7 PyTorch framework** for Remote Sensing Image Change Captioning (RSICC) evaluated on LEVIR-CC and Indian domain datasets.

---

## 📦 Project Structure

```text
.
├── Levir-CC-dataset/              # LEVIR-CC dataset (images and annotations)
│   ├── LevirCCcaptions.json      # Annotations with train/val/test splits
│   └── images/                   # Split folders: train/, val/, test/
│
├── src/                          # Cleaned Phase 7 Modular Python library
│   ├── config.py                 # Dataclasses for Data, Model & Training configs
│   ├── dataset.py                # Dataset loaders, Vocabulary, and transforms
│   ├── codeaug_dataset.py        # Phase 7 GLI Union Masking & 2:1 Balanced Sampler
│   ├── metrics.py                # 7-Metric Evaluation Suite (BLEU, METEOR, ROUGE, CIDEr, Cosine, SFPR, SFNR)
│   ├── training.py               # Training loops, loss calculation, checkpoint save/load
│   ├── utils.py                  # Reproducibility seeds and device setup
│   └── models/                   # Model architectures package
│       ├── __init__.py           # Clean Phase 7 exports
│       └── codeaug/              # Phase 7 CodeAug Architecture (ViT-L-14 + LoRA + Q-Former + Qwen2-VL-2B)
│
├── phase7.ipynb                  # 🚀 Phase 7 CodeAug Two-Stage Training & 7-Metric Evaluation Notebook
├── execute_phase7.sh             # ⚙️ Production SLURM batch execution script for phase7.ipynb
├── check_vram_calibration.py     # 🧪 Mandatory Empirical VRAM & Parameter Calibration Script
├── generate_codeaug_pdf.py       # 📄 Generator for the formal CodeAug.pdf master plan
├── CodeAug.pdf                   # 📄 3-Page CodeAug Master Plan Reference
├── README_CODEAUG.md             # 📖 Detailed CodeAug Architecture & Training Documentation
├── README.md                     # This top-level overview
└── requirements.txt              # Unified Phase 7 Python dependencies
```

---

## 🚀 Key Architectural Breakthroughs in Phase 7

1. **Pretrained Causal Multimodal Decoder (`Qwen2-VL-2B-Instruct`)**:
   Replaces scratch-trained decoders with a 4-bit NF4 quantized large language model (`bnb_4bit_compute_dtype="float16"` for Pascal CC 6.1 compatibility).
2. **Q-Former Visual Compression (`num_queries=64`)**:
   Compresses 324 spatial patch tokens from frozen **OpenCLIP ViT-L-14 (FP16)** down to 64 learnable latent queries (**61.3% token reduction**).
3. **Visual LoRA (`r=16, alpha=32`)**:
   Injects rank-16 low-rank adapters into all 48 MLP projection layers (`c_fc`, `c_proj`) across the 24 ViT-L-14 blocks.
4. **Bi-Temporal Union GLI Masking ($M_{\text{veg}} = M_A \cup M_B$)**:
   Solves seasonal vegetation false positives by flagging vegetation in either dry or monsoon frames and applying symmetric jitter to both.
5. **Single-Lever Imbalance Control**:
   Uses a **2:1 WeightedRandomSampler** with unweighted Cross-Entropy ($\gamma=1.0$) to eliminate compounding bias and enforce **SFPR $< 5\%$**.

---

## ⚡ Running Phase 7 on the ADA Cluster

### 1. Verification & Calibration (Mandatory Pre-Flight)
Before launching SLURM training, verify that VRAM headroom fits within the GTX 1080 Ti (11.0 GB) envelope:
```bash
python check_vram_calibration.py
```
*Empirical Peak VRAM:* **3.48 GB** (7.52 GB headroom on 11 GB GTX 1080 Ti).

### 2. SLURM Execution
Submit the production pipeline script to SLURM (`gnode004`, 1 GPU, `num_workers=0`):
```bash
sbatch execute_phase7.sh
```
This executes `phase7.ipynb` sequentially through:
- Stage 1: LEVIR-CC Pre-Training (Master Syntax & Base Semantics)
- Stage 2: Indian Few-Shot Domain Adaptation (Urban Morphology & Seasonal Shifts)
- Evaluation: 7-Metric Benchmark Suite (BLEU-1 to 4, METEOR, ROUGE-L, Anchored CIDEr, Semantic Cosine Sim, SFPR, SFNR)
