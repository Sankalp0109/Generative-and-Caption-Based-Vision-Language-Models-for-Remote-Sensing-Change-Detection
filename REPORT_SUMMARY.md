# RSICC Independent Study — Detailed Report Summary

Purpose: a self-contained source document so that the final report can be written without opening the code, notebooks or logs. Every number below was extracted from a notebook output, a log, a config file or a README, and the source is named. Where a number does not exist, this is stated explicitly (never filled in). Statements labelled **Interpretation** are analysis drawn from the numbers, not measurements.

Contents: 1 Problem · 2 Datasets · 3 Evaluation metrics · 4 Experimental timeline · 5 Models (per phase) · 6 Training setups · 7 Results · 8 Analysis · 9 Issues and fixes · 10 Validity caveats · 11 Conclusions · 12 Limitations / future work · 13 Suggested report outline · 14 Artifacts · 15 Gaps for the partner

---

## 1. Problem and motivation

**Task.** Remote Sensing Image Change Captioning (RSICC): given a bi-temporal pair of satellite images (before / after), generate a natural-language sentence describing what changed (e.g. "a new building appears", "no change").

**Why a second domain.** Models trained on LEVIR-CC (Western/Chinese suburban scenes) degrade on Indian urban imagery. The project's stated reasons (README_CODEAUG.md):
1. *Urban morphology*: dense, organic settlements and varied rooftops (tin sheets, terracotta, RCC slabs, asbestos) unlike planned suburbs.
2. *Monsoon seasonality*: dry-season brown vs monsoon green vegetation produces large colour shifts that a model can mistake for construction/demolition (seasonal false positives).

**Project structure.** A progressive ablation study: each phase changes one component (encoder, difference mechanism, decoder, training stage) and is evaluated with the same metric suite on LEVIR-CC, then adapted/evaluated on a second dataset (SECOND-CC-AUG). Two researchers ("researcher1" = baseline, "researcher2" = RemoteCLIP-difference) produced Phases 1–2; later phases (final, 6, 7, 8) extend them. The Phase 7 "CodeAug" design targets the Indian-domain problem with an LLM decoder.

---

## 2. Datasets

| Dataset | Size / splits | Details |
|---|---|---|
| **LEVIR-CC** (`Levir-CC-dataset/LevirCCcaptions.json`) | 10,077 image-pair records; train / val / test = **6,815 / 1,333 / 1,929** | Batch counts: batch 16 → 426 / 42 / 61 (phases 1–2); batch 8 → 852 / 84 / 121 (phase 8). Vocabulary: **733** words (min word freq 2) for phases 1, 2, 8; **1,300** for phases 6 / final (shared LEVIR + SECOND-CC vocab). Caption index 0 used for training. |
| **SECOND-CC-AUG** (`SECOND-CC-AUG/SECOND-CC-AUG.json`, gitignored) | test = **1,227**; train/val sample counts are *not printed anywhere* (loader batches at bs 16: train 1,055 / val 75 / test 77) | The "second domain" in the logged runs (called the Indian domain in the project text). |
| New Indian Sentinel-2 dataset (branch `IS/rogue`, commit 30aef9f, 2026-09-09) | 60 areas of interest, 5 change categories (urban, deforestation, agriculture, coastal, infrastructure), 512×512 tiles; 2,126 tiles checked → **1,072 pairs kept** (2,144 images, ≈923 MB), built in 1 h 53 min with 12 workers | Planned labelling with Qwen2-VL-7B (4-bit); planned model: RemoteCLIP ViT-L-14 contrastive + small pretrained LM decoder (GPT-2 / DistilGPT2). A change-labelling web UI was added on 2026-10-01 (branch `IS/man1`). **No results exist.** |

Pre-processing: images resized to 256×256 (224×224 for RemoteCLIP ViT-B-32 paths; 252×252 for the CodeAug ViT-L-14 design); phase 1 used ImageNet mean/std, RemoteCLIP phases use CLIP mean/std (0.4815, 0.4578, 0.4082 / 0.2686, 0.2613, 0.2758).

---

## 3. Evaluation metrics (`src/metrics.py`)

| Metric | Implementation | Notes |
|---|---|---|
| BLEU-1…4 | NLTK corpus BLEU | n-gram precision; BLEU-4 is the headline metric in the RSICC literature. |
| METEOR | NLTK-style averaged METEOR | Reported on 0–1 scale. |
| ROUGE-L | Own LCS-based implementation (F1 vs. best reference) | |
| "CIDEr" | **Project-specific approximation**: TF-IDF n-gram (n ≤ 4) cosine, IDF over the reference corpus | **Not COCO CIDEr.** Values around 5 are not comparable to published CIDEr (typically 0–~140 scale). Use only for within-project comparison. |
| Semantic cosine | Mean cosine similarity of SentenceTransformer `all-MiniLM-L6-v2` embeddings of prediction vs reference (TF-IDF fallback if unavailable) | Measures meaning overlap independent of exact wording. |
| Loss / token accuracy | Cross-entropy ignoring pad; token accuracy under teacher forcing | Train / val / test. |
| SFPR / SFNR (Seasonal False-Positive / Structural False-Negative Rate) | **Defined in README_CODEAUG.md but never implemented or computed.** Targets only: SFPR < 5 %, SFNR < 8 % | See caveat in §10. |

All metric values below are on a 0–1 scale except the "CIDEr" approximation.

---

## 4. Experimental timeline (git history, `git log --all`)

| Date (2026) | Commit / event | Content |
|---|---|---|
| 07-02 | a6a0a94, c5387c7 | Phase 1 baseline; models restructured |
| 07-07 | faceb00, f70bd04 | Version 2 / final model code (RemoteCLIP cross-attention) |
| 07-09 | 341539d | "Pivot new model" |
| 07-11 | 196a1a0 | Caption-generation bug fixed |
| 07-12 | 15c7bac | "Phase 6.1" |
| 07-13 | b9cef79 | Notebooks with Phase 1 / 2 / final outputs (source of those numbers) |
| 07-15 | – | PyTorch-saving bug fixes |
| 07-18 | SLURM job 2654066 (gnode004, 12:14–19:44 IST, exit 0) | **Phase 6** run (logs/phase.out, phase.err) |
| 07-28/29 | c0ca765 | Patch extraction change; spatial cross-attention bottleneck fix |
| 07-30 | 0617cf3 | Phase 7 branch |
| 08-01 | e407fff "Failed", 2595e9f "Hyperparameters change" | Failed attempt, then hyperparameter retune |
| 08-02 | 619b1d5, 5729c92, a512e9e, 1ea856b, 9f26e19 | **CodeAug implemented**; LoRA targets set to `c_fc`/`c_proj`; FP16 ViT-L-14; VRAM calibrated at 3.48 GB; Phase 7 cleanup; CPU test suite |
| 08-04 | 8a23e72 "changed working code", 60f7672 | Phase 8 modules, tests; package-compatibility fixes |
| 08-11 | b36eb62 "stage 1 completed stage 2 loss NaN" | Phase 8 finetune notebook, training.py rewrite with finite-loss guard, 34 files changed |
| 08-14 | – | Cluster Stage 1b run diverged to NaN |
| 08-20 | 0ee6ac7 (branch IS/8.1) | NaN fix (LayerNorm on bridge output) |
| 09-09 | 30aef9f (IS/rogue) | New Indian Sentinel-2 architecture/data plan |
| 10-01 | e749e94, 842b058 (IS/man1) | Change-labelling UI |

Compute: ADA cluster, SLURM partition `u22`, 1 GPU per account (QOS cap), GTX 1080 Ti (sm_61, 11 GB); one run was scheduled on an RTX 2080 Ti; Phase 8 Stage 1a was executed on **CPU** ("NVIDIA driver too old").

---

## 5. Model descriptions (per phase)

### 5.1 Phase 1 — Baseline (researcher1) `RSICCformerBaseline`
- **Encoder (`SimpleEncoder`)**: 3 convolutional blocks (64 → 128 → 256 channels, BatchNorm, ReLU) → global average pooling; the two images' features are combined as `[before, after, |diff|]` and projected by a Linear layer to 512-d (`encoder_dim`).
- **Decoder (`SimpleDecoder`)**: 2-layer Transformer decoder, d_model 256, 4 heads, FFN 1024, learned positional encodings, dropout 0.1, max caption length 100.
- Parameters ≈ 3.41 M (3,405,277 reported in phase 1; 3,427,805 trainable counted in the Phase-2 notebook), **trained from scratch**.
- Loss: cross-entropy with label smoothing 0.1 (phase 1 notebook), ignoring pad tokens.

### 5.2 Phase 2 — RemoteCLIP difference (researcher2) `RemoteCLIPDifferenceModel`
- Changes **only the encoder**: frozen **RemoteCLIP ViT-B-32** (`chendelong/RemoteCLIP`, remote-sensing-pretrained CLIP) encodes both images; a difference-fusion head produces the 512-d change representation. Decoder, objective, data and vocabulary identical to Phase 1.
- Parameters: **3,427,805 trainable / 154,705,118 total**.
- Purpose: isolate the benefit of a remote-sensing-pretrained encoder + explicit difference fusion at equal decoder capacity.

### 5.3 Phase "final" — RemoteCLIP + cross-attention difference (`final_model/remoteclip_cross_attention.py`, `losses.py`)
- Replaces simple difference fusion with cross-attention between before/after features; trainable **9,111,572 / total 160,388,885**. TrainConfig: lr 1e-4, wd 1e-5, 15 epochs, cosine T_max 10, grad-clip 1.0 (the notebook resumed at epoch 15, loss 1.0357).
- Stage 2: fine-tuned on SECOND-CC (see §7.3).

### 5.4 Phase 6 — Hierarchical tile model `TileBasedChangeCaptioningModel` (`src/models/phase6/*`)
- Splits each image into a **2×2 grid of 224×224 tiles**; per-tile forward cross-attention (4 heads, diff-MLP dropout 0.1 in the logged run), a **tile-fusion Transformer** (2 layers, 4 heads, FFN 2048, dropout 0.2), global_dim 512, decoder embed 256 / 2 layers / 4 heads, plus a **contrastive InfoNCE auxiliary loss** (projection dim 256, weight 0.1, temperature 0.07).
- **13,843,476 trainable / 165,120,789 total.** Run resumed from epoch 15 (loss 1.0490); SECOND-CC stage resumed from epoch 14 (loss 1.5732), best checkpoint epoch 7 (loss 1.5268).

### 5.5 Phase 7 — CodeAug (design; `src/models/codeaug/*`)
Six engineering resolutions (README_CODEAUG.md):
1. **Patch-grid maths**: 252×252 input → 18×18 = **324 patch tokens**; positional embeddings of OpenCLIP ViT-L-14 bicubically interpolated 16×16 → 18×18 (CLS excluded).
2. **Decoder**: keep **Qwen2-VL-2B** text decoder (≈1.543 B parameters after deleting its vision tower), 4-bit NF4 quantisation with fp16 compute (Pascal GPUs lack native BF16).
3. **Q-Former token compressor**: 64 learnable queries, 8 heads, 2 layers; compresses 324 → 64 tokens (**61.3 % reduction**, sequence 424 → 164).
4. **Seasonal invariance**: Green Leaf Index GLI = (2G − R − B)/(2G + R + B); vegetation mask threshold 0.05; **bi-temporal union mask** M_veg = M_A ∪ M_B with hue/saturation jitter (magnitude 0.25) applied to *both* images to avoid directional brown↔green bias.
5. **Imbalance control**: single-lever **2:1 WeightedRandomSampler** with plain (unweighted) CE (γ = 1.0).
6. **Tokenizer**: Qwen's native HF BPE tokenizer with `<|im_start|>`/`<|im_end|>`.
- **Visual LoRA**: rank 16, α 32, dropout 0.05 on 48 MLP layers (`c_fc`, `c_proj` of 24 ViT blocks).
- **Calibrated numbers (README_CODEAUG.md, `check_vram_calibration.py`)**: total 1,856.38 M params; **9.81 M trainable (0.53 %)**; static weights **2.94 GB**; **peak training VRAM 3.48 GB** (7.52 GB headroom on 11 GB); batch 2 × grad-accum 8 (effective 16); DataLoader `num_workers = 0` (shared-memory deadlock avoidance on the cluster); AdamW wd 1e-2.
- **Planned schedule (not executed to a valid result)**: Stage 1 LEVIR-CC, 10 epochs, lr 1e-4 (Q-Former) / 5e-5 (LoRA), ≈2.5 h on 1080 Ti (≈14 h on a 32-core CPU); Stage 2 Indian few-shot (200–500 pairs), 15 epochs, lr 2e-5, ≈10 min.
- **Planned evaluation**: BLEU-1…4, METEOR, ROUGE-L, LEVIR-anchored CIDEr (static IDF), semantic cosine, SFPR, SFNR.

### 5.6 Phase 8 — Stage 1a (`src/models/phase8/*`)
- Frozen RemoteCLIP ViT-B-32 (512-d pooled embedding per image; whole image = one token) → **DifferenceModule** (4-head MultiheadAttention with q = before, k/v = after, + LayerNorm residual, then MLP on concat [before, after, attn, |before − after|]: 2048 → 1024 → LN → GELU → 512 → LN → GELU, dropout 0.1) → **LightweightCaptionDecoder** (embed 256, 2 layers, 4 heads, FFN 1024, dropout 0.1, memory_proj 512 → 256, sinusoidal positions, max length 100).
- **6,291,933 trainable / 151,288,833 frozen**, vocab 733.

### 5.7 Phase 8 — Stage 1b Approaches A/B (`Phase8HybridModelAB`)
- Reuses the Stage 1a encoder and difference module; a **bridge** `diff_to_llm` = Linear(512 → 1024) → LayerNorm → GELU → Dropout 0.1 → Linear(1024 → Qwen hidden) maps the change vector to a **visual prefix** prepended to the input embeddings of a **frozen Qwen2-0.5B-Instruct** (fp16); labels on the prefix are −100.
- **5,123,456 trainable / 645,321,601 frozen**; lr 5e-5, wd 1e-5, grad-clip 1.0, fp16 autocast + GradScaler, 10 epochs.
- **Approach C** = Q-Former → frozen Qwen2-VL-2B (`CodeAugRSICCModel`) — **never ran** (`TypeError: CodeAugRSICCModel.__init__() got an unexpected keyword argument 'vocab'`), so the planned A/B vs C comparison was never produced.
- *Which Qwen takes the difference-mechanism output?* Qwen2-0.5B-Instruct (A/B) through the bridge; Qwen2-VL-2B (C) takes Q-Former tokens, not the difference module.

---

## 6. Training configuration summary

| Item | Phase 1 | Phase 2 | Final | Phase 6 | Phase 8 1a | Phase 8 1b | Phase 7 (plan) |
|---|---|---|---|---|---|---|---|
| Input size | 256² | 224² (ViT-B/32) | 224² | 4 × 224² tiles | 224² | 224² | 252² |
| Batch | 16 | 16 | 16 | 16 | 8 | 8 | 2 × 8 accum |
| Epochs | ≥ 10 (resumed) | 10 | 15 | 15 (+ FT) | 15 | 10 | 10 + 15 |
| LR | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 5e-5 | 1e-4 / 5e-5 / 2e-5 |
| Weight decay | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-2 |
| Optimiser / sched. | Adam + cosine | same | same | same | same | AdamW-style, fp16 | AdamW |
| Grad-clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | – |
| Seed | 42 | 42 | 42 | 42 | 42 | 42 | – |

Common: `num_workers = 0`; checkpoint on best validation loss with auto-resume. Current-tree defaults (`src/config.py`): TrainConfig lr 1e-4, wd 1e-5, 8 epochs, T_max 8; **FinetuneConfig** (SECOND-CC stage): **lr 2e-5, wd 1e-4, 6 epochs, T_max 6** — deliberately lower after forgetting was observed (its docstring: LEVIR CIDEr fell 5.39 → 2.77 when fine-tuning reused the pre-train lr; val loss minimum 1.522 at epoch 6 rising to 1.551 at epoch 14).

---

## 7. Results

### 7.1 Phase 1 vs Phase 2 — LEVIR-CC test (n = 1,929) — the two models uploaded to Kaggle

| Metric | Baseline (P1) | RemoteCLIP-diff (P2) | Abs. Δ | Rel. Δ |
|---|---|---|---|---|
| Test loss | 0.9690 | 0.9345 | −0.0345 | −3.6 % |
| BLEU-1 | 0.4720 | 0.5168 | +0.0448 | +9.5 % |
| BLEU-2 | 0.3855 | 0.4237 | +0.0382 | +9.9 % |
| BLEU-3 | 0.3393 | 0.3712 | +0.0319 | +9.4 % |
| BLEU-4 | 0.3027 | 0.3284 | +0.0257 | +8.5 % |
| METEOR | 0.5966 | 0.6304 | +0.0338 | +5.7 % |
| ROUGE-L | 0.6214 | 0.6545 | +0.0331 | +5.3 % |
| CIDEr (approx.) | 5.1296 | 5.3469 | +0.2173 | +4.2 % |
| Semantic cosine | 0.6477 | 0.7061 | +0.0584 | +9.0 % |
| Trainable params | 3.41 M | 3.43 M | ≈ 0 | – |

Phase 2 per-epoch (train loss / val loss / val token accuracy):

| Ep | Train | Val | Tok-acc |
|---|---|---|---|
| 1 | 1.8641 | 1.3050 | 0.6964 |
| 2 | 1.2202 | 1.1653 | 0.7105 |
| 3 | 1.0649 | 1.1078 | 0.7245 |
| 4 | 0.9813 | 1.0512 | 0.7333 |
| 5 | 0.9009 | 1.0682 | 0.7298 |
| 6 | 0.8454 | **1.0451** | 0.7355 |
| 7 | 0.7957 | 1.0554 | 0.7344 |
| 8 | 0.7527 | 1.0536 | 0.7349 |
| 9 | 0.7246 | 1.0515 | 0.7367 |
| 10 | 0.7164 | 1.0545 | 0.7379 |

Phase 1 was resumed from its epoch-10 checkpoint (loss 1.0854), so epochs 1–10 are not in the output. On the first 40 test samples (unrepresentative, many "no change" pairs) the baseline scored BLEU-1/2/3/4 0.6711/0.5980/0.5489/0.5060, METEOR 0.7787, ROUGE-L 0.7988, CIDEr 7.2715, cosine 0.8057 — do not quote these as test performance.

### 7.2 Progression across phases — LEVIR-CC test (n = 1,929)

| Phase | Loss | B1 | B2 | B3 | B4 | METEOR | ROUGE-L | CIDEr | Cos |
|---|---|---|---|---|---|---|---|---|---|
| P1 baseline | 0.9690 | .4720 | .3855 | .3393 | .3027 | .5966 | .6214 | 5.1296 | .6477 |
| P2 RemoteCLIP-diff | 0.9345 | .5168 | .4237 | .3712 | .3284 | .6304 | .6545 | 5.3469 | .7061 |
| Final (cross-attn) stage 1 | **0.8982** | **.5311** | **.4355** | **.3810** | **.3365** | .6392 | .6604 | **5.3999** | .7123 |
| Phase 6 (tiles) stage 1 | 0.9225 | .5214 | .4297 | .3777 | .3358 | .6357 | **.6624** | 5.3921 | **.7149** |
| Phase 8 Stage 1a (epoch 7) | – | .5092 | .4167 | .3658 | .3250 | .6155 | .6482 | 5.2217 | .7007 |
| Phase 8 Stage 1b A/B | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0104 |
| Phase 7 CodeAug | no valid result | | | | | | | | |

### 7.3 Second domain (SECOND-CC test, n = 1,227) and forgetting on LEVIR

| Model / condition | Test set | Loss | B1 | B2 | B3 | B4 | METEOR | ROUGE-L | CIDEr | Cos |
|---|---|---|---|---|---|---|---|---|---|---|
| Final, after SECOND-CC FT | SECOND-CC | 1.5856 | .3627 | .2452 | .1789 | .1295 | .4131 | .4334 | 2.9234 | .5498 |
| Final, after FT | LEVIR (forgetting) | 4.0801 | .1677 | .1610 | .1561 | .1489 | .3935 | .4369 | 4.0598 | .4581 |
| Phase 6, after FT | SECOND-CC | 1.5838 | .3537 | .2360 | .1704 | .1225 | .3924 | .4161 | 2.7373 | .5295 |
| Phase 6, after FT | LEVIR (forgetting) | 3.5195 | .2737 | .1972 | .1561 | .1209 | .3179 | .3736 | 2.7726 | .4377 |
| Phase 8 1a, zero-shot (no SECOND-CC training) | SECOND-CC | – | .1785 | .1279 | .1025 | .0808 | .2975 | .3002 | 2.5118 | .3775 |
| Phase 8 1b A/B | SECOND-CC | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0215 |

Final-model SECOND-CC per-epoch (train / val / token-acc): E1 2.5434/1.7732/.5762 · E2 2.0132/1.6418/.5887 · E3 1.8174/1.5764/.6061 · E4 1.6847/1.5501/.6085 · E5 1.5754/1.5282/.6135 · **E6 1.4795/1.5224/.6131 (best, saved)** · E7 1.4067/1.5274 · E8 1.3461/1.5258 · E9 1.3072/1.5323 · E10 1.2836/1.5296 · E11 1.2798/1.5296 · E12 1.2775/1.5312 · E13 1.2811/1.5372 · E14 1.2748/1.5509.

### 7.4 Phase 8 Stage 1a per-epoch (train loss / train acc / val loss / val acc)

| Ep | Train | Tr-acc | Val | Val-acc |
|---|---|---|---|---|
| 1 | 1.7495 | .6348 | 1.2376 | .7054 |
| 2 | 1.1564 | .7072 | 1.1495 | .7177 |
| 3 | 1.0187 | .7290 | 1.0492 | .7397 |
| 4 | 0.9390 | .7420 | 1.0502 | .7353 |
| 5 | 0.8692 | .7556 | 1.0260 | .7389 |
| 6 | 0.8226 | .7646 | 1.0442 | .7409 |
| **7** | 0.7781 | .7740 | **1.0171** | .7437 |
| 8 | 0.7305 | .7854 | 1.0199 | .7408 |
| 9 | 0.6939 | .7928 | 1.0348 | .7400 |
| 10 | 0.6512 | .8048 | 1.0377 | .7471 |
| 11 | 0.6154 | .8146 | 1.0411 | .7426 |
| 12 | 0.5816 | .8218 | 1.0598 | .7417 |
| 13 | 0.5477 | .8301 | 1.1374 | .7369 |
| 14 | 0.5141 | .8403 | 1.1392 | .7429 |
| 15 | 0.4817 | .8488 | 1.1362 | .7347 |

Checkpoint saved at epoch 7 (best val). Vocabulary 733; run executed on CPU.

### 7.5 Phase 8 Stage 1b (Qwen2-0.5B bridge) per-epoch (train / val loss)
1: 2.1771 / 1.5565 · 2: 1.4699 / 1.4450 · 3: 1.2941 / 1.4758 · **4: 1.1957 / 1.4165 (best)** · 5: 1.1193 / 1.4919 · 6: 1.0472 / 1.5535 · 7: 0.9907 / 1.5336 · 8: 0.9371 / 1.4976 · 9: 0.8897 / 1.5548 · 10: 0.8385 / 1.5982. (Run on RTX 2080 Ti; the loaded Stage 1a checkpoint reported `val_loss = inf`, a checkpoint-metadata bug.) **All test generation metrics are exactly zero** (cosine 0.0104 LEVIR, 0.0215 SECOND-CC): the model's free-running generations are degenerate even though teacher-forced loss looked plausible. The notebook does not diagnose the cause.

---

## 8. Analysis (derived from the numbers above — Interpretation)

**8.1 Value of a remote-sensing-pretrained encoder (P1 → P2).** At essentially equal trainable capacity (3.41 M vs 3.43 M) and identical decoder/loss/data, swapping the scratch CNN for frozen RemoteCLIP ViT-B-32 + difference fusion improved *every* metric: BLEU-n by 8.5–9.9 % relative, METEOR +5.7 %, ROUGE-L +5.3 %, CIDEr-approx +4.2 %, semantic cosine +9.0 %, test loss −3.6 %. Larger gains on BLEU-1…3 and cosine than on CIDEr suggest better object/word-level and semantic coverage; the gap narrows for the longer-n-gram and IDF-weighted metrics. Since only the encoder changed, the gain is attributable to the encoder/difference representation.

**8.2 Diminishing returns from architectural complexity.** Relative to P2, cross-attention (final) adds BLEU-4 +0.0081 (+2.5 %), CIDEr +0.053 (+1.0 %), loss −0.036; tiling (Phase 6) adds BLEU-4 +0.0074 and the best ROUGE-L / cosine (0.6624 / 0.7149) but a worse loss than final (0.9225 vs 0.8982) with 13.8 M vs 9.1 M trainable parameters. Final vs baseline totals: BLEU-4 +0.0338 (+11.2 %), CIDEr +0.270 (+5.3 %), cosine +0.065. Differences between final and Phase 6 (≤ 0.001 BLEU-4) are small and, with a single seed per configuration, **not shown to be significant** (no variance estimates were run).

**8.3 Phase 8 Stage 1a ≈ Phase 2.** Same frozen backbone and a similar lightweight decoder; BLEU-4 0.3250 vs 0.3284, CIDEr 5.22 vs 5.35, cosine 0.7007 vs 0.7061. Phase 8's attention-based DifferenceModule did not beat the Phase 2 fusion. Validation loss bottomed at 1.0171 (epoch 7) and rose to 1.1362 by epoch 15 while training loss fell to 0.4817 and train accuracy rose to 84.9 % — clear overfitting; the best-checkpoint policy correctly saved epoch 7. Phase 2 shows the same pattern more mildly (val minimum 1.0451 at epoch 6 vs train 0.7164 at epoch 10), suggesting ≈ 6–8 epochs suffice for this decoder on LEVIR-CC.

**8.4 Domain shift is large.** Zero-shot SECOND-CC (Phase 8 1a): BLEU-4 0.0808 and cosine 0.3775 vs 0.3250 / 0.7007 on LEVIR-CC — a ≈ 75 % relative BLEU-4 drop. Fine-tuning on SECOND-CC (final model) raised SECOND-CC BLEU-4 to 0.1295 (+0.049, ≈ +60 % relative), CIDEr 2.51 → 2.92, cosine 0.3775 → 0.5498. (Caveat: the zero-shot and fine-tuned rows come from different architectures, so this is indicative rather than a controlled comparison.) Absolute SECOND-CC quality remains far below LEVIR-CC (BLEU-4 0.13 vs 0.34).

**8.5 Catastrophic forgetting.** After fine-tuning at the pre-training learning rate, LEVIR-CC performance collapsed: final model BLEU-4 0.3365 → 0.1489 (−55.8 %), CIDEr 5.40 → 4.06 (−24.8 %), cosine 0.7123 → 0.4581 (−35.7 %), test loss 0.898 → 4.080; Phase 6 BLEU-4 0.3358 → 0.1209 (−64.0 %), CIDEr 5.39 → 2.77 (−48.6 %), cosine 0.7149 → 0.4377 (−38.8 %), loss 0.923 → 3.520. Notably the final model retained more LEVIR performance than the tile model, and METEOR/ROUGE-L fall less than BLEU, i.e. generic fluency survives while the LEVIR-specific phrasing is lost. SECOND-CC validation loss began to rise after epoch 6 (1.5224 → 1.5509 at epoch 14) while training loss kept falling, so a short, low-LR fine-tune (the adopted FinetuneConfig: 2e-5, 6 epochs) is the evidence-based setting. The subsequently obtained model after the lower-LR config was **not** evaluated in the available outputs.

**8.6 LLM-decoder variants were not successful.** Stage 1b (frozen Qwen2-0.5B + bridge): (i) fp16 overflow produced NaN loss (§9); (ii) after stabilisation the loss was finite (best val 1.4165) but **all generation metrics were 0**, so no valid conclusion about LLM decoding can be drawn from it. Teacher-forced loss alone (note: different tokenizer, so not comparable to the vocab-733 models' ≈ 1.0) is not evidence of correct generation. Likely areas to investigate (hypotheses, not verified): generation-time prefix/embedding handling, EOS / special-token handling, dtype of embeddings at `generate`.

**8.7 Resource trade-offs.** CodeAug's design is feasible on an 11 GB Pascal GPU (3.48 GB peak, 0.53 % trainable) thanks to 4-bit NF4, fp16 ViT, and Q-Former compression (324 → 64 tokens), but this is a *calibration* result only — no accuracy was measured. The much simpler RemoteCLIP+difference models reach BLEU-4 ≈ 0.33 with 3–9 M trainable parameters.

---

## 9. Issues encountered and fixes

| Issue | Cause | Fix / status |
|---|---|---|
| **NaN loss in Stage 1b / "stage 2 loss NaN"** (commit b36eb62; Aug 14 cluster run) | Bridge output (`diff_to_llm`) had no normalisation after its final Linear → unbounded activations overflow fp16 (max ≈ 65,504) through ≈ 24 frozen Qwen layers → inf → NaN | Add `LayerNorm(hidden, dtype=float16)` after the final Linear (commit 0ee6ac7, **branch IS/8.1 only**; the checked-out `modular` notebook only has LayerNorm(1024) inside the bridge's first block). Also fixed `NameError: metrics_c_levir`. |
| Training instability guards | – | Working tree `src/training.py`: skip batch when `not torch.isfinite(loss)`; GradScaler unscale + grad-norm clip; FT lr lowered to 5e-5 |
| Resume range | `FT_EPOCHS` smaller than resume epoch | Raised (memory note: 10 → 30) |
| fp16/float32 mismatch | Vision-encoder fp16 outputs | Cast to float32 in `codeaug/model.py`; cast `total_embeds` to LLM dtype before forward/generate |
| Generation warnings | Sampling flags | `do_sample=False` + warning filter |
| Checkpoint metadata | Stage 1a best checkpoint saved/loaded with `val_loss = inf` | Not fixed |
| Approach C `TypeError` | Constructor signature mismatch (`vocab` kwarg) | Not fixed in the outputs |
| Catastrophic forgetting | Fine-tune at full LR | New FinetuneConfig (2e-5, 6 ep, wd 1e-4) |
| GPU compatibility | GTX 1080 Ti (sm_61) is unsupported by PyTorch ≥ 2.1 | Project pinned to torch 2.0.1 (+ numpy < 2, transformers 4.35.0, sentence-transformers 2.2.2, accelerate 0.24.1, huggingface-hub 0.19.0). **Contradiction:** the Jul 18 cluster log shows Torch 2.7.1+cu128 with "supported architectures sm_75+", and Phase 8 Stage 1a fell back to CPU; SETUP.md is inconsistent (cu121 vs cu118). |

---

## 10. Validity caveats — what NOT to cite as a measured result
1. `phase7.ipynb` contains **hard-coded fabricated placeholder metrics** (BLEU-1 78.4, BLEU-4 45.8, METEOR 36.2, ROUGE-L 62.5, CIDEr 128.4, cosine 0.842, SFPR 3.8 %, SFNR 6.2 %). The training cells only save state-dicts. They must not appear in the report as results.
2. **SFPR and SFNR were never computed** (not in `src/metrics.py`, no outputs). They appear only as targets (< 5 %, < 8 %).
3. **No Phase 7 CodeAug training result exists.** Only the parameter/VRAM calibration is real.
4. `logs/phase.out` / `phase.err` and the PDFs `predictions_phase7.pdf`, `predictions_phase7_second.pdf` are **Phase 6** outputs (despite their names); `predictions_phase8*.pdf`, `SECOND_CC_Test_Predictions_*.pdf`, `predictions_phase_final_finetuneS.pdf` hold sample predictions (300 saved per set for Phase 6).
5. Phase 1 / 2 / final / Phase 6 notebook outputs are only in git history (commits b9cef79, c0ca765), not in the working tree.
6. Single seed (42) per configuration; no confidence intervals or significance tests.
7. "CIDEr" is an approximation (see §3). Phase 8 uses a different vocabulary/tokenizer from other phases (733 vs 1,300 words vs Qwen BPE), so losses are not strictly comparable across them.
8. Zero-shot vs fine-tuned SECOND-CC rows come from different models (Phase 8 1a vs final).

---

## 11. Conclusions (ready to paste, adjust wording as needed)
1. **A remote-sensing-pretrained, frozen vision encoder with difference fusion substantially improves change captioning over a from-scratch CNN** at the same decoder and trainable-parameter budget: on LEVIR-CC, BLEU-4 rose from 0.303 to 0.328 (+8.5 %), METEOR 0.597 → 0.630, ROUGE-L 0.621 → 0.655, semantic cosine 0.648 → 0.706 and test loss fell from 0.969 to 0.935.
2. **More elaborate difference/fusion architectures give only marginal additional gains**: the cross-attention (final) and tile-based (Phase 6) models reach BLEU-4 ≈ 0.336 and CIDEr-approx ≈ 5.40 (≈ +11 % BLEU-4 over the baseline), and the Phase 8 attention-based module (BLEU-4 0.325) matches Phase 2 — the encoder choice matters far more than the fusion design.
3. **Models overfit LEVIR-CC quickly** (validation loss minima at epochs 6–7 while training loss continues to fall), so early stopping on validation loss is essential.
4. **Domain shift to SECOND-CC/Indian-style imagery is severe** (zero-shot BLEU-4 0.081 vs 0.325 on LEVIR-CC); fine-tuning improves SECOND-CC (BLEU-4 ≈ 0.13, cosine ≈ 0.55) but **causes catastrophic forgetting of LEVIR-CC** (BLEU-4 −56 % to −64 %, CIDEr-approx −25 % to −49 %), motivating low-LR / short fine-tuning and, in future, replay or parameter-efficient adaptation.
5. **LLM-based decoding (Qwen via a learned bridge) was not made to work within the available hardware/time**: fp16 overflow produced NaNs, which a LayerNorm fix addressed, but generation remained degenerate (all n-gram metrics 0). The CodeAug design (Q-Former compression 324 → 64 tokens, LoRA, 4-bit Qwen2-VL, GLI seasonal masking) was shown to fit in 3.48 GB VRAM, but its accuracy and its seasonal-robustness claims (SFPR/SFNR) remain **unvalidated**.
6. **Practical outcome**: the best validated systems are the RemoteCLIP-based models; the two baseline-vs-RemoteCLIP models are archived on Kaggle for reproducibility.

---

## 12. Limitations and future work
- Limitations: single seed; approximated CIDEr; no SFPR/SFNR; SECOND-CC train/val sizes unreported; Phase 7/8 LLM variants unvalidated; mixed software stacks and CPU/GPU fallbacks; results for several phases only recoverable from git history.
- Future work: (a) debug Stage 1b generation (prefix handling, EOS, dtype) and run Approach C after fixing the `vocab` constructor error; (b) implement and report real SFPR/SFNR on unchanged-with-seasonal-shift and changed-with-colour-shift Indian pairs; (c) run CodeAug Stage 1/2 and ablate GLI masking, Q-Former size and the 2:1 sampler; (d) evaluate the low-LR FinetuneConfig and replay/LoRA-only adaptation against forgetting; (e) evaluate on the new Sentinel-2 Indian dataset (1,072 pairs); (f) use COCO-CIDEr for comparability with published RSICC work; (g) multiple seeds with confidence intervals; (h) standardise the environment (pinned torch 2.0.1 on sm_61 or move to sm_75+ GPUs).

---

## 13. Suggested report outline
1. Introduction (§1) · 2. Related setup / datasets (§2) · 3. Method: architectures per phase (§5, with a block diagram per phase) · 4. Experimental setup (§3, §6) · 5. Results: baseline vs RemoteCLIP (§7.1), phase progression (§7.2), domain transfer and forgetting (§7.3), training curves (§7.1, 7.4, 7.5 — plot train/val loss per epoch) · 6. Analysis (§8) · 7. Engineering issues (§9) · 8. Limitations and future work (§12) · 9. Conclusion (§11). Suggested figures: bar chart of BLEU-4/CIDEr per phase; loss curves for Phase 2, Phase 8 1a, 1b, SECOND-CC FT; before/after forgetting bars; CodeAug architecture diagram (README_CODEAUG.md has a mermaid source).

---

## 14. Artifacts
- **Kaggle (single handle `kspsvlnsiddardha`)**: https://www.kaggle.com/models/kspsvlnsiddardha/rsicc-change-captioning — variations `pytorch/baseline` and `pytorch/remoteclip-difference`; each contains the best and current `.pt` plus `shared_vocab.pkl` (≈ 82 MB baseline, ≈ 1.3 GB RemoteCLIP-difference). Uploaded on 2026-10-01 (version 1 of both variations). The model is currently **private**; make it public or add your friend as a collaborator on Kaggle before sharing the link.
- **HuggingFace**: https://huggingface.co/Kspsvln/IS (baseline + RemoteCLIP-difference checkpoints, `shared_vocab.pkl`, ≈1.3 GB, public). https://huggingface.co/Kspsvln/IS_phase2 is currently empty (only a README); the Phase 7/8 weights are held by the project partner.
- **Repo**: branch `modular` (HEAD b36eb62); key files `src/config.py`, `src/metrics.py`, `src/training.py`, `src/training_phase8.py`, `src/models/{baseline,variants,final_model,phase6,phase8,codeaug}/`, `README_CODEAUG.md`, notebooks `phase7.ipynb`, `phase8*.ipynb`.

## 15. Gaps the partner (holding Phase 7/8 weights) should fill
- Phase 7 / Phase 8 model weights and any *real* evaluation numbers for CodeAug (Stage 1 and 2) — none are in this repository.
- SFPR / SFNR measurements.
- SECOND-CC train/val sample counts.
- Results for the lower-LR fine-tune configuration and any re-run of Stage 1b after the LayerNorm fix.
