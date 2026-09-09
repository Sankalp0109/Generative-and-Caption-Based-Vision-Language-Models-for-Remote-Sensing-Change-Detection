# Indian Bi-Temporal Change Captioning — Architecture Plan

## Context
The team previously trained a change-captioning pipeline (Qwen2-VL-2B + Visual LoRA + Q-Former, "CodeAug Phase 7") on **LEVIR-CC**, but that dataset is foreign-satellite imagery with low-quality machine-generated captions, capping how good the model can get and not transferring to Indian geography. This plan builds a purpose-built Indian bi-temporal dataset, labels it with a strong open-weight VLM, and trains a **fresh** architecture on top — deliberately not reusing Phase 7's Q-Former+LoRA+Qwen2-VL decoder design, since that choice was made for a bad dataset and isn't the right fit now that we have good data.

Deliverable: a model that takes two Indian satellite images and outputs a precise natural-language description of what changed — a real generated caption, not just an embedding-space demo.

## Ada cluster facts (verified by actually running jobs, not assumed)
- **Compute:** SLURM, partition `u22` (GPU, mostly 3-4× GTX 1080 Ti/node, Pascal sm_61) and `u22-cpu`. QOS tier **`low`** caps this account at **1 GPU total across ALL running jobs, cluster-wide** (`MaxTRESPU: gres/gpu=1`) — not per-job. `MaxJobsPU=5` only lets you queue jobs; a 2nd GPU job just sits `PD` until the 1st finishes. CPU (10 cores) and memory (32,000MB) caps are similarly aggregate, not per-job. **Net effect: every GPU-dependent stage runs strictly sequentially — no concurrency, no job-sharding throughput trick. Total wall-clock is additive across stages.**
- **One job so far landed on an RTX 2080 Ti** (Turing, sm_75), not the more-common 1080 Ti — treat Pascal as the design baseline regardless (per the physical spec), and don't assume the next job lands on the same card type. SLURM remaps whichever physical GPU it assigns to `CUDA_VISIBLE_DEVICES=0` inside a job, so code can always just target `cuda:0`.
- **Storage — confirmed adequate, not a real constraint:** `/archive/home` 25GB (healthy, ~1GB used), `/share1/$USER/` 24.4GB (root `/share1/` itself is NOT writable — must use the per-user subdir). `/scratch` is 1.8TB local SSD per node, observed to persist across 2 jobs on the same node but not guaranteed to (node placement isn't guaranteed to repeat) — treat as ephemeral working space, copy final artifacts to `/share1/$USER/` before a job ends.
- **Software: nothing pre-existing.** The venv assumed to exist at `/home2/sankalp0109/src_IS/venv` (per `ADA_CLUSTER_SPECIFICATION.md`) does not actually exist on this account — confirmed via a failed `import torch`. Must be built from scratch (see Immediate Next Actions).
- `num_workers=0` required in every DataLoader (cluster-specific shared-memory deadlock hit previously). Neither GPU type supports bf16 (needs Ampere+) — use fp16/fp32 only.

## Pipeline

```
Stage 1                Stage 2                      Stage 3
Data collection    →    VLM auto-captioning     →    Train our own model
(DONE: 1,072 pairs)     (Qwen2-VL-7B labels          (RemoteCLIP encoder +
                         Stage 1's pairs)              small pretrained decoder)
```

---

## Stage 1 — Data collection ✅ DONE

`data_collection/collect_dataset.py`: pulls Sentinel-2 before/after pairs (years apart) over 60 hand-picked Indian AOIs across 5 change categories (urban, deforestation, agriculture, coastal, infrastructure), grid-tiled into 512×512 patches, filtered by a change-magnitude threshold so only tiles with real visible change are kept. Fully resumable (`progress.jsonl`, flushed per-tile, safe to Ctrl+C and rerun). Kept pairs are saved as `dataset/p{N}_before.png` / `p{N}_after.png` with dense, resume-stable numbering.

**Result of the actual run:** 2,126 tiles checked → **1,072 pairs kept** (2,144 images, ~923MB) in **1h53m** at 12 concurrent workers, zero failures.

**Known gotcha, already fixed in code:** Microsoft Planetary Computer's SAS-token signing endpoint (not the STAC search) is rate-limited under sustained heavy concurrency — hit this during benchmarking at 32-64 workers. Fixed with exponential backoff+jitter retry (`with_rate_limit_retry` in `collect_dataset.py`). 12 workers is the empirically-validated safe concurrency (8 and 16 both ran clean in testing; 32+ triggered throttling).

**Scale note:** this 1,072-pair run is a full pipeline validation. The final production dataset is intended to be 10K+ pairs — done by widening the AOI boxes and/or shrinking `TILE_SIZE_DEG` in `collect_dataset.py` (no structural changes needed) and re-running the same script.

---

## Stage 2 — VLM auto-captioning (labeling tool, not part of the final model)

`captioning/caption_dataset.py` + `captioning/caption_dataset.sbatch`: loads **Qwen2-VL-7B-Instruct** in 4-bit NF4, shows it each before/after pair, and asks it to describe the change in 1-2 sentences. Fully resumable (`captions.jsonl`, same flush-per-item pattern as Stage 1).

**Important distinction:** Qwen2-VL-7B here is purely a **labeling tool** used once to generate training captions for Stage 3 — it is NOT part of the final deployed model, so using it doesn't reintroduce the "old architecture" we're moving away from.

**Time estimate:** ~5-8s/pair on the account's single GPU (unverified on Pascal specifically — needs the calibration below); for 1,072 pairs, roughly 1.5-2.5 hours, run as one sequential job (no concurrency available under this account's QOS).

**Quality control:** VLM captions every pair (bulk); a curated subset (~100-150 of the 1,072) should get a human pass to serve as a clean, trustworthy val/test split, since automated captions will have some noise.

---

## Stage 3 — Train the actual model (fresh design, not Phase 7's)

Two components, trained in sequence:

1. **RemoteCLIP encoder, contrastively fine-tuned on our data.** RemoteCLIP (an OpenCLIP ViT-L-14 checkpoint, loadable via the already-planned `open-clip-torch`) is pretrained on general remote-sensing scene/object data (detection/segmentation/classification datasets converted to pseudo-captions) — it understands satellite imagery well but has **never seen a before/after pair or any notion of change**, since its training data was single static images. Stage 3a fine-tunes it (InfoNCE contrastive loss between the before/after representation and the Stage 2 caption's text embedding) specifically on our Indian change data, so it learns to represent *what changed*, not just *what's present*.

2. **A small pretrained language model as the caption decoder** — e.g. GPT-2 small / DistilGPT2 (~80-125M params), fine-tuned to attend to the fine-tuned RemoteCLIP's change representation (via cross-attention) and generate the actual caption sentence.
   - **Why not train a decoder from scratch:** with only ~1,072 pairs (and even at the eventual 10K+ target), a from-scratch decoder doesn't have enough data to learn fluent English from zero — it tends to produce repetitive, templated output. A small *pretrained* LM already knows how to write sentences; our data only has to teach it what to say about satellite changes, which is a much easier learning problem at this data scale.
   - **Why not reuse Qwen2-VL/Q-Former/LoRA (the Phase 7 approach):** that architecture choice was made to work around LEVIR-CC's bad data, not because it's the best fit here. It's also unnecessarily heavy (2B+ params) for a decoder task that a 100M-parameter pretrained LM handles well once properly conditioned.
   - This same design holds at both the current 1,072-pair pilot scale and the eventual 10K+ scale — no architecture change needed as more data is added, which matters since the pilot is meant to validate the *real* pipeline, not a stand-in.

**Job structure:** strictly sequential on Ada (1 GPU total for the account) — Stage 3a fully finishes, then Stage 3b starts. Both are lightweight enough (RemoteCLIP ViT-L-14 + a 100M-class decoder) to comfortably fit an 11GB card with room to spare.

---

## Evaluation
BLEU-1..4, METEOR, ROUGE-L, CIDEr against the human-verified test subset; report per-change-category breakdown (urban/deforestation/agriculture/coastal/infrastructure), and a zero-shot-vs-fine-tuned comparison (the existing LEVIR-CC-trained Phase 7 checkpoint tested on this Indian set, vs. this new model) to quantify the domain-gap improvement.

---

## File map (current state)
```
IS_Rogue/
├── ADA_CLUSTER_SPECIFICATION.md      -- ground-truth cluster facts from prior job logs
├── data_collection/
│   ├── fetch_pair.py                 -- original single-pair proof of concept
│   ├── collect_dataset.py            -- Stage 1, DONE, resumable, rate-limit-safe
│   └── india_dataset/dataset/        -- 1,072 kept pairs (p1..p1072 _before/_after.png)
├── captioning/
│   ├── caption_dataset.py            -- Stage 2 script, resumable
│   └── caption_dataset.sbatch        -- Stage 2 Ada job (QOS-compliant: gpu:1, -c10, 30000M)
└── ada_preflight/
    ├── 00_storage_check.sh           -- verifies /archive/home + /share1/$USER writability
    ├── setup_venv.sh                 -- builds the BASE venv (torch/transformers/bitsandbytes/open-clip-torch/Stage-1 deps)
    ├── setup_qwen_caption.sh         -- adds qwen-vl-utils + pre-downloads Qwen2-VL-7B (~15GB) into /share1/$USER/models_cache
    ├── 01_gpu_and_scratch.sbatch     -- confirms GPU visibility + /scratch behavior
    ├── 02_bnb_pascal_calibration.sbatch -- bnb 4-bit + FP16/FP32 timing, needs to land on a real 1080 Ti
    └── README.md                     -- run order + what to do with each result
```

## Immediate next actions
1. On Ada: `bash ada_preflight/setup_venv.sh` (base environment — required before anything else).
2. Upload the 923MB dataset: `rsync -avz data_collection/india_dataset/dataset/ <ada_account>@ada.iiit.ac.in:/share1/$USER/india_dataset/dataset/`
3. `bash ada_preflight/setup_qwen_caption.sh` (adds Qwen2-VL-7B + pre-downloads its weights — only needed before Stage 2).
4. Run `02_bnb_pascal_calibration.sbatch` (repeat submissions until it lands on a 1080 Ti — one 2080 Ti data point isn't enough) to confirm the Stage 2 timing estimate holds on Pascal.
5. Calibrate Stage 2 on a small sample first: `python captioning/caption_dataset.py --dataset-dir ... --out ... --max-pairs 20` before committing to the full 1,072-pair run — checks the transformers/Qwen2-VL compatibility risk (transformers 4.39.3 predates Qwen2-VL's official upstream support) before spending real walltime.
6. Once Stage 2 completes: build Stage 3a/3b training scripts (RemoteCLIP contrastive fine-tune, then small-LM decoder fine-tune) — not yet written.
