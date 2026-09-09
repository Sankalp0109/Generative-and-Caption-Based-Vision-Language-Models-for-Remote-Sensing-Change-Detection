# ADA Cluster — Specification (as evidenced by this project's logs/scripts)

Compiled from actual SLURM job logs, batch scripts, and calibration output found in this project (`phase.out`, `phase8_training.out`, `execute.sh`, `execute_phase7.sh` git history, `README_CODEAUG.md`). Where a number wasn't directly observed in any log, it's marked as such rather than guessed.

## 1. Nodes observed / referenced

| Node | Role in this project | GPU | Evidence |
|---|---|---|---|
| `gnode004` | Target/calibration node named in `README_CODEAUG.md` and the original `execute_phase7.sh` | NVIDIA **GTX 1080 Ti**, 11GB VRAM, Pascal architecture, compute capability `sm_61` | README_CODEAUG.md hardware table + `check_vram_calibration.py`'s stated target |
| `gnode060` | Actually ran the Aug 14, 2026 fine-tuning job (SLURM job `2668262`) | NVIDIA **GeForce RTX 2080 Ti**, 11264 MiB (~11GB) VRAM, Turing architecture, compute capability `(7, 5)` i.e. `sm_75` | `phase.out` — `nvidia-smi` output, driver `570.211.01`, CUDA driver version `12.8` |
| `gnode068` | Ran an environment-setup/package-verification job (SLURM job `2668148`) | RTX 2080 Ti (per install log context) | `phase8_training.out` |

Job scheduler: **SLURM** (`sbatch`, `#SBATCH` directives, `SLURM_JOB_ID`, `SLURM_NODELIST`, `SLURM_JOB_GPUS`).

## 2. Per-node GPU capability (measured, from `nvidia-smi` / `torch.cuda`)

**gnode060** (from `phase.out`):
```
GPU: NVIDIA GeForce RTX 2080 Ti
Memory: 11264 MiB (~11 GB)
Driver Version: 570.211.01
CUDA Version (driver): 12.8
Compute Capability: (7, 5)
Torch CUDA build: 11.8 (torch 2.0.1+cu118)
Supported arches (this torch build): sm_37, sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
SLURM_JOB_GPUS: 3   ← job was allocated 3 GPUs on this node, though only 1 was requested in execute.sh (see §4)
```

**gnode004** (documented target, GTX 1080 Ti — not directly captured via `nvidia-smi` in any log we have, but is the basis for the project's own empirical VRAM calibration):

| Metric | Value | Source |
|---|---|---|
| Total model parameters (CodeAug Phase 7 pipeline) | 1,856.38M (~1.86B) | `README_CODEAUG.md` calibration table |
| Trainable parameters | 9.81M (0.53%) — Visual LoRA (r=16, 48 layers) + Q-Former queries | same |
| Static loaded weights VRAM | 2.94 GB (FP16 ViT-L-14 + 4-bit NF4 Qwen2-VL-2B) | same |
| Peak training VRAM (fwd+bwd+optimizer.step) | 3.48 GB | same |
| Free headroom on 11GB card | 7.52 GB | same |

## 3. CPU / RAM (as requested in SLURM batch scripts — no `lscpu`/`free` output was captured in any log, so these are allocation requests, not directly measured hardware limits)

| Script | CPUs requested | RAM requested | GPUs requested | Wall-time limit | Partition/node pin |
|---|---|---|---|---|---|
| `execute.sh` (current, `origin/modular`) | 4 (`--cpus-per-task=4`) | 16G | 1 (`--gres=gpu:1`) | 24:00:00 | none pinned |
| `execute_phase6.sh` (main branch) | 4 | 16G | 1 | 12:00:00 | none pinned |
| `execute_phase7.sh` (historical, superseded by `execute.sh`) | 16 (`--cpus-per-task=16`) | 32G | 1 | 24:00:00 | `--partition=gnode`, `--nodelist=gnode004` |

Documented (not log-measured) CPU-only compute node profile, per `README_CODEAUG.md`'s hardware table:
- 16–32 x86_64 CPU cores
- 36GB+ RAM
- No CUDA (CPU-only, FP32 full precision)
- Must explicitly request `--cpus-per-task=16` or `32` to override the SLURM default of 4 cores

Shared cluster-wide constraint noted in the same table: **all DataLoaders must use `num_workers=0`**, to avoid a shared-memory multiprocessing deadlock the team hit on this cluster.

## 4. Observed discrepancy worth flagging

`execute.sh` (the script that produced the Aug 14 run) requests `--gres=gpu:1` — one GPU — but the resulting job log (`phase.out`) shows `SLURM_JOB_GPUS = 3`, i.e., the job actually ran with 3 GPUs allocated on `gnode060`. This mismatch between the committed script and the observed allocation means either a different/edited version of the script was actually submitted on the cluster, or the allocation was overridden manually at submission time (e.g. `sbatch --gres=gpu:3 execute.sh`) — the committed `execute.sh` alone doesn't explain what ran.

## 5. Software stack (from the Aug 14 install log, `phase8_training.out`)

| Package | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.0.1+cu118 |
| torchvision | 0.15.2+cu118 |
| torchaudio | 2.0.2+cu118 |
| transformers | 4.39.3 (pinned in `requirements.txt`) |
| bitsandbytes | 0.45.5 |
| accelerate | 1.14.0 |
| open-clip-torch | 2.32.0 |
| huggingface-hub | 0.36.2 |
| numpy | 1.26.4 |
| nltk | 3.10.3 |
| scikit-learn | 1.7.2 |
| jupyter / nbconvert / ipykernel | present — used to headlessly execute `.ipynb` notebooks via `jupyter nbconvert --execute` |

Environment management: a project-local Python **venv** at `/home2/sankalp0109/src_IS/venv` (some scripts also try `conda activate mlenv` as a fallback before falling back to the venv). Working directory on the cluster: `/home2/sankalp0109/src_IS`.

## 6. Running time (observed, not requested)

| Job | SLURM Job ID | Node | Start | Duration | Outcome |
|---|---|---|---|---|---|
| Env/package verification | 2668148 | gnode068 | Aug 14, 2026 10:30 AM IST | ~11 min | Completed (dependency install + sanity check only — not a training run) |
| `phase8_finetune.ipynb` execution | 2668262 | gnode060 | Aug 14, 2026 12:29 PM IST | ~3h 05min (12:29→15:34) | **Failed, exit code 1** — crashed on the `metrics_c_levir` NameError (fixed on branch `8.1`) |

Both jobs requested up to 24:00:00 wall time; actual usage was far under that ceiling in both cases.

## 7. What's not available in any log/script (not fabricated here)

- Exact CPU model (e.g. specific Xeon/EPYC part number) for any node.
- Exact total RAM or disk/storage ("ROM") capacity of `gnode004`, `gnode060`, or `gnode068` — no `lscpu`/`free -h`/`df -h` output exists in any captured log; only SLURM *requests* (§3) are known.
- Cluster-wide node count, interconnect, or storage architecture (e.g. shared filesystem type/capacity) — never referenced in this project's scripts or logs.
- Whether `gnode004` is Pascal-architecture GTX 1080 Ti was independently `nvidia-smi`-confirmed anywhere in this project's logs — it is asserted by `README_CODEAUG.md` and `check_vram_calibration.py`'s design target, but that specific node's `nvidia-smi` output isn't present in the files reviewed.
