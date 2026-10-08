# Stage 0 — Ada Pre-Flight Verification

**Status: partially run and verified (job `2691804` on `gnode084`, plus `00_storage_check.sh` and `sacctmgr`).** Results below are real, not assumptions — see `architecture_plan.md`'s "Known Ada cluster facts" section for the full writeup. This README reflects the corrected order/parameters after those results.

## Verified so far

| Check | Result |
|---|---|
| `/archive/home` | Healthy, ~1.1GB/25GB used |
| `/share1` | Root not writable — **must use `/share1/$USER/...`** |
| `/share2`-`/share6` | Visible cluster-wide (write access untested) |
| QOS tier | **`low`**: `MaxTRESPU: gres/gpu=1` — 1 GPU **total across all running jobs** (not per-job), 10 CPU cores + 32,000MB mem also aggregate. `MaxJobsPU=5` is a submission-slot limit only — a 2nd GPU job just queues (`PD`) until the 1st finishes |
| GPU landed | RTX 2080 Ti (Turing, sm_75) on `gnode084` — one data point, not proof most jobs get this |
| Device indexing | SLURM remaps whichever physical GPU it assigns to `CUDA_VISIBLE_DEVICES=0` inside the job — always safe to just target `cuda:0` |
| `/scratch` | Persisted across 2 jobs on the same node — less ephemeral than assumed, but node placement isn't guaranteed to repeat, so still copy final outputs to `/share1/$USER/` before a job ends |
| venv at `/home2/sankalp0109/src_IS/venv` | **Does not exist** — must be built (see step 2 below) |

**Resolved — no longer open:** confirmed via `sacctmgr` that `MaxTRESPU: gres/gpu=1` caps this account to 1 GPU total, cluster-wide, across all running jobs. Job-level sharding across concurrent GPU jobs is not possible on this account; every GPU-dependent stage runs strictly sequentially. `architecture_plan.md`'s Stage 2/3 time estimates have been updated to reflect this as final, not provisional.

## Order to run

1. **`bash 00_storage_check.sh`** (login node, no `sbatch`) — re-run after the path fix to confirm `/share1/$USER/` write access explicitly (the original run inferred this from context; this version tests it directly).

2. **`bash setup_venv.sh`** (login node) — **new step**, required because the assumed-existing venv doesn't exist. Builds the **base** venv at `/home2/sankalp0109/src_IS/venv` with just the foundation stack from `ADA_CLUSTER_SPECIFICATION.md` (torch 2.0.1+cu118, transformers 4.39.3, bitsandbytes 0.45.5, accelerate 1.14.0, open-clip-torch 2.32.0, huggingface-hub 0.36.2) plus the STAC/rasterio packages for Stage 1. Pass a different path as `$1` if `/home2/sankalp0109/src_IS/venv` isn't right for your account. **Does not** install anything Qwen-specific — that's a separate script (see 2b), so the base environment can be built/tested independently of Stage 2.

2b. **`bash setup_qwen_caption.sh`** (login node, after 2) — adds `qwen-vl-utils` to the same venv and pre-downloads the Qwen2-VL-7B-Instruct weights (~15GB) into `/share1/$USER/models_cache`, so the actual Stage 2 job doesn't burn walltime downloading. Only needed before Stage 2 — the base venv from step 2 already covers Stage 1 and the calibration below.

3. **`sbatch 01_gpu_and_scratch.sbatch`** — updated to request `--gres=gpu:1 -c 10 --mem=30000M` (the QOS-compliant parameters that actually worked in job `2691804`). Confirms GPU visibility and re-tests `/scratch` persistence if you want a second data point on a different node.

4. **`sbatch 02_bnb_pascal_calibration.sbatch`** — now fails fast with a clear message if the venv from step 2 isn't there yet, instead of silently warning and crashing later on `import torch`. **Run this repeatedly** until it lands on a GTX 1080 Ti (Pascal, sm_61) node — job `2691804` landed on a 2080 Ti (Turing), so the Pascal-specific numbers this script exists to produce still haven't been measured.

## What to do with the results

| Check | If it passes as expected | If it doesn't |
|---|---|---|
| `/share1/$USER` write test | Proceed with the plan's `/scratch` → `/share1/$USER/` flow | Escalate to cluster admins — this account's `/share1` allocation may need reprovisioning |
| `/scratch` persistence | Confirmed persists on same node, but keep the copy-to-`/share1/$USER/` step anyway since node placement isn't guaranteed to repeat | N/A — already observed to persist once |
| bnb 4-bit NF4 on Pascal | If it works and timing is close to the Turing numbers already informally expected: the "3-8s/pair" Stage 2 estimate can be trusted | If notably slower (>10-15s/pair) or fails on sm_61: fall back to fp16-only or 8-bit instead of 4-bit, and recompute the Stage 2 wall-clock estimate (still single-job/sequential either way — no concurrency to plan around) |

Concurrency is resolved (1 GPU total, sequential-only). Once the Pascal calibration has actually run on Pascal hardware, the Stage 1-3 time estimates in `architecture_plan.md` should be treated as final rather than provisional.
