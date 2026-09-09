#!/bin/bash
#SBATCH --job-name=preflight_gpu
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30000M
#SBATCH --time=00:15:00
#SBATCH --output=preflight_gpu_%j.out

# UPDATED after running this on the account: QOS "low" caps jobs at 1 GPU,
# 10 CPUs, ~31.25GB (32000MB) mem -- the original --gres=gpu:4/-c16/--mem=32G
# request fails outright under this QOS. This version uses the parameters
# that actually worked (job 2691804 on gnode084). Multi-GPU throughput now
# has to come from submitting several of these jobs concurrently instead of
# one job requesting multiple GPUs -- see architecture_plan.md.
#
# Stage 0 preflight: GPU allocation + /scratch check.
# Confirms what's ACTUALLY visible inside a single-GPU job —
# the architecture plan flags a known discrepancy (execute.sh requested 1
# GPU but SLURM_JOB_GPUS showed 3 in a past run), so never trust the
# request alone.

echo "== SLURM allocation =="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "requested --cpus-per-task: 10, --mem: 30000M, --gres=gpu:1 (QOS low caps: 10 CPU / ~31.25GB / 1 GPU per job)"
echo "nproc (actually visible cores): $(nproc)"
free -h
echo

echo "== nvidia-smi =="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

echo
echo "== torch view of the allocation =="
python3 - <<'PYEOF'
import torch
n = torch.cuda.device_count()
print(f"torch.cuda.device_count(): {n}")
for i in range(n):
    name = torch.cuda.get_device_name(i)
    cc = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}  compute_capability={cc}  vram={mem:.1f}GB")
    if cc == (6, 1):
        print(f"    -> Pascal (GTX 1080 Ti class) — this is the baseline the plan assumes")
    elif cc == (7, 5):
        print(f"    -> Turing (RTX 2080 Ti class) — faster path, treat as a bonus not the default")
    else:
        print(f"    -> UNEXPECTED compute capability, re-check the architecture plan's assumptions")
PYEOF

echo
echo "== /scratch check (node-local, only visible inside a job) =="
if [ -d /scratch ]; then
  echo "/scratch exists on $(hostname)"
  TESTDIR="/scratch/${USER}_preflight_$SLURM_JOB_ID"
  mkdir -p "$TESTDIR"
  dd if=/dev/urandom of="$TESTDIR/testfile.bin" bs=1M count=200 status=none && echo "wrote 200MB to /scratch: OK"
  df -h /scratch
  echo "marker file left at $TESTDIR/marker.txt to test persistence across jobs:"
  date > "$TESTDIR/marker.txt"
  echo "$TESTDIR/marker.txt"
  echo "-> submit a second, throwaway job later and check whether this path still exists;"
  echo "   if it's gone, /scratch is confirmed job-ephemeral (as the plan assumes) and every"
  echo "   pipeline stage MUST copy final artifacts to /share1 before the job ends."
else
  echo "/scratch NOT found on $(hostname) — re-check the node/partition, this contradicts the spec doc."
fi
