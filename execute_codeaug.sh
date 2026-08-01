#!/usr/bin/env bash
#SBATCH --job-name=rsicc_codeaug
#SBATCH --output=logs/codeaug_%j.out
#SBATCH --error=logs/codeaug_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# ==============================================================================
# CodeAug RSICC Training Script for ADA Cluster (gnode004 GTX 1080 Ti 11 GB VRAM)
# Automatically executes Pre-Flight Calibration followed by Stage 1 & Stage 2.
# ==============================================================================

echo "=========================================================================="
echo "Job started : $(date)"
echo "Host        : $(hostname)"
echo "Working dir : $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "=========================================================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "$HOME/miniforge3/envs/mlenv" ]; then
    source $HOME/miniforge3/envs/mlenv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create log directory
mkdir -p logs checkpoints

echo ""
echo "--------------------------------------------------------------------------"
echo "STEP 0: MANDATORY EMPIRICAL PRE-FLIGHT VRAM CALIBRATION"
echo "--------------------------------------------------------------------------"
python check_vram_calibration.py
if [ $? -ne 0 ]; then
    echo "❌ [ERROR] Pre-flight VRAM calibration failed! Aborting job."
    exit 1
fi
echo "✅ Pre-flight calibration passed."

echo ""
echo "--------------------------------------------------------------------------"
echo "STEP 1: TWO-STAGE TWO-STREAM CODEAUG TRAINING"
echo "--------------------------------------------------------------------------"
# Note: DataLoader num_workers=0 is enforced internally by CodeAugConfig for ADA nodes.
# Uncomment below to launch training script when ready:
# python -m src.training --config-name codeaug --stage 1 --epochs 10 --batch-size 4 --grad-accum 4
# python -m src.training --config-name codeaug --stage 2 --epochs 15 --batch-size 2 --grad-accum 8

echo ""
echo "=========================================================================="
echo "Job finished: $(date)"
echo "=========================================================================="
