#!/bin/bash
#==============================================================================
# PHASE 7 (CODEAUG) production SLURM Execution Script for ADA Cluster
# Project: Remote Sensing Image Change Captioning (RSICC)
# Architecture: OpenCLIP ViT-L-14 (FP16) + Visual LoRA + Q-Former + Qwen2-VL-2B (4-bit NF4)
#==============================================================================

#SBATCH --job-name=rsicc_phase7
#SBATCH --output=logs/phase7_%j.out
#SBATCH --error=logs/phase7_%j.err
#SBATCH --partition=gnode
#SBATCH --nodelist=gnode004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 1. Environment & CUDA setup
echo "======================================================================"
echo "STARTING PHASE 7 (CODEAUG) SLURM JOB ON ADA CLUSTER"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Start Time   : $(date)"
echo "======================================================================"

# Activate Python environment
source ~/.bashrc
conda activate mlenv || source venv/bin/activate

# Ensure logs and checkpoints directories exist
mkdir -p logs checkpoints

# 2. Pre-Flight Hardware & VRAM Calibration Check
echo "[Phase 7] Running mandatory empirical VRAM calibration..."
python check_vram_calibration.py
if [ $? -ne 0 ]; then
    echo "❌ FATAL: VRAM calibration failed. Check hardware/VRAM headroom."
    exit 1
fi
echo "✅ VRAM calibration passed."

# 3. Execute Phase 7 Jupyter Notebook
echo "[Phase 7] Executing phase7.ipynb pipeline..."
jupyter nbconvert --to notebook --execute phase7.ipynb --output phase7_output.ipynb

if [ $? -eq 0 ]; then
    echo "======================================================================"
    echo "✅ PHASE 7 EXECUTION COMPLETED SUCCESSFULLY: $(date)"
    echo "Output Notebook: phase7_output.ipynb"
    echo "======================================================================"
else
    echo "❌ ERROR: phase7.ipynb execution failed: $(date)"
    exit 1
fi
