#!/usr/bin/env bash
# SAFE execution script with PINNED dependencies
# Ensures PyTorch 2.0.1 is used (not auto-upgraded)

#SBATCH --job-name=rsicc_phase8
#SBATCH --output=logs/phase8.out
#SBATCH --error=logs/phase8.err
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "PHASE 8 - SAFE EXECUTION (PyTorch 2.0.1)"
echo "=========================================="
echo "Job started : $(date)"
echo "Host        : $(hostname)"
echo "Working dir : $(pwd)"
echo "=========================================="

# Go to project
cd /home2/sankalp0109/src_IS

# Create fresh virtual environment
echo "========== Creating fresh venv =========="
rm -rf venv 2>/dev/null || true
python -m venv venv
source venv/bin/activate

# CRITICAL: Install from requirements.txt with CORRECT index
# This ensures PyTorch 2.0.1 (not 2.1+) is installed for GTX 1080 Ti sm_61
echo "========== Installing dependencies from requirements.txt =========="
echo "Using PyTorch wheel index: https://download.pytorch.org/whl/cu121"
pip install -q --upgrade pip setuptools wheel
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export NLTK_DATA="$HOME/nltk_data"
export REMOTECLIP_DOWNLOAD_IF_MISSING=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Verify setup
echo ""
echo "========== Verifying Setup =========="
python verify_setup.py

if [ $? -ne 0 ]; then
    echo "ERROR: Setup verification failed"
    exit 1
fi

echo ""
echo "========== Executing Notebook =========="
date

python -u -m jupyter nbconvert \
    --to notebook \
    --execute phase8_finetune.ipynb \
    --output phase8_output.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1

STATUS=$?

echo ""
echo "=========================================="
echo "Notebook finished with exit code: $STATUS"
echo "Finished at: $(date)"
echo "=========================================="

exit $STATUS
