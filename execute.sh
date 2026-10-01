#!/usr/bin/env bash
#SBATCH --job-name=rsicc_phase8
#SBATCH --output=logs/phase.out
#SBATCH --error=logs/phase_2.err
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job started : $(date)"
echo "Host        : $(hostname)"
echo "Working dir : $(pwd)"
echo "=========================================="

# Go to project
cd /home2/sankalp0109/src_IS

# Clean old environment
echo "========== Cleaning old environment =========="
rm -rf venv 2>/dev/null || true

# Create fresh virtual environment
echo "========== Creating fresh venv =========="
python -m venv venv
source venv/bin/activate

# Install all dependencies from requirements.txt with PINNED versions
# CRITICAL: Use correct PyTorch wheel index to ensure PyTorch 2.0.1 (NOT 2.1+)
# PyTorch 2.1+ dropped sm_61 support (GTX 1080 Ti)
echo "========== Installing dependencies =========="
echo "Installing from requirements.txt with PyTorch wheel index: cu121"
pip install -q --upgrade pip setuptools wheel
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies from requirements.txt"
    echo "Make sure you have internet access and correct index URL"
    exit 1
fi

# Flush Python output immediately
export PYTHONUNBUFFERED=1

# NLTK
export NLTK_DATA="$HOME/nltk_data"
export REMOTECLIP_DOWNLOAD_IF_MISSING=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

echo ""
echo "========== Python =========="
which python
python --version

echo ""
echo "========== GPU =========="
nvidia-smi || true

echo ""
echo "========== PyTorch & CUDA Info =========="
python -u <<'PY'
import sys
import torch

sys.stdout.reconfigure(line_buffering=True)

print("=" * 50, flush=True)
print("Torch Version:", torch.__version__, flush=True)
print("CUDA Runtime:", torch.version.cuda, flush=True)
print("CUDA Available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Capability:", torch.cuda.get_device_capability(0), flush=True)
    print("Supported Architectures:", torch.cuda.get_arch_list(), flush=True)
else:
    print("WARNING: Running on CPU", flush=True)

print("=" * 50, flush=True)
PY

echo ""
echo "========== Executing Notebook =========="
date

python -u -m jupyter nbconvert \
    --to notebook \
    --execute phase8_finetune.ipynb \
    --output phase8_output.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    --debug

STATUS=$?

echo ""
echo "=========================================="
echo "Notebook finished with exit code: $STATUS"
echo "Finished at: $(date)"
echo "=========================================="

exit $STATUS
