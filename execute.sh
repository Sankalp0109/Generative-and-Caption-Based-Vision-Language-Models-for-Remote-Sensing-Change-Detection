#!/usr/bin/env bash
#SBATCH --job-name=rsicc_phase1
#SBATCH --output=logs/phase.out
#SBATCH --error=logs/phase_2.err
#SBATCH --time=24:00:00
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
# Create fresh virtual environment
echo "========== Creating fresh venv =========="
source venv/bin/activate

# Upgrade pip
# Install all dependencies
echo "========== Installing dependencies =========="
# Flush Python output immediately
export PYTHONUNBUFFERED=1

# NLTK
export NLTK_DATA="$HOME/nltk_data"
export REMOTECLIP_DOWNLOAD_IF_MISSING=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
echo "========== Python =========="
which python
python --version

echo "========== GPU =========="
nvidia-smi || true

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
    print("Running on CPU", flush=True)

print("=" * 50, flush=True)
PY

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

echo "=========================================="
echo "Notebook finished with exit code: $STATUS"
echo "Finished at: $(date)"
echo "=========================================="

exit $STATUS
