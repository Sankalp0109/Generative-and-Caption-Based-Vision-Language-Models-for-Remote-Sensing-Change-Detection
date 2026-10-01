#!/usr/bin/env bash
#SBATCH --job-name=phase8_training
#SBATCH --output=logs/phase8_training.out
#SBATCH --error=logs/phase8_training.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "PHASE 8 FINE-TUNING TRAINING"
echo "Job started : $(date)"
echo "Host        : $(hostname)"
echo "Working dir : $(pwd)"
echo "=========================================="

cd /home2/sankalp0109/src_IS
source venv/bin/activate

echo ""
echo "========== System Info =========="
which python
python --version
nvidia-smi

echo ""
echo "========== Installing PyTorch 2.0.1 for CUDA 12.8 (GTX 1080 Ti sm_61) =========="
echo "⚠️  CRITICAL: PyTorch 2.0.1 is REQUIRED for GTX 1080 Ti (sm_61)"
echo "⚠️  PyTorch 2.1+ dropped sm_61 support - will fall back to CPU!"
echo ""
pip cache purge
pip uninstall torch torchvision torchaudio -y
# Use cu121 index (covers CUDA 12.1+, includes 12.8)
pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "========== Installing Requirements from requirements.txt =========="
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "========== PyTorch Verification =========="
python -u <<'PYEOF'
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
print("="*60, flush=True)
print(f"PyTorch Version: {torch.__version__}", flush=True)

# Check PyTorch version
major, minor = int(torch.__version__.split('.')[0]), int(torch.__version__.split('.')[1])
if major == 2 and minor == 0:
    print("✅ PyTorch 2.0.x - CORRECT for GTX 1080 Ti (sm_61)", flush=True)
elif major == 2 and minor >= 1:
    print(f"❌ ERROR: PyTorch {torch.__version__} - TOO NEW!", flush=True)
    print("PyTorch 2.1+ dropped sm_61 support - MUST USE 2.0.1", flush=True)
    sys.exit(1)

print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Compute Capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})", flush=True)
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)

    # Test CUDA
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✅ CUDA Operations: WORKING!", flush=True)
    except RuntimeError as e:
        print(f"❌ CUDA Error: {e}", flush=True)
        sys.exit(1)
else:
    print("❌ ERROR - CUDA NOT available! Will run on CPU (100x slower)", flush=True)
    sys.exit(1)
print("="*60, flush=True)
PYEOF

if [ $? -ne 0 ]; then
    echo "CUDA verification FAILED!"
    exit 1
fi

echo ""
echo "========== Environment Setup =========="
export PYTHONUNBUFFERED=1
export NLTK_DATA="$HOME/nltk_data"
export REMOTECLIP_DOWNLOAD_IF_MISSING=1

echo ""
echo "========== Running Phase 8 Fine-tuning Notebook =========="

python -u -m jupyter nbconvert \
    --to notebook \
    --execute phase8_finetune.ipynb \
    --output phase8_output.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1

STATUS=$?

echo ""
echo "=========================================="
if [ $STATUS -eq 0 ]; then
  echo "✅ Training COMPLETED successfully!"
else
  echo "❌ Training FAILED with exit code: $STATUS"
fi
echo "Finished at: $(date)"
echo "=========================================="

exit $STATUS
