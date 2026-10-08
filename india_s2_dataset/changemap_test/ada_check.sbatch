#!/bin/bash
# Change-map test on Ada (Qwen3-VL-8B 4-bit), one RTX 2080 Ti. Independent of the API run.
#   cd ~/changemap && sbatch ada_check.sbatch
#
# Ada storage: /share1 is NOT mounted on GPU nodes, so
#   code, data, Python env -> home (~/haze, ~/envs/vlm), visible on every node
#   model weights (~17 GB) -> node-local /ssd_scratch, downloaded on first use on that node
#SBATCH --job-name=changemap
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
BATCH=1
export HF_HOME=/ssd_scratch/$USER/hf UV_NO_CACHE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 8-bit leaves ~1 GB headroom on 11 GB
mkdir -p "$HF_HOME"
echo "node $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

if [ ! -x ~/envs/vlm/bin/python ]; then            # one-time environment, ~6 GB in home
  [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv --python 3.11 ~/envs/vlm
  ~/.local/bin/uv pip install --python ~/envs/vlm/bin/python torch==2.8.0 torchvision \
      --index-url https://download.pytorch.org/whl/cu126
  ~/.local/bin/uv pip install --python ~/envs/vlm/bin/python "transformers>=4.57" accelerate \
      bitsandbytes pillow huggingface_hub hf_transfer
fi
~/envs/vlm/bin/python -c "import torch, transformers, bitsandbytes as b; print('torch', torch.__version__, '| transformers', transformers.__version__, '| bnb', b.__version__, '| gpu ok', torch.cuda.is_available())"
# Some nodes show the GPU in nvidia-smi but fail CUDA init ("CUDA unknown error", e.g. gnode076):
# stop at once so the job can be resubmitted with --exclude=<node>, instead of failing after setup.
~/envs/vlm/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" \
  || { echo "CUDA not usable on $(hostname); resubmit with --exclude=$(hostname)"; exit 3; }

HF_HUB_ENABLE_HF_TRANSFER=1 ~/envs/vlm/bin/hf download Qwen/Qwen3-VL-8B-Instruct > /dev/null   # no-op if cached
export HF_HUB_OFFLINE=1
~/envs/vlm/bin/python ada_check.py
