#!/bin/bash
# Stage 0.5a: build the base venv that ADA_CLUSTER_SPECIFICATION.md documents
# as already existing at /home2/sankalp0109/src_IS/venv -- confirmed MISSING
# when 02_bnb_pascal_calibration.sbatch tried to source it (job on gnode084,
# `import torch` found nothing). Run this once, on a login node (no GPU
# needed just to create the venv and pip-install).
#
# This installs only the FOUNDATION stack shared by every stage: torch,
# transformers, bitsandbytes, accelerate, open-clip-torch (for RemoteCLIP),
# plus the Stage 1 data-collection packages (pystac-client, rasterio, etc).
#
# Stage-specific extras (e.g. Qwen2-VL for captioning) live in their own
# separate setup script -- see setup_qwen_caption.sh -- so the pipeline can
# be installed and updated in pieces instead of one monolithic script.
#
# Versions pinned to exactly what the prior working Phase 7 logs recorded,
# so behavior matches what was already proven for the LEVIR-CC run.
#
# Usage: bash setup_venv.sh [venv_path]
#
# NOTE: Ada's system python3 was found to be 3.6.8 with no python3.8+ and no
# module system to get one (confirmed on sankalp0109's account) -- torch
# 2.0.1 needs >=3.8, so `python3 -m venv` cannot work here. This script
# installs its own Miniforge (bundles Python) into $HOME if conda isn't
# already available, then creates the env with conda instead of venv.

set -e
VENV_PATH="${1:-/home2/sankalp0109/src_IS/venv}"
MINIFORGE_DIR="$HOME/miniforge3"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$MINIFORGE_DIR/bin/conda" ]; then
    echo "conda not on PATH but found at $MINIFORGE_DIR, using it directly."
  else
    echo "No conda found -- installing Miniforge to $MINIFORGE_DIR (bundles its own Python, ~500MB)..."
    curl -L -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
    rm -f /tmp/miniforge.sh
  fi
  eval "$("$MINIFORGE_DIR/bin/conda" shell.bash hook)"
else
  eval "$(conda shell.bash hook)"
fi

echo "Creating Python 3.10 env at $VENV_PATH ..."
conda create -y -p "$VENV_PATH" python=3.10
export PATH="$VENV_PATH/bin:$PATH"  # equivalent of activating, but works uniformly for scripts sourcing this env later

pip install --upgrade pip

# torch/vision/audio pinned to the cu118 wheels the prior logs used
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install \
    transformers==4.39.3 \
    bitsandbytes==0.45.5 \
    accelerate==1.14.0 \
    open-clip-torch==2.32.0 \
    huggingface-hub==0.36.2 \
    numpy==1.26.4 \
    nltk==3.10.3 \
    scikit-learn==1.7.2 \
    jupyter nbconvert ipykernel \
    pystac-client planetary-computer rasterio pillow

echo
echo "=== sanity check ==="
python3 -c "
import torch, transformers, bitsandbytes, accelerate, open_clip
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('bitsandbytes', bitsandbytes.__version__)
print('accelerate', accelerate.__version__)
print('open_clip', open_clip.__version__)
"
echo "base venv ready at $VENV_PATH."
echo "Next: run setup_qwen_caption.sh (same venv path) before Stage 2 captioning,"
echo "or 02_bnb_pascal_calibration.sbatch to calibrate on this GPU."
