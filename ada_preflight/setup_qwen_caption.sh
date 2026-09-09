#!/bin/bash
# Stage 0.5b: add the Qwen2-VL captioning extras to an EXISTING venv (run
# setup_venv.sh first). Kept separate from the base venv script on purpose --
# the base stack (torch/transformers/bitsandbytes/open-clip-torch) is needed
# by every stage, but qwen-vl-utils and the Qwen2-VL model weights are only
# needed for Stage 2 (auto-captioning). Splitting them means:
#   - the base environment can be installed/tested independently of Stage 2
#   - re-running just this script (e.g. to bump qwen-vl-utils) never touches
#     the already-working base stack
#
# Also pre-downloads the Qwen2-VL-7B-Instruct weights into HF_HOME so the
# actual captioning job doesn't spend its walltime on a ~15GB download.
#
# Usage: bash setup_qwen_caption.sh [venv_path] [hf_cache_dir]

set -e
VENV_PATH="${1:-/home2/sankalp0109/src_IS/venv}"
HF_CACHE_DIR="${2:-/share1/$USER/models_cache}"

if [ ! -x "$VENV_PATH/bin/python3" ]; then
  echo "FATAL: no env found at $VENV_PATH -- run setup_venv.sh first."
  exit 1
fi
export PATH="$VENV_PATH/bin:$PATH"  # conda env (created with -p), not a venv -- no bin/activate to source

pip install qwen-vl-utils

# NOTE: transformers 4.39.3 (pinned in setup_venv.sh) predates Qwen2-VL's
# official upstream merge (~4.45+). Qwen2-VL-2B was reportedly working on
# this exact pin before via trust_remote_code=True (see architecture_plan.md
# "Software environment"), but that hasn't been re-verified for Qwen2-VL-7B
# specifically. If loading the 7B model fails on import/from_pretrained, the
# fix is either bumping transformers to >=4.45 (re-validate the rest of the
# pinned stack still works) or confirming trust_remote_code extends cleanly
# -- don't assume it just works, this is exactly what the calibration run
# (caption_dataset.py --max-pairs 20) is for.

mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
echo "HF_HOME set to $HF_CACHE_DIR (~15GB for Qwen2-VL-7B-Instruct weights --"
echo "make sure this points at /share1/\$USER, NOT \$HOME, or it'll collide with the venv's quota)."

echo
echo "=== pre-downloading Qwen2-VL-7B-Instruct (one-time, ~15GB) ==="
python3 -c "
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
MODEL_ID = 'Qwen/Qwen2-VL-7B-Instruct'
print('downloading processor...')
AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print('downloading model weights (this is the ~15GB part)...')
Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True, device_map='cpu')
print('done -- weights cached at', '$HF_CACHE_DIR')
"

echo "Qwen2-VL-7B-Instruct ready in $HF_CACHE_DIR."
echo "Remember to 'export HF_HOME=$HF_CACHE_DIR' in caption_dataset.sbatch (or before running it interactively)."
echo "After Stage 2 finishes, these weights can be deleted (rm -rf $HF_CACHE_DIR/hub/models--Qwen--Qwen2-VL-7B-Instruct)"
echo "to free ~15GB back up for Stage 3, which uses the smaller Qwen2-VL-2B / a small decoder instead."
