#!/bin/bash
#SBATCH --job-name=caption_dataset
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30000M
#SBATCH --time=12:00:00
#SBATCH --output=caption_dataset_%j.out

# Stage 2 job: run caption_dataset.py on Ada. QOS-compliant resource request
# (confirmed working parameters -- see architecture_plan.md "QOS low tier").
# This account has only 1 GPU total across all running jobs, so this is a
# strictly sequential job -- don't submit a second one expecting it to run
# concurrently, it'll just queue (PD) until this one finishes.

VENV_PATH="/home2/sankalp0109/src_IS/venv"
if [ ! -x "$VENV_PATH/bin/python3" ]; then
  echo "FATAL: no env found at $VENV_PATH -- run ../ada_preflight/setup_venv.sh then ../ada_preflight/setup_qwen_caption.sh first."
  exit 1
fi
export PATH="$VENV_PATH/bin:$PATH"  # conda env (created with -p), not a venv -- no bin/activate to source

# Must match the HF_CACHE_DIR used in setup_qwen_caption.sh, or this job
# won't find the pre-downloaded ~15GB Qwen2-VL-7B weights and will try to
# re-download them (and into $HOME's small quota, by HF's default, at that).
export HF_HOME="${HF_HOME:-/share1/$USER/models_cache}"

echo "Node: $(hostname)"
echo "HF_HOME: $HF_HOME"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

cd "$(dirname "$0")"

# Copy final dataset from /share1/$USER (or wherever it was uploaded) to
# /scratch for fastest I/O during the run, then copy captions.jsonl back
# to /share1/$USER before the job ends -- adjust these paths to match
# wherever the dataset actually landed on this account.
DATASET_DIR="${1:-/share1/$USER/india_dataset/dataset}"
OUT_FILE="${2:-/share1/$USER/captions.jsonl}"

python3 caption_dataset.py --dataset-dir "$DATASET_DIR" --out "$OUT_FILE"

echo "done -- captions saved to $OUT_FILE"
