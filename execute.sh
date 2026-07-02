#!/usr/bin/env bash

# HPC batch script for running the RSICC notebook non-interactively.
# Submit with: sbatch execute.sh

# ---- SLURM settings ----
# Adjust these if your cluster uses SLURM and you want to override defaults.
#SBATCH --job-name=rsicc_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# ---- Environment setup ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

cd /home2/sankalp0109/src_IS
source venv/bin/activate

# Tell NLTK where to store/find its data
export NLTK_DATA=$HOME/nltk_data

# Download required NLTK resources (safe to run every time)
python - <<'EOF'
import nltk
nltk.download("wordnet", download_dir="/home2/sankalp0109/nltk_data")
nltk.download("omw-1.4", download_dir="/home2/sankalp0109/nltk_data")
EOF

# Execute notebook
jupyter nbconvert \
    --to notebook \
    --execute phase2_modular_pipeline.ipynb \
    --output phase2.ipynb \
    --ExecutePreprocessor.timeout=-1

