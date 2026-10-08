#!/usr/bin/env bash
#SBATCH --job-name=rsicc_phase1
#SBATCH --output=logs/phase_final.out
#SBATCH --error=logs/phase_final.err
#SBATCH --time=12:00:00
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

# Activate environment
source venv/bin/activate

# Flush Python output immediately
export PYTHONUNBUFFERED=1

# NLTK
export NLTK_DATA="$HOME/nltk_data"
export REMOTECLIP_DOWNLOAD_IF_MISSING=1

echo "========== Python =========="
which python
python --version

echo "========== GPU =========="
nvidia-smi || true


echo "========== Executing Notebook =========="
date

python -u -m jupyter nbconvert \
    --to notebook \
    --execute phase_final.ipynb \
    --output phase_final_output.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    --debug

STATUS=$?

echo "=========================================="
echo "Notebook finished with exit code: $STATUS"
echo "Finished at: $(date)"
echo "=========================================="

exit $STATUS