#!/bin/bash
#SBATCH --job-name=ml_smoke_test
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "[$(date +%T)] Start joba"
module load uv/0.11.2
cd /scratch/fpietrew/dna-assembler/ml

VENV_DIR=$SLURM_SUBMIT_DIR/venv-smoke-$SLURM_JOB_ID
echo "[$(date +%T)] Tworze venv..."
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[$(date +%T)] Instaluje zaleznosci z requirements.txt (moze potrwac kilka minut)..."
uv pip install -r requirements.txt
echo "[$(date +%T)] Zaleznosci zainstalowane."

echo "[$(date +%T)] Import check..."
python3 -c "from training.train import train_model; print('OK: importy dzialaja')"

echo "[$(date +%T)] Start 1-epokowego treningu (device=cpu)..."
python3 main.py train \
    --data_path ../data/training/group_a/farthest_train \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 4 \
    --device cpu \
    --ckpt_dir ../results/checkpoints_smoke \
    --run_name smoke_test_v2
echo "[$(date +%T)] Koniec."
