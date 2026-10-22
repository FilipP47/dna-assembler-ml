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

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/venv-smoke-$SLURM_JOB_ID"
echo "[$(date +%T)] Synchronizuje zaleznosci (uv sync, moze potrwac kilka minut)..."
uv sync --project "$ROOT/ml" $UV_SYNC_EXTRA_ARGS
echo "[$(date +%T)] Zaleznosci zainstalowane."

echo "[$(date +%T)] Import check..."
(cd "$ROOT/ml" && uv run python3 -c "from training.train import train_model; print('OK: importy dzialaja')")

echo "[$(date +%T)] Start 1-epokowego treningu (device=cpu)..."
uv run --project "$ROOT/ml" python3 "$ROOT/ml/main.py" train \
    --data_path "$ROOT/data/training/group_a/farthest_train" \
    --ckpt_dir "$ROOT/results/checkpoints_smoke" \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 4 \
    --device cpu \
    --run_name smoke_test_v2
echo "[$(date +%T)] Koniec."
