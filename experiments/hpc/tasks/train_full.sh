#!/bin/bash
# Task body for the 4-way group A/B x farthest/random training run.
# Portable: no #SBATCH, no module load, no venv setup. Called by both
# train_full_array.sh (under SLURM, one task per array index) and
# run_local.sh train_full 4 (plain loop, same SLURM_ARRAY_TASK_ID contract).
#
# Requires: ROOT, DEVICE set (see config.sh) and SLURM_ARRAY_TASK_ID in 1..4.
set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    1) DATA_PATH="$ROOT/data/training/group_a/farthest_train"; RUN_NAME="group_a_farthest" ;;
    2) DATA_PATH="$ROOT/data/training/group_a/random_train";   RUN_NAME="group_a_random"   ;;
    3) DATA_PATH="$ROOT/data/training/group_b/farthest_train"; RUN_NAME="group_b_farthest" ;;
    4) DATA_PATH="$ROOT/data/training/group_b/random_train";   RUN_NAME="group_b_random"   ;;
    *) echo "train_full.sh: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected 1-4)" >&2; exit 1 ;;
esac

echo "[$(date +%T)] Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (${DATA_PATH})"

uv run --project "$ROOT/ml" python3 "$ROOT/ml/main.py" train \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$ROOT/results/checkpoints" \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 50 \
    --early_stop 10 \
    --device "$DEVICE" \
    --run_name "$RUN_NAME"

echo "[$(date +%T)] Koniec: ${RUN_NAME}"
