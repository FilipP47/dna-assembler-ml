#!/bin/bash
# Task body for zero-shot evaluation of the 4 main group A/B transfer
# checkpoints, each against every held-out organism in its test split.
# Portable: called by evaluate_zero_shot.sh (SLURM) and
# run_local.sh eval_zero_shot 4.
#
# Requires: ROOT set, SLURM_ARRAY_TASK_ID in 1..4.
set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    1) TEST_DIR="$ROOT/data/training/group_a/farthest_test"; CKPT="$ROOT/results/checkpoints/mv1_siamese_cnn_group_a_farthest_best.pt"; OUT_PREFIX="group_a_farthest" ;;
    2) TEST_DIR="$ROOT/data/training/group_a/random_test";   CKPT="$ROOT/results/checkpoints/mv1_siamese_cnn_group_a_random_best.pt";   OUT_PREFIX="group_a_random"   ;;
    3) TEST_DIR="$ROOT/data/training/group_b/farthest_test"; CKPT="$ROOT/results/checkpoints/mv1_siamese_cnn_group_b_farthest_best.pt"; OUT_PREFIX="group_b_farthest" ;;
    4) TEST_DIR="$ROOT/data/training/group_b/random_test";   CKPT="$ROOT/results/checkpoints/mv1_siamese_cnn_group_b_random_best.pt";   OUT_PREFIX="group_b_random"   ;;
    *) echo "eval_zero_shot.sh: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected 1-4)" >&2; exit 1 ;;
esac

echo "[$(date +%T)] Ewaluacja ${OUT_PREFIX} na kazdym z organizmow testowych osobno"

RESULTS_DIR="$ROOT/results/eval_${OUT_PREFIX}"
mkdir -p "$RESULTS_DIR"

for csv_file in "$TEST_DIR"/*.csv; do
    acc=$(basename "$csv_file" _training.csv)
    echo "[$(date +%T)] -- $acc --"
    uv run --project "$ROOT/ml" python3 "$ROOT/ml/main.py" evaluate \
        --model_path "$CKPT" \
        --test_csv "$csv_file" \
        --context_len 150 \
        --device "$DEVICE" \
        --output_json "${RESULTS_DIR}/${acc}.json"
done

echo "[$(date +%T)] Koniec: ${OUT_PREFIX}"
