#!/bin/bash
# Task body for evaluating the 3-seed repeat-variant checkpoints. Portable:
# called by evaluate_repeats.sh (SLURM) and run_local.sh eval_repeats <n>.
#
# Requires: ROOT, DEVICE set, SLURM_ARRAY_TASK_ID is a 1-based line number
# into repeat_variant_tasks.txt.
set -euo pipefail

TASKS_FILE="$ROOT/repeat_variant_tasks.txt"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")
if [ -z "$LINE" ]; then
    echo "eval_repeats.sh: no line ${SLURM_ARRAY_TASK_ID} in $TASKS_FILE" >&2
    exit 1
fi
VARIANT=$(echo "$LINE" | cut -d, -f1)
DATA_PATH=$(echo "$LINE" | cut -d, -f2)
SEED=$(echo "$LINE" | cut -d, -f3)

TEST_DIR="$ROOT/$(echo "$DATA_PATH" | sed 's/_train$/_test/')"
CKPT="$ROOT/results/checkpoints_repeats/mv1_siamese_cnn_${VARIANT}_seed${SEED}_best.pt"
RESULTS_DIR="$ROOT/results/eval_repeats/${VARIANT}_seed${SEED}"

echo "[$(date +%T)] Ewaluacja ${VARIANT} seed=${SEED} na ${TEST_DIR}"
mkdir -p "$RESULTS_DIR"

for csv_file in "$TEST_DIR"/*.csv; do
    acc=$(basename "$csv_file" _training.csv)
    uv run --project "$ROOT/ml" python3 "$ROOT/ml/main.py" evaluate \
        --model_path "$CKPT" \
        --test_csv "$csv_file" \
        --context_len 150 \
        --device "$DEVICE" \
        --output_json "${RESULTS_DIR}/${acc}.json"
done

echo "[$(date +%T)] Koniec: ${VARIANT} seed=${SEED}"
