#!/bin/bash
# End-to-end smoke test for the Docker image: generate -> train (2 epochs)
# -> evaluate, on 3 small local genomes. Deliberately does not touch NCBI
# or any other network service - uses genomes/*.fna already in the repo,
# so this also works with `docker run --network none`.
#
# Run from the host:
#   docker run --rm -v "$PWD/work:/work" dna-assembler-ml \
#       experiments/hpc/smoke_test_docker.sh
#
# Writes everything under $ROOT/work (bind-mount target), nothing outside it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

GENOMES=(hpylori bsubtilis mtuberculosis)
WORK_DIR="${SMOKE_WORK_DIR:-$ROOT/work/smoke_test}"
TRAIN_DATA_DIR="$WORK_DIR/data/training"
RESULTS_DIR="$WORK_DIR/results"

rm -rf "$WORK_DIR"
mkdir -p "$TRAIN_DATA_DIR" "$RESULTS_DIR/checkpoints"

echo "[1/3] Generating training CSVs for: ${GENOMES[*]}"
for g in "${GENOMES[@]}"; do
    echo "  -- $g --"
    "$ROOT/target/release/dna-assembler" pipeline \
        --reference "$ROOT/genomes/${g}.fna" \
        --k 21 --coverage 100 \
        --gen-train-data \
        --train-data-dir "$TRAIN_DATA_DIR" \
        --results-dir "$RESULTS_DIR/${g}"
done

echo "[2/3] Training (2 epochs, device=$DEVICE)"
uv run --project "$ROOT/ml" python3 "$ROOT/ml/main.py" train \
    --data_path "$TRAIN_DATA_DIR" \
    --ckpt_dir "$RESULTS_DIR/checkpoints" \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 2 \
    --device "$DEVICE" \
    --run_name smoke_test

echo "[3/3] Evaluating on ${GENOMES[0]}_training.csv"
uv run --project "$ROOT/ml" python3 "$ROOT/ml/main.py" evaluate \
    --model_path "$RESULTS_DIR/checkpoints/mv1_siamese_cnn_smoke_test_best.pt" \
    --test_csv "$TRAIN_DATA_DIR/${GENOMES[0]}_training.csv" \
    --context_len 150 \
    --device "$DEVICE" \
    --output_json "$RESULTS_DIR/smoke_eval.json"

echo "=== smoke_eval.json ==="
cat "$RESULTS_DIR/smoke_eval.json"
echo
echo "SMOKE TEST OK"
