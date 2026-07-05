#!/usr/bin/env bash
# Fine-tune model_src on organism_tgt and evaluate.
# Usage: bash experiments/transfer/run_finetune.sh <src> <tgt>
# Example: bash experiments/transfer/run_finetune.sh ecoli cerevisiae
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 experiments/transfer/run_experiments.py --organisms "$1" "$2"
