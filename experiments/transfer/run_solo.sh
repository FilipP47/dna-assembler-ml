#!/usr/bin/env bash
# Train a single organism solo.
# Usage: bash experiments/transfer/run_solo.sh <organism_name>
# Example: bash experiments/transfer/run_solo.sh ecoli
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 experiments/transfer/run_experiments.py --only-solo --organisms "$1"
