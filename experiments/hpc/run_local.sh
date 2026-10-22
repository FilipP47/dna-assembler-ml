#!/bin/bash
# Run one of experiments/hpc/tasks/*.sh locally, without SLURM: a plain
# parallel loop (xargs -P CORES) over task indices 1..N, setting
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_JOB_ID the same way sbatch would so the
# task body (and ml/main.py underneath it) can't tell the difference. This
# is the same logic the *_array.sh scripts run under SLURM: see
# experiments/hpc/tasks/<name>.sh for the actual work.
#
# Usage: ./run_local.sh <task> <n_tasks> [max_task_index]
#   task       one of: gen_group_a gen_group_b train_full train_solo
#              train_repeat eval_zero_shot eval_repeats
#   n_tasks    how many indices to run, i.e. 1..n_tasks
#
# Examples:
#   ./run_local.sh train_full 4
#   CORES=2 DEVICE=cpu ./run_local.sh gen_group_a 100
#
# All python tasks share one local venv (ml/.venv, via UV_PROJECT_ENVIRONMENT)
# instead of the per-array-task venv the SLURM scripts use - there's no
# shared-filesystem contention to avoid locally, and re-syncing per task
# would just be slower.
set -euo pipefail

usage() {
    echo "Usage: $0 <task> <n_tasks>" >&2
    echo "  task: gen_group_a | gen_group_b | train_full | train_solo | train_repeat | eval_zero_shot | eval_repeats" >&2
    exit 1
}

TASK="${1:-}"
N_TASKS="${2:-}"
[ -z "$TASK" ] && usage
[ -z "$N_TASKS" ] && usage

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

TASK_SCRIPT="$HPC_DIR/tasks/${TASK}.sh"
if [ ! -f "$TASK_SCRIPT" ]; then
    echo "run_local.sh: unknown task '$TASK' (no $TASK_SCRIPT)" >&2
    usage
fi

case "$TASK" in
    train_full|train_solo|train_repeat|eval_zero_shot|eval_repeats)
        export UV_PROJECT_ENVIRONMENT="$ROOT/ml/.venv"
        echo "[$(date +%T)] uv sync ($ROOT/ml, group matching DEVICE=$DEVICE)..."
        uv sync --project "$ROOT/ml" $UV_SYNC_EXTRA_ARGS
        ;;
esac

echo "[$(date +%T)] Running '$TASK' for tasks 1..$N_TASKS with $CORES parallel workers (DEVICE=$DEVICE)"

seq 1 "$N_TASKS" | xargs -P "$CORES" -I{} env \
    SLURM_ARRAY_TASK_ID={} \
    SLURM_ARRAY_JOB_ID=local \
    ROOT="$ROOT" \
    DEVICE="$DEVICE" \
    UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-}" \
    bash "$TASK_SCRIPT"

echo "[$(date +%T)] run_local.sh: '$TASK' done (1..$N_TASKS)"
