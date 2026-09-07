"""Train the group A generalist model and evaluate it zero-shot on held-out E. coli strains.

Group A tests the "same-species" point of the transfer gradient (A/B/C by
genomic distance): train on ~90 E. coli genomes, evaluate zero-shot on
10 held-out strains chosen as the phylogenetically farthest from the
training pool (see select_heldout.py). Compares the generalist model against
the max-coverage baseline, per held-out organism and aggregated. Results and
rationale: RAPORT_TRANSFER.md.

Solo-reference training (one model per held-out organism, trained and
evaluated on that organism alone) and the 10/30/60/90 training-size ablation
are deliberately NOT part of this first pass - they're the natural next step
once this baseline comparison is in.

Usage (from repo root, after generate_training_data.sh has produced the CSVs):
    python3 experiments/group_a/run_group_a.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CONFIG = Path(__file__).parent / "config" / "group_a.yaml"

sys.path.insert(0, str(ML_DIR))
from data.loader import load_training_csv

BASES = ["A", "C", "G", "T"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Load the group A YAML config (data paths + training hyperparameters)."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command, streaming its output, raising on failure."""
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def train_generalist(cfg: dict) -> Path:
    """Train the generalist model on the full group A train directory.

    Returns:
        Checkpoint directory (resumable via a `.done` sentinel, matching
        `experiments/transfer/run_experiments.py`).
    """
    ckpt_dir = RESULTS_DIR / "generalist"
    sentinel = ckpt_dir / ".done"
    tr = cfg["training"]

    if sentinel.exists():
        logger.info("SKIP generalist training (already done)")
        return ckpt_dir

    train_dir = REPO_ROOT / cfg["data"]["train_dir"]
    if not any(train_dir.glob("*.csv")):
        raise SystemExit(
            f"No training CSVs in {train_dir} - run generate_training_data.sh first."
        )

    run_cmd(
        [
            sys.executable,
            str(ML_DIR / "main.py"),
            "train",
            "--data_path",
            str(train_dir),
            "--ckpt_dir",
            str(ckpt_dir),
            "--run_name",
            "ecoli_group_a_generalist",
            "--epochs",
            str(tr["epochs"]),
            "--batch_size",
            str(tr["batch_size"]),
            "--lr",
            str(tr["lr"]),
            "--early_stop",
            str(tr["early_stop"]),
            "--seed",
            str(tr["seed"]),
            "--val_split",
            str(tr["val_split"]),
            "--embed_dim",
            str(tr["embed_dim"]),
            "--hidden_dim",
            str(tr["hidden_dim"]),
            *(["--augment"] if tr["augment"] else []),
        ]
    )

    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    return ckpt_dir


def find_checkpoint(ckpt_dir: Path) -> Path:
    """Find the trained checkpoint in `ckpt_dir` (any `*_best.pt` file)."""
    candidates = list(ckpt_dir.glob("*_best.pt"))
    if not candidates:
        raise SystemExit(f"No checkpoint found in {ckpt_dir}")
    return max(candidates, key=lambda p: len(p.name))


def evaluate_zero_shot(ckpt_path: Path, csv_path: Path, out_json: Path) -> dict:
    """Evaluate the generalist checkpoint on one held-out CSV (or directory)."""
    if out_json.exists():
        import json

        return json.loads(out_json.read_text())

    run_cmd(
        [
            sys.executable,
            str(ML_DIR / "main.py"),
            "evaluate",
            "--model_path",
            str(ckpt_path),
            "--test_csv",
            str(csv_path),
            "--output_json",
            str(out_json),
        ]
    )
    import json

    return json.loads(out_json.read_text())


def compute_baseline(csv_path: Path) -> tuple[float, int]:
    """Compute max-coverage heuristic accuracy on a training CSV.

    Mirrors the baseline logic in `ml/experiments/cov_ratio_analysis.py`:
    for each labeled junction, predict the branch with the highest read
    coverage and compare against `ground_truth`.

    Args:
        csv_path: Path to a training CSV or a directory of CSVs.

    Returns:
        Tuple of ``(accuracy, n_junctions)``.
    """
    df, _stats = load_training_csv(csv_path, only_labeled=True)
    cov_cols = [f"branch_{b}_cov" for b in BASES]
    correct = 0
    total = 0
    for row in df.select(["ground_truth", *cov_cols]).iter_rows(named=True):
        cov = {b: (row[f"branch_{b}_cov"] or 0.0) for b in BASES}
        if not any(v > 0 for v in cov.values()):
            continue
        total += 1
        if max(cov, key=cov.get) == row["ground_truth"]:
            correct += 1
    accuracy = correct / total if total else 0.0
    return accuracy, total


def print_comparison_table(rows: list[dict]) -> None:
    """Print and save the final per-organism + aggregate comparison table.

    Format matches the existing tables in RAPORT.md: percentages with one
    decimal place, signed delta in percentage points.

    `n_baseline` and `n_generalist` are reported separately (not collapsed
    into one `n`) because they come from different filters: `compute_baseline`
    additionally skips junctions with zero coverage on every branch (the
    heuristic has nothing to rank there), while the model evaluates every
    labeled junction. Logs a warning per row where they differ, since that
    means baseline_pct and generalist_pct are not computed on the same
    population of junctions.
    """
    import csv as csv_module

    out_path = RESULTS_DIR / "comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv_module.DictWriter(
            f,
            fieldnames=[
                "organism",
                "n_baseline",
                "n_generalist",
                "baseline_pct",
                "generalist_pct",
                "delta_pp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["n_baseline"] != row["n_generalist"]:
            logger.warning(
                "%s: n_baseline=%d != n_generalist=%d - baseline and model are "
                "scored on different junction populations, delta_pp is not apples-to-apples",
                row["organism"],
                row["n_baseline"],
                row["n_generalist"],
            )

    header = (
        f"{'organism':<20} {'n_base':>7} {'n_gen':>7} {'baseline':>10} "
        f"{'generalist':>12} {'delta':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for row in rows:
        logger.info(
            "%-20s %7d %7d %9.1f%% %11.1f%% %+7.1fpp",
            row["organism"],
            row["n_baseline"],
            row["n_generalist"],
            row["baseline_pct"],
            row["generalist_pct"],
            row["delta_pp"],
        )
    logger.info("Saved -> %s", out_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Path to group_a.yaml"
    )
    return parser


def main() -> None:
    """Train the generalist model and evaluate it against the baseline on held-out strains."""
    args = build_parser().parse_args()
    cfg = load_config(Path(args.config))

    ckpt_dir = train_generalist(cfg)
    ckpt_path = find_checkpoint(ckpt_dir)

    heldout_dir = REPO_ROOT / cfg["data"]["heldout_dir"]
    heldout_csvs = sorted(heldout_dir.glob("*.csv"))
    if not heldout_csvs:
        raise SystemExit(f"No held-out CSVs in {heldout_dir}")

    rows = []
    for csv_path in heldout_csvs:
        organism = csv_path.stem.replace("_training", "")
        out_json = RESULTS_DIR / "eval" / f"{organism}.json"
        eval_result = evaluate_zero_shot(ckpt_path, csv_path, out_json)
        baseline_acc, n_baseline = compute_baseline(csv_path)
        rows.append(
            {
                "organism": organism,
                "n_baseline": n_baseline,
                "n_generalist": eval_result["n_junctions"],
                "baseline_pct": baseline_acc * 100,
                "generalist_pct": eval_result["junction_accuracy"] * 100,
                "delta_pp": (eval_result["junction_accuracy"] - baseline_acc) * 100,
            }
        )

    # Aggregate over the whole held-out directory (one combined evaluation).
    agg_json = RESULTS_DIR / "eval" / "aggregate.json"
    agg_result = evaluate_zero_shot(ckpt_path, heldout_dir, agg_json)
    agg_baseline_acc, agg_n_baseline = compute_baseline(heldout_dir)
    rows.append(
        {
            "organism": "ALL (aggregate)",
            "n_baseline": agg_n_baseline,
            "n_generalist": agg_result["n_junctions"],
            "baseline_pct": agg_baseline_acc * 100,
            "generalist_pct": agg_result["junction_accuracy"] * 100,
            "delta_pp": (agg_result["junction_accuracy"] - agg_baseline_acc) * 100,
        }
    )

    print_comparison_table(rows)


if __name__ == "__main__":
    main()
