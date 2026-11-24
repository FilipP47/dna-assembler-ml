"""Accuracy breakdown of zero-shot predictions.

Reads the per-junction predictions written by ``main.py evaluate``
(``<organism>_predictions.csv``, see :mod:`training.predictions`) and reports
model and max-coverage baseline accuracy per:

- ``label_type``: Known / Ambiguous / NotFound (accuracy only for Known;
  the other two have no ground truth and are reported as counts),
- ``n_branches``: number of branches at the junction,
- ``repeat_len``: ``< threshold`` and ``>= threshold`` (default 21),
- ``cov_tie``: the two highest branch coverages are equal or not,
- ``cov_ratio_max``: share of the dominant branch in total coverage, binned,
- ``organism``: per source genome.

Junctions of all organisms are pooled, so ``overall`` equals the mean of the
per-genome ``junction_accuracy`` weighted by ``n_junctions``. Accuracies are
computed on the junctions that have ``correct`` set (the ones counted in the
evaluation JSON); the baseline is scored on the same junctions.

Usage::

    cd ml
    uv run python3 analysis/error_analysis.py ../results/eval_group_a_farthest \\
        --out_csv ../results/error_analysis_group_a_farthest.csv

Writes the table to ``--out_csv`` and a short text summary to
``<out_csv without .csv>_summary.txt`` (also logged).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

PREDICTIONS_GLOB = "*_predictions.csv"
LABEL_TYPES = ["Known", "Ambiguous", "NotFound"]
COV_RATIO_BREAKS = [0.5, 0.6, 0.75]
COV_RATIO_LABELS = ["0.00-0.50", "0.50-0.60", "0.60-0.75", "0.75-1.00"]
TABLE_COLUMNS = [
    "dimension",
    "bucket",
    "n_junctions",
    "n_evaluated",
    "model_correct",
    "model_acc",
    "baseline_acc",
    "delta_pp",
]


def find_prediction_files(paths: list[Path]) -> list[Path]:
    """Expand directories to their ``*_predictions.csv`` files.

    Args:
        paths: Prediction CSV files and/or directories that contain them.

    Returns:
        Sorted list of prediction files.

    Raises:
        FileNotFoundError: If a path does not exist or nothing is found.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob(PREDICTIONS_GLOB)))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    if not files:
        raise FileNotFoundError(f"No {PREDICTIONS_GLOB} files in {paths}")
    return files


def load_predictions(paths: list[Path]) -> pl.DataFrame:
    """Read and concatenate prediction CSVs.

    Args:
        paths: Prediction CSV files and/or directories.

    Returns:
        One DataFrame with the rows of all files.
    """
    files = find_prediction_files(paths)
    logger.info("Reading %d prediction file(s)", len(files))
    frames = [
        pl.read_csv(
            f,
            schema_overrides={"ground_truth": pl.String, "organism": pl.String},
        )
        for f in files
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def _bucket_expressions(repeat_threshold: int) -> dict[str, pl.Expr]:
    """Return one bucket expression per dimension."""
    repeat = (
        pl.when(pl.col("repeat_len").is_null())
        .then(pl.lit("missing"))
        .when(pl.col("repeat_len") < repeat_threshold)
        .then(pl.lit(f"<{repeat_threshold}"))
        .otherwise(pl.lit(f">={repeat_threshold}"))
    )
    cov_ratio = (
        pl.col("cov_ratio_max")
        .cut(COV_RATIO_BREAKS, labels=COV_RATIO_LABELS, left_closed=True)
        .cast(pl.String)
        .fill_null("missing")
    )
    return {
        "overall": pl.lit("all"),
        "label_type": pl.col("label_type"),
        "n_branches": pl.col("n_branches").cast(pl.String).fill_null("missing"),
        "repeat_len": repeat,
        "cov_tie": pl.col("cov_tie").cast(pl.String).fill_null("missing"),
        "cov_ratio_max": cov_ratio,
        "organism": pl.col("organism"),
    }


def accuracy_table(df: pl.DataFrame, repeat_threshold: int = 21) -> pl.DataFrame:
    """Compute model and baseline accuracy per dimension and bucket.

    Args:
        df: Predictions (output of :func:`load_predictions`).
        repeat_threshold: Boundary for the ``repeat_len`` buckets.

    Returns:
        DataFrame with :data:`TABLE_COLUMNS`. ``n_junctions`` counts all
        rows in the bucket, ``n_evaluated`` those with ``correct`` set;
        accuracies are fractions over ``n_evaluated`` (null when 0) and
        ``delta_pp`` is ``(model_acc - baseline_acc) * 100``.
    """
    evaluated = pl.col("correct").is_not_null()
    parts = []
    for dimension, bucket in _bucket_expressions(repeat_threshold).items():
        part = (
            df.with_columns(bucket.alias("bucket"))
            .group_by("bucket")
            .agg(
                pl.len().alias("n_junctions"),
                evaluated.sum().alias("n_evaluated"),
                pl.col("correct").sum().alias("model_correct"),
                pl.col("correct").mean().alias("model_acc"),
                pl.col("baseline_correct")
                .filter(evaluated)
                .mean()
                .alias("baseline_acc"),
            )
            .with_columns(pl.lit(dimension).alias("dimension"))
            .sort("bucket")
        )
        parts.append(part)
    table = pl.concat(parts, how="vertical_relaxed").with_columns(
        pl.col("n_junctions").cast(pl.Int64),
        pl.col("n_evaluated").cast(pl.Int64),
        pl.col("model_correct").cast(pl.Int64),
        ((pl.col("model_acc") - pl.col("baseline_acc")) * 100).alias("delta_pp"),
    )
    return table.select(TABLE_COLUMNS).with_columns(
        pl.col("model_acc", "baseline_acc", "delta_pp").round(6)
    )


def _pct(value: float | None) -> str:
    """Format a fraction as a percentage with two decimals."""
    return "n/a" if value is None else f"{value * 100:.2f}%"


def format_summary(table: pl.DataFrame) -> str:
    """Render the accuracy table as short plain-text lines.

    Args:
        table: Output of :func:`accuracy_table`.

    Returns:
        Multi-line summary, one block per dimension.
    """
    lines: list[str] = []
    for dimension in table["dimension"].unique(maintain_order=True).to_list():
        lines.append(f"[{dimension}]")
        for row in table.filter(pl.col("dimension") == dimension).iter_rows(named=True):
            delta = "n/a" if row["delta_pp"] is None else f"{row['delta_pp']:+.2f} pp"
            lines.append(
                f"  {row['bucket']:<12} n={row['n_junctions']:<6} "
                f"evaluated={row['n_evaluated']:<6} model={_pct(row['model_acc']):>7} "
                f"baseline={_pct(row['baseline_acc']):>7} delta={delta}"
            )
    lines.append(
        "Accuracy is pooled over all junctions (weighted by n_junctions); "
        "Ambiguous and NotFound junctions have no ground truth (evaluated=0)."
    )
    return "\n".join(lines)


def run(paths: list[Path], out_csv: Path, repeat_threshold: int = 21) -> pl.DataFrame:
    """Load predictions, write the table and the text summary.

    Args:
        paths: Prediction CSV files and/or directories.
        out_csv: Destination of the table.
        repeat_threshold: Boundary for the ``repeat_len`` buckets.

    Returns:
        The accuracy table.
    """
    table = accuracy_table(load_predictions(paths), repeat_threshold)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_csv)
    summary = format_summary(table)
    summary_path = out_csv.with_name(f"{out_csv.stem}_summary.txt")
    summary_path.write_text(summary + "\n")
    logger.info("Error analysis table: %s", out_csv)
    logger.info("Summary (%s):\n%s", summary_path, summary)
    return table


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser for :func:`main`.
    """
    parser = argparse.ArgumentParser(
        description="Accuracy breakdown of per-junction zero-shot predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=f"Prediction CSVs or directories with {PREDICTIONS_GLOB}",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output table (default: <first path's directory>/error_analysis.csv)",
    )
    parser.add_argument(
        "--repeat_threshold",
        type=int,
        default=21,
        help="repeat_len bucket boundary (< and >=)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name (``sys.argv`` when ``None``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    out_csv = args.out_csv
    if out_csv is None:
        first = args.paths[0]
        out_csv = (first if first.is_dir() else first.parent) / "error_analysis.csv"
    run(args.paths, out_csv, args.repeat_threshold)


if __name__ == "__main__":
    main()
