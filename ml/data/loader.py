"""CSV loading, validation, and train/val/test splitting for junction training data.

Uses Polars for fast columnar I/O and scikit-learn for stratified splitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

BASES = ["A", "C", "G", "T"]

REQUIRED_COLS = [
    "junction_id", "contig_id", "depth_in_contig",
    "out_degree", "in_degree",
    "node_seq", "context_seq",
    "branch_A_seq", "branch_A_cov",
    "branch_C_seq", "branch_C_cov",
    "branch_G_seq", "branch_G_cov",
    "branch_T_seq", "branch_T_cov",
    "ground_truth", "gt_ambiguous",
    "cov_ratio_max", "total_reads",
]


@dataclass
class DatasetStats:
    """Summary statistics computed after loading and filtering a training CSV.

    Attributes:
        total_rows: Number of rows before any filtering.
        usable_rows: Rows with an unambiguous ground-truth label.
        ambiguous_rows: Rows whose ``ground_truth`` is ``"ambiguous"``.
        not_found_rows: Rows with a null ``ground_truth``.
        label_distribution: Count of each ground-truth base label.
        avg_context_len: Mean context sequence length in base pairs.
        avg_branch_len: Mean maximum branch sequence length in base pairs.
        avg_out_degree: Mean junction out-degree.
        coverage_stats: Min/mean/max per-branch read coverage.
    """
    total_rows:         int
    usable_rows:        int
    ambiguous_rows:     int
    not_found_rows:     int
    label_distribution: Dict[str, int]
    avg_context_len:    float
    avg_branch_len:     float
    avg_out_degree:     float
    coverage_stats:     Dict[str, float]

    def print(self) -> None:
        """Print a human-readable summary of the dataset statistics."""
        print("=" * 55)
        print("Dataset statistics (Polars Engine):")
        print(f"  Total rows:          {self.total_rows}")
        print(f"  Usable (labeled):    {self.usable_rows}  "
              f"({self.usable_rows/max(self.total_rows,1)*100:.1f}%)")
        print(f"  Ambiguous:           {self.ambiguous_rows}")
        print(f"  Not in reference:    {self.not_found_rows}")
        print(f"  Label distribution:  {self.label_distribution}")
        print(f"  Avg context length:  {self.avg_context_len:.1f} bp")
        print(f"  Avg branch length:   {self.avg_branch_len:.1f} bp")
        print(f"  Avg out-degree:      {self.avg_out_degree:.2f}")
        print(f"  Coverage (min/mean/max): "
              f"{self.coverage_stats['min']:.0f} / "
              f"{self.coverage_stats['mean']:.0f} / "
              f"{self.coverage_stats['max']:.0f}")
        print("=" * 55)


def _read_raw_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=0, null_values=["NA", "na", "NaN", "nan", ""])


def load_training_csv(
    csv_path: Path,
    min_context_len: int = 10,
    min_branch_len:  int = 5,
    only_labeled:    bool = True,
) -> Tuple[pl.DataFrame, DatasetStats]:
    """Load training data produced by the Rust pipeline, validate it, and compute statistics.

    Accepts either a single CSV file or a directory of CSV files.  When a directory
    is given, all ``*.csv`` files inside are concatenated and ``junction_id`` values
    are offset per file so IDs remain globally unique across genomes.

    Bioinformatics null markers (``NA``, ``NaN``, empty strings) are mapped to
    native Polars nulls.  Numeric columns are cast with ``strict=False`` so
    malformed values become null rather than raising an error.

    Args:
        csv_path: Path to a single training CSV file **or** a directory that
            contains one or more CSV files (e.g. one per reference genome).
        min_context_len: Minimum context sequence length in base pairs;
            shorter junctions are discarded.
        min_branch_len: Minimum branch sequence length in base pairs;
            junctions where all branches are shorter are discarded.
        only_labeled: When ``True``, return only rows with an unambiguous
            ground-truth label (excludes ``"ambiguous"`` and null rows).

    Returns:
        A tuple of ``(df, stats)`` where ``df`` is the filtered Polars
        DataFrame and ``stats`` is a :class:`DatasetStats` instance.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist or (for a directory)
            contains no CSV files.
        ValueError: If any required column is missing from the CSV.
    """
    csv_path = Path(csv_path)

    if csv_path.is_dir():
        csv_files = sorted(csv_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")
        logger.info("Loading %d CSV files from %s", len(csv_files), csv_path)
        parts = []
        for i, f in enumerate(csv_files):
            part = _read_raw_csv(f)
            # Offset junction_id per file so IDs stay unique across genomes.
            part = part.with_columns(pl.lit(i).alias("_source_file_idx"))
            parts.append(part)
            logger.info("  [%d/%d] %s — %d rows", i + 1, len(csv_files), f.name, part.height)
        df = pl.concat(parts, how="diagonal_relaxed")
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {csv_path}")
        logger.info("Loading training data from %s", csv_path)
        df = _read_raw_csv(csv_path)
        df = df.with_columns(pl.lit(0).alias("_source_file_idx"))

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    total_rows = df.height
    logger.info("Loaded %d rows total", total_rows)

    int_cols = ["junction_id", "contig_id", "depth_in_contig", "out_degree", "in_degree", "total_reads"]
    float_cols = ["cov_ratio_max"]
    cov_cols = [f"branch_{b}_cov" for b in BASES]

    # strict=False mirrors pandas errors='coerce': invalid entries become null
    df = df.with_columns([
        pl.col(col).cast(pl.Int64, strict=False).fill_null(0) for col in int_cols
    ] + [
        pl.col(col).cast(pl.Float32, strict=False) for col in float_cols + cov_cols
    ] + [
        (pl.col("gt_ambiguous").str.to_lowercase() == "true").alias("gt_ambiguous")
    ])

    # Ensure junction_ids are globally unique when multiple files were loaded.
    df = df.with_columns(
        (pl.col("junction_id") + pl.col("_source_file_idx") * 1_000_000).alias("junction_id")
    ).drop("_source_file_idx")

    df = df.with_columns(
        pl.col("context_seq").fill_null("").str.len_chars().alias("context_len")
    )

    for b in BASES:
        df = df.with_columns(
            pl.col(f"branch_{b}_seq").fill_null("").str.len_chars().alias(f"branch_{b}_len")
        )

    df = df.with_columns([
        pl.max_horizontal([f"branch_{b}_len" for b in BASES]).alias("max_branch_len"),
        pl.sum_horizontal([pl.col(f"branch_{b}_seq").is_not_null() for b in BASES]).alias("n_active_branches")
    ])

    df = df.filter(
        (pl.col("context_len") >= min_context_len) &
        (pl.col("max_branch_len") >= min_branch_len)
    )
    logger.info(f"After length filtering: {df.height} rows")

    ambiguous_rows = df.filter(pl.col("ground_truth") == "ambiguous").height
    not_found_rows = df.filter(pl.col("ground_truth").is_null()).height

    usable_df = df.filter(
        pl.col("ground_truth").is_not_null() & (pl.col("ground_truth") != "ambiguous")
    )
    usable_rows = usable_df.height

    label_counts = usable_df["ground_truth"].value_counts()
    label_dist = {row["ground_truth"]: row["count"] for row in label_counts.iter_rows(named=True)}

    all_covs = df.select(cov_cols).melt().select("value").drop_nulls()
    if all_covs.height > 0:
        cov_stats = {
            "min":  float(all_covs["value"].min()),
            "mean": float(all_covs["value"].mean()),
            "max":  float(all_covs["value"].max()),
        }
    else:
        cov_stats = {"min": 0.0, "mean": 0.0, "max": 0.0}

    stats = DatasetStats(
        total_rows=total_rows,
        usable_rows=usable_rows,
        ambiguous_rows=ambiguous_rows,
        not_found_rows=not_found_rows,
        label_distribution=label_dist,
        avg_context_len=float(df["context_len"].mean() or 0.0),
        avg_branch_len=float(df["max_branch_len"].mean() or 0.0),
        avg_out_degree=float(df["out_degree"].mean() or 0.0),
        coverage_stats=cov_stats,
    )

    if only_labeled:
        df = usable_df
        logger.info(f"After label filtering: {df.height} usable rows")

    return df, stats


def split_dataset(
    df: pl.DataFrame,
    val_split:  float = 0.15,
    test_split: float = 0.15,
    seed:       int   = 42,
    stratify:   bool  = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a Polars DataFrame into train, validation, and test subsets.

    Uses scikit-learn's ``train_test_split`` for reproducible stratified
    index sampling, then selects rows with ``df.gather``.

    Args:
        df: Source Polars DataFrame (all rows, post-filtering).
        val_split: Fraction of the full dataset reserved for validation.
        test_split: Fraction of the full dataset reserved for testing.
            Pass ``0.0`` to skip the test split.
        seed: Random seed for reproducibility.
        stratify: When ``True``, stratify splits by ``ground_truth`` label.

    Returns:
        A tuple ``(df_train, df_val, df_test)``.  ``df_test`` is empty
        when ``test_split == 0``.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(df.height)
    stratify_array = df["ground_truth"].fill_null("MISSING").to_numpy() if stratify else None

    if test_split > 0.0:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_split,
            random_state=seed,
            stratify=stratify_array,
        )
    else:
        train_val_idx = indices
        test_idx = np.array([], dtype=int)

    val_fraction = val_split / (1.0 - test_split) if test_split < 1.0 else 0.0
    stratify_tv = df.gather(train_val_idx)["ground_truth"].fill_null("MISSING").to_numpy() if stratify else None

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_tv,
    )

    df_train = df.gather(train_idx)
    df_val   = df.gather(val_idx)
    df_test  = df.gather(test_idx)

    logger.info(
        f"Split: train={df_train.height} | val={df_val.height} | test={df_test.height}"
    )
    return df_train, df_val, df_test
