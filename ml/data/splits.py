"""Train/validation splits for ``main.py train --split_by``.

``row`` is the original stratified row split (:func:`data.loader.split_dataset`).
``node_seq`` and ``genome`` are group splits: every group of rows sharing the
same ``node_seq`` (or the same source CSV, i.e. genome) lands entirely in
train or entirely in val. In *S. cerevisiae* 71% of labeled rows share their
``node_seq`` with another row, so a row split puts near-duplicates on both
sides and overstates val accuracy (docs/RAPORT_transfer_A_B.md, section 7.3).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from data.loader import split_dataset

logger = logging.getLogger(__name__)

SPLIT_MODES = ("row", "node_seq", "genome")
_GROUP_COLUMN = {"node_seq": "node_seq", "genome": "source_file"}


def group_split(
    df: pl.DataFrame, group_col: str, val_split: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split rows so that no value of ``group_col`` is on both sides.

    Groups are visited in a seeded random order and moved to val until val
    holds at least ``val_split`` of the rows, so the val fraction is measured
    in rows, not in groups. Rows with a null group value each form their own
    group. Row order inside each side follows ``df``.

    Args:
        df: Labeled rows (output of ``load_training_csv``).
        group_col: Column that defines the groups.
        val_split: Target fraction of rows in val, in ``(0, 1)``.
        seed: Seed for the group order.

    Returns:
        Tuple ``(df_train, df_val)``.

    Raises:
        ValueError: If ``group_col`` is missing, or there are fewer than two
            groups, or ``val_split`` is not in ``(0, 1)``.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} needed for a group split is missing")
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    row_idx = pl.int_range(pl.len()).cast(pl.String)
    keys = df.select(
        pl.when(pl.col(group_col).is_null())
        .then(pl.lit("__row_") + row_idx)
        .otherwise(pl.col(group_col).cast(pl.String))
        .alias("key")
    )["key"]
    groups = keys.unique().sort().to_list()
    if len(groups) < 2:
        raise ValueError(
            f"split by {group_col!r} needs at least 2 groups, found {len(groups)}"
        )

    sizes = dict(keys.value_counts().iter_rows())
    order = np.random.default_rng(seed).permutation(len(groups))
    target = val_split * df.height
    val_groups: set[str] = set()
    n_val = 0
    for i in order:
        # Keep at least one group in train.
        if n_val >= target or len(val_groups) == len(groups) - 1:
            break
        val_groups.add(groups[i])
        n_val += sizes[groups[i]]

    in_val = keys.is_in(list(val_groups))
    df_train, df_val = df.filter(~in_val), df.filter(in_val)
    logger.info(
        "Group split by %s: train=%d rows | val=%d rows (%d of %d groups, %.1f%%)",
        group_col,
        df_train.height,
        df_val.height,
        len(val_groups),
        len(groups),
        100.0 * df_val.height / max(df.height, 1),
    )
    return df_train, df_val


def split_train_val(
    df: pl.DataFrame, split_by: str, val_split: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split labeled rows into train and val according to ``--split_by``.

    ``row`` calls :func:`data.loader.split_dataset` with the same arguments as
    ``train.py`` always did (stratified, no test set), so it gives exactly the
    original split.

    Args:
        df: Labeled rows (output of ``load_training_csv``).
        split_by: One of :data:`SPLIT_MODES`.
        val_split: Fraction of rows for validation.
        seed: Split seed (``--seed``).

    Returns:
        Tuple ``(df_train, df_val)``.

    Raises:
        ValueError: Unknown ``split_by``, or a group split that cannot be made.
    """
    if split_by == "row":
        df_train, df_val, _ = split_dataset(
            df=df, val_split=val_split, test_split=0.0, seed=seed, stratify=True
        )
        return df_train, df_val
    if split_by not in _GROUP_COLUMN:
        raise ValueError(f"split_by must be one of {SPLIT_MODES}, got {split_by!r}")
    return group_split(df, _GROUP_COLUMN[split_by], val_split, seed)
