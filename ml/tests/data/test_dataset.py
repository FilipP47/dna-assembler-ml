"""Tests for ``ml/data/dataset.py`` pair construction."""

from __future__ import annotations

import random

import polars as pl

from data.dataset import JunctionDataset

BASES = ["A", "C", "G", "T"]


def _make_df(n_junctions: int, seed: int = 0) -> pl.DataFrame:
    """Build a synthetic one-row-per-junction DataFrame with two branches each.

    Args:
        n_junctions: Number of junction rows to generate.
        seed: Seed for the random sequence generator.

    Returns:
        Polars DataFrame with the columns ``JunctionDataset`` reads when
        ``use_cov_feat=False``.
    """
    rng = random.Random(seed)
    rows = []
    for jid in range(n_junctions):
        gt = rng.choice(BASES)
        other = rng.choice([b for b in BASES if b != gt])
        row = {
            "junction_id": jid,
            "ground_truth": gt,
            "context_seq": "".join(rng.choice(BASES) for _ in range(100)),
        }
        for base in BASES:
            if base in (gt, other):
                row[f"branch_{base}_seq"] = "".join(
                    rng.choice(BASES) for _ in range(100)
                )
            else:
                row[f"branch_{base}_seq"] = "NA"
        rows.append(row)
    return pl.DataFrame(rows)


def _labels(ds: JunctionDataset) -> list[int]:
    """Return the label of every sample in the dataset."""
    return [s.label for s in ds.samples]


def test_augment_false_label_distribution_is_balanced() -> None:
    """augment=False must not degenerate to a single label value.

    Regression test: previously the correct branch always landed on
    ``branch_a`` (label 0) unless ``augment=True``, so any val/test set built
    with ``augment=False`` was 100% label 0.
    """
    df = _make_df(500)
    ds = JunctionDataset(df, augment=False, use_cov_feat=False, pair_seed=0)
    labels = _labels(ds)

    assert len(labels) == 500
    ones = sum(labels)
    frac = ones / len(labels)
    assert 0 < ones < len(labels), "label distribution degenerated to one value"
    assert 0.4 <= frac <= 0.6, f"label distribution not ~50/50: {frac:.3f}"


def test_pair_position_assignment_is_deterministic() -> None:
    """Same (pair_seed, junction_id) must yield the same position every build."""
    df = _make_df(300)
    a = _labels(JunctionDataset(df, augment=False, use_cov_feat=False, pair_seed=7))
    b = _labels(JunctionDataset(df, augment=False, use_cov_feat=False, pair_seed=7))
    assert a == b

    # A different seed should shuffle at least some positions.
    c = _labels(JunctionDataset(df, augment=False, use_cov_feat=False, pair_seed=8))
    assert a != c


def test_augment_true_is_exactly_balanced_and_full_coverage() -> None:
    """augment=True keeps the 4-variant structure and exact 50/50 labels."""
    df = _make_df(200)
    ds = JunctionDataset(df, augment=True, use_cov_feat=False, pair_seed=3)
    labels = _labels(ds)

    assert len(labels) == 4 * 200
    assert sum(labels) == len(labels) // 2

    # Each junction contributes 2 label=0 and 2 label=1 samples.
    per_junction: dict[int, list[int]] = {}
    for s in ds.samples:
        per_junction.setdefault(s.junction_id, []).append(s.label)
    for jid, ls in per_junction.items():
        assert sorted(ls) == [0, 0, 1, 1], f"junction {jid} not fully covered: {ls}"
