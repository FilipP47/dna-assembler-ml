"""Tests for ``ml/data/loader.py``: filters (REQ005), split (REQ009), CSV schema (REQ026)."""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest
import torch

from data.dataset import JunctionDataset, one_hot, reverse_complement
from data.loader import REQUIRED_COLS, load_training_csv, split_dataset
from tests.helpers import make_junction_df

ALL_COLS = [*REQUIRED_COLS, "k", "repeat_len"]


def _base_rows(n: int) -> list[dict[str, str]]:
    """Return ``n`` valid CSV rows as string dicts (as the Rust writer emits them)."""
    return [
        {key: str(value) for key, value in row.items()}
        for row in make_junction_df(n).iter_rows(named=True)
    ]


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> Path:
    """Write ``rows`` verbatim to ``path`` with the given column order."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _branch_bases(row: dict[str, str]) -> list[str]:
    """Bases of the active (non-NA) branches of a row."""
    return [b for b in "ACGT" if row[f"branch_{b}_seq"] != "NA"]


# --- REQ005: loader filters -------------------------------------------------


def test_length_filters_drop_short_context_and_short_branches(tmp_path: Path) -> None:
    """Context < 10 or longest branch < 5 is dropped; exactly 10 and 5 are kept."""
    rows = _base_rows(4)
    rows[0]["context_seq"] = "A" * 9
    rows[1]["context_seq"] = "A" * 10
    first, second = _branch_bases(rows[2])
    rows[2][f"branch_{first}_seq"] = "C" * 4
    rows[2][f"branch_{second}_seq"] = "G" * 3
    first, second = _branch_bases(rows[3])
    rows[3][f"branch_{first}_seq"] = "C" * 5
    rows[3][f"branch_{second}_seq"] = "G"
    path = _write_csv(tmp_path / "t.csv", rows, ALL_COLS)

    df, stats = load_training_csv(path)

    assert sorted(df["junction_id"].to_list()) == [1, 3]
    assert stats.total_rows == 4


def test_custom_length_thresholds(tmp_path: Path) -> None:
    """min_context_len and min_branch_len are parameters, not constants."""
    rows = _base_rows(2)
    rows[0]["context_seq"] = "A" * 30
    path = _write_csv(tmp_path / "t.csv", rows, ALL_COLS)

    df, _ = load_training_csv(path, min_context_len=31)
    assert df["junction_id"].to_list() == [1]

    df, _ = load_training_csv(path, min_branch_len=101)
    assert df.height == 0


def test_only_labeled_drops_ambiguous_and_na(tmp_path: Path) -> None:
    """only_labeled=True drops ground_truth 'ambiguous' and NA; False keeps them."""
    rows = _base_rows(3)
    rows[0]["ground_truth"] = "ambiguous"
    rows[1]["ground_truth"] = "NA"
    path = _write_csv(tmp_path / "t.csv", rows, ALL_COLS)

    labeled, stats = load_training_csv(path, only_labeled=True)
    everything, _ = load_training_csv(path, only_labeled=False)

    assert labeled["junction_id"].to_list() == [2]
    assert sorted(everything["junction_id"].to_list()) == [0, 1, 2]
    assert (stats.ambiguous_rows, stats.not_found_rows, stats.usable_rows) == (1, 1, 1)


def test_no_upper_length_limit_and_truncation_at_encoding(tmp_path: Path) -> None:
    """Long sequences pass the loader; JunctionDataset cuts them to context_len/branch_len."""
    rows = _base_rows(1)
    near_junction = "ACGTTGCAAG" * 5  # last 50 nt of the context
    long_ctx = "C" * 250 + near_junction
    rows[0]["context_seq"] = long_ctx
    gt = rows[0]["ground_truth"]
    long_branch = "G" * 40 + "T" * 210
    rows[0][f"branch_{gt}_seq"] = long_branch
    path = _write_csv(tmp_path / "t.csv", rows, ALL_COLS)

    df, _ = load_training_csv(path)
    assert df["context_seq"][0] == long_ctx

    dataset = JunctionDataset(
        df, context_len=50, branch_len=40, augment=True, use_cov_feat=False
    )
    item = dataset[0]
    assert item["context"].shape == (4, 50)
    assert item["branch_a"].shape == (4, 40)
    assert item["branch_b"].shape == (4, 40)
    # Context keeps its last context_len bases (next to the junction).
    assert torch.equal(item["context"], one_hot(near_junction, 50))
    # The reverse-complement variant is built from the same bases.
    rc_contexts = {s.context_seq for s in dataset.samples[2:]}
    assert rc_contexts == {reverse_complement(near_junction)}
    # Branches still keep their start (the bases right after the junction).
    correct = item["branch_a"] if dataset.samples[0].label == 0 else item["branch_b"]
    assert torch.equal(correct, one_hot(long_branch[:40], 40))


# --- REQ009: train/val split by junction ------------------------------------


@pytest.mark.parametrize("test_split", [0.0, 0.2])
def test_split_is_disjoint_by_junction_id(test_split: float) -> None:
    """Every junction_id lands in exactly one split, before pairs are built."""
    df = make_junction_df(60)

    parts = split_dataset(df, val_split=0.2, test_split=test_split, seed=3)
    id_sets = [set(p["junction_id"].to_list()) for p in parts]

    assert sum(len(s) for s in id_sets) == df.height
    assert set().union(*id_sets) == set(df["junction_id"].to_list())
    for i in range(3):
        for j in range(i + 1, 3):
            assert not id_sets[i] & id_sets[j]

    # Pairs (with augmentation) of one junction stay in that junction's split.
    pair_ids = [
        {
            s.junction_id
            for s in JunctionDataset(p, augment=True, use_cov_feat=False).samples
        }
        for p in parts
    ]
    for ids, part_ids in zip(pair_ids, id_sets):
        assert ids <= part_ids
    assert not pair_ids[0] & pair_ids[1]


# --- REQ026: CSV schema in the loader ---------------------------------------


def test_required_cols_are_the_19_rust_columns() -> None:
    """REQUIRED_COLS is the Rust schema without the optional k and repeat_len."""
    assert len(REQUIRED_COLS) == 19
    assert "k" not in REQUIRED_COLS and "repeat_len" not in REQUIRED_COLS


@pytest.mark.parametrize("missing", REQUIRED_COLS)
def test_missing_required_column_raises(tmp_path: Path, missing: str) -> None:
    """Dropping any required column raises ValueError naming that column."""
    columns = [c for c in ALL_COLS if c != missing]
    path = _write_csv(tmp_path / "t.csv", _base_rows(2), columns)

    with pytest.raises(ValueError, match=missing):
        load_training_csv(path)


def test_k_and_repeat_len_are_optional(tmp_path: Path) -> None:
    """An older CSV without k and repeat_len loads."""
    path = _write_csv(tmp_path / "t.csv", _base_rows(3), REQUIRED_COLS)

    df, stats = load_training_csv(path)

    assert df.height == 3
    assert "k" not in df.columns and "repeat_len" not in df.columns
    assert stats.k_value is None


@pytest.mark.parametrize("marker", ["NA", "na", "NaN", "nan", ""])
def test_null_markers_become_null(tmp_path: Path, marker: str) -> None:
    """NA, na, NaN, nan and empty fields are read as null."""
    rows = _base_rows(1)
    free = next(b for b in "ACGT" if rows[0][f"branch_{b}_seq"] == "NA")
    rows[0][f"branch_{free}_seq"] = marker
    rows[0][f"branch_{free}_cov"] = marker
    rows[0]["node_seq"] = marker
    path = _write_csv(tmp_path / "t.csv", rows, ALL_COLS)

    df, _ = load_training_csv(path)

    assert df[f"branch_{free}_seq"][0] is None
    assert df[f"branch_{free}_cov"][0] is None
    assert df["node_seq"][0] is None
    assert df["n_active_branches"][0] == 2


def test_bad_numbers_are_coerced_without_error(tmp_path: Path) -> None:
    """Bad integers become 0; bad coverages and cov_ratio_max become null."""
    rows = _base_rows(1)
    active = _branch_bases(rows[0])[0]
    rows[0].update(
        {
            "depth_in_contig": "abc",
            "total_reads": "1.5x",
            "k": "?",
            "repeat_len": "long",
            f"branch_{active}_cov": "high",
            "cov_ratio_max": "zz",
        }
    )
    path = _write_csv(tmp_path / "t.csv", rows, ALL_COLS)

    df, _ = load_training_csv(path)

    row = df.row(0, named=True)
    assert (
        row["depth_in_contig"],
        row["total_reads"],
        row["k"],
        row["repeat_len"],
    ) == (
        0,
        0,
        0,
        0,
    )
    assert row[f"branch_{active}_cov"] is None
    assert row["cov_ratio_max"] is None
    assert df.schema["cov_ratio_max"] == pl.Float32
