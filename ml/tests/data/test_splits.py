"""Tests for ``ml/data/splits.py`` (--split_by, REQ034)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from data.loader import load_training_csv, split_dataset
from data.splits import group_split, split_train_val
from tests.helpers import make_junction_df


def _loaded(tmp_path: Path, n: int = 60) -> pl.DataFrame:
    """Synthetic junctions where groups of 3 rows share node_seq, as loaded by train.py."""
    df = make_junction_df(n).with_columns(
        (pl.lit("NODE") + (pl.col("junction_id") // 3).cast(pl.String)).alias(
            "node_seq"
        )
    )
    path = tmp_path / "toy_training.csv"
    df.write_csv(path)
    return load_training_csv(path)[0]


def _genome_dir(tmp_path: Path, n_genomes: int = 5) -> Path:
    """A directory with one CSV per genome, of different sizes."""
    data_dir = tmp_path / "genomes"
    data_dir.mkdir()
    for i in range(n_genomes):
        make_junction_df(10 + 5 * i, seed=i).write_csv(data_dir / f"g{i}_training.csv")
    return data_dir


def test_row_split_is_the_original_split_dataset(tmp_path: Path) -> None:
    """--split_by row returns exactly what train.py got from split_dataset before."""
    df = _loaded(tmp_path)
    expected_train, expected_val, _ = split_dataset(
        df=df, val_split=0.15, test_split=0.0, seed=42, stratify=True
    )
    train, val = split_train_val(df, "row", 0.15, 42)
    assert train.equals(expected_train)
    assert val.equals(expected_val)


def test_node_seq_split_keeps_groups_together(tmp_path: Path) -> None:
    """No node_seq appears in both train and val; every row is used once."""
    df = _loaded(tmp_path)
    train, val = split_train_val(df, "node_seq", 0.2, 42)

    assert set(train["node_seq"]).isdisjoint(set(val["node_seq"]))
    assert train.height + val.height == df.height
    assert val.height >= 0.2 * df.height
    assert sorted(
        train["junction_id"].to_list() + val["junction_id"].to_list()
    ) == sorted(df["junction_id"].to_list())


def test_node_seq_split_is_seeded(tmp_path: Path) -> None:
    """Same seed, same split; another seed gives another split."""
    df = _loaded(tmp_path)
    _, val_a = split_train_val(df, "node_seq", 0.2, 1)
    _, val_b = split_train_val(df, "node_seq", 0.2, 1)
    _, val_c = split_train_val(df, "node_seq", 0.2, 2)
    assert val_a.equals(val_b)
    assert not val_a.equals(val_c)


def test_loader_adds_source_file(tmp_path: Path) -> None:
    """Each row records the CSV it came from, for a file and for a directory."""
    df, _ = load_training_csv(_genome_dir(tmp_path))
    assert set(df["source_file"]) == {f"g{i}_training.csv" for i in range(5)}

    single = tmp_path / "one_training.csv"
    make_junction_df(4).write_csv(single)
    assert set(load_training_csv(single)[0]["source_file"]) == {"one_training.csv"}


def test_genome_split_keeps_genomes_together(tmp_path: Path) -> None:
    """Every genome is entirely in train or entirely in val."""
    df, _ = load_training_csv(_genome_dir(tmp_path))
    train, val = split_train_val(df, "genome", 0.15, 42)

    assert val.height > 0 and train.height > 0
    assert set(train["source_file"]).isdisjoint(set(val["source_file"]))
    assert train.height + val.height == df.height


def test_genome_split_needs_two_genomes(tmp_path: Path) -> None:
    """A single CSV cannot be split by genome."""
    with pytest.raises(ValueError, match="at least 2 groups"):
        split_train_val(_loaded(tmp_path), "genome", 0.15, 42)


def test_group_split_keeps_one_group_in_train(tmp_path: Path) -> None:
    """Even a large val_split leaves at least one group for training."""
    df, _ = load_training_csv(_genome_dir(tmp_path, n_genomes=2))
    train, val = group_split(df, "source_file", 0.95, 0)
    assert train.height > 0 and val.height > 0


def test_group_split_null_values_are_separate_groups() -> None:
    """Rows with a null group value do not form one big group."""
    df = pl.DataFrame({"g": [None] * 10, "x": list(range(10))})
    train, val = group_split(df, "g", 0.3, 0)
    assert val.height == 3 and train.height == 7


def test_unknown_split_mode_is_rejected(tmp_path: Path) -> None:
    """Only row, node_seq and genome are accepted."""
    with pytest.raises(ValueError, match="split_by"):
        split_train_val(_loaded(tmp_path), "contig", 0.15, 42)
