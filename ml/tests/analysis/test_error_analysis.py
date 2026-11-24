"""Tests for ``ml/analysis/error_analysis.py`` (REQ036)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from analysis.error_analysis import (
    TABLE_COLUMNS,
    accuracy_table,
    load_predictions,
    main,
)


def _predictions(organism: str, rows: list[tuple]) -> pl.DataFrame:
    """Build a predictions frame from (label_type, n_branches, repeat_len,
    cov_ratio_max, cov_tie, correct, baseline_correct) tuples."""
    return pl.DataFrame(
        [
            {
                "organism": organism,
                "junction_id": i,
                "label_type": r[0],
                "ground_truth": {"Known": "A", "Ambiguous": "ambiguous"}.get(r[0]),
                "n_branches": r[1],
                "repeat_len": r[2],
                "cov_ratio_max": r[3],
                "cov_tie": r[4],
                "correct": r[5],
                "baseline_correct": r[6],
            }
            for i, r in enumerate(rows)
        ],
        schema_overrides={"correct": pl.Int64, "baseline_correct": pl.Int64},
    )


ORG1 = [
    ("Known", 2, 5, 0.5, True, 1, 0),
    ("Known", 2, 30, 0.9, False, 0, 1),
    ("Known", 3, 10, 0.55, False, 1, 1),
    ("Ambiguous", 2, 40, 0.5, True, None, None),
]
ORG2 = [
    ("Known", 2, 21, 0.7, False, 1, 0),
    ("NotFound", 4, 2, 0.3, False, None, None),
]


def _write(tmp_path: Path) -> Path:
    """Write the two organisms as evaluate would and return the directory."""
    out = tmp_path / "eval"
    out.mkdir()
    _predictions("org1", ORG1).write_csv(out / "org1_predictions.csv")
    _predictions("org2", ORG2).write_csv(out / "org2_predictions.csv")
    (out / "org1.json").write_text("{}")
    return out


def _cell(table: pl.DataFrame, dimension: str, bucket: str) -> dict:
    """Return one row of the table as a dict."""
    rows = table.filter(
        (pl.col("dimension") == dimension) & (pl.col("bucket") == bucket)
    )
    assert rows.height == 1, (dimension, bucket)
    return rows.row(0, named=True)


def test_accuracy_table_buckets(tmp_path: Path) -> None:
    """Pooled accuracy per label type, branch count, repeat length and coverage."""
    table = accuracy_table(load_predictions([_write(tmp_path)]))
    assert table.columns == TABLE_COLUMNS

    overall = _cell(table, "overall", "all")
    assert (overall["n_junctions"], overall["n_evaluated"]) == (6, 4)
    assert overall["model_acc"] == 0.75
    assert overall["baseline_acc"] == 0.5
    assert overall["delta_pp"] == 25.0

    assert _cell(table, "label_type", "Known")["n_evaluated"] == 4
    ambiguous = _cell(table, "label_type", "Ambiguous")
    assert (ambiguous["n_junctions"], ambiguous["n_evaluated"]) == (1, 0)
    assert ambiguous["model_acc"] is None
    assert _cell(table, "label_type", "NotFound")["n_junctions"] == 1

    two = _cell(table, "n_branches", "2")
    assert (two["n_evaluated"], two["model_acc"]) == (3, pytest.approx(2 / 3, abs=1e-6))
    assert _cell(table, "n_branches", "3")["model_acc"] == 1.0

    short = _cell(table, "repeat_len", "<21")
    long = _cell(table, "repeat_len", ">=21")
    assert (short["n_evaluated"], short["model_acc"]) == (2, 1.0)
    assert (long["n_evaluated"], long["model_acc"], long["baseline_acc"]) == (
        2,
        0.5,
        0.5,
    )

    assert _cell(table, "cov_tie", "true")["n_junctions"] == 2
    assert _cell(table, "cov_ratio_max", "0.50-0.60")["n_evaluated"] == 2
    assert _cell(table, "cov_ratio_max", "0.75-1.00")["model_acc"] == 0.0
    assert _cell(table, "organism", "org2")["model_acc"] == 1.0


def test_repeat_threshold_is_configurable(tmp_path: Path) -> None:
    """--repeat_threshold moves the repeat_len boundary."""
    table = accuracy_table(load_predictions([_write(tmp_path)]), repeat_threshold=25)
    assert _cell(table, "repeat_len", "<25")["n_evaluated"] == 3


def test_main_writes_csv_and_summary(tmp_path: Path) -> None:
    """The CLI writes the table and a text summary next to it."""
    out_csv = tmp_path / "report" / "ea.csv"
    main([str(_write(tmp_path)), "--out_csv", str(out_csv)])

    assert pl.read_csv(out_csv).columns == TABLE_COLUMNS
    summary = (tmp_path / "report" / "ea_summary.txt").read_text()
    assert "[repeat_len]" in summary
    assert "model= 75.00%" in summary
    assert "+25.00 pp" in summary


def test_missing_input_is_an_error(tmp_path: Path) -> None:
    """An empty directory gives a clear error."""
    with pytest.raises(FileNotFoundError, match="_predictions.csv"):
        load_predictions([tmp_path])
