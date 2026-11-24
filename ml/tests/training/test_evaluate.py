"""Tests for ``ml/training/evaluate.py`` output JSON and predictions (REQ010, REQ031, REQ035)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch

from models.mv1 import Mv1JunctionModel
from tests.helpers import make_junction_df
from training.evaluate import evaluate_model
from training.predictions import (
    PREDICTION_COLUMNS,
    baseline_branch,
    coverage_tie,
    label_type,
)

EXPECTED_KEYS = {
    "checkpoint",
    "git_commit",
    "eval_csv",
    "organism",
    "junction_accuracy",
    "pair_accuracy",
    "f1",
    "precision",
    "recall",
    "loss",
    "n_junctions",
    "n_pairs",
}

_HPARAMS = {
    "context_embed_dim": 8,
    "branch_embed_dim": 8,
    "cov_feat_dim": 0,
    "hidden_dim": 8,
}


def _write_checkpoint(path: Path, extra: dict) -> None:
    """Save a small untrained model checkpoint with the given extra keys."""
    torch.manual_seed(0)
    model = Mv1JunctionModel(**_HPARAMS)
    model.save_checkpoint(path, extra={"epoch": 1, "model_hparams": _HPARAMS, **extra})


def _evaluate(tmp_path: Path, ckpt: Path) -> dict:
    """Run evaluate_model on a synthetic CSV and return the written JSON."""
    csv_path = tmp_path / "toy_training.csv"
    make_junction_df(12).write_csv(csv_path)
    out_json = tmp_path / "out" / "toy.json"
    args = argparse.Namespace(
        model_path=str(ckpt),
        test_csv=str(csv_path),
        output_json=str(out_json),
        device="cpu",
    )
    returned = evaluate_model(args)
    written = json.loads(out_json.read_text())
    assert written == returned
    return written


def test_output_json_keys_with_new_checkpoint(tmp_path: Path) -> None:
    """JSON has the REQ010 keys and the checkpoint's git_commit."""
    ckpt = tmp_path / "new.pt"
    _write_checkpoint(
        ckpt, {"git_commit": "abc123", "git_dirty": False, "config": {"seed": 42}}
    )

    result = _evaluate(tmp_path, ckpt)

    assert set(result) == EXPECTED_KEYS
    assert result["git_commit"] == "abc123"
    assert result["organism"] == "toy"
    assert result["n_junctions"] == 12
    assert result["n_pairs"] == 12


def test_old_checkpoint_without_provenance_still_loads(tmp_path: Path) -> None:
    """A checkpoint saved before REQ031 evaluates fine; git_commit is null."""
    ckpt = tmp_path / "old.pt"
    _write_checkpoint(ckpt, {"val_junction_accuracy": 0.5, "val_f1": 0.5})

    result = _evaluate(tmp_path, ckpt)

    assert set(result) == EXPECTED_KEYS
    assert result["git_commit"] is None


# --- REQ035: per-junction predictions ----------------------------------------


def _mixed_csv(tmp_path: Path) -> Path:
    """12 junctions: 8 Known, 2 Ambiguous, 2 NotFound."""
    df = make_junction_df(12)
    gt = df["ground_truth"].to_list()
    gt[8:10] = ["ambiguous", "ambiguous"]
    gt[10:12] = ["NA", "NA"]
    df = df.with_columns(pl.Series("ground_truth", gt))
    path = tmp_path / "mixed_training.csv"
    df.write_csv(path)
    return path


def _evaluate_mixed(tmp_path: Path, **extra: object) -> tuple[dict, Path]:
    """Evaluate a small checkpoint on the mixed CSV, writing the JSON."""
    ckpt = tmp_path / "m.pt"
    _write_checkpoint(ckpt, {})
    out_json = tmp_path / "out" / "mixed.json"
    args = argparse.Namespace(
        model_path=str(ckpt),
        test_csv=str(_mixed_csv(tmp_path)),
        output_json=str(out_json),
        device="cpu",
        **extra,
    )
    return evaluate_model(args), out_json


def test_predictions_written_next_to_json(tmp_path: Path) -> None:
    """Default file is <json stem>_predictions.csv; the JSON keys are unchanged."""
    result, out_json = _evaluate_mixed(tmp_path)

    assert set(json.loads(out_json.read_text())) == EXPECTED_KEYS
    pred = pl.read_csv(tmp_path / "out" / "mixed_predictions.csv")
    assert pred.columns == PREDICTION_COLUMNS
    assert pred.height == 12
    assert (
        pred["label_type"].to_list()
        == ["Known"] * 8 + ["Ambiguous"] * 2 + ["NotFound"] * 2
    )
    assert result["n_junctions"] == 8


def test_predictions_match_json_accuracy(tmp_path: Path) -> None:
    """Mean of `correct` over Known junctions equals junction_accuracy in the JSON."""
    result, _ = _evaluate_mixed(tmp_path)
    pred = pl.read_csv(tmp_path / "out" / "mixed_predictions.csv")

    known = pred.filter(pl.col("label_type") == "Known")
    assert known["correct"].null_count() == 0
    assert round(known["correct"].mean(), 6) == result["junction_accuracy"]
    # With two branches the argmax of the branch scores is the pair decision.
    assert (
        (known["predicted_branch"] == known["ground_truth"]).cast(int)
        == known["correct"]
    ).all()

    unlabeled = pred.filter(pl.col("label_type") != "Known")
    assert unlabeled["correct"].null_count() == 4
    assert unlabeled["baseline_correct"].null_count() == 4
    assert unlabeled["predicted_branch"].null_count() == 0


def test_explicit_predictions_path(tmp_path: Path) -> None:
    """--output_predictions overrides the default location."""
    target = tmp_path / "elsewhere" / "p.csv"
    _evaluate_mixed(tmp_path, output_predictions=str(target))
    assert target.exists()
    assert not (tmp_path / "out" / "mixed_predictions.csv").exists()


def test_baseline_tie_break_and_label_type() -> None:
    """Baseline picks the max coverage, ties in A, C, G, T order."""
    row = {f"branch_{b}_cov": None for b in "ACGT"}
    row.update({"branch_C_cov": 7.0, "branch_T_cov": 7.0, "branch_G_cov": 3.0})
    assert baseline_branch(row) == "C"
    assert coverage_tie(row) is True
    row["branch_T_cov"] = 8.0
    assert baseline_branch(row) == "T"
    assert coverage_tie(row) is False
    assert [label_type(v) for v in ("G", "ambiguous", None)] == [
        "Known",
        "Ambiguous",
        "NotFound",
    ]
