"""Tests for ``ml/training/evaluate.py`` output JSON (REQ010, REQ031)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.mv1 import Mv1JunctionModel
from tests.helpers import make_junction_df
from training.evaluate import evaluate_model

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
