"""Tests for ``ml/training/train.py``: model input with --no_cov (REQ008), seeds (REQ028)."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

import training.train as train_module
from data.dataset import JunctionDataset, collate_junction_batch
from data.loader import load_training_csv
from main import build_parser
from models.mv1 import Mv1JunctionModel
from tests.helpers import make_junction_df
from training.train import train_model


def _write_toy_csv(tmp_path: Path, n_junctions: int = 40) -> Path:
    """Write a synthetic training CSV and return its path."""
    path = tmp_path / "toy_training.csv"
    make_junction_df(n_junctions).write_csv(path)
    return path


def _train_args(csv_path: Path, ckpt_dir: Path, *extra: str) -> object:
    """Parse ``main.py train`` arguments for a tiny CPU run."""
    return build_parser().parse_args(
        [
            "train",
            "--data_path",
            str(csv_path),
            "--ckpt_dir",
            str(ckpt_dir),
            "--epochs",
            "1",
            "--batch_size",
            "16",
            "--embed_dim",
            "8",
            "--hidden_dim",
            "8",
            "--device",
            "cpu",
            *extra,
        ]
    )


def _loaded_df(tmp_path: Path) -> pl.DataFrame:
    """Synthetic junctions passed through load_training_csv, as in train_model."""
    df, _ = load_training_csv(_write_toy_csv(tmp_path, 20))
    return df


def _perturb_non_sequence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Change repeat_len, cov_ratio_max and coverage columns, keep sequences."""
    cov_cols = [f"branch_{b}_cov" for b in "ACGT"]
    return df.with_columns(
        pl.col("repeat_len") * 37 + 5,
        (1.0 - pl.col("cov_ratio_max")).alias("cov_ratio_max"),
        pl.col("total_reads") * 9,
        *[pl.col(c) * 50 + 1 for c in cov_cols],
    )


def _model_scores(model: Mv1JunctionModel, dataset: JunctionDataset) -> torch.Tensor:
    """Run the model in eval mode on all pairs and stack (sim_a, sim_b)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_junction_batch)
    out = []
    with torch.no_grad():
        for batch in loader:
            sim_a, sim_b = model(
                batch["context"],
                batch["branch_a"],
                batch["branch_b"],
                batch["cov_feat"],
            )
            out.append(torch.stack([sim_a, sim_b], dim=1))
    return torch.cat(out)


# --- REQ008: model input with --no_cov --------------------------------------


def test_no_cov_flag_disables_coverage_features() -> None:
    """--no_cov stores use_cov=False; the default keeps coverage on."""
    assert build_parser().parse_args(["train", "--no_cov"]).use_cov is False
    assert build_parser().parse_args(["train"]).use_cov is True


def test_no_cov_dataset_has_empty_cov_feat() -> None:
    """With use_cov_feat=False every sample's coverage vector has length 0."""
    dataset = JunctionDataset(make_junction_df(5), use_cov_feat=False)
    assert all(dataset[i]["cov_feat"].shape == (0,) for i in range(len(dataset)))


def test_no_cov_output_ignores_repeat_len_and_coverage(tmp_path: Path) -> None:
    """Under --no_cov, changing repeat_len, cov_ratio_max and coverages changes nothing."""
    df = _loaded_df(tmp_path)
    perturbed = _perturb_non_sequence_columns(df)
    torch.manual_seed(0)
    model = Mv1JunctionModel(
        context_embed_dim=8, branch_embed_dim=8, cov_feat_dim=0, hidden_dim=8
    )

    original = _model_scores(model, JunctionDataset(df, use_cov_feat=False))
    changed = _model_scores(model, JunctionDataset(perturbed, use_cov_feat=False))

    assert torch.equal(original, changed)


def test_repeat_len_is_not_an_input_with_coverage_on(tmp_path: Path) -> None:
    """repeat_len does not reach the model even when coverage features are on."""
    df = _loaded_df(tmp_path)
    changed_df = df.with_columns(pl.col("repeat_len") * 37 + 5)
    torch.manual_seed(0)
    model = Mv1JunctionModel(
        context_embed_dim=8, branch_embed_dim=8, cov_feat_dim=6, hidden_dim=8
    )

    original = _model_scores(model, JunctionDataset(df, use_cov_feat=True))
    changed = _model_scores(model, JunctionDataset(changed_df, use_cov_feat=True))

    assert torch.equal(original, changed)

    # Sanity check: the same perturbation of coverage does move the cov model.
    cov_changed = _model_scores(
        model, JunctionDataset(_perturb_non_sequence_columns(df), use_cov_feat=True)
    )
    assert not torch.equal(original, cov_changed)


def test_train_no_cov_saves_cov_feat_dim_zero(tmp_path: Path) -> None:
    """train --no_cov builds the model with cov_feat_dim=0 and stores it for evaluate."""
    args = _train_args(_write_toy_csv(tmp_path), tmp_path / "ckpt", "--no_cov")

    train_model(args)

    (ckpt_path,) = (tmp_path / "ckpt").glob("*_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["model_hparams"]["cov_feat_dim"] == 0


# --- REQ028: training seeds -------------------------------------------------


def test_same_seed_gives_same_first_epoch_metrics(tmp_path: Path) -> None:
    """Two CPU runs with the same seed give identical epoch-1 metrics."""
    csv_path = _write_toy_csv(tmp_path)

    runs = []
    for name in ("run1", "run2"):
        args = _train_args(csv_path, tmp_path / name, "--seed", "7", "--augment")
        history = train_model(args)
        runs.append(
            (dataclasses.asdict(history.train[0]), dataclasses.asdict(history.val[0]))
        )

    assert runs[0] == runs[1]


def test_seed_and_cudnn_flags_set_before_data_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """random, numpy and torch are seeded and cudnn is deterministic before the CSV is read."""
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    seen: dict[str, object] = {}
    real_load = train_module.load_training_csv

    def spy_load(*args: object, **kwargs: object) -> object:
        seen["random"] = random.getstate()
        seen["numpy"] = np.random.get_state()[1].copy()
        seen["torch"] = torch.initial_seed()
        seen["deterministic"] = torch.backends.cudnn.deterministic
        seen["benchmark"] = torch.backends.cudnn.benchmark
        return real_load(*args, **kwargs)

    monkeypatch.setattr(train_module, "load_training_csv", spy_load)
    args = _train_args(_write_toy_csv(tmp_path), tmp_path / "ckpt", "--seed", "7")

    train_model(args)

    random.seed(7)
    np.random.seed(7)
    assert seen["random"] == random.getstate()
    assert np.array_equal(seen["numpy"], np.random.get_state()[1])
    assert seen["torch"] == 7
    assert seen["deterministic"] is True
    assert seen["benchmark"] is False
