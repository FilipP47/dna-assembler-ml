"""Tests for the model-tuning CLI flags of ``main.py train`` (REQ032-REQ034, REQ037)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from main import build_parser
from tests.helpers import make_junction_df
from training.evaluate import evaluate_model
from training.train import train_model

# Flag -> (value without the flag, meaning the original behaviour).
NEW_FLAG_DEFAULTS = {
    "lr_schedule": "plateau",
    "warmup_epochs": 0,
    "width_mult": 1.0,
    "n_blocks": 3,
    "residual": False,
    "batchnorm": False,
    "split_by": "row",
}
# Flags that existed before and keep their defaults.
OLD_FLAG_DEFAULTS = {"epochs": 50, "weight_decay": 1e-4, "dropout": 0.3, "lr": 1e-3}


def _train_dests() -> set[str]:
    """All argument names of the train sub-command."""
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {a.dest for a in sub.choices["train"]._actions if a.dest != "help"}


def test_new_flags_default_to_original_behaviour() -> None:
    """Without the new flags the parsed values describe the old training setup."""
    args = build_parser().parse_args(["train"])
    for name, value in {**NEW_FLAG_DEFAULTS, **OLD_FLAG_DEFAULTS}.items():
        assert getattr(args, name) == value, name


def test_new_flags_parse() -> None:
    """Every new flag is accepted with the documented type."""
    args = build_parser().parse_args(
        [
            "train",
            "--lr_schedule",
            "cosine",
            "--warmup_epochs",
            "5",
            "--epochs",
            "200",
            "--weight_decay",
            "0.01",
            "--dropout",
            "0.1",
            "--width_mult",
            "1.5",
            "--n_blocks",
            "4",
            "--residual",
            "--batchnorm",
            "--split_by",
            "node_seq",
        ]
    )
    assert (args.lr_schedule, args.warmup_epochs, args.epochs) == ("cosine", 5, 200)
    assert (args.weight_decay, args.dropout) == (0.01, 0.1)
    assert (args.width_mult, args.n_blocks, args.residual, args.batchnorm) == (
        1.5,
        4,
        True,
        True,
    )
    assert args.split_by == "node_seq"


@pytest.mark.parametrize(
    "argv", [["--lr_schedule", "step"], ["--split_by", "contig"], ["--n_blocks", "x"]]
)
def test_invalid_flag_values_are_rejected(argv: list[str]) -> None:
    """argparse rejects values outside the choices or of the wrong type."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", *argv])


def _tiny_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    """Arguments for a 2-epoch CPU run on synthetic data."""
    csv_path = tmp_path / "toy_training.csv"
    make_junction_df(40).write_csv(csv_path)
    return build_parser().parse_args(
        [
            "train",
            "--data_path",
            str(csv_path),
            "--ckpt_dir",
            str(tmp_path / "ckpt"),
            "--epochs",
            "2",
            "--batch_size",
            "16",
            "--embed_dim",
            "8",
            "--hidden_dim",
            "8",
            "--device",
            "cpu",
            "--run_name",
            "tune",
            *extra,
        ]
    )


TUNED = [
    "--lr_schedule",
    "cosine",
    "--warmup_epochs",
    "1",
    "--width_mult",
    "0.5",
    "--n_blocks",
    "2",
    "--residual",
    "--batchnorm",
    "--dropout",
    "0.1",
    "--weight_decay",
    "0.001",
    "--split_by",
    "node_seq",
]


def test_metrics_json_has_full_config_and_best_epoch(tmp_path: Path) -> None:
    """<run_name>_metrics.json records every train flag and the best epoch."""
    train_model(_tiny_args(tmp_path, *TUNED))

    metrics = json.loads((tmp_path / "ckpt" / "tune_metrics.json").read_text())
    config = metrics["config"]
    assert _train_dests() <= config.keys()
    assert config["lr_schedule"] == "cosine"
    assert config["warmup_epochs"] == 1
    assert config["width_mult"] == 0.5
    assert config["n_blocks"] == 2
    assert config["residual"] is True
    assert config["batchnorm"] is True
    assert config["split_by"] == "node_seq"
    assert config["dropout"] == 0.1
    assert config["weight_decay"] == 0.001
    assert 1 <= metrics["best_epoch"] <= metrics["epochs_run"] == 2
    assert {"git_commit", "git_dirty", "best_val"} <= metrics.keys()


def test_tuned_checkpoint_stores_architecture_and_evaluates(tmp_path: Path) -> None:
    """The checkpoint keeps the new architecture so evaluate rebuilds it."""
    args = _tiny_args(tmp_path, *TUNED)
    train_model(args)

    (ckpt_path,) = (tmp_path / "ckpt").glob("*_best.pt")
    hparams = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "model_hparams"
    ]
    assert hparams["width_mult"] == 0.5
    assert hparams["n_blocks"] == 2
    assert hparams["residual"] is True
    assert hparams["scorer_batchnorm"] is True

    result = evaluate_model(
        argparse.Namespace(
            model_path=str(ckpt_path), test_csv=args.data_path, device="cpu"
        )
    )
    assert result["n_junctions"] == 40


def test_default_checkpoint_hparams_describe_original_model(tmp_path: Path) -> None:
    """Without the new flags the stored architecture is the original MV1."""
    train_model(_tiny_args(tmp_path))
    (ckpt_path,) = (tmp_path / "ckpt").glob("*_best.pt")
    hparams = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "model_hparams"
    ]
    assert hparams["width_mult"] == 1.0
    assert hparams["n_blocks"] == 3
    assert hparams["residual"] is False
    assert hparams["scorer_batchnorm"] is False
