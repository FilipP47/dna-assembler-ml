"""Tests for run provenance (REQ031): git info, fallback, checkpoint and metrics JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import JunctionDataset, collate_junction_batch
from models.mv1 import Mv1JunctionModel
from tests.helpers import make_junction_df
from training import provenance
from training.metrics import EpochMetrics
from training.provenance import collect_provenance, get_git_info, run_config
from training.trainer import Trainer

PROVENANCE_KEYS = {"git_commit", "git_dirty", "config"}


def _git(repo: Path, *args: str) -> None:
    """Run a git command in ``repo``, failing the test on error."""
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A fresh git repo with one committed file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("a\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _train_args(tmp_path: Path) -> argparse.Namespace:
    """Namespace with the train arguments that the config must record."""
    return argparse.Namespace(
        data_path=tmp_path / "data",
        augment=True,
        seed=42,
        lr=1e-3,
        epochs=1,
        context_len=150,
        use_cov=False,
        run_name="prov",
    )


def test_git_info_clean_then_dirty(git_repo: Path) -> None:
    """Commit is the 40-char HEAD; only tracked modifications make it dirty."""
    commit, dirty = get_git_info(git_repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit == head and len(commit) == 40
    assert dirty is False

    (git_repo / "untracked.log").write_text("x\n")
    assert get_git_info(git_repo)[1] is False

    (git_repo / "tracked.txt").write_text("b\n")
    assert get_git_info(git_repo)[1] is True


def test_fallback_without_git_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a git binary the commit comes from GIT_COMMIT, dirty is unknown."""

    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", no_git)
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    assert get_git_info() == ("abc123", None)

    monkeypatch.delenv("GIT_COMMIT")
    assert get_git_info() == ("unknown", None)


def test_fallback_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory that is not a git repo falls back instead of raising."""
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert get_git_info(tmp_path) == ("unknown", None)


def test_run_config_records_train_args(tmp_path: Path) -> None:
    """Config keeps every argument, is JSON-serialisable and adds no_cov."""
    config = run_config(_train_args(tmp_path))

    assert config["data_path"] == str(tmp_path / "data")
    assert config["augment"] is True
    assert config["seed"] == 42
    assert config["lr"] == 1e-3
    assert config["epochs"] == 1
    assert config["context_len"] == 150
    assert config["no_cov"] is True
    assert config["run_name"] == "prov"
    json.dumps(config)


def _trainer(tmp_path: Path, prov: dict) -> Trainer:
    """A CPU trainer with a small model and the given provenance."""
    model = Mv1JunctionModel(
        context_embed_dim=8, branch_embed_dim=8, cov_feat_dim=0, hidden_dim=8
    )
    return Trainer(
        model,
        tmp_path / "ckpt",
        epochs=1,
        device="cpu",
        run_name="prov",
        provenance=prov,
    )


def test_checkpoint_stores_provenance(tmp_path: Path) -> None:
    """The best checkpoint contains git_commit, git_dirty and the full config."""
    prov = collect_provenance(_train_args(tmp_path))
    path = _trainer(tmp_path, prov)._save_best(1, EpochMetrics())

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert PROVENANCE_KEYS <= ckpt.keys()
    assert ckpt["git_commit"] == prov["git_commit"]
    assert ckpt["git_dirty"] == prov["git_dirty"]
    assert ckpt["config"] == prov["config"]


def test_fit_writes_metrics_json_and_keeps_history_csv(tmp_path: Path) -> None:
    """fit() writes <run_name>_metrics.json next to training_history.csv."""
    torch.manual_seed(0)
    prov = {"git_commit": "abc123", "git_dirty": False, "config": {"seed": 0}}
    dataset = JunctionDataset(make_junction_df(16), use_cov_feat=False)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_junction_batch)

    _trainer(tmp_path, prov).fit(loader, loader)

    ckpt_dir = tmp_path / "ckpt"
    assert (ckpt_dir / "training_history.csv").exists()
    metrics = json.loads((ckpt_dir / "prov_metrics.json").read_text())
    assert metrics["git_commit"] == "abc123"
    assert metrics["git_dirty"] is False
    assert metrics["config"] == {"seed": 0}
    assert metrics["epochs_run"] == 1
    assert {"best_epoch", "best_val", "final_train", "final_val"} <= metrics.keys()
