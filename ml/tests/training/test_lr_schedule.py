"""Tests for --lr_schedule and --warmup_epochs in the Trainer (REQ032)."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.dataset import JunctionDataset, collate_junction_batch
from models.mv1 import Mv1JunctionModel
from tests.helpers import make_junction_df
from training.metrics import MetricsTracker
from training.trainer import Trainer, scheduled_lr


def _loader() -> DataLoader:
    """A small deterministic loader (no shuffling)."""
    dataset = JunctionDataset(make_junction_df(24), use_cov_feat=False, augment=True)
    return DataLoader(dataset, batch_size=16, collate_fn=collate_junction_batch)


def _model() -> Mv1JunctionModel:
    """A small seeded model."""
    torch.manual_seed(0)
    return Mv1JunctionModel(
        context_embed_dim=8, branch_embed_dim=8, cov_feat_dim=0, hidden_dim=8
    )


def _legacy_fit(
    model: Mv1JunctionModel, loader: DataLoader, epochs: int
) -> list[float]:
    """The training loop of the old Trainer (Adam + ReduceLROnPlateau), stripped down.

    Returns:
        Learning rate after every epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()
    lrs = []
    for _ in range(epochs):
        for training in (True, False):
            model.train(training)
            tracker = MetricsTracker()
            with torch.enable_grad() if training else torch.no_grad():
                for batch in loader:
                    sim_a, sim_b = model(
                        batch["context"],
                        batch["branch_a"],
                        batch["branch_b"],
                        batch["cov_feat"],
                    )
                    logits = torch.stack([sim_a, sim_b], dim=1)
                    loss = criterion(logits, batch["label"])
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    tracker.update(
                        logits.argmax(dim=1),
                        batch["label"],
                        batch["junction_id"],
                        loss.item(),
                    )
            if not training:
                scheduler.step(tracker.compute().junction_accuracy)
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_default_trainer_matches_legacy_loop(tmp_path: Path) -> None:
    """Default schedule (plateau, no warm-up) gives the same weights and LRs as the old loop."""
    epochs = 20
    loader = _loader()

    legacy = _model()
    torch.manual_seed(5)
    legacy_lrs = _legacy_fit(legacy, loader, epochs)
    # The run must include at least one plateau reduction to test the scheduler.
    assert legacy_lrs[-1] < legacy_lrs[0]

    model = _model()
    trainer = Trainer(model, tmp_path, epochs=epochs, early_stop=100, device="cpu")
    assert isinstance(trainer.scheduler, optim.lr_scheduler.ReduceLROnPlateau)
    assert trainer.scheduler.patience == 5 and trainer.scheduler.factor == 0.5
    torch.manual_seed(5)
    trainer.fit(loader, loader)

    for key, value in legacy.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key
    assert trainer.optimizer.param_groups[0]["lr"] == legacy_lrs[-1]


def test_scheduled_lr_defaults_leave_lr_to_plateau() -> None:
    """With plateau and no warm-up the optimiser LR is never overwritten."""
    assert all(scheduled_lr(1e-3, e, 50, "plateau", 0) is None for e in range(1, 51))


def test_scheduled_lr_warmup_is_linear() -> None:
    """Warm-up epochs use lr * epoch / warmup_epochs, then the schedule starts at lr."""
    for schedule in ("constant", "cosine", "plateau"):
        lrs = [scheduled_lr(1.0, e, 20, schedule, 4) for e in range(1, 6)]
        assert lrs == [0.25, 0.5, 0.75, 1.0, 1.0]
    assert scheduled_lr(1.0, 6, 20, "plateau", 4) is None


def test_scheduled_lr_constant() -> None:
    """constant keeps --lr for every epoch."""
    assert {scheduled_lr(0.01, e, 200, "constant", 0) for e in range(1, 201)} == {0.01}


def test_scheduled_lr_cosine_over_200_epochs() -> None:
    """cosine starts at lr, halves mid-way, decreases monotonically and stays > 0."""
    lrs = [scheduled_lr(1.0, e, 200, "cosine", 0) for e in range(1, 201)]
    assert lrs[0] == 1.0
    assert lrs[100] == pytest.approx(0.5)
    assert all(a > b for a, b in pairwise(lrs))
    assert 0 < lrs[-1] < 1e-3


def test_scheduled_lr_cosine_after_warmup() -> None:
    """With warm-up, the cosine phase spans epochs - warmup_epochs epochs."""
    assert scheduled_lr(1.0, 11, 110, "cosine", 10) == 1.0
    expected = 0.5 * (1 + math.cos(math.pi * 50 / 100))
    assert scheduled_lr(1.0, 61, 110, "cosine", 10) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("schedule", "warmup", "expected"),
    [
        ("constant", 0, [1e-3, 1e-3, 1e-3, 1e-3]),
        ("cosine", 0, [1e-3 * 0.5 * (1 + math.cos(math.pi * t / 4)) for t in range(4)]),
        ("cosine", 2, [5e-4, 1e-3, 1e-3, 5e-4]),
        ("plateau", 2, [5e-4, 1e-3, 1e-3, 1e-3]),
    ],
)
def test_fit_applies_schedule(
    tmp_path: Path, schedule: str, warmup: int, expected: list[float]
) -> None:
    """Trainer.fit sets the scheduled LR before each epoch."""
    trainer = Trainer(
        _model(),
        tmp_path,
        epochs=4,
        early_stop=100,
        device="cpu",
        lr_schedule=schedule,
        warmup_epochs=warmup,
    )
    assert (trainer.scheduler is None) == (schedule != "plateau")
    seen: list[float] = []
    real_run_epoch = trainer._run_epoch

    def spy(loader: DataLoader, training: bool) -> object:
        if training:
            seen.append(trainer.optimizer.param_groups[0]["lr"])
        return real_run_epoch(loader, training)

    trainer._run_epoch = spy  # type: ignore[method-assign]
    trainer.fit(_loader(), _loader())
    assert seen == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr_schedule": "step"}, "lr_schedule"),
        ({"warmup_epochs": -1}, "warmup_epochs"),
    ],
)
def test_trainer_rejects_invalid_schedule(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    """Unknown schedules and negative warm-up are rejected."""
    with pytest.raises(ValueError, match=match):
        Trainer(_model(), tmp_path, device="cpu", **kwargs)
