"""Tests for ``ml/training/metrics.py``."""

from __future__ import annotations

import pytest
import torch

from training.metrics import MetricsTracker


def test_junction_correct_only_when_all_pairs_correct() -> None:
    """A junction counts as resolved only if every one of its pairs is correct.

    Junction 0: both pairs correct -> resolved.
    Junction 1: one of two pairs wrong -> not resolved.
    Junction 2: single pair, correct -> resolved.
    Expected: junction_accuracy = 2/3, pair_accuracy = 4/5.
    """
    tracker = MetricsTracker()
    labels = torch.tensor([0, 1, 0, 1, 0])
    preds = torch.tensor([0, 1, 0, 0, 0])
    junction_ids = torch.tensor([0, 0, 1, 1, 2])
    tracker.update(preds, labels, junction_ids, loss=0.5)

    metrics = tracker.compute()

    assert metrics.junction_accuracy == pytest.approx(2 / 3)
    assert metrics.pair_accuracy == pytest.approx(4 / 5)


def test_junction_pairs_split_across_batches() -> None:
    """Pairs of one junction spread over two batches are still aggregated together."""
    tracker = MetricsTracker()
    tracker.update(torch.tensor([0]), torch.tensor([0]), torch.tensor([7]), loss=0.1)
    tracker.update(torch.tensor([1]), torch.tensor([0]), torch.tensor([7]), loss=0.1)

    assert tracker.compute().junction_accuracy == pytest.approx(0.0)
