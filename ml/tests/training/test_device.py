"""Tests for training.device.resolve_device."""

from __future__ import annotations

import pytest
import torch

from training.device import resolve_device


def _no_gpu(monkeypatch: pytest.MonkeyPatch, cuda_build: str | None) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", cuda_build)


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch, None)
    assert resolve_device(None) == torch.device("cpu")


def test_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch, None)
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_on_cpu_build_points_to_cu128(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch, None)
    with pytest.raises(RuntimeError, match="CPU-only build.*--group cu128"):
        resolve_device("cuda")


def test_cuda_build_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch, "12.8")
    with pytest.raises(RuntimeError, match="no GPU is visible"):
        resolve_device("cuda:0")
