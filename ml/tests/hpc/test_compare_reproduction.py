"""Tests for ``experiments/hpc/compare_reproduction.py`` (TST018 helper)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "hpc"
    / "compare_reproduction.py"
)


def _load_module() -> ModuleType:
    """Import the script as a module without running ``main``.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("compare_reproduction", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(eval_dir: Path, accs: dict[str, tuple[float, int]]) -> None:
    """Write per-organism evaluation JSON files.

    Args:
        eval_dir: Target directory (created if missing).
        accs: Mapping organism -> (junction_accuracy, n_junctions).
    """
    eval_dir.mkdir(parents=True)
    for org, (acc, n) in accs.items():
        (eval_dir / f"{org}.json").write_text(
            json.dumps({"junction_accuracy": acc, "n_junctions": n})
        )


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, new_acc: float, *extra: str
) -> int:
    """Run ``main`` on a two-organism reference and a shifted new run.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary directory.
        new_acc: Accuracy of both organisms in the new run.
        *extra: Additional command-line arguments.

    Returns:
        Exit status of ``main``.
    """
    ref, new = tmp_path / "ref", tmp_path / "new"
    _write(ref, {"a": (0.72, 100), "b": (0.72, 300)})
    _write(new, {"a": (new_acc, 100), "b": (new_acc, 300)})
    argv = ["prog", "--ref_dir", str(ref), "--new_dir", str(new), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return _load_module().main()


def test_weighted_mean() -> None:
    """The weighted mean uses n_junctions as weights."""
    plain, weighted = _load_module().means(
        {
            "a": {"junction_accuracy": 0.5, "n_junctions": 100},
            "b": {"junction_accuracy": 1.0, "n_junctions": 300},
        }
    )
    assert plain == pytest.approx(0.75)
    assert weighted == pytest.approx(0.875)


def test_within_default_tolerance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A -0.4 pp shift (RTX 4090 re-runs) passes the default 1 pp tolerance."""
    assert _run(monkeypatch, tmp_path, 0.716) == 0


def test_tolerance_is_two_sided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A +1.3 pp shift (RTX PRO 6000 run) fails the default tolerance."""
    assert _run(monkeypatch, tmp_path, 0.733) == 1


@pytest.mark.parametrize("flag", ["--tolerance", "--tolerance_pp"])
def test_tolerance_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
) -> None:
    """Both the new flag and the old alias set the tolerance."""
    assert _run(monkeypatch, tmp_path, 0.733, flag, "1.5") == 0
