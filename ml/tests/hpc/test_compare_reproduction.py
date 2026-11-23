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


@pytest.mark.parametrize(
    ("new_acc", "expected"),
    [
        (0.716, 0),  # -0.4 pp, glasser re-runs
        (0.733, 0),  # +1.3 pp, h32 re-run
        (0.745, 1),  # +2.5 pp, outside the default 2 pp
        (0.695, 1),  # -2.5 pp, the tolerance is two-sided
    ],
)
def test_default_tolerance_is_two_sided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, new_acc: float, expected: int
) -> None:
    """The default 2 pp tolerance accepts the observed spread and rejects 2.5 pp."""
    assert _run(monkeypatch, tmp_path, new_acc) == expected


@pytest.mark.parametrize("flag", ["--tolerance", "--tolerance_pp"])
def test_tolerance_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
) -> None:
    """Both the new flag and the old alias set the tolerance."""
    assert _run(monkeypatch, tmp_path, 0.733, flag, "1.0") == 1
    assert _run(monkeypatch, tmp_path / "x", 0.745, flag, "3.0") == 0
