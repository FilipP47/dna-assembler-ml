"""Consistency checks for the committed genome split lists (groups A and B).

The splits are read from ``experiments/hpc/data/genomes_*/output/*.txt`` and
are never re-drawn by the pipeline, so these files are the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HPC_DATA = Path(__file__).resolve().parents[3] / "experiments" / "hpc" / "data"
_GROUPS = ["genomes_ecoli", "genomes_enterobacteriaceae"]
_VARIANTS = ["farthest", "random"]


def _read_list(path: Path) -> list[str]:
    """Read one accession per line, skipping blank lines.

    Args:
        path: Path to a split list file.

    Returns:
        List of accession strings in file order.
    """
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("group", _GROUPS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_test_and_train_are_disjoint_and_cover_pool(group: str, variant: str) -> None:
    """Test and train lists are disjoint, sized 10/90, and together equal the pool."""
    out = _HPC_DATA / group / "output"
    test = _read_list(out / f"test_{variant}_10.txt")
    train = _read_list(out / f"train_90_for_{variant}_split.txt")
    pool = _read_list(_HPC_DATA / group / "full_pool_100.txt")

    assert len(test) == len(set(test)) == 10
    assert len(train) == len(set(train)) == 90
    assert not set(test) & set(train)
    assert set(test) | set(train) == set(pool)


@pytest.mark.parametrize("group", _GROUPS)
def test_dropped_near_duplicates_not_in_pool(group: str) -> None:
    """Genomes dropped as near-duplicates (Mash) never enter the pool."""
    dropped = _read_list(_HPC_DATA / group / "output" / "dropped_near_duplicates.txt")
    pool = _read_list(_HPC_DATA / group / "full_pool_100.txt")

    assert not set(dropped) & set(pool)
