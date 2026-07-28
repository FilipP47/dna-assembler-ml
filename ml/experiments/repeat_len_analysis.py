"""Repeat-length distribution: does labeled == short repeat, ambiguous == long repeat?

Loads the full cerevisiae k=21 CSV (labeled + ambiguous) and compares the distribution
of `repeat_len` (backward distance from the junction to the upstream in-junction, i.e. how
many bases of shared repeat precede the branch point) between the two groups. The hypothesis
from the report: `Known` labels have their disambiguating flank inside the 100 bp context
window (short repeats), while `Ambiguous` junctions sit inside repeats longer than the window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_training_csv  # noqa: E402

DATA = (
    Path(__file__).resolve().parent.parent.parent
    / "data/training/cerevisiae_k21_training.csv"
)


def summarize(name: str, vals: np.ndarray) -> None:
    if len(vals) == 0:
        print(f"{name:>12}: brak wierszy")
        return
    pct = np.percentile(vals, [50, 75, 90, 95])
    print(
        f"{name:>12}: n={len(vals):>5}  mean={vals.mean():6.1f}  "
        f"median={pct[0]:5.0f}  p75={pct[1]:5.0f}  p90={pct[2]:5.0f}  p95={pct[3]:5.0f}  "
        f"max={vals.max():.0f}"
    )


def main() -> None:
    df, stats = load_training_csv(DATA, only_labeled=False)
    print(f"Rows (po filtrze dlugosci): {df.height}, k={stats.k_value}")
    if "repeat_len" not in df.columns:
        print("BRAK kolumny repeat_len - zregeneruj dane nowym binarnym.")
        return

    labeled = df.filter(
        pl.col("ground_truth").is_not_null() & (pl.col("ground_truth") != "ambiguous")
    )
    ambiguous = df.filter(pl.col("ground_truth") == "ambiguous")

    r_lab = labeled["repeat_len"].to_numpy()
    r_amb = ambiguous["repeat_len"].to_numpy()

    print("\nRozklad repeat_len:")
    summarize("labeled", r_lab)
    summarize("ambiguous", r_amb)

    # Fraction with repeat inside vs beyond the 100 bp context window.
    for thr in (50, 100):
        lab_below = (r_lab < thr).mean() * 100
        amb_below = (r_amb < thr).mean() * 100
        print(
            f"\n  repeat_len < {thr}:  labeled {lab_below:.1f}%   ambiguous {amb_below:.1f}%"
        )

    # Binned counts.
    edges = [0, 10, 20, 40, 60, 100, 300, 10_000]
    lbls = ["<10", "10-20", "20-40", "40-60", "60-100", "100-300", ">=300"]
    print(f"\n{'repeat_len':>12}  {'labeled':>10}  {'ambiguous':>10}")
    print("-" * 38)
    for lo, hi, lb in zip(edges[:-1], edges[1:], lbls):
        nl = int(((r_lab >= lo) & (r_lab < hi)).sum())
        na = int(((r_amb >= lo) & (r_amb < hi)).sum())
        print(f"{lb:>12}  {nl:>10}  {na:>10}")


if __name__ == "__main__":
    main()
