#!/usr/bin/env python3
"""Compare a re-run zero-shot evaluation with the committed reference results.

Reads the per-organism JSON files written by ``main.py evaluate`` for one
variant (default ``group_a_farthest``) from ``$ROOT/results/eval_<variant>/``
(the new run) and from ``experiments/hpc/results/eval_<variant>/`` (the
reference), and prints both the plain mean over organisms and the mean
weighted by ``n_junctions``. The pass/fail check uses the weighted mean,
which is the number quoted in the report (71.98% for group A farthest).

Exits with status 1 when the weighted means differ by more than
``--tolerance_pp`` percentage points, or when the organism sets differ.

Usage (from the repo root)::

    python3 experiments/hpc/compare_reproduction.py --variant group_a_farthest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_HPC_DIR = Path(__file__).resolve().parent


def load_results(eval_dir: Path) -> dict[str, dict]:
    """Load every per-organism evaluation JSON in a directory.

    Args:
        eval_dir: Directory containing ``<organism>.json`` files.

    Returns:
        Mapping from organism (file stem) to the parsed JSON dict.

    Raises:
        FileNotFoundError: If the directory is missing or has no JSON files.
    """
    files = sorted(eval_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No evaluation JSON files in {eval_dir}")
    return {f.stem: json.loads(f.read_text()) for f in files}


def means(results: dict[str, dict]) -> tuple[float, float]:
    """Compute the plain and the n_junctions-weighted mean junction accuracy.

    Args:
        results: Mapping from organism to evaluation dict.

    Returns:
        Tuple ``(plain_mean, weighted_mean)`` as fractions in [0, 1].
    """
    accs = [r["junction_accuracy"] for r in results.values()]
    total = sum(r["n_junctions"] for r in results.values())
    weighted = (
        sum(r["junction_accuracy"] * r["n_junctions"] for r in results.values()) / total
    )
    return statistics.mean(accs), weighted


def main() -> int:
    """Parse arguments, compare the two result sets, and report.

    Returns:
        Process exit status (0 = within tolerance, 1 = not).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(os.environ.get("ROOT", _HPC_DIR.parents[1]))

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="group_a_farthest")
    ap.add_argument(
        "--new_dir",
        type=Path,
        default=None,
        help="Default: $ROOT/results/eval_<variant>",
    )
    ap.add_argument(
        "--ref_dir",
        type=Path,
        default=None,
        help="Default: experiments/hpc/results/eval_<variant>",
    )
    ap.add_argument("--tolerance_pp", type=float, default=1.0)
    args = ap.parse_args()

    new_dir = args.new_dir or root / "results" / f"eval_{args.variant}"
    ref_dir = args.ref_dir or _HPC_DIR / "results" / f"eval_{args.variant}"

    new = load_results(new_dir)
    ref = load_results(ref_dir)
    if set(new) != set(ref):
        logger.error(
            "Organism sets differ: only new=%s, only ref=%s",
            sorted(set(new) - set(ref)),
            sorted(set(ref) - set(new)),
        )
        return 1

    logger.info("%-22s %10s %10s %8s", "organism", "ref", "new", "diff_pp")
    for org in sorted(ref):
        r, n = ref[org]["junction_accuracy"], new[org]["junction_accuracy"]
        logger.info(
            "%-22s %9.2f%% %9.2f%% %+8.2f", org, r * 100, n * 100, (n - r) * 100
        )

    ref_plain, ref_w = means(ref)
    new_plain, new_w = means(new)
    diff_pp = (new_w - ref_w) * 100
    logger.info("")
    logger.info(
        "plain mean:    ref=%.2f%%  new=%.2f%%", ref_plain * 100, new_plain * 100
    )
    logger.info(
        "weighted mean: ref=%.2f%%  new=%.2f%%  diff=%+.2f pp (tolerance +/-%.2f pp)",
        ref_w * 100,
        new_w * 100,
        diff_pp,
        args.tolerance_pp,
    )

    if abs(diff_pp) > args.tolerance_pp:
        logger.error("FAIL: weighted mean outside tolerance")
        return 1
    logger.info("OK: weighted mean within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
