"""Tests for ``experiments/hpc/baseline_max_coverage.py``.

The baseline must score the same junction population as ``main.py evaluate``,
so its hardcoded length thresholds have to match the evaluation defaults.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

from data.loader import load_training_csv
from main import build_parser

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASELINE_PATH = _REPO_ROOT / "experiments" / "hpc" / "baseline_max_coverage.py"

_HEADER = (
    "junction_id,contig_id,depth_in_contig,out_degree,in_degree,node_seq,context_seq,"
    "branch_A_seq,branch_A_cov,branch_C_seq,branch_C_cov,branch_G_seq,branch_G_cov,"
    "branch_T_seq,branch_T_cov,ground_truth,gt_ambiguous,cov_ratio_max,total_reads,k,repeat_len"
)


def _load_baseline() -> ModuleType:
    """Import the baseline script as a module without running ``main()``.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(
        "baseline_max_coverage", _BASELINE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thresholds_match_loader_and_evaluate_defaults() -> None:
    """MIN_CONTEXT_LEN / MIN_BRANCH_LEN equal the loader and evaluate CLI defaults."""
    baseline = _load_baseline()

    loader_params = inspect.signature(load_training_csv).parameters
    assert baseline.MIN_CONTEXT_LEN == loader_params["min_context_len"].default
    assert baseline.MIN_BRANCH_LEN == loader_params["min_branch_len"].default

    eval_args = build_parser().parse_args(
        ["evaluate", "--model_path", "x", "--test_csv", "y"]
    )
    assert baseline.MIN_CONTEXT_LEN == eval_args.min_context_len
    assert baseline.MIN_BRANCH_LEN == eval_args.min_branch_len


def test_coverage_tie_is_broken_alphabetically(tmp_path: Path) -> None:
    """On equal coverage the baseline picks the first base in A, C, G, T order."""
    baseline = _load_baseline()
    ctx = "A" * 20
    branch = "C" * 10
    # Tie between C and G; ground truth is G, so the alphabetical pick (C) is wrong.
    tie_row = (
        f"0,0,1,2,1,{ctx},{ctx},NA,NA,{branch},5,{branch},5,NA,NA,G,false,0.5,10,21,3"
    )
    csv_path = tmp_path / "tie_training.csv"
    csv_path.write_text(_HEADER + "\n" + tie_row + "\n")

    result = baseline.evaluate_file(csv_path)

    assert result["usable"] == 1
    assert result["correct"] == 0
