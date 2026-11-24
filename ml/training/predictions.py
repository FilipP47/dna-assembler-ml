"""Per-junction predictions written next to the evaluation JSON.

``main.py evaluate`` writes one row per junction to a separate CSV (the JSON
format is unchanged). Every junction left after the length filters is
included, not only the labeled ones, so the file also covers ``Ambiguous`` and
``NotFound`` junctions; for those only the predicted branch is known, and
``correct`` / ``baseline_correct`` are empty. ``ml/analysis/error_analysis.py``
reads these files.

Columns:

- ``organism``, ``junction_id``: source CSV stem and junction id.
- ``label_type``: ``Known`` (ground truth is a base), ``Ambiguous``
  (``ground_truth == "ambiguous"``) or ``NotFound`` (ground truth missing),
  as in ``src/pipeline/training_data/ground_truth.rs``.
- ``ground_truth``: the CSV value (a base, ``ambiguous`` or empty).
- ``n_branches``, ``out_degree``, ``repeat_len``, ``k``, ``cov_ratio_max``,
  ``total_reads``: copied from the CSV (``repeat_len`` and ``k`` are empty
  for CSVs written before those columns existed).
- ``cov_tie``: the two highest branch coverages are equal.
- ``predicted_branch``: base with the highest model score
  (``score_<base>``, the scorer output for that branch).
- ``correct``: ``1`` if every (correct, wrong) pair of the junction was
  classified correctly, the same definition as ``junction_accuracy`` in the
  JSON; empty for junctions without a pair.
- ``baseline_branch``, ``baseline_correct``: max-coverage baseline, ties
  broken in A, C, G, T order as in ``experiments/hpc/baseline_max_coverage.py``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch

from data.dataset import BASES, coverage_features, one_hot
from models.base import BaseJunctionModel

PREDICTION_COLUMNS = [
    "organism",
    "junction_id",
    "label_type",
    "ground_truth",
    "n_branches",
    "out_degree",
    "repeat_len",
    "k",
    "cov_ratio_max",
    "total_reads",
    "cov_tie",
    "predicted_branch",
    "correct",
    "baseline_branch",
    "baseline_correct",
    *[f"score_{b}" for b in BASES],
]


def default_predictions_path(output_json: str | Path) -> Path:
    """Return ``<output_json without .json>_predictions.csv``.

    Args:
        output_json: Path given to ``--output_json``.

    Returns:
        Path of the predictions CSV in the same directory.
    """
    path = Path(output_json)
    return path.with_name(f"{path.stem}_predictions.csv")


def label_type(ground_truth: str | None) -> str:
    """Map a ``ground_truth`` CSV value to ``Known``, ``Ambiguous`` or ``NotFound``.

    Args:
        ground_truth: Value of the ``ground_truth`` column (``None`` for NA).

    Returns:
        The label type name.
    """
    if ground_truth is None:
        return "NotFound"
    if ground_truth == "ambiguous":
        return "Ambiguous"
    return "Known" if ground_truth.upper() in BASES else "NotFound"


def _active_branches(row: dict) -> dict[str, str]:
    """Return ``{base: sequence}`` for the branches present in a row."""
    out = {}
    for base in BASES:
        seq = row.get(f"branch_{base}_seq")
        if seq and seq != "NA":
            out[base] = seq
    return out


def _coverages(row: dict) -> dict[str, float]:
    """Return per-base coverage, ``-1`` when missing (as the baseline script)."""
    covs = {}
    for base in BASES:
        cov = row.get(f"branch_{base}_cov")
        covs[base] = float(cov) if cov is not None else -1.0
    return covs


def baseline_branch(row: dict) -> str:
    """Pick the max-coverage branch, ties broken in A, C, G, T order.

    Args:
        row: Row dict from ``pl.DataFrame.iter_rows(named=True)``.

    Returns:
        The chosen base.
    """
    covs = _coverages(row)
    return max(BASES, key=lambda b: covs[b])


def coverage_tie(row: dict) -> bool:
    """Return whether the two highest coverages of the active branches are equal.

    Args:
        row: Row dict from ``pl.DataFrame.iter_rows(named=True)``.

    Returns:
        ``True`` for a tie; ``False`` with fewer than two coverages.
    """
    covs = sorted((c for c in _coverages(row).values() if c >= 0), reverse=True)
    return len(covs) >= 2 and covs[0] == covs[1]


def score_branches(
    model: BaseJunctionModel,
    df: pl.DataFrame,
    context_len: int,
    branch_len: int,
    use_cov: bool,
    device: torch.device,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    """Score every active branch of every row with the model's scorer.

    The scorer of a Siamese model treats each branch on its own, so the
    score of ``branch`` is ``model(context, branch, branch, cov)[0]``. Inputs
    are encoded as in :class:`data.dataset.JunctionDataset` (last
    ``context_len`` context bases, first ``branch_len`` branch bases).

    Args:
        model: Model in eval mode, already on ``device``.
        df: Rows to score (output of ``load_training_csv``).
        context_len: Context window (``--context_len``).
        branch_len: Branch window (``--branch_len``).
        use_cov: Whether the model takes the 6 coverage features.
        device: Device of the model.
        batch_size: Number of (row, branch) items per forward pass.

    Returns:
        One ``{base: score}`` dict per row, in ``df`` order.
    """
    items: list[tuple[int, str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i, row in enumerate(df.iter_rows(named=True)):
        context = one_hot(str(row.get("context_seq") or "")[-context_len:], context_len)
        cov = coverage_features(row) if use_cov else torch.zeros(0)
        for base, seq in _active_branches(row).items():
            items.append((i, base, context, one_hot(seq, branch_len), cov))

    scores: list[dict[str, float]] = [{} for _ in range(df.height)]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            context = torch.stack([it[2] for it in chunk]).to(device)
            branch = torch.stack([it[3] for it in chunk]).to(device)
            cov = torch.stack([it[4] for it in chunk]).to(device)
            sim, _ = model(context, branch, branch, cov)
            for (i, base, *_), value in zip(chunk, sim.cpu().tolist(), strict=True):
                scores[i][base] = value
    return scores


def build_predictions(
    df: pl.DataFrame,
    scores: list[dict[str, float]],
    junction_correct: dict[int, bool],
    organism: str,
) -> pl.DataFrame:
    """Assemble the per-junction predictions table.

    Args:
        df: All rows after the length filters (labeled or not).
        scores: Output of :func:`score_branches` for ``df``.
        junction_correct: ``junction_id -> all pairs correct`` from the
            evaluation loop (labeled junctions only).
        organism: Organism name written to every row.

    Returns:
        DataFrame with :data:`PREDICTION_COLUMNS`.
    """
    rows = []
    for row, row_scores in zip(df.iter_rows(named=True), scores, strict=True):
        gt = row.get("ground_truth")
        kind = label_type(gt)
        jid = int(row["junction_id"])
        base_pick = baseline_branch(row)
        correct = junction_correct.get(jid)
        rows.append(
            {
                "organism": organism,
                "junction_id": jid,
                "label_type": kind,
                "ground_truth": gt,
                "n_branches": row.get("n_active_branches"),
                "out_degree": row.get("out_degree"),
                "repeat_len": row.get("repeat_len"),
                "k": row.get("k"),
                "cov_ratio_max": row.get("cov_ratio_max"),
                "total_reads": row.get("total_reads"),
                "cov_tie": coverage_tie(row),
                "predicted_branch": (
                    max(row_scores, key=row_scores.__getitem__) if row_scores else None
                ),
                "correct": None if correct is None else int(correct),
                "baseline_branch": base_pick,
                "baseline_correct": (
                    int(base_pick == gt.upper()) if kind == "Known" else None
                ),
                **{
                    f"score_{b}": (round(row_scores[b], 6) if b in row_scores else None)
                    for b in BASES
                },
            }
        )
    schema = {
        "organism": pl.String,
        "junction_id": pl.Int64,
        "label_type": pl.String,
        "ground_truth": pl.String,
        "n_branches": pl.Int64,
        "out_degree": pl.Int64,
        "repeat_len": pl.Int64,
        "k": pl.Int64,
        "cov_ratio_max": pl.Float64,
        "total_reads": pl.Int64,
        "cov_tie": pl.Boolean,
        "predicted_branch": pl.String,
        "correct": pl.Int64,
        "baseline_branch": pl.String,
        "baseline_correct": pl.Int64,
        **{f"score_{b}": pl.Float64 for b in BASES},
    }
    return pl.DataFrame(rows, schema=schema)
