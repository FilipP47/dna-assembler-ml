"""Load and aggregate transfer learning experiment results for notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_eval_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_solo_evals(results_dir: Path, organisms: list[str]) -> pd.DataFrame:
    """Load all eval_on_<org>.json files from solo/<org>/ directories.

    Returns DataFrame with columns:
        model_organism, eval_organism, junction_accuracy, pair_accuracy,
        f1, precision, recall, loss, n_junctions, n_pairs
    """
    rows = []
    for model_org in organisms:
        solo_dir = results_dir / "solo" / model_org
        for eval_org in organisms:
            json_path = solo_dir / f"eval_on_{eval_org}.json"
            if not json_path.exists():
                continue
            d = load_eval_json(json_path)
            rows.append({
                "model_organism": model_org,
                "eval_organism":  eval_org,
                **{k: d[k] for k in
                   ("junction_accuracy", "pair_accuracy", "f1",
                    "precision", "recall", "loss", "n_junctions", "n_pairs")
                   if k in d},
            })
    return pd.DataFrame(rows)


def collect_finetune_evals(results_dir: Path, organisms: list[str]) -> pd.DataFrame:
    """Load all eval JSONs from finetune/<src>_to_<tgt>/ directories.

    Returns DataFrame with columns:
        source_organism, target_organism, eval_organism,
        junction_accuracy, pair_accuracy, f1, precision, recall, loss,
        n_junctions, n_pairs
    """
    rows = []
    for src in organisms:
        for tgt in organisms:
            if src == tgt:
                continue
            ft_dir = results_dir / "finetune" / f"{src}_to_{tgt}"
            for eval_org in organisms:
                json_path = ft_dir / f"eval_on_{eval_org}.json"
                if not json_path.exists():
                    continue
                d = load_eval_json(json_path)
                rows.append({
                    "source_organism": src,
                    "target_organism": tgt,
                    "eval_organism":   eval_org,
                    **{k: d[k] for k in
                       ("junction_accuracy", "pair_accuracy", "f1",
                        "precision", "recall", "loss", "n_junctions", "n_pairs")
                       if k in d},
                })
    return pd.DataFrame(rows)


def build_transfer_matrix(
    finetune_df: pd.DataFrame,
    solo_df: pd.DataFrame,
    organisms: list[str],
) -> np.ndarray:
    """Compute the NxN joint transfer score matrix.

    transfer_score[i, j] = delta_target[i,j] + min(delta_source[i,j], 0)

    where:
        delta_target[i,j] = acc(finetune_i_on_j evaluated on j) - acc(solo_j on j)
        delta_source[i,j] = acc(finetune_i_on_j evaluated on i) - acc(solo_i on i)

    A positive transfer_score means: fine-tuning src on tgt improved tgt performance
    without hurting src more than the gain.

    Args:
        finetune_df: From collect_finetune_evals().
        solo_df: From collect_solo_evals().
        organisms: Ordered list of organism names (defines matrix rows/cols).

    Returns:
        NxN float array. Diagonal entries are NaN (no self-fine-tuning).
    """
    n = len(organisms)
    idx = {org: i for i, org in enumerate(organisms)}
    matrix = np.full((n, n), np.nan)

    solo_self = {}
    for _, row in solo_df[solo_df["model_organism"] == solo_df["eval_organism"]].iterrows():
        solo_self[row["model_organism"]] = row["junction_accuracy"]

    for i, src in enumerate(organisms):
        for j, tgt in enumerate(organisms):
            if src == tgt:
                continue

            # delta_target: did the fine-tuned model get better on tgt?
            ft_on_tgt = finetune_df[
                (finetune_df["source_organism"] == src) &
                (finetune_df["target_organism"] == tgt) &
                (finetune_df["eval_organism"]   == tgt)
            ]
            if ft_on_tgt.empty or tgt not in solo_self:
                continue
            delta_tgt = ft_on_tgt.iloc[0]["junction_accuracy"] - solo_self[tgt]

            # delta_source: did the fine-tuned model forget src?
            ft_on_src = finetune_df[
                (finetune_df["source_organism"] == src) &
                (finetune_df["target_organism"] == tgt) &
                (finetune_df["eval_organism"]   == src)
            ]
            if ft_on_src.empty or src not in solo_self:
                delta_src = 0.0
            else:
                delta_src = ft_on_src.iloc[0]["junction_accuracy"] - solo_self[src]

            matrix[i, j] = delta_tgt + min(delta_src, 0.0)

    return matrix


def load_training_history(ckpt_dir: Path) -> pd.DataFrame:
    """Load training_history.csv from a checkpoint directory."""
    csv_path = Path(ckpt_dir) / "training_history.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)
