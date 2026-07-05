"""Evaluation pipeline for trained junction-resolution checkpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import JunctionDataset, collate_junction_batch
from data.loader import load_training_csv
from models.mv1 import Mv1JunctionModel
from training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


def evaluate_model(args) -> dict:
    """Load a checkpoint and evaluate it on a CSV file.

    Evaluates on the entire CSV (no train/val split) - intended for
    cross-organism evaluation where the model has never seen this organism.

    Args:
        args: Namespace with: ``model_path``, ``test_csv``,
            ``output_json`` (optional), ``context_len``, ``branch_len``,
            ``min_context_len``, ``min_branch_len``, ``batch_size``,
            ``num_workers``, ``device``.

    Returns:
        Dict with evaluation metrics.
    """
    ckpt_path = Path(args.model_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hparams = checkpoint.get("model_hparams", {})
    model = Mv1JunctionModel(
        context_embed_dim=hparams.get("context_embed_dim", 64),
        branch_embed_dim=hparams.get("branch_embed_dim", 64),
        cov_feat_dim=hparams.get("cov_feat_dim", 6),
        hidden_dim=hparams.get("hidden_dim", 64),
    )
    model.load_state_dict(checkpoint["model_state"])
    use_cov = hparams.get("cov_feat_dim", 6) > 0

    device_str = getattr(args, "device", None)
    if device_str is None:
        device_str = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    logger.info("Evaluating %s on %s (device=%s)", ckpt_path.name, args.test_csv, device)

    csv_path = Path(args.test_csv)
    df, stats = load_training_csv(
        csv_path=csv_path,
        min_context_len=getattr(args, "min_context_len", 10),
        min_branch_len=getattr(args, "min_branch_len", 5),
        only_labeled=True,
    )
    stats.print()

    dataset = JunctionDataset(
        df=df,
        context_len=getattr(args, "context_len", 100),
        branch_len=getattr(args, "branch_len", 100),
        augment=False,
        use_cov_feat=use_cov,
    )
    loader = DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size", 64),
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=collate_junction_batch,
    )

    criterion = nn.CrossEntropyLoss()
    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in loader:
            context  = batch["context"].to(device)
            b_a      = batch["branch_a"].to(device)
            b_b      = batch["branch_b"].to(device)
            cov_feat = batch["cov_feat"].to(device)
            labels   = batch["label"].to(device)
            j_ids    = batch["junction_id"]

            sim_a, sim_b = model(context, b_a, b_b, cov_feat)
            logits = torch.stack([sim_a, sim_b], dim=1)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            tracker.update(preds, labels, j_ids, loss.item())

    metrics = tracker.compute()

    organism = csv_path.stem.replace("_training", "")
    n_junctions = df["junction_id"].n_unique()
    result = {
        "checkpoint": str(ckpt_path),
        "eval_csv":   str(csv_path),
        "organism":   organism,
        "junction_accuracy": round(metrics.junction_accuracy, 6),
        "pair_accuracy":     round(metrics.pair_accuracy, 6),
        "f1":                round(metrics.f1, 6),
        "precision":         round(metrics.precision, 6),
        "recall":            round(metrics.recall, 6),
        "loss":              round(metrics.loss, 6),
        "n_junctions":       int(n_junctions),
        "n_pairs":           len(dataset),
    }

    logger.info(
        "junction_accuracy=%.4f  pair_accuracy=%.4f  F1=%.4f  n_junctions=%d",
        result["junction_accuracy"], result["pair_accuracy"],
        result["f1"], result["n_junctions"],
    )

    output_json = getattr(args, "output_json", None)
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Evaluation results saved to %s", out_path)

    return result
