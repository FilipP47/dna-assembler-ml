"""Metrics for evaluating junction-resolution model quality.

Tracks three levels of accuracy:

1. **Pair accuracy** — fraction of ``(branch_correct, branch_wrong)`` pairs
   classified correctly.
2. **Junction accuracy** — fraction of junctions where *all* pairs are correct.
3. **F1 / Precision / Recall** — standard binary classification metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Aggregated metrics for a single training or evaluation epoch.

    Attributes:
        loss: Mean cross-entropy loss over all batches.
        pair_accuracy: Fraction of branch pairs classified correctly.
        junction_accuracy: Fraction of junctions where every pair is correct.
        precision: Binary precision (positive class = branch index 0).
        recall: Binary recall.
        f1: Binary F1 score.
        tp: True positive count.
        fp: False positive count.
        tn: True negative count.
        fn: False negative count.
    """
    loss:              float = 0.0
    pair_accuracy:     float = 0.0
    junction_accuracy: float = 0.0
    precision:         float = 0.0
    recall:            float = 0.0
    f1:                float = 0.0

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def __str__(self) -> str:
        """Return a compact one-line summary of the epoch metrics."""
        return (
            f"loss={self.loss:.4f} | "
            f"pair_acc={self.pair_accuracy*100:.2f}% | "
            f"junc_acc={self.junction_accuracy*100:.2f}% | "
            f"F1={self.f1:.4f}"
        )


class MetricsTracker:
    """Accumulate per-batch predictions and labels, then compute epoch metrics.

    Typical usage::

        tracker = MetricsTracker()
        for batch in loader:
            ...
            tracker.update(preds, labels, junction_ids, loss.item())
        metrics = tracker.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated state in preparation for a new epoch."""
        self._all_preds:        list[int] = []
        self._all_labels:       list[int] = []
        self._all_junction_ids: list[int] = []
        self._total_loss:       float = 0.0
        self._n_batches:        int   = 0

    def update(
        self,
        preds:        torch.Tensor,
        labels:       torch.Tensor,
        junction_ids: torch.Tensor,
        loss:         float,
    ) -> None:
        """Accumulate predictions and loss from a single batch.

        Args:
            preds: Predicted branch indices of shape ``(B,)``, values 0 or 1.
            labels: Ground-truth branch indices of shape ``(B,)``, values 0 or 1.
            junction_ids: Junction identifiers of shape ``(B,)`` used for
                junction-level accuracy aggregation.
            loss: Scalar batch loss value.
        """
        self._all_preds.extend(preds.cpu().numpy().tolist())
        self._all_labels.extend(labels.cpu().numpy().tolist())
        self._all_junction_ids.extend(junction_ids.cpu().numpy().tolist())
        self._total_loss += loss
        self._n_batches  += 1

    def compute(self) -> EpochMetrics:
        """Compute and return epoch-level metrics from all accumulated data.

        Returns:
            An :class:`EpochMetrics` instance.  Returns a zeroed instance if
            no data has been accumulated.
        """
        if not self._all_preds:
            return EpochMetrics()

        preds  = np.array(self._all_preds)
        labels = np.array(self._all_labels)
        j_ids  = np.array(self._all_junction_ids)

        correct = (preds == labels)
        pair_acc = correct.mean()

        # label=0 means branch_correct is at position 0
        tp = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 1) & (labels == 1)).sum())
        fn = int(((preds == 1) & (labels == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # A junction is "resolved" only when ALL its pairs are correct
        junction_correct: dict[int, bool] = {}
        for jid, is_correct in zip(j_ids, correct):
            if jid not in junction_correct:
                junction_correct[jid] = bool(is_correct)
            else:
                junction_correct[jid] = junction_correct[jid] and bool(is_correct)

        junction_acc = (np.array(list(junction_correct.values()))).mean()

        avg_loss = self._total_loss / max(self._n_batches, 1)

        return EpochMetrics(
            loss=avg_loss,
            pair_accuracy=float(pair_acc),
            junction_accuracy=float(junction_acc),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            tp=tp, fp=fp, tn=tn, fn=fn,
        )


@dataclass
class TrainingHistory:
    """Container for per-epoch train and validation metrics across a full run.

    Attributes:
        train: List of :class:`EpochMetrics` for the training set.
        val: List of :class:`EpochMetrics` for the validation set.
    """
    train: list[EpochMetrics] = field(default_factory=list)
    val:   list[EpochMetrics] = field(default_factory=list)

    def append(self, train_m: EpochMetrics, val_m: EpochMetrics) -> None:
        """Append metrics from one completed epoch.

        Args:
            train_m: Training metrics for the epoch.
            val_m: Validation metrics for the epoch.
        """
        self.train.append(train_m)
        self.val.append(val_m)

    def best_val_epoch(self, metric: str = "f1") -> int:
        """Return the zero-based index of the epoch with the highest validation metric.

        Args:
            metric: Attribute name of :class:`EpochMetrics` to maximise.

        Returns:
            Zero-based epoch index.
        """
        values = [getattr(m, metric) for m in self.val]
        return int(np.argmax(values))

    def print_summary(self) -> None:
        """Print a summary of the best validation epoch to stdout."""
        best_ep = self.best_val_epoch("junction_accuracy")
        best    = self.val[best_ep]
        print(f"\nBest epoch: {best_ep + 1}")
        print(f"  Val junction accuracy: {best.junction_accuracy*100:.2f}%")
        print(f"  Val pair accuracy:     {best.pair_accuracy*100:.2f}%")
        print(f"  Val F1:                {best.f1:.4f}")
        print(f"  Val loss:              {best.loss:.4f}")

    def to_csv(self, path) -> None:
        """Write the full training history to a CSV file.

        Args:
            path: Destination file path.  Parent directories are created
                automatically.
        """
        import csv, pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (t, v) in enumerate(zip(self.train, self.val)):
            rows.append({
                "epoch": i + 1,
                "train_loss": t.loss, "train_pair_acc": t.pair_accuracy,
                "train_junc_acc": t.junction_accuracy, "train_f1": t.f1,
                "val_loss":   v.loss, "val_pair_acc":   v.pair_accuracy,
                "val_junc_acc":   v.junction_accuracy, "val_f1":   v.f1,
            })
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Training history saved to {path}")
