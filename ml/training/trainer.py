"""Architecture-agnostic training loop for junction-resolution models.

Accepts any model implementing :class:`~models.base.BaseJunctionModel`.

Training strategy:

- Loss: CrossEntropy on ``(sim_a, sim_b)`` treated as logits.
- Optimiser: Adam with ``ReduceLROnPlateau`` scheduler.
- Early stopping on validation ``junction_accuracy``.
- Best-model checkpoint saved automatically.
- Optional Weights & Biases logging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.metrics import EpochMetrics, MetricsTracker, TrainingHistory
from models.base import BaseJunctionModel

logger = logging.getLogger(__name__)


class Trainer:
    """Train a junction-resolution model and manage checkpoints.

    Example::

        trainer = Trainer(model, checkpoint_dir, use_wandb=True)
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model:          BaseJunctionModel,
        checkpoint_dir: Path,
        learning_rate:  float = 1e-3,
        epochs:         int   = 50,
        early_stop:     int   = 10,
        lr_patience:    int   = 5,
        lr_factor:      float = 0.5,
        grad_clip:      float = 1.0,
        device:         Optional[str] = None,
        use_wandb:      bool  = False,
        wandb_project:  str   = "dna-assembler-ml",
        wandb_config:   Optional[dict] = None,
    ):
        """Initialise the trainer, optimiser, scheduler, and optional W&B run.

        Args:
            model: Model implementing :class:`~models.base.BaseJunctionModel`.
            checkpoint_dir: Directory where checkpoints are saved.
            learning_rate: Initial Adam learning rate.
            epochs: Maximum number of training epochs.
            early_stop: Number of epochs without improvement before stopping.
            lr_patience: Epochs without improvement before LR is reduced.
            lr_factor: Multiplicative factor for LR reduction.
            grad_clip: Max gradient norm for clipping (disabled when ``<= 0``).
            device: Compute device (``"cuda"``, ``"mps"``, or ``"cpu"``).
                Auto-detected when ``None``.
            use_wandb: Enable Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_config: Extra key-value pairs logged to the W&B run config.
        """
        self.model     = model
        self.ckpt_dir  = Path(checkpoint_dir)
        self.epochs    = epochs
        self.grad_clip = grad_clip
        self.early_stop_patience = early_stop
        self.use_wandb = use_wandb

        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                     "mps"  if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model: {model.model_name()} "
                    f"({model.count_parameters():,} parameters)")

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max",
            patience=lr_patience, factor=lr_factor,
        )
        self.criterion = nn.CrossEntropyLoss()

        if use_wandb:
            self._init_wandb(wandb_project, learning_rate, wandb_config or {})

    def _init_wandb(self, project: str, lr: float, extra_config: dict) -> None:
        """Initialise a Weights & Biases run and attach gradient logging.

        Falls back gracefully to no-op logging if ``wandb`` is not installed.

        Args:
            project: W&B project name.
            lr: Learning rate included in the run config.
            extra_config: Additional config entries merged into the W&B config.
        """
        try:
            import wandb
            hparams = self.model.hparams() if hasattr(self.model, "hparams") else {}
            wandb.init(
                project=project,
                config={
                    "model":        self.model.model_name(),
                    "parameters":   self.model.count_parameters(),
                    "device":       str(self.device),
                    "epochs":       self.epochs,
                    "early_stop":   self.early_stop_patience,
                    "learning_rate": lr,
                    "grad_clip":    self.grad_clip,
                    **hparams,
                    **extra_config,
                },
            )
            wandb.watch(self.model, log="gradients", log_freq=50)
            logger.info("Weights & Biases run: %s", wandb.run.url)
        except ImportError:
            logger.warning("wandb not installed — disabling W&B logging")
            self.use_wandb = False

    def _wandb_log(self, epoch: int, train: EpochMetrics, val: EpochMetrics, lr: float) -> None:
        """Log per-epoch train/val metrics to Weights & Biases.

        Args:
            epoch: Current epoch number (1-based).
            train: Training metrics for this epoch.
            val: Validation metrics for this epoch.
            lr: Current learning rate.
        """
        try:
            import wandb
            wandb.log({
                "epoch": epoch,
                "lr": lr,
                "train/loss":          train.loss,
                "train/pair_acc":      train.pair_accuracy,
                "train/junction_acc":  train.junction_accuracy,
                "train/f1":            train.f1,
                "train/precision":     train.precision,
                "train/recall":        train.recall,
                "val/loss":            val.loss,
                "val/pair_acc":        val.pair_accuracy,
                "val/junction_acc":    val.junction_accuracy,
                "val/f1":              val.f1,
                "val/precision":       val.precision,
                "val/recall":          val.recall,
            }, step=epoch)
        except Exception:
            pass

    def _wandb_log_best(self, epoch: int, val: EpochMetrics, ckpt_path: Path) -> None:
        """Update W&B run summary and upload the best checkpoint as an artifact.

        Args:
            epoch: Epoch number of the new best checkpoint.
            val: Validation metrics at that epoch.
            ckpt_path: Path to the saved checkpoint file.
        """
        try:
            import wandb
            wandb.summary["best_epoch"]             = epoch
            wandb.summary["best_val_junction_acc"]  = val.junction_accuracy
            wandb.summary["best_val_pair_acc"]      = val.pair_accuracy
            wandb.summary["best_val_f1"]            = val.f1
            artifact = wandb.Artifact(
                name=f"{self.model.model_name()}_best",
                type="model",
                description=f"Best checkpoint (epoch {epoch}, val_junc_acc={val.junction_accuracy:.4f})",
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)
        except Exception:
            pass

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> TrainingHistory:
        """Run the full training loop and return the metric history.

        Saves the best checkpoint whenever ``val junction_accuracy`` improves.
        Applies early stopping when no improvement is seen for
        ``early_stop_patience`` consecutive epochs.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.

        Returns:
            :class:`~training.metrics.TrainingHistory` containing per-epoch
            train and validation metrics.
        """
        history = TrainingHistory()
        best_val_junc_acc = 0.0
        no_improve_count  = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics   = self._run_epoch(val_loader,   training=False)

            history.append(train_metrics, val_metrics)
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} ({elapsed:.1f}s) | "
                f"Train: {train_metrics} | Val: {val_metrics} | lr={lr:.2e}"
            )

            if self.use_wandb:
                self._wandb_log(epoch, train_metrics, val_metrics, lr)

            self.scheduler.step(val_metrics.junction_accuracy)

            if val_metrics.junction_accuracy > best_val_junc_acc:
                best_val_junc_acc = val_metrics.junction_accuracy
                no_improve_count  = 0
                ckpt_path = self._save_best(epoch, val_metrics)
                logger.info(f"  New best val junction accuracy: "
                            f"{best_val_junc_acc*100:.2f}%")
                if self.use_wandb:
                    self._wandb_log_best(epoch, val_metrics, ckpt_path)
            else:
                no_improve_count += 1
                if no_improve_count >= self.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch} "
                                f"(no improvement for {no_improve_count} epochs)")
                    break

        history.print_summary()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(self.ckpt_dir / "training_history.csv")

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> EpochMetrics:
        """Execute one pass over a DataLoader and return aggregated metrics.

        Args:
            loader: DataLoader to iterate over.
            training: When ``True``, runs backpropagation and gradient clipping.

        Returns:
            :class:`~training.metrics.EpochMetrics` for this pass.
        """
        self.model.train(training)
        tracker = MetricsTracker()

        ctx_mgr = torch.enable_grad if training else torch.no_grad
        with ctx_mgr():
            for batch in loader:
                context  = batch["context"].to(self.device)
                b_a      = batch["branch_a"].to(self.device)
                b_b      = batch["branch_b"].to(self.device)
                cov_feat = batch["cov_feat"].to(self.device)
                labels   = batch["label"].to(self.device)
                j_ids    = batch["junction_id"]

                sim_a, sim_b = self.model(context, b_a, b_b, cov_feat)
                logits = torch.stack([sim_a, sim_b], dim=1)
                loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                preds = logits.argmax(dim=1)
                tracker.update(preds, labels, j_ids, loss.item())

        return tracker.compute()

    def _save_best(self, epoch: int, metrics: EpochMetrics) -> Path:
        """Save a checkpoint for the current best model.

        Args:
            epoch: Current epoch number (used as metadata in the checkpoint).
            metrics: Validation metrics at this epoch.

        Returns:
            Path to the saved checkpoint file.
        """
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"{self.model.model_name()}_best.pt"
        self.model.save_checkpoint(path, extra={
            "epoch": epoch,
            "val_junction_accuracy": metrics.junction_accuracy,
            "val_f1": metrics.f1,
            "model_hparams": self.model.hparams() if hasattr(self.model, "hparams") else {},
        })
        return path
