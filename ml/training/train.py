"""End-to-end training pipeline.

Loads data, builds the model, and runs :class:`~training.trainer.Trainer`.
Called from ``main.py`` via ``handle_train(args)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import JunctionDataset, collate_junction_batch
from data.loader import load_training_csv, split_dataset
from models.mv1 import Mv1JunctionModel
from training.trainer import Trainer
from training.metrics import TrainingHistory

logger = logging.getLogger(__name__)


def train_model(args) -> TrainingHistory:
    """Run the full training pipeline from parsed CLI arguments.

    Loads the training CSV, splits it into train/val sets, builds
    :class:`~models.mv1.Mv1JunctionModel`, and runs
    :class:`~training.trainer.Trainer`.

    Args:
        args: Namespace returned by ``argparse``  (see ``main.py`` for
            the full list of expected attributes: ``data_path``,
            ``min_context_len``, ``min_branch_len``, ``val_split``,
            ``seed``, ``context_len``, ``branch_len``, ``augment``,
            ``use_cov``, ``batch_size``, ``num_workers``, ``embed_dim``,
            ``hidden_dim``, ``ckpt_dir``, ``lr``, ``epochs``,
            ``early_stop``, ``device``, ``wandb``, ``wandb_project``).

    Returns:
        :class:`~training.metrics.TrainingHistory` with per-epoch metrics.
    """
    csv_path = Path(args.data_path)

    df, stats = load_training_csv(
        csv_path=csv_path,
        min_context_len=args.min_context_len,
        min_branch_len=args.min_branch_len,
        only_labeled=True,
    )
    stats.print()

    df_train, df_val, _ = split_dataset(
        df=df,
        val_split=args.val_split,
        test_split=0.0,
        seed=args.seed,
        stratify=True,
    )

    train_dataset = JunctionDataset(
        df=df_train,
        context_len=args.context_len,
        branch_len=args.branch_len,
        augment=args.augment,
        use_cov_feat=args.use_cov,
    )
    val_dataset = JunctionDataset(
        df=df_val,
        context_len=args.context_len,
        branch_len=args.branch_len,
        augment=False,
        use_cov_feat=args.use_cov,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_junction_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_junction_batch,
    )

    cov_dim = 6 if args.use_cov else 0
    model = Mv1JunctionModel(
        context_embed_dim=args.embed_dim,
        branch_embed_dim=args.embed_dim,
        cov_feat_dim=cov_dim,
        hidden_dim=args.hidden_dim,
    )

    wandb_config = {
        "data_path":   str(Path(args.data_path).name),
        "context_len": args.context_len,
        "branch_len":  args.branch_len,
        "augment":     args.augment,
        "use_cov":     args.use_cov,
        "batch_size":  args.batch_size,
        "val_split":   args.val_split,
        "seed":        args.seed,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
    }

    trainer = Trainer(
        model=model,
        checkpoint_dir=Path(args.ckpt_dir),
        learning_rate=args.lr,
        epochs=args.epochs,
        early_stop=args.early_stop,
        device=args.device,
        use_wandb=getattr(args, "wandb", False),
        wandb_project=getattr(args, "wandb_project", "dna-assembler-ml"),
        wandb_config=wandb_config,
    )

    history = trainer.fit(train_loader, val_loader)
    logger.info("Training finished. Best checkpoint saved to %s", args.ckpt_dir)
    return history
