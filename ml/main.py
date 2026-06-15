"""Command-line interface for the DNA Assembler ML module.

Available sub-commands:

- ``train``    — train the Siamese CNN on junction data from the Rust pipeline.
- ``export``   — export a trained checkpoint to ONNX for inference in Rust.
- ``evaluate`` — evaluate on a held-out test set (not yet implemented).

Example::

    cd ml/
    python3 main.py train \\
        --data_csv ../output/tmp/train/ecoli_training.csv --augment
    python3 main.py export \\
        --model_path ../output/checkpoints/mv1_siamese_cnn_best.pt \\
        --output_onnx ../output/model.onnx
"""

from __future__ import annotations

import argparse
import logging

from training.train import train_model
from training.export import export_to_onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all sub-commands.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="DNA Junction Resolver — ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train",
        help="Train the siamese CNN network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_data = p_train.add_argument_group("data")
    g_data.add_argument("--data_path",       default="../data/training", help="Path to training CSV file or folder (default: ../data/training/)")
    g_data.add_argument("--ckpt_dir",        default="../results/checkpoints", help="Directory to save checkpoints")
    g_data.add_argument("--context_len",     type=int, default=100,  help="Max context sequence length (bp)")
    g_data.add_argument("--branch_len",      type=int, default=100,  help="Max branch sequence length (bp)")
    g_data.add_argument("--min_context_len", type=int, default=10,   help="Discard junctions with shorter context")
    g_data.add_argument("--min_branch_len",  type=int, default=5,    help="Discard junctions with shorter branches")
    g_data.add_argument("--val_split",       type=float, default=0.15, help="Fraction of data for validation")
    g_data.add_argument("--augment",         action="store_true",    help="Enable reverse-complement augmentation")
    g_data.add_argument("--no_cov",          dest="use_cov", action="store_false",
                        help="Disable coverage features (cov_feat_dim=0)")

    g_model = p_train.add_argument_group("model")
    g_model.add_argument("--embed_dim",  type=int, default=64, help="CNN output embedding size")
    g_model.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden layer size")

    g_opt = p_train.add_argument_group("training")
    g_opt.add_argument("--batch_size",  type=int,   default=64,   help="Batch size")
    g_opt.add_argument("--epochs",      type=int,   default=50,   help="Maximum number of epochs")
    g_opt.add_argument("--lr",          type=float, default=1e-3, help="Adam learning rate")
    g_opt.add_argument("--early_stop",  type=int,   default=10,   help="Early-stopping patience (epochs)")
    g_opt.add_argument("--seed",        type=int,   default=42,   help="Random seed for splits")
    g_opt.add_argument("--device",        type=str,   default=None,               help="Device: cuda / mps / cpu (auto-detect if None)")
    g_opt.add_argument("--num_workers",   type=int,   default=0,                  help="DataLoader worker processes")
    g_opt.add_argument("--wandb",         action="store_true",                    help="Enable Weights & Biases logging")
    g_opt.add_argument("--wandb_project", type=str,   default="dna-assembler-ml", help="W&B project name")

    p_export = sub.add_parser(
        "export",
        help="Export trained model to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_export.add_argument("--model_path",   required=True,         help="Path to best_model.pt checkpoint")
    p_export.add_argument("--output_onnx",  default="model.onnx",  help="Output path for the .onnx file")
    p_export.add_argument("--context_len",  type=int, default=100, help="Dummy context length for ONNX trace")
    p_export.add_argument("--branch_len",   type=int, default=100, help="Dummy branch length for ONNX trace")
    # Fallback architecture args — used only when checkpoint has no model_hparams
    p_export.add_argument("--embed_dim",    type=int, default=64,  help="(fallback) CNN embedding size")
    p_export.add_argument("--hidden_dim",   type=int, default=64,  help="(fallback) MLP hidden size")
    p_export.add_argument("--cov_feat_dim", type=int, default=6,   help="(fallback) coverage feature dim")

    p_eval = sub.add_parser("evaluate", help="Evaluate model on a held-out test CSV (TODO)")
    p_eval.add_argument("--model_path", required=True, help="Path to best_model.pt")
    p_eval.add_argument("--test_csv",   required=True, help="Path to test CSV")

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate sub-command handler."""
    args = build_parser().parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "export":
        export_to_onnx(args)
    elif args.command == "evaluate":
        logger.warning("'evaluate' not yet implemented.")


if __name__ == "__main__":
    main()
