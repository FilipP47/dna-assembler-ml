"""ONNX export for trained junction-resolution models.

The exported ONNX graph has dynamic axes on the sequence-length dimension
(axis 2) because the CNN uses Global Average Pooling, so the model accepts
sequences of any length at inference time.

Called from ``main.py`` via ``handle_export(args)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from models.mv1 import Mv1JunctionModel

logger = logging.getLogger(__name__)


def export_to_onnx(args) -> None:
    """Load a ``.pt`` checkpoint and export the model to ONNX format.

    Architecture hyperparameters are read from the ``model_hparams`` field
    stored inside the checkpoint.  When that field is absent (old checkpoint
    format), the values fall back to the CLI arguments
    ``--embed_dim``, ``--hidden_dim``, and ``--cov_feat_dim``.

    Args:
        args: Namespace returned by ``argparse`` with the following attributes:
            ``model_path``, ``output_onnx``, ``context_len``, ``branch_len``,
            ``embed_dim``, ``hidden_dim``, ``cov_feat_dim``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    ckpt_path = Path(args.model_path)
    out_path  = Path(args.output_onnx)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    saved_hparams = checkpoint.get("model_hparams", {})
    hparams = {
        "context_embed_dim": saved_hparams.get("context_embed_dim", args.embed_dim),
        "branch_embed_dim":  saved_hparams.get("branch_embed_dim",  args.embed_dim),
        "cov_feat_dim":      saved_hparams.get("cov_feat_dim",      args.cov_feat_dim),
        "hidden_dim":        saved_hparams.get("hidden_dim",         args.hidden_dim),
    }
    logger.info("Model hparams: %s", hparams)

    model = Mv1JunctionModel(**hparams)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_junction_accuracy", float("nan"))
    logger.info("Checkpoint: epoch=%s  val_junction_accuracy=%.4f", epoch, val_acc)

    B = 1
    ctx_len = args.context_len
    br_len  = args.branch_len
    cov_dim = hparams["cov_feat_dim"]

    dummy_context  = torch.zeros(B, 4, ctx_len)
    dummy_branch_a = torch.zeros(B, 4, br_len)
    dummy_branch_b = torch.zeros(B, 4, br_len)
    dummy_cov      = torch.zeros(B, cov_dim)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX: %s", out_path)
    torch.onnx.export(
        model,
        (dummy_context, dummy_branch_a, dummy_branch_b, dummy_cov),
        str(out_path),
        input_names=["context", "branch_a", "branch_b", "cov_feat"],
        output_names=["sim_a", "sim_b"],
        dynamic_axes={
            "context":  {0: "batch", 2: "context_len"},
            "branch_a": {0: "batch", 2: "branch_len"},
            "branch_b": {0: "batch", 2: "branch_len"},
            "cov_feat": {0: "batch"},
            "sim_a":    {0: "batch"},
            "sim_b":    {0: "batch"},
        },
        opset_version=17,
    )
    logger.info("Export complete -> %s", out_path)
