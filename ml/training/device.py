"""Compute-device selection shared by training and evaluation."""

from __future__ import annotations

import torch


def resolve_device(device: str | None) -> torch.device:
    """Turn a ``--device`` value into a usable ``torch.device``.

    With ``None`` the best available backend is picked (cuda, then mps,
    then cpu). An explicitly requested ``cuda`` is checked up front, so a
    CPU-only torch build fails here with a hint instead of with an
    ``AssertionError`` deep inside ``model.to()``.

    Args:
        device: ``"cuda"``, ``"mps"``, ``"cpu"`` (optionally with an index,
            e.g. ``"cuda:1"``), or ``None`` for auto-detection.

    Returns:
        The selected device.

    Raises:
        RuntimeError: ``cuda`` was requested but is not available.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        if torch.version.cuda is None:
            reason = (
                f"the installed torch {torch.__version__} is a CPU-only build. "
                "Sync the CUDA group: uv sync --no-group cpu --group cu128 "
                "(experiments/hpc/config.sh does this for DEVICE=cuda), and run "
                "with 'uv run --no-sync' so uv does not swap torch back to the "
                "default cpu group"
            )
        else:
            reason = (
                f"torch {torch.__version__} is built for CUDA {torch.version.cuda}, "
                "but no GPU is visible (check --gres=gpu:1 / CUDA_VISIBLE_DEVICES "
                "and nvidia-smi)"
            )
        raise RuntimeError(f"--device {device} requested, but {reason}.")
    return resolved
