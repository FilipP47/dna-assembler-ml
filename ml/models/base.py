"""Abstract model interface for junction-resolution models.

Every model (MV1, MV2, coverage-only baseline, …) must implement this
interface so that ``trainer.py`` remains architecture-agnostic.

Contract:
    ``forward(context, branch_a, branch_b, cov_feat)``
    returns ``(sim_a, sim_b)``: similarity of the context to each branch.
    ``argmax([sim_a, sim_b])`` gives the predicted branch index
    (0 = branch_a, 1 = branch_b).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseJunctionModel(nn.Module, ABC):
    """Abstract base class for all junction-resolution models.

    Subclasses must implement :meth:`forward` (returning similarity scores
    for both branches) and :meth:`model_name`.
    """

    @abstractmethod
    def forward(
        self,
        context:  torch.Tensor,
        branch_a: torch.Tensor,
        branch_b: torch.Tensor,
        cov_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity scores for both branches.

        Args:
            context: One-hot encoded context tensor of shape ``(B, 4, context_len)``.
            branch_a: One-hot encoded branch A tensor of shape ``(B, 4, branch_len)``.
            branch_b: One-hot encoded branch B tensor of shape ``(B, 4, branch_len)``.
            cov_feat: Coverage feature vector of shape ``(B, cov_feat_dim)``.

        Returns:
            A tuple ``(sim_a, sim_b)`` where each element has shape ``(B,)``
            and represents the similarity score of the context to the
            corresponding branch.
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return a short identifier used for logging and checkpoint filenames."""
        ...

    def predict(
        self,
        context:  torch.Tensor,
        branch_a: torch.Tensor,
        branch_b: torch.Tensor,
        cov_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return the index of the predicted correct branch.

        Wraps :meth:`forward` with ``torch.no_grad`` for inference use.

        Args:
            context: One-hot encoded context tensor of shape ``(B, 4, context_len)``.
            branch_a: One-hot encoded branch A tensor of shape ``(B, 4, branch_len)``.
            branch_b: One-hot encoded branch B tensor of shape ``(B, 4, branch_len)``.
            cov_feat: Coverage feature vector of shape ``(B, cov_feat_dim)``.

        Returns:
            Long tensor of shape ``(B,)`` with values 0 (branch_a predicted) or
            1 (branch_b predicted).
        """
        with torch.no_grad():
            sim_a, sim_b = self.forward(context, branch_a, branch_b, cov_feat)
            return torch.where(sim_a >= sim_b,
                               torch.zeros_like(sim_a, dtype=torch.long),
                               torch.ones_like(sim_a, dtype=torch.long))

    def count_parameters(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            Total count of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        """Serialise the model state dict to a PyTorch checkpoint file.

        Args:
            path: Destination file path.  Parent directories are created
                automatically.
            extra: Optional dict of additional metadata (e.g. epoch number,
                validation metrics) merged into the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.state_dict(),
            "model_name":  self.model_name(),
            **(extra or {}),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: Path, **model_kwargs) -> "BaseJunctionModel":
        """Instantiate a model from a saved checkpoint file.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            **model_kwargs: Keyword arguments forwarded to the subclass
                constructor.

        Returns:
            A model instance with weights loaded from the checkpoint.
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["model_state"])
        return model
