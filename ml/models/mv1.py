"""Siamese CNN model (MV1) for de Bruijn junction resolution.

Inherits from :class:`~models.base.BaseJunctionModel` and uses a shared
1-D CNN encoder to embed context and branch sequences into fixed-size
vectors, which are then scored by a shared MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import BaseJunctionModel


class DNASequenceEncoder(nn.Module):
    """Lightweight 1-D CNN that maps a one-hot DNA sequence to a dense embedding.

    Used as the shared (Siamese) core for encoding the context node and both
    branch candidates.
    """

    def __init__(self, embed_dim: int = 64):
        """Initialise the convolutional encoder.

        Args:
            embed_dim: Number of output channels (embedding dimension).
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of one-hot sequences into embedding vectors.

        Args:
            x: Float tensor of shape ``(B, 4, length)``.

        Returns:
            Float tensor of shape ``(B, embed_dim)``.  Global average pooling
            over the length dimension makes the encoder length-agnostic,
            so sequences of any length are accepted at inference time.
        """
        x = self.conv_block(x)
        x = torch.mean(x, dim=2)
        return x


class Mv1JunctionModel(BaseJunctionModel):
    """MV1 Siamese CNN for junction branch selection.

    Encodes the context, branch A, and branch B with a shared CNN encoder,
    concatenates each branch embedding with the context and coverage features,
    and scores both paths through a single shared MLP to preserve Siamese
    symmetry.
    """

    def __init__(
        self,
        context_embed_dim: int = 64,
        branch_embed_dim:  int = 64,
        cov_feat_dim:      int = 6,
        hidden_dim:        int = 64,
    ):
        """Initialise the MV1 model.

        Args:
            context_embed_dim: Output embedding size of the context CNN encoder.
            branch_embed_dim: Output embedding size of the branch CNN encoder.
            cov_feat_dim: Dimensionality of the coverage feature vector.
                Pass ``0`` to disable coverage features.
            hidden_dim: Hidden layer width of the scoring MLP.
        """
        super().__init__()
        self._context_embed_dim = context_embed_dim
        self._branch_embed_dim  = branch_embed_dim
        self._cov_feat_dim      = cov_feat_dim
        self._hidden_dim        = hidden_dim

        self.context_encoder = DNASequenceEncoder(embed_dim=context_embed_dim)
        self.branch_encoder  = DNASequenceEncoder(embed_dim=branch_embed_dim)

        # A single scorer shared across both branches preserves Siamese symmetry:
        # branch_a and branch_b receive identical treatment with identical weights.
        mlp_in_dim = context_embed_dim + branch_embed_dim + cov_feat_dim

        self.scorer = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def model_name(self) -> str:
        """Return the model identifier string."""
        return "mv1_siamese_cnn"

    def hparams(self) -> dict:
        """Return a dict of hyperparameters for logging and checkpoint storage.

        Returns:
            Dictionary with keys ``context_embed_dim``, ``branch_embed_dim``,
            ``cov_feat_dim``, and ``hidden_dim``.
        """
        return {
            "context_embed_dim": self._context_embed_dim,
            "branch_embed_dim":  self._branch_embed_dim,
            "cov_feat_dim":      self._cov_feat_dim,
            "hidden_dim":        self._hidden_dim,
        }

    def forward(
        self,
        context:  torch.Tensor,
        branch_a: torch.Tensor,
        branch_b: torch.Tensor,
        cov_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity scores for both branch candidates.

        Args:
            context: One-hot context tensor of shape ``(B, 4, context_len)``.
            branch_a: One-hot branch-A tensor of shape ``(B, 4, branch_len)``.
            branch_b: One-hot branch-B tensor of shape ``(B, 4, branch_len)``.
            cov_feat: Coverage features of shape ``(B, cov_feat_dim)``.

        Returns:
            A tuple ``(sim_a, sim_b)`` of shape ``(B,)`` tensors containing
            the raw similarity logits for each branch.
        """
        ctx_emb = self.context_encoder(context)
        a_emb   = self.branch_encoder(branch_a)
        b_emb   = self.branch_encoder(branch_b)

        feat_path_a = torch.cat([ctx_emb, a_emb, cov_feat], dim=1)
        feat_path_b = torch.cat([ctx_emb, b_emb, cov_feat], dim=1)

        sim_a = self.scorer(feat_path_a).squeeze(1)
        sim_b = self.scorer(feat_path_b).squeeze(1)

        return sim_a, sim_b
