"""Siamese CNN model (MV1) for de Bruijn junction resolution.

Inherits from :class:`~models.base.BaseJunctionModel` and uses a shared
1-D CNN encoder to embed context and branch sequences into fixed-size
vectors, which are then scored by a shared MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import BaseJunctionModel


def encoder_channels(n_blocks: int, width_mult: float, embed_dim: int) -> list[int]:
    """Return the output channel count of every convolutional block.

    The base widths are 32 for the first block and 64 for every following
    hidden block; ``width_mult`` scales them. The last block always outputs
    ``embed_dim`` channels, so ``width_mult`` does not change the embedding
    size. With the defaults (``n_blocks=3``, ``width_mult=1.0``) this gives
    ``[32, 64, embed_dim]``, the original MV1 encoder.

    Args:
        n_blocks: Number of convolutional blocks (>= 1).
        width_mult: Multiplier for the hidden block widths (> 0).
        embed_dim: Output channels of the last block.

    Returns:
        List of ``n_blocks`` channel counts.

    Raises:
        ValueError: If ``n_blocks < 1`` or ``width_mult <= 0``.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    if width_mult <= 0:
        raise ValueError(f"width_mult must be > 0, got {width_mult}")
    hidden = [max(1, round(base * width_mult)) for base in [32] + [64] * (n_blocks - 2)]
    return hidden[: n_blocks - 1] + [embed_dim]


class ResidualConvBlock(nn.Module):
    """Conv -> ReLU -> BatchNorm block with a skip connection.

    The skip is the identity when the channel count is unchanged and a 1x1
    convolution otherwise.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        """Initialise the block.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
            kernel_size: Odd kernel size; padding keeps the length unchanged.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``body(x) + skip(x)``.

        Args:
            x: Float tensor of shape ``(B, in_ch, length)``.

        Returns:
            Float tensor of shape ``(B, out_ch, length)``.
        """
        return self.body(x) + self.skip(x)


class DNASequenceEncoder(nn.Module):
    """Lightweight 1-D CNN that maps a one-hot DNA sequence to a dense embedding.

    Used as the shared (Siamese) core for encoding the context node and both
    branch candidates.
    """

    def __init__(
        self,
        embed_dim:  int = 64,
        dropout:    float = 0.3,
        width_mult: float = 1.0,
        n_blocks:   int = 3,
        residual:   bool = False,
    ):
        """Initialise the convolutional encoder.

        The first block uses kernel size 5, the others kernel size 3. Each
        block is Conv -> ReLU -> BatchNorm. With the default arguments the
        module tree, parameter names and initialisation order are identical
        to the original three-block MV1 encoder, so old checkpoints load and
        a seeded model gets the same weights.

        Args:
            embed_dim: Number of output channels (embedding dimension).
            dropout: Dropout probability applied after global average pooling.
            width_mult: Multiplier for the hidden block widths (see
                :func:`encoder_channels`).
            n_blocks: Number of convolutional blocks.
            residual: Wrap every block in a skip connection
                (:class:`ResidualConvBlock`).
        """
        super().__init__()
        channels = encoder_channels(n_blocks, width_mult, embed_dim)
        in_channels = [4] + channels[:-1]
        kernels = [5] + [3] * (n_blocks - 1)

        if residual:
            self.conv_block = nn.Sequential(*[
                ResidualConvBlock(c_in, c_out, k)
                for c_in, c_out, k in zip(in_channels, channels, kernels)
            ])
        else:
            layers: list[nn.Module] = []
            for c_in, c_out, k in zip(in_channels, channels, kernels):
                layers += [
                    nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=k, padding=k // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(c_out),
                ]
            self.conv_block = nn.Sequential(*layers)
        self.pool_drop = nn.Dropout(dropout)

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
        return self.pool_drop(x)


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
        dropout:           float = 0.3,
        width_mult:        float = 1.0,
        n_blocks:          int = 3,
        residual:          bool = False,
        scorer_batchnorm:  bool = False,
    ):
        """Initialise the MV1 model.

        Args:
            context_embed_dim: Output embedding size of the context CNN encoder.
            branch_embed_dim: Output embedding size of the branch CNN encoder.
            cov_feat_dim: Dimensionality of the coverage feature vector.
                Pass ``0`` to disable coverage features.
            hidden_dim: Hidden layer width of the scoring MLP.
            dropout: Dropout probability used in CNN encoders and the MLP scorer.
            width_mult: Multiplier for the hidden convolution widths of both
                encoders (the embedding size stays ``*_embed_dim``).
            n_blocks: Number of convolutional blocks per encoder.
            residual: Use residual convolutional blocks in both encoders.
            scorer_batchnorm: Add ``BatchNorm1d`` after each hidden ``Linear``
                of the scorer MLP (Linear -> BatchNorm -> ReLU -> Dropout).

        The defaults reproduce the original MV1 (49 473 parameters with
        ``cov_feat_dim=0``, 49 857 with ``cov_feat_dim=6``), including
        parameter names and initialisation order.
        """
        super().__init__()
        self._context_embed_dim = context_embed_dim
        self._branch_embed_dim  = branch_embed_dim
        self._cov_feat_dim      = cov_feat_dim
        self._hidden_dim        = hidden_dim
        self._dropout           = dropout
        self._width_mult        = width_mult
        self._n_blocks          = n_blocks
        self._residual          = residual
        self._scorer_batchnorm  = scorer_batchnorm

        encoder_kwargs = {"dropout": dropout, "width_mult": width_mult,
                          "n_blocks": n_blocks, "residual": residual}
        self.context_encoder = DNASequenceEncoder(embed_dim=context_embed_dim, **encoder_kwargs)
        self.branch_encoder  = DNASequenceEncoder(embed_dim=branch_embed_dim,  **encoder_kwargs)

        # A single scorer shared across both branches preserves Siamese symmetry:
        # branch_a and branch_b receive identical treatment with identical weights.
        mlp_in_dim = context_embed_dim + branch_embed_dim + cov_feat_dim

        scorer_layers: list[nn.Module] = []
        for d_in, d_out in [(mlp_in_dim, hidden_dim), (hidden_dim, hidden_dim // 2)]:
            scorer_layers.append(nn.Linear(d_in, d_out))
            if scorer_batchnorm:
                scorer_layers.append(nn.BatchNorm1d(d_out))
            scorer_layers += [nn.ReLU(), nn.Dropout(dropout)]
        scorer_layers.append(nn.Linear(hidden_dim // 2, 1))
        self.scorer = nn.Sequential(*scorer_layers)

    @classmethod
    def from_hparams(cls, hparams: dict) -> "Mv1JunctionModel":
        """Build a model from a checkpoint's ``model_hparams`` dict.

        Keys missing from ``hparams`` (checkpoints saved before a key was
        added) fall back to the constructor defaults, i.e. the original
        MV1 architecture.

        Args:
            hparams: Dict as returned by :meth:`hparams`.

        Returns:
            An untrained model with the stored architecture.
        """
        known = {"context_embed_dim", "branch_embed_dim", "cov_feat_dim", "hidden_dim",
                 "dropout", "width_mult", "n_blocks", "residual", "scorer_batchnorm"}
        return cls(**{k: v for k, v in hparams.items() if k in known})

    def model_name(self) -> str:
        """Return the model identifier string."""
        return "mv1_siamese_cnn"

    def hparams(self) -> dict:
        """Return a dict of hyperparameters for logging and checkpoint storage.

        Returns:
            Dictionary with keys ``context_embed_dim``, ``branch_embed_dim``,
            ``cov_feat_dim``, ``hidden_dim``, ``dropout``, ``width_mult``,
            ``n_blocks``, ``residual`` and ``scorer_batchnorm``.
        """
        return {
            "context_embed_dim": self._context_embed_dim,
            "branch_embed_dim":  self._branch_embed_dim,
            "cov_feat_dim":      self._cov_feat_dim,
            "hidden_dim":        self._hidden_dim,
            "dropout":           self._dropout,
            "width_mult":        self._width_mult,
            "n_blocks":          self._n_blocks,
            "residual":          self._residual,
            "scorer_batchnorm":  self._scorer_batchnorm,
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
