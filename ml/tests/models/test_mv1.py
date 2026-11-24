"""Tests for the MV1 architecture options (REQ033).

``_LegacyMv1`` is a verbatim copy of the encoder and scorer before the
tuning flags were added; the default model must match it exactly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.mv1 import (
    DNASequenceEncoder,
    Mv1JunctionModel,
    ResidualConvBlock,
    encoder_channels,
)


class _LegacyEncoder(nn.Module):
    """The MV1 encoder as it was before --width_mult/--n_blocks/--residual."""

    def __init__(self, embed_dim: int = 64, dropout: float = 0.3) -> None:
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
            nn.BatchNorm1d(embed_dim),
        )
        self.pool_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_drop(torch.mean(self.conv_block(x), dim=2))


class _LegacyMv1(nn.Module):
    """The MV1 model as it was before the tuning flags."""

    def __init__(self, embed_dim: int, cov_feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context_encoder = _LegacyEncoder(embed_dim, 0.3)
        self.branch_encoder = _LegacyEncoder(embed_dim, 0.3)
        self.scorer = nn.Sequential(
            nn.Linear(2 * embed_dim + cov_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, ctx: torch.Tensor, a: torch.Tensor, b: torch.Tensor, cov: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.context_encoder(ctx)
        a_emb = self.branch_encoder(a)
        b_emb = self.branch_encoder(b)
        sim_a = self.scorer(torch.cat([c, a_emb, cov], 1)).squeeze(1)
        sim_b = self.scorer(torch.cat([c, b_emb, cov], 1)).squeeze(1)
        return sim_a, sim_b


def _inputs(cov_dim: int, batch: int = 5) -> tuple[torch.Tensor, ...]:
    """Random one-hot-shaped inputs."""
    gen = torch.Generator().manual_seed(1)
    return (
        torch.rand(batch, 4, 150, generator=gen),
        torch.rand(batch, 4, 100, generator=gen),
        torch.rand(batch, 4, 100, generator=gen),
        torch.rand(batch, cov_dim, generator=gen),
    )


@pytest.mark.parametrize("cov_dim", [0, 6])
def test_default_model_matches_legacy_architecture(cov_dim: int) -> None:
    """Same seed -> same parameter names, values and outputs (train and eval mode)."""
    torch.manual_seed(3)
    legacy = _LegacyMv1(64, cov_dim, 64)
    torch.manual_seed(3)
    model = Mv1JunctionModel(cov_feat_dim=cov_dim)

    legacy_sd, new_sd = legacy.state_dict(), model.state_dict()
    assert list(new_sd) == list(legacy_sd)
    assert all(torch.equal(new_sd[k], legacy_sd[k]) for k in legacy_sd)

    inputs = _inputs(cov_dim)
    for train_mode in (True, False):
        legacy.train(train_mode)
        model.train(train_mode)
        torch.manual_seed(9)
        expected = legacy(*inputs)
        torch.manual_seed(9)
        got = model(*inputs)
        assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])


def test_default_parameter_counts() -> None:
    """49 473 parameters with --no_cov, 49 857 with coverage features."""
    assert Mv1JunctionModel(cov_feat_dim=0).count_parameters() == 49_473
    assert Mv1JunctionModel(cov_feat_dim=6).count_parameters() == 49_857


def test_legacy_checkpoint_loads_into_default_model(tmp_path) -> None:
    """A state dict saved by the old architecture loads strictly."""
    legacy = _LegacyMv1(64, 6, 64)
    model = Mv1JunctionModel.from_hparams(
        {
            "context_embed_dim": 64,
            "branch_embed_dim": 64,
            "cov_feat_dim": 6,
            "hidden_dim": 64,
        }
    )
    model.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize(
    ("n_blocks", "width_mult", "expected"),
    [
        (1, 1.0, [16]),
        (2, 1.0, [32, 16]),
        (3, 1.0, [32, 64, 16]),
        (4, 1.0, [32, 64, 64, 16]),
        (3, 1.5, [48, 96, 16]),
        (3, 0.5, [16, 32, 16]),
    ],
)
def test_encoder_channels(
    n_blocks: int, width_mult: float, expected: list[int]
) -> None:
    """width_mult scales hidden widths; the last block is always embed_dim."""
    assert encoder_channels(n_blocks, width_mult, 16) == expected


@pytest.mark.parametrize(("n_blocks", "width_mult"), [(0, 1.0), (3, 0.0), (3, -1.0)])
def test_encoder_channels_rejects_invalid(n_blocks: int, width_mult: float) -> None:
    """n_blocks < 1 and width_mult <= 0 are errors."""
    with pytest.raises(ValueError):
        encoder_channels(n_blocks, width_mult, 16)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("n_blocks", [1, 2, 5])
def test_encoder_variants_output_embed_dim(n_blocks: int, residual: bool) -> None:
    """Every block count and residual setting maps (B, 4, L) to (B, embed_dim)."""
    enc = DNASequenceEncoder(
        embed_dim=16, width_mult=0.5, n_blocks=n_blocks, residual=residual
    )
    convs = [m for m in enc.conv_block.modules() if isinstance(m, nn.Conv1d)]
    assert len([c for c in convs if c.kernel_size != (1,)]) == n_blocks
    assert enc(torch.rand(3, 4, 37)).shape == (3, 16)


def test_residual_block_skip() -> None:
    """Identity skip for equal widths, 1x1 conv otherwise; output = body + skip."""
    same = ResidualConvBlock(8, 8, 3)
    assert isinstance(same.skip, nn.Identity)
    proj = ResidualConvBlock(4, 8, 5)
    assert isinstance(proj.skip, nn.Conv1d) and proj.skip.kernel_size == (1,)
    proj.eval()
    x = torch.rand(2, 4, 11)
    assert torch.allclose(proj(x), proj.body(x) + proj.skip(x))


def test_scorer_batchnorm_flag() -> None:
    """--batchnorm adds BatchNorm1d after each hidden Linear of the scorer only."""
    plain = Mv1JunctionModel(cov_feat_dim=0)
    with_bn = Mv1JunctionModel(cov_feat_dim=0, scorer_batchnorm=True)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in plain.scorer)
    kinds = [type(m) for m in with_bn.scorer]
    assert kinds == [
        nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
        nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
        nn.Linear,
    ]  # fmt: skip
    with_bn.eval()
    sim_a, sim_b = with_bn(*_inputs(0))
    assert sim_a.shape == sim_b.shape == (5,)


def test_hparams_round_trip_through_from_hparams() -> None:
    """from_hparams(model.hparams()) rebuilds the same architecture."""
    model = Mv1JunctionModel(
        context_embed_dim=16,
        branch_embed_dim=16,
        cov_feat_dim=6,
        hidden_dim=8,
        width_mult=1.5,
        n_blocks=4,
        residual=True,
        scorer_batchnorm=True,
    )
    rebuilt = Mv1JunctionModel.from_hparams(model.hparams())
    assert rebuilt.hparams() == model.hparams()
    rebuilt.load_state_dict(model.state_dict())
