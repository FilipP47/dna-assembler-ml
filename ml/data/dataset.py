"""PyTorch Dataset for the Siamese junction-resolution network.

Optimised for Polars DataFrames and RAM-efficient one-hot encoding
(sequences are stored as raw strings and encoded on-the-fly in ``__getitem__``).
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Dict, List, NamedTuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


class RawSample(NamedTuple):
    """Lightweight container that stores raw sequence strings instead of tensors.

    Attributes:
        context_seq: Sequence of the junction context node.
        branch_a_seq: Sequence of branch A.
        branch_b_seq: Sequence of branch B.
        cov_feat: Pre-computed coverage feature tensor.
        label: 0 if branch_a is correct, 1 if branch_b is correct.
        junction_id: Identifier of the source junction.
    """
    context_seq: str
    branch_a_seq: str
    branch_b_seq: str
    cov_feat: torch.Tensor
    label: int
    junction_id: int


def one_hot(seq: str, length: int) -> torch.Tensor:
    """Encode a DNA sequence as a one-hot tensor of shape ``(4, length)``.

    Positions beyond ``len(seq)`` are left as zero (padding).

    Args:
        seq: DNA sequence string (A/C/G/T characters).
        length: Target tensor width; sequences longer than this are truncated.

    Returns:
        Float32 tensor of shape ``(4, length)``.
    """
    t = torch.zeros(4, length, dtype=torch.float32)
    for i, base in enumerate(seq[:length]):
        idx = BASE_TO_IDX.get(base.upper())
        if idx is not None:
            t[idx, i] = 1.0
    return t


def coverage_features(row: dict) -> torch.Tensor:
    """Build a coverage feature vector from a single Polars row dictionary.

    Features (in order):

    1. ``cov_ratio_max`` - dominant-branch coverage ratio.
    2. ``total_reads`` - log-normalised total read count.
    3. ``branch_{A,C,G,T}_cov`` - log-normalised per-branch coverage (4 values).

    Args:
        row: A dictionary produced by ``pl.DataFrame.iter_rows(named=True)``.
            Null values are represented as ``None``.

    Returns:
        Float32 tensor of shape ``(6,)``.
    """
    feats = []

    feats.append(float(row.get("cov_ratio_max") or 0.0))

    total = float(row.get("total_reads") or 0.0)
    feats.append(np.log1p(total) / 10.0)

    for base in BASES:
        cov = row.get(f"branch_{base}_cov")
        # Polars returns None for nulls, unlike pandas NA
        feats.append(np.log1p(float(cov)) / 10.0 if cov is not None else 0.0)

    return torch.tensor(feats, dtype=torch.float32)


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Args:
        seq: DNA sequence string.

    Returns:
        Reverse-complemented sequence string.
    """
    return seq.translate(_COMPLEMENT)[::-1]


class JunctionDataset(Dataset):
    """PyTorch Dataset that generates ``(branch_a, branch_b)`` pairs for the Siamese network.

    One-hot encoding is deferred to ``__getitem__`` to minimise RAM usage.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        context_len: int = 100,
        branch_len: int = 100,
        augment: bool = False,
        use_cov_feat: bool = True,
        pair_seed: int = 0,
    ):
        """Initialise the dataset and build all sample pairs.

        Args:
            df: Polars DataFrame with one row per junction (output of
                ``load_training_csv``).
            context_len: Maximum context sequence length in base pairs; longer
                sequences are truncated during encoding.
            branch_len: Maximum branch sequence length in base pairs.
            augment: When ``True``, each correct/wrong pair is extended with
                three additional augmented variants (swapped positions and
                reverse-complement strands).
            use_cov_feat: When ``False``, coverage features are replaced with
                an empty tensor (``cov_feat_dim = 0``).
            pair_seed: Seed for the per-pair position assignment (which of the
                correct/wrong sequences lands on ``branch_a``). The assignment
                is derived deterministically from ``(pair_seed, junction_id,
                wrong-branch index)``, so it is balanced in expectation yet
                reproducible across runs and worker counts, independent of the
                ``augment`` flag.
        """
        self.context_len  = context_len
        self.branch_len   = branch_len
        self.augment      = augment
        self.use_cov_feat = use_cov_feat
        self.pair_seed    = pair_seed

        self.samples: List[RawSample] = []
        self._build_pairs(df)

        logger.info(
            f"JunctionDataset: {len(self.samples)} pairs built from {df.height} junctions"
            f" (augment={augment})"
        )

    def _position_swapped(self, junction_id: int, wrong_idx: int) -> bool:
        """Decide whether the correct branch should start on ``branch_b``.

        The choice is a deterministic function of ``(pair_seed, junction_id,
        wrong_idx)`` rather than of a shared iteration-order RNG state, so it is
        stable across runs, DataLoader worker counts, and row ordering.

        Args:
            junction_id: Identifier of the source junction.
            wrong_idx: Index of the wrong branch within the junction's list of
                wrong branches (0 when there is only one).

        Returns:
            ``True`` if the base pair should place the correct branch on
            ``branch_b`` (base label 1), ``False`` to keep it on ``branch_a``
            (base label 0).
        """
        key = f"{self.pair_seed}:{junction_id}:{wrong_idx}".encode("ascii")
        digest = hashlib.sha256(key).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return rng.random() < 0.5

    def _build_pairs(self, df: pl.DataFrame) -> None:
        """Populate ``self.samples`` by iterating over the Polars DataFrame.

        Args:
            df: Polars DataFrame with junction data.
        """
        for row in df.iter_rows(named=True):
            gt_base = str(row.get("ground_truth") or "NA").upper()
            if gt_base not in BASES:
                continue

            context_seq = str(row.get("context_seq") or "")
            junction_id = int(row.get("junction_id") or 0)

            correct_seq = str(row.get(f"branch_{gt_base}_seq") or "")
            if not correct_seq or correct_seq == "NA":
                continue

            wrong_seqs = []
            for base in BASES:
                if base == gt_base:
                    continue
                seq = str(row.get(f"branch_{base}_seq") or "")
                if seq and seq != "NA":
                    wrong_seqs.append(seq)

            if not wrong_seqs:
                continue

            cov_tensor = coverage_features(row) if self.use_cov_feat else torch.zeros(0)

            for wrong_idx, wrong_seq in enumerate(wrong_seqs):
                # Assign which sequence starts on branch_a for every dataset, not
                # only augment=True. Without this an augment=False set (val in
                # train.py, test in evaluate.py) would be 100% label=0 instead
                # of balanced in expectation.
                if self._position_swapped(junction_id, wrong_idx):
                    a_seq, b_seq, base_label = wrong_seq, correct_seq, 1
                else:
                    a_seq, b_seq, base_label = correct_seq, wrong_seq, 0

                # Base pair, in the drawn position.
                self.samples.append(
                    RawSample(context_seq, a_seq, b_seq, cov_tensor, base_label, junction_id)
                )

                if self.augment:
                    swap_label = 1 - base_label

                    # Position swap of the base pair.
                    self.samples.append(
                        RawSample(context_seq, b_seq, a_seq, cov_tensor, swap_label, junction_id)
                    )

                    # Reverse-complement of the base pair.
                    self.samples.append(
                        RawSample(
                            reverse_complement(context_seq),
                            reverse_complement(a_seq),
                            reverse_complement(b_seq),
                            cov_tensor,
                            base_label,
                            junction_id
                        )
                    )

                    # Reverse-complement of the position-swapped pair.
                    self.samples.append(
                        RawSample(
                            reverse_complement(context_seq),
                            reverse_complement(b_seq),
                            reverse_complement(a_seq),
                            cov_tensor,
                            swap_label,
                            junction_id
                        )
                    )

    def __len__(self) -> int:
        """Return the total number of sample pairs in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single encoded sample dict.

        One-hot encoding is performed here (on-the-fly) to avoid storing
        large float tensors in memory between batches.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary with keys ``context``, ``branch_a``, ``branch_b``,
            ``cov_feat``, ``label``, and ``junction_id``.
        """
        sample = self.samples[idx]

        return {
            "context":  one_hot(sample.context_seq, self.context_len),
            "branch_a": one_hot(sample.branch_a_seq, self.branch_len),
            "branch_b": one_hot(sample.branch_b_seq, self.branch_len),
            "cov_feat": sample.cov_feat,
            "label":    torch.tensor(sample.label, dtype=torch.long),
            "junction_id": torch.tensor(sample.junction_id, dtype=torch.long),
        }


def collate_junction_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-sample dicts into a batch of tensors.

    Args:
        batch: List of dicts as returned by ``JunctionDataset.__getitem__``.

    Returns:
        Dictionary with the same keys, where each value is a stacked tensor
        with leading batch dimension.
    """
    return {
        "context":     torch.stack([b["context"]  for b in batch]),
        "branch_a":    torch.stack([b["branch_a"] for b in batch]),
        "branch_b":    torch.stack([b["branch_b"] for b in batch]),
        "cov_feat":    torch.stack([b["cov_feat"] for b in batch]),
        "label":       torch.stack([b["label"]    for b in batch]),
        "junction_id": torch.stack([b["junction_id"] for b in batch]),
    }
